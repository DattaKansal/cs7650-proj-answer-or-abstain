"""Single-forward-pass inference: answer-letter logits and last-token hidden states.

For each question we read the next-token logits at the answer position and keep one logit
per letter A-K. Optionally we also save the last-token residual stream at every 4th layer.
Output is written in resumable chunks under <out>/<model>/<prompt tag>/.

Usage:
    python -m src.infer --model Qwen/Qwen3-1.7B --prompt neutral --save-hidden
    python -m src.infer --model Qwen/Qwen3-1.7B --prompt p1 --t 0.75
    python -m src.infer --model Qwen/Qwen3-1.7B --prompt neutral --limit 200   # pilot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from . import data, prompts

LAYER_STRIDE = 4
LOGIT_COLUMNS = [f"logit_{letter}" for letter in prompts.ALL_LETTERS]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        name = {"cuda": "bf16", "mps": "fp16"}.get(device, "fp32")
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def prompt_tag(kind: str, t: float | None) -> str:
    return kind if kind == "neutral" else f"p1_t{t:g}"


def build_prompt(question: dict, kind: str, t: float | None) -> str:
    if kind == "neutral":
        return prompts.neutral_prompt(question["question"], question["options"])
    if kind == "p1":
        return prompts.p1_prompt(question["question"], question["options"], t)
    raise ValueError(f"unknown prompt kind: {kind}")


DEFAULT_PREFILL = "Answer:"


def render_chat(tokenizer, user_text: str, prefill: str = DEFAULT_PREFILL) -> str:
    """Wrap the prompt in the model's chat template, ending where the answer letter begins.

    `prefill` starts the assistant's reply so the next token is the answer letter. Without
    it, models often open calculation questions with reasoning text instead of a letter.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Qwen3: skip the reasoning block; ignored by other templates
    )
    return text + prefill


def load_model(model_name: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"  # so position -1 is always the answer position
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    return tokenizer, model


def encode(tokenizer, texts: list[str], device: str):
    # The chat template already added the special tokens.
    return tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)


def letter_token_ids(tokenizer) -> list[list[int]]:
    """Token ids for each letter A-K, covering the bare and space-prefixed variants."""
    ids = []
    for letter in prompts.ALL_LETTERS:
        variants = set()
        for text in (letter, " " + letter):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) == 1:
                variants.add(tokens[0])
        if not variants:
            raise ValueError(f"no single-token encoding for letter {letter}")
        ids.append(sorted(variants))
    return ids


def letter_logits(next_token_logits: torch.Tensor, token_ids: list[list[int]]):
    """Collapse vocabulary logits to one logit per letter, plus total probability on letters."""
    per_letter = torch.stack(
        [torch.logsumexp(next_token_logits[:, ids], dim=-1) for ids in token_ids], dim=-1
    )
    log_total = torch.logsumexp(next_token_logits, dim=-1)
    letter_mass = torch.exp(torch.logsumexp(per_letter, dim=-1) - log_total)
    return per_letter, letter_mass


def select_questions(questions: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is None or limit >= len(questions):
        return questions
    keep = np.random.default_rng(seed).choice(len(questions), size=limit, replace=False)
    return [questions[i] for i in sorted(keep)]


@torch.inference_mode()
def run(args: argparse.Namespace) -> Path:
    device = args.device or pick_device()
    dtype = pick_dtype(args.dtype, device)
    out_dir = args.out / args.model.replace("/", "__") / prompt_tag(args.prompt, args.t)
    if args.limit:
        out_dir = out_dir.with_name(out_dir.name + f"_pilot{args.limit}")
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = select_questions(data.load_mmlu_pro(), args.limit, args.seed)

    tokenizer, model = load_model(args.model, device, dtype)

    token_ids = letter_token_ids(tokenizer)
    n_layers = model.config.num_hidden_layers
    layers = list(range(LAYER_STRIDE, n_layers + 1, LAYER_STRIDE))

    texts = [
        render_chat(tokenizer, build_prompt(q, args.prompt, args.t), args.prefill)
        for q in questions
    ]
    lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
    order = sorted(range(len(questions)), key=lambda i: -lengths[i])  # longest first

    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "prompt": args.prompt,
                "t": args.t,
                "prefill": args.prefill,
                "dtype": str(dtype),
                "device": device,
                "n_questions": len(questions),
                "hidden_layers": layers if args.save_hidden else [],
                "letter_token_ids": token_ids,
                "example_prompt": texts[order[-1]],
            },
            indent=2,
        )
    )

    chunks = [order[i : i + args.chunk_size] for i in range(0, len(order), args.chunk_size)]
    for chunk_index, chunk in enumerate(tqdm(chunks, desc=out_dir.name)):
        table_path = out_dir / f"part-{chunk_index:05d}.parquet"
        hidden_path = out_dir / f"hidden-{chunk_index:05d}.npy"
        if table_path.exists() and (hidden_path.exists() or not args.save_hidden):
            continue

        logits_out, mass_out, hidden_out = [], [], []
        for start in range(0, len(chunk), args.batch_size):
            batch = chunk[start : start + args.batch_size]
            enc = encode(tokenizer, [texts[i] for i in batch], device)
            out = model(**enc, output_hidden_states=args.save_hidden)
            per_letter, mass = letter_logits(out.logits[:, -1, :].float(), token_ids)
            logits_out.append(per_letter.cpu().numpy())
            mass_out.append(mass.cpu().numpy())
            if args.save_hidden:
                last = torch.stack([out.hidden_states[l][:, -1, :] for l in layers], dim=1)
                hidden_out.append(last.float().cpu().numpy().astype(np.float16))

        frame = pd.DataFrame(np.concatenate(logits_out), columns=LOGIT_COLUMNS)
        frame.insert(0, "qid", [questions[i]["qid"] for i in chunk])
        frame["letter_mass"] = np.concatenate(mass_out)
        frame["n_options"] = [len(questions[i]["options"]) for i in chunk]
        frame["answer_index"] = [questions[i]["answer_index"] for i in chunk]
        frame["category"] = [questions[i]["category"] for i in chunk]

        if args.save_hidden:  # written first, so a finished table implies finished hidden states
            tmp = hidden_path.with_suffix(".tmp.npy")
            np.save(tmp, np.concatenate(hidden_out))
            tmp.rename(hidden_path)
        tmp = table_path.with_suffix(".tmp")
        frame.to_parquet(tmp, index=False)
        tmp.rename(table_path)

    return out_dir


def load_run(out_dir: Path) -> pd.DataFrame:
    """Load a finished run and add the model's choice among the valid letters."""
    frame = pd.concat(
        [pd.read_parquet(p) for p in sorted(out_dir.glob("part-*.parquet"))], ignore_index=True
    )
    meta = json.loads((out_dir / "meta.json").read_text())
    logits = frame[LOGIT_COLUMNS].to_numpy(dtype=np.float64)
    n_options = frame["n_options"].to_numpy()

    idk_index = len(prompts.ANSWER_LETTERS)
    valid = np.arange(logits.shape[1])[None, :] < n_options[:, None]
    if meta["prompt"] == "p1":
        valid[:, idk_index] = True
    masked = np.where(valid, logits, -np.inf)

    frame["choice"] = masked.argmax(axis=1)
    frame["abstain"] = frame["choice"] == idk_index
    # The answer letter the model prefers, ignoring IDK (equals `choice` unless it abstained).
    frame["answer_choice"] = masked[:, :idk_index].argmax(axis=1)
    frame["correct"] = frame["answer_choice"] == frame["answer_index"]
    # Softmax over the valid answer letters only (IDK excluded): the P3 confidence.
    answer_logits = masked[:, :idk_index]
    shifted = answer_logits - answer_logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    frame["top_prob"] = probs.max(axis=1)
    return frame


def print_summary(out_dir: Path) -> None:
    frame = load_run(out_dir)
    print(f"\n{out_dir}  ({len(frame)} questions)")
    print(f"  non-finite logits        : {int((~np.isfinite(frame[LOGIT_COLUMNS])).sum().sum())}")
    print(f"  mean prob mass on letters: {frame['letter_mass'].mean():.4f}")
    print(f"  min prob mass on letters : {frame['letter_mass'].min():.4f}")
    print(f"  abstention rate          : {frame['abstain'].mean():.4f}")
    print(f"  accuracy of answer letter: {frame['correct'].mean():.4f}")
    print(f"  mean top answer prob     : {frame['top_prob'].mean():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--prompt", choices=["neutral", "p1"], default="neutral")
    parser.add_argument("--t", type=float, default=None, help="confidence target for p1")
    parser.add_argument("--save-hidden", action="store_true")
    parser.add_argument("--prefill", default=DEFAULT_PREFILL, help="start of the assistant reply")
    parser.add_argument("--limit", type=int, default=None, help="random subset size for pilots")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.prompt == "p1" and args.t is None:
        parser.error("--t is required with --prompt p1")
    if args.prompt == "neutral":
        args.t = None

    print_summary(run(args))


if __name__ == "__main__":
    main()
