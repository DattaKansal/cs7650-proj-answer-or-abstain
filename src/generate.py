"""P2: generate "Answer: <letter>, Confidence: <0-100>" replies and parse them.

Greedy decoding of a few tokens per question. Output is written in resumable chunks under
<out>/<model>/p2/, one row per question with the raw reply and the parsed fields.

Usage:
    python -m src.generate --model Qwen/Qwen3-1.7B
    python -m src.generate --model Qwen/Qwen3-1.7B --limit 200   # pilot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from . import data, infer, prompts

MAX_NEW_TOKENS = 12


def parse_reply(reply: str, n_options: int, prefill: str) -> tuple[int, float, bool]:
    """Return (answer index, confidence, parsed). Unparseable replies get (-1, 0.0, False)."""
    index, confidence = prompts.parse_p2(prefill + reply, n_options)
    if index is None:
        return -1, 0.0, False
    return index, confidence, True


@torch.inference_mode()
def run(args: argparse.Namespace) -> Path:
    device = args.device or infer.pick_device()
    dtype = infer.pick_dtype(args.dtype, device)
    out_dir = args.out / args.model.replace("/", "__") / "p2"
    if args.limit:
        out_dir = out_dir.with_name(out_dir.name + f"_pilot{args.limit}")
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = infer.select_questions(data.load_mmlu_pro(), args.limit, args.seed)
    tokenizer, model = infer.load_model(args.model, device, dtype)
    texts = [
        infer.render_chat(tokenizer, prompts.p2_prompt(q["question"], q["options"]), args.prefill)
        for q in questions
    ]
    lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
    order = sorted(range(len(questions)), key=lambda i: -lengths[i])

    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "prompt": "p2",
                "prefill": args.prefill,
                "max_new_tokens": MAX_NEW_TOKENS,
                "dtype": str(dtype),
                "n_questions": len(questions),
                "example_prompt": texts[order[-1]],
            },
            indent=2,
        )
    )

    chunks = [order[i : i + args.chunk_size] for i in range(0, len(order), args.chunk_size)]
    for chunk_index, chunk in enumerate(tqdm(chunks, desc=out_dir.name)):
        table_path = out_dir / f"part-{chunk_index:05d}.parquet"
        if table_path.exists():
            continue
        replies = []
        for start in range(0, len(chunk), args.batch_size):
            batch = chunk[start : start + args.batch_size]
            enc = infer.encode(tokenizer, [texts[i] for i in batch], device)
            generated = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = generated[:, enc["input_ids"].shape[1] :]
            replies += tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        parsed = [
            parse_reply(reply, len(questions[i]["options"]), args.prefill)
            for reply, i in zip(replies, chunk)
        ]
        frame = pd.DataFrame(
            {
                "qid": [questions[i]["qid"] for i in chunk],
                "reply": replies,
                "answer_choice": [p[0] for p in parsed],
                "confidence": [p[1] for p in parsed],
                "parsed": [p[2] for p in parsed],
                "n_options": [len(questions[i]["options"]) for i in chunk],
                "answer_index": [questions[i]["answer_index"] for i in chunk],
                "category": [questions[i]["category"] for i in chunk],
            }
        )
        tmp = table_path.with_suffix(".tmp")
        frame.to_parquet(tmp, index=False)
        tmp.rename(table_path)
    return out_dir


def load_run(out_dir: Path) -> pd.DataFrame:
    frame = pd.concat(
        [pd.read_parquet(p) for p in sorted(out_dir.glob("part-*.parquet"))], ignore_index=True
    )
    frame["correct"] = frame["parsed"] & (frame["answer_choice"] == frame["answer_index"])
    return frame


def print_summary(out_dir: Path) -> None:
    frame = load_run(out_dir)
    print(f"\n{out_dir}  ({len(frame)} questions)")
    print(f"  parsed replies      : {frame['parsed'].mean():.4f}")
    print(f"  accuracy            : {frame['correct'].mean():.4f}")
    print(f"  mean confidence     : {frame.loc[frame['parsed'], 'confidence'].mean():.4f}")
    values = frame.loc[frame["parsed"], "confidence"].round(2).value_counts().head(6)
    print("  most common confidences:", {float(k): int(v) for k, v in values.items()})
    print("  example replies     :", frame["reply"].head(5).tolist())
    unparsed = frame.loc[~frame["parsed"], "reply"].head(5).tolist()
    if unparsed:
        print("  unparsed examples   :", unparsed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--prefill", default=infer.DEFAULT_PREFILL)
    parser.add_argument("--limit", type=int, default=None, help="random subset size for pilots")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if (args.device or infer.pick_device()) != "cuda":
        raise SystemExit("Run this on a CUDA GPU (e.g. Colab); it is not meant for laptops.")
    print_summary(run(args))


if __name__ == "__main__":
    main()
