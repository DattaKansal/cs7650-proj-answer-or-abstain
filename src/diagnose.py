"""Check where the model puts next-token probability at the answer position.

Compares assistant-reply prefills on a random subset: how much probability lands on answer
letters, and how accurate the letter-restricted choice is. Loads the model once.

Usage:
    python -m src.diagnose --model Qwen/Qwen3-1.7B --n 200
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from . import data, infer, prompts


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--show", type=int, default=6, help="questions to print top tokens for")
    parser.add_argument("--prefills", nargs="+", default=["", infer.DEFAULT_PREFILL])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    args = parser.parse_args()

    device = infer.pick_device()
    if device != "cuda":
        raise SystemExit("Run this on a CUDA GPU (e.g. Colab); it is not meant for laptops.")
    tokenizer, model = infer.load_model(args.model, device, infer.pick_dtype(args.dtype, device))
    token_ids = infer.letter_token_ids(tokenizer)
    print("letter token ids:", dict(zip(prompts.ALL_LETTERS, token_ids)))

    questions = infer.select_questions(data.load_mmlu_pro(), args.n, args.seed)
    n_options = np.array([len(q["options"]) for q in questions])
    gold = np.array([q["answer_index"] for q in questions])
    math_heavy = np.array([q["math_heavy"] for q in questions])
    valid = np.arange(len(prompts.ANSWER_LETTERS))[None, :] < n_options[:, None]

    for prefill in args.prefills:
        texts = [
            infer.render_chat(
                tokenizer, prompts.neutral_prompt(q["question"], q["options"]), prefill
            )
            for q in questions
        ]
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        logits = np.zeros((len(texts), len(prompts.ALL_LETTERS)))
        mass = np.zeros(len(texts))
        top_tokens: dict[int, list] = {}
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            enc = infer.encode(tokenizer, [texts[i] for i in batch], device)
            next_logits = model(**enc).logits[:, -1, :].float()
            per_letter, letter_mass = infer.letter_logits(next_logits, token_ids)
            logits[batch] = per_letter.cpu().numpy()
            mass[batch] = letter_mass.cpu().numpy()
            probs = next_logits.softmax(-1)
            for row, i in enumerate(batch):
                if i < args.show:
                    top = probs[row].topk(5)
                    top_tokens[i] = [
                        (tokenizer.decode(t), round(p.item(), 3))
                        for p, t in zip(top.values, top.indices)
                    ]

        answer_logits = np.where(valid, logits[:, : len(prompts.ANSWER_LETTERS)], -np.inf)
        correct = answer_logits.argmax(axis=1) == gold
        print(f"\n=== prefill {prefill!r} ===")
        print("prompt ends with:", repr(texts[0][-60:]))
        print(f"mean probability on letters : {mass.mean():.3f}")
        print(f"share of questions above 0.9: {(mass > 0.9).mean():.3f}")
        print(f"accuracy (all / math-heavy / other): {correct.mean():.3f} / "
              f"{correct[math_heavy].mean():.3f} / {correct[~math_heavy].mean():.3f}")
        print(f"letter mass (math-heavy / other)   : {mass[math_heavy].mean():.3f} / "
              f"{mass[~math_heavy].mean():.3f}")
        for i in sorted(top_tokens):
            q = questions[i]
            print(f"  [{q['category']}] gold={prompts.ANSWER_LETTERS[q['answer_index']]} "
                  f"top={top_tokens[i]}")


if __name__ == "__main__":
    main()
