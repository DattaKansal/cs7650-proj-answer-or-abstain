"""MMLU-Pro loading and stratified train/val/test splits.

Usage:
    python -m src.data            # download MMLU-Pro and write data/mmlu_pro_splits.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATASET = "TIGER-Lab/MMLU-Pro"
MAX_OPTIONS = 10
MATH_HEAVY = frozenset({"math", "physics", "chemistry", "engineering"})
SPLIT_FRACTIONS = (("train", 0.6), ("val", 0.2), ("test", 0.2))
DEFAULT_SPLIT_PATH = Path("data/mmlu_pro_splits.json")
DEFAULT_SEED = 0


def clean_options(options: list[str], answer_index: int) -> tuple[list[str], int]:
    """Drop "N/A" padding options and return the options with the remapped answer index."""
    kept = [(i, o) for i, o in enumerate(options) if o.strip() != "N/A"]
    positions = [pos for pos, (i, _) in enumerate(kept) if i == answer_index]
    if not positions:
        raise ValueError("the correct option was removed as N/A padding")
    return [o for _, o in kept], positions[0]


def load_mmlu_pro() -> list[dict]:
    """Return the MMLU-Pro test set as a list of question dicts."""
    from datasets import load_dataset

    questions = []
    for row in load_dataset(DATASET, split="test"):
        options, answer_index = clean_options(row["options"], row["answer_index"])
        if not 2 <= len(options) <= MAX_OPTIONS:
            raise ValueError(f"question {row['question_id']} has {len(options)} options")
        questions.append(
            {
                "qid": int(row["question_id"]),
                "question": row["question"],
                "options": options,
                "answer_index": answer_index,
                "category": row["category"],
                "math_heavy": row["category"] in MATH_HEAVY,
            }
        )
    return questions


def make_splits(questions: list[dict], seed: int = DEFAULT_SEED) -> dict[int, str]:
    """Assign each question to train/val/test (60/20/20), stratified by category."""
    by_category: dict[str, list[int]] = defaultdict(list)
    for q in questions:
        by_category[q["category"]].append(q["qid"])

    rng = np.random.default_rng(seed)
    assignment: dict[int, str] = {}
    for category in sorted(by_category):
        qids = np.array(sorted(by_category[category]))
        qids = qids[rng.permutation(len(qids))]
        n_train = round(SPLIT_FRACTIONS[0][1] * len(qids))
        n_val = round(SPLIT_FRACTIONS[1][1] * len(qids))
        for qid in qids[:n_train]:
            assignment[int(qid)] = "train"
        for qid in qids[n_train : n_train + n_val]:
            assignment[int(qid)] = "val"
        for qid in qids[n_train + n_val :]:
            assignment[int(qid)] = "test"
    return assignment


def save_splits(assignment: dict[int, str], path: Path = DEFAULT_SPLIT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in sorted(assignment.items())}))


def load_splits(path: Path = DEFAULT_SPLIT_PATH) -> dict[int, str]:
    return {int(k): v for k, v in json.loads(path.read_text()).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_SPLIT_PATH)
    args = parser.parse_args()

    questions = load_mmlu_pro()
    assignment = make_splits(questions, seed=args.seed)
    save_splits(assignment, args.out)

    print(f"{len(questions)} questions -> {args.out}")
    print("split sizes:", dict(Counter(assignment.values())))
    print("options per question:", dict(sorted(Counter(len(q['options']) for q in questions).items())))
    per_category = Counter(q["category"] for q in questions)
    for category, n in sorted(per_category.items()):
        print(f"  {category:20s} {n:5d}{'  (math-heavy)' if category in MATH_HEAVY else ''}")


if __name__ == "__main__":
    main()
