"""Fit P3b and P4, score every policy on the test split, and write tables and figures.

Expects the runs produced by src.infer for one model: neutral (with hidden states) and
p1_t<t> for each confidence target.

Usage:
    python -m src.report --run-dir outputs/Qwen__Qwen3-1.7B
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, infer, policies, prompts, scoring

POLICY_LABELS = {
    "P1": "P1 prompt-only",
    "P3": "P3 answer probability",
    "P3b": "P3b + temperature",
    "P4": "P4 probe",
}
# Categorical slots 1-4 of the validated palette; baselines are neutral gray.
POLICY_COLORS = {"P1": "#2a78d6", "P3": "#eb6834", "P3b": "#1baf7a", "P4": "#eda100"}
POLICY_MARKERS = {"P1": "o", "P3": "s", "P3b": "^", "P4": "D"}
BASELINE_COLOR = "#6b6a63"
INK, MUTED, GRID = "#1f1f1d", "#6b6a63", "#e4e3dd"


def load_p1(run_dir: Path, t: float, qids: pd.Series) -> pd.DataFrame:
    frame = infer.load_run(run_dir / infer.prompt_tag("p1", t)).set_index("qid")
    return frame.loc[qids].reset_index()


def evaluate(run_dir: Path, seed: int = data.DEFAULT_SEED):
    neutral = infer.load_run(run_dir / "neutral")
    meta = json.loads((run_dir / "neutral" / "meta.json").read_text())
    hidden = policies.load_hidden(run_dir / "neutral")
    if len(hidden) != len(neutral):
        raise ValueError("hidden states and logits have different numbers of rows")

    assignment = data.make_splits(neutral[["qid", "category"]].to_dict("records"), seed=seed)
    split = neutral["qid"].map(assignment).to_numpy()
    train, val, test = (split == s for s in ("train", "val", "test"))
    reference = neutral["correct"].to_numpy()
    gold = neutral["answer_index"].to_numpy()
    math_heavy = neutral["category"].isin(data.MATH_HEAVY).to_numpy()

    logits = policies.answer_logits(neutral)
    temperature = policies.fit_temperature(logits[val], gold[val])
    probe_confidence, probe_table = policies.fit_probe(
        hidden, reference, train, val, meta["hidden_layers"]
    )
    confidence = {
        "P3": policies.top_probability(logits),
        "P3b": policies.top_probability(logits, temperature),
        "P4": probe_confidence,
    }

    subsets = {"all": test, "math_heavy": test & math_heavy, "other": test & ~math_heavy}
    rows, flips = [], {}
    for t in prompts.THRESHOLDS:
        p1 = load_p1(run_dir, t, neutral["qid"])
        decisions = {"P1": (p1["correct"].to_numpy(), p1["abstain"].to_numpy())}
        for name, conf in confidence.items():
            decisions[name] = (reference, scoring.threshold_policy(conf, t))
        flips[t] = float((p1["answer_choice"] != neutral["answer_choice"])[test].mean())
        for subset, mask in subsets.items():
            for name, (correct, abstain) in decisions.items():
                row = scoring.summarize(correct[mask], abstain[mask], reference[mask], t)
                rows.append({"policy": name, "subset": subset, "n": int(mask.sum()), **row})
    results = pd.DataFrame(rows)

    from sklearn.metrics import roc_auc_score

    signal_rows = [
        {
            "policy": name,
            "auroc": roc_auc_score(reference[test], conf[test]),
            "aurc": scoring.aurc(conf[test], reference[test]),
            "ece": scoring.expected_calibration_error(conf[test], reference[test]),
            "mean_confidence": float(conf[test].mean()),
        }
        for name, conf in confidence.items()
    ]
    summary = {
        "model": meta["model"],
        "n_test": int(test.sum()),
        "test_accuracy": float(reference[test].mean()),
        "temperature": temperature,
        "probe": probe_table[probe_table["selected"]].iloc[0].to_dict(),
        "p1_answer_flip_rate": flips,
    }
    curves = {name: scoring.risk_coverage(conf[test], reference[test]) for name, conf in confidence.items()}
    return results, pd.DataFrame(signal_rows), probe_table, summary, curves


def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _policy_lines(ax, results: pd.DataFrame, column: str) -> None:
    positions = range(len(prompts.THRESHOLDS))
    for name in POLICY_LABELS:
        rows = results[(results["policy"] == name) & (results["subset"] == "all")].sort_values("t")
        ax.plot(positions, rows[column], color=POLICY_COLORS[name], marker=POLICY_MARKERS[name],
                markersize=7, linewidth=2, label=POLICY_LABELS[name],
                markeredgecolor="white", markeredgewidth=1.5)
    ax.set_xticks(list(positions), [f"{t:g}" for t in prompts.THRESHOLDS])


def make_figures(results, probe_table, summary, curves, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = summary["model"].split("/")[-1]
    overall = results[results["subset"] == "all"]
    baselines = overall[overall["policy"] == "P1"].sort_values("t")
    positions = range(len(prompts.THRESHOLDS))

    fig, ax = plt.subplots(figsize=(6, 4))
    _policy_lines(ax, results, "abstention_rate")
    ax.plot(positions, [1 - summary["test_accuracy"]] * len(prompts.THRESHOLDS), color=BASELINE_COLOR,
            linestyle="--", linewidth=1.5, label="Oracle (abstain when wrong)")
    ax.set_ylim(0, 1)
    _style(ax, "Confidence target t", "Abstention rate", f"Abstention rate by target ({model}, MMLU-Pro test)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "abstention_vs_t.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    _policy_lines(ax, results, "score")
    ax.plot(positions, baselines["oracle_score"], color=BASELINE_COLOR, linestyle="--",
            linewidth=1.5, label="Oracle")
    ax.plot(positions, baselines["always_answer_score"], color=BASELINE_COLOR, linestyle=":",
            linewidth=1.5, label="Always answer")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_yticks([-8, -4, -2, -1, 0, 1], ["-8", "-4", "-2", "-1", "0", "1"])
    ax.minorticks_off()
    ax.axhline(0, color=MUTED, linewidth=0.8)
    _style(ax, "Confidence target t", "Penalized score per question (symlog)",
           f"Penalized score by target ({model}, MMLU-Pro test)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "score_vs_t.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (coverage, risk) in curves.items():
        ax.plot(coverage, risk, color=POLICY_COLORS[name], linewidth=2, label=POLICY_LABELS[name])
    ax.axhline(1 - summary["test_accuracy"], color=BASELINE_COLOR, linestyle=":", linewidth=1.5,
               label="Always answer")
    ax.set_xlim(0.02, 1)
    ax.set_ylim(0, 1)
    _style(ax, "Coverage (share of questions answered)", "Selective risk (error rate on answered)",
           f"Risk-coverage ({model}, MMLU-Pro test)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "risk_coverage.png", dpi=200)
    plt.close(fig)

    best_c = probe_table.loc[probe_table.groupby("layer")["val_log_loss"].idxmin()]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(best_c["layer"], best_c["val_auroc"], color=POLICY_COLORS["P4"], marker="D",
            markersize=7, linewidth=2, markeredgecolor="white", markeredgewidth=1.5)
    chosen = probe_table[probe_table["selected"]].iloc[0]
    ax.annotate(f"selected: layer {int(chosen['layer'])}", (chosen["layer"], chosen["val_auroc"]),
                textcoords="offset points", xytext=(0, 10), ha="center", color=INK, fontsize=8)
    ax.set_xticks(list(best_c["layer"]))
    _style(ax, "Layer", "Validation AUROC", f"Probe: predicting correctness by layer ({model})")
    fig.tight_layout()
    fig.savefig(out_dir / "probe_by_layer.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="default: <run-dir>/report")
    args = parser.parse_args()
    out_dir = args.out or args.run_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    results, signals, probe_table, summary, curves = evaluate(args.run_dir)
    results.to_csv(out_dir / "results.csv", index=False)
    signals.to_csv(out_dir / "signals.csv", index=False)
    probe_table.to_csv(out_dir / "probe_layers.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    make_figures(results, probe_table, summary, curves, out_dir)

    pd.set_option("display.width", 200)
    print(json.dumps(summary, indent=2))
    print("\nConfidence signals (test split):")
    print(signals.round(3).to_string(index=False))
    for subset in ("all", "math_heavy", "other"):
        print(f"\nPolicies on test split, subset = {subset}:")
        view = results[results["subset"] == subset]
        print(view[["policy", "t", "score", "abstention_rate", "selective_accuracy", "regret"]]
              .round(3).to_string(index=False))
    baseline = results[(results["subset"] == "all") & (results["policy"] == "P1")]
    print("\nBaselines (all):")
    print(baseline[["t", "always_answer_score", "random_matched_score", "oracle_score"]]
          .round(3).to_string(index=False))
    print(f"\nWrote tables and figures to {out_dir}")


if __name__ == "__main__":
    main()
