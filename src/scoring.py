"""Penalized scoring, baselines, regret, risk-coverage and calibration metrics.

Conventions: `correct` and `abstain` are boolean arrays with one entry per question.
`correct[i]` says whether the answer the policy would give is right; it is ignored where
`abstain[i]` is True. `reference_correct` is correctness under the neutral prompt, which
defines the oracle.
"""
from __future__ import annotations

import numpy as np

from .prompts import penalty


def _as_bool(x) -> np.ndarray:
    return np.asarray(x, dtype=bool)


def per_question_scores(correct, abstain, t: float) -> np.ndarray:
    """+1 if correct, -t/(1-t) if wrong, 0 if abstained."""
    correct, abstain = _as_bool(correct), _as_bool(abstain)
    scores = np.where(correct, 1.0, -penalty(t))
    return np.where(abstain, 0.0, scores)


def penalized_score(correct, abstain, t: float) -> float:
    return float(per_question_scores(correct, abstain, t).mean())


def abstention_rate(abstain) -> float:
    return float(_as_bool(abstain).mean())


def selective_accuracy(correct, abstain) -> float:
    """Accuracy on answered questions; NaN if every question was abstained on."""
    correct, abstain = _as_bool(correct), _as_bool(abstain)
    answered = ~abstain
    return float(correct[answered].mean()) if answered.any() else float("nan")


def threshold_policy(confidence, t: float) -> np.ndarray:
    """Abstain where confidence is below t (so nothing is abstained on at t = 0)."""
    return np.asarray(confidence, dtype=float) < t


def oracle_score(reference_correct) -> float:
    """Answer exactly the questions the model gets right: +1 each, 0 otherwise, for any t."""
    return float(_as_bool(reference_correct).mean())


def always_answer_score(reference_correct, t: float) -> float:
    reference_correct = _as_bool(reference_correct)
    return penalized_score(reference_correct, np.zeros_like(reference_correct), t)


def random_abstention_score(reference_correct, rate: float, t: float) -> float:
    """Expected score when abstaining uniformly at random on a fraction `rate` of questions."""
    return (1 - rate) * always_answer_score(reference_correct, t)


def regret(score: float, reference_correct) -> float:
    return oracle_score(reference_correct) - score


def summarize(correct, abstain, reference_correct, t: float) -> dict[str, float]:
    """All headline numbers for one policy at one threshold."""
    score = penalized_score(correct, abstain, t)
    rate = abstention_rate(abstain)
    return {
        "t": t,
        "score": score,
        "abstention_rate": rate,
        "selective_accuracy": selective_accuracy(correct, abstain),
        "always_answer_score": always_answer_score(reference_correct, t),
        "always_abstain_score": 0.0,
        "random_matched_score": random_abstention_score(reference_correct, rate, t),
        "oracle_score": oracle_score(reference_correct),
        "regret": regret(score, reference_correct),
    }


def risk_coverage(confidence, correct) -> tuple[np.ndarray, np.ndarray]:
    """Coverage and selective risk as questions are answered from most to least confident."""
    confidence = np.asarray(confidence, dtype=float)
    correct = _as_bool(correct)
    order = np.argsort(-confidence, kind="stable")
    errors = np.cumsum(~correct[order])
    answered = np.arange(1, len(correct) + 1)
    return answered / len(correct), errors / answered


def aurc(confidence, correct) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    _, risk = risk_coverage(confidence, correct)
    return float(risk.mean())


def expected_calibration_error(confidence, correct, n_bins: int = 10) -> float:
    """ECE with equal-width confidence bins."""
    confidence = np.asarray(confidence, dtype=float)
    correct = _as_bool(correct)
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)
