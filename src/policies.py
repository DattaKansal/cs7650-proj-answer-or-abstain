"""Confidence signals for the externally thresholded policies: P3, P3b and P4.

Each function returns a confidence in [0, 1] per question; the policy abstains where that
confidence is below the target t (see scoring.threshold_policy).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from . import infer, prompts

N_ANSWERS = len(prompts.ANSWER_LETTERS)


def answer_logits(frame: pd.DataFrame) -> np.ndarray:
    """Logits over answer letters A-J, with -inf where a question has fewer options."""
    logits = frame[infer.LOGIT_COLUMNS[:N_ANSWERS]].to_numpy(dtype=np.float64)
    valid = np.arange(N_ANSWERS)[None, :] < frame["n_options"].to_numpy()[:, None]
    return np.where(valid, logits, -np.inf)


def log_probs(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    return scaled - logsumexp(scaled, axis=1, keepdims=True)


def top_probability(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """P3 (temperature 1) and P3b (fitted temperature) confidence."""
    return np.exp(log_probs(logits, temperature).max(axis=1))


def nll(logits: np.ndarray, gold: np.ndarray, temperature: float) -> float:
    return float(-log_probs(logits, temperature)[np.arange(len(gold)), gold].mean())


def fit_temperature(logits: np.ndarray, gold: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): one scalar minimizing NLL of the gold answer."""
    result = minimize_scalar(
        lambda log_t: nll(logits, gold, float(np.exp(log_t))), bounds=(-3, 5), method="bounded"
    )
    return float(np.exp(result.x))


def load_hidden(run_dir: Path) -> np.ndarray:
    """Hidden states [questions, layers, width], in the same row order as infer.load_run."""
    parts = sorted(run_dir.glob("hidden-*.npy"))
    if not parts:
        raise FileNotFoundError(f"no hidden states in {run_dir}; rerun with --save-hidden")
    return np.concatenate([np.load(p) for p in parts]).astype(np.float32)


PROBE_REGULARIZATION = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)  # inverse strength C; smaller = stronger


def fit_probe(hidden, correct, train, val, layers, regularization=PROBE_REGULARIZATION):
    """P4: logistic-regression probes predicting correctness from one layer's hidden state.

    Fits one probe per (layer, C) on the train rows and picks the pair with the lowest
    validation log loss, since the policy thresholds the predicted probability directly.
    Returns the chosen probe's p(correct) for every question and the selection table.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(correct, dtype=int)
    rows, best = [], None
    for layer_position, layer in enumerate(layers):
        x = hidden[:, layer_position, :]
        for c in regularization:
            probe = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
            probe.fit(x[train], y[train])
            p_val = probe.predict_proba(x[val])[:, 1]
            row = {
                "layer": layer,
                "C": c,
                "val_log_loss": log_loss(y[val], p_val),
                "val_auroc": roc_auc_score(y[val], p_val),
                "train_auroc": roc_auc_score(y[train], probe.predict_proba(x[train])[:, 1]),
            }
            rows.append(row)
            if best is None or row["val_log_loss"] < best[0]["val_log_loss"]:
                best = (row, probe.predict_proba(x)[:, 1])
    table = pd.DataFrame(rows)
    table["selected"] = (table["layer"] == best[0]["layer"]) & (table["C"] == best[0]["C"])
    return best[1], table
