import numpy as np
import pandas as pd
import pytest

from src import infer, policies


def test_answer_logits_masks_missing_options():
    frame = pd.DataFrame([[0.0] * 11], columns=infer.LOGIT_COLUMNS)
    frame["n_options"] = 4
    logits = policies.answer_logits(frame)
    assert logits.shape == (1, 10)
    assert np.isfinite(logits[0, :4]).all() and np.isinf(logits[0, 4:]).all()
    assert policies.top_probability(logits)[0] == pytest.approx(0.25)


def test_fit_temperature_recovers_known_scale():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(5000, 10)) * 2
    probs = np.exp(true_logits) / np.exp(true_logits).sum(axis=1, keepdims=True)
    gold = np.array([rng.choice(10, p=p) for p in probs])
    overconfident = true_logits * 4  # a calibrated model would need temperature 4
    assert policies.fit_temperature(overconfident, gold) == pytest.approx(4, rel=0.1)
    assert policies.top_probability(overconfident).mean() > policies.top_probability(
        overconfident, 4
    ).mean()


def test_fit_probe_finds_the_informative_layer():
    rng = np.random.default_rng(0)
    n = 600
    correct = rng.random(n) < 0.4
    hidden = rng.normal(size=(n, 2, 16)).astype(np.float32)
    hidden[:, 1, 0] += 3 * correct  # only the second layer carries the signal
    index = np.arange(n)
    train, val = index < 360, (index >= 360) & (index < 480)
    confidence, table = policies.fit_probe(hidden, correct, train, val, layers=[4, 8])
    assert int(table[table["selected"]]["layer"].iloc[0]) == 8
    test = index >= 480
    assert confidence[test][correct[test]].mean() > confidence[test][~correct[test]].mean() + 0.3
