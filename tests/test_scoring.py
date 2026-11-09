import numpy as np
import pytest

from src import scoring


def test_proposal_example_sixty_percent_accuracy():
    # A model that is right 60% of the time and always answers: +0.2 at t=0.5, -0.6 at t=0.75.
    correct = [True, True, True, False, False]
    abstain = [False] * 5
    assert scoring.penalized_score(correct, abstain, 0.5) == pytest.approx(0.2)
    assert scoring.penalized_score(correct, abstain, 0.75) == pytest.approx(-0.6)


def test_penalties_match_thresholds():
    wrong, answered = [False], [False]
    for t, pen in [(0.0, 0.0), (0.5, 1.0), (0.75, 3.0), (0.9, 9.0)]:
        assert scoring.penalized_score(wrong, answered, t) == pytest.approx(-pen)


def test_abstention_scores_zero_and_is_excluded_from_selective_accuracy():
    correct = [True, False, False, True]
    abstain = [False, True, True, False]
    assert scoring.penalized_score(correct, abstain, 0.9) == pytest.approx(0.5)
    assert scoring.abstention_rate(abstain) == 0.5
    assert scoring.selective_accuracy(correct, abstain) == 1.0


def test_selective_accuracy_nan_when_everything_abstained():
    assert np.isnan(scoring.selective_accuracy([True, False], [True, True]))


def test_threshold_policy_never_abstains_at_zero():
    conf = np.array([0.0, 0.1, 0.5, 0.75, 1.0])
    assert not scoring.threshold_policy(conf, 0.0).any()
    assert scoring.threshold_policy(conf, 0.75).tolist() == [True, True, True, False, False]


def test_oracle_has_zero_regret_and_beats_always_answer():
    reference = np.array([True, False, True, False, False])
    oracle_abstain = ~reference
    for t in (0.0, 0.5, 0.75, 0.9):
        score = scoring.penalized_score(reference, oracle_abstain, t)
        assert score == pytest.approx(scoring.oracle_score(reference))
        assert scoring.regret(score, reference) == pytest.approx(0.0)
        assert score >= scoring.always_answer_score(reference, t)


def test_random_abstention_matches_simulation():
    rng = np.random.default_rng(0)
    reference = rng.random(2000) < 0.4
    expected = scoring.random_abstention_score(reference, 0.3, 0.75)
    simulated = np.mean(
        [scoring.penalized_score(reference, rng.random(2000) < 0.3, 0.75) for _ in range(300)]
    )
    assert simulated == pytest.approx(expected, abs=0.02)


def test_summarize_keys_and_consistency():
    reference = np.array([True, True, False, False])
    out = scoring.summarize(reference, [False, False, True, False], reference, 0.5)
    assert out["score"] == pytest.approx(0.25)
    assert out["regret"] == pytest.approx(0.25)
    assert out["abstention_rate"] == 0.25


def test_risk_coverage_and_aurc():
    conf = [0.9, 0.8, 0.7, 0.6]
    coverage, risk = scoring.risk_coverage(conf, [True, True, False, False])
    assert coverage.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert risk.tolist() == pytest.approx([0, 0, 1 / 3, 0.5])
    # A confidence ranking that puts errors last has lower AURC than the reverse.
    assert scoring.aurc(conf, [True, True, False, False]) < scoring.aurc(
        conf, [False, False, True, True]
    )


def test_ece_zero_when_calibrated_and_large_when_overconfident():
    assert scoring.expected_calibration_error([1.0, 1.0, 0.0], [True, True, False]) == 0.0
    assert scoring.expected_calibration_error([0.95] * 4, [False] * 4) == pytest.approx(0.95)
