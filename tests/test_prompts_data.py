from collections import Counter

import pytest

from src import data, prompts

QUESTION = "What is 2 + 2?"
OPTIONS = ["3", "4", "5", "22"]


def test_neutral_prompt_has_no_idk_option():
    text = prompts.neutral_prompt(QUESTION, OPTIONS)
    assert "A. 3" in text and "D. 22" in text
    assert "K." not in text and prompts.IDK_TEXT not in text


def test_p1_prompt_uses_fixed_idk_letter_even_with_few_options():
    text = prompts.p1_prompt(QUESTION, OPTIONS, 0.75)
    assert text.rstrip().endswith("K. I don't know")
    assert "E." not in text
    assert "more than 75% confident" in text
    assert "penalized 3 points" in text


@pytest.mark.parametrize("t,expected", [(0.0, "0 points"), (0.5, "1 point,"), (0.9, "9 points")])
def test_rule_text_penalties(t, expected):
    assert f"penalized {expected}" in prompts.rule_text(t)


def test_too_many_options_rejected():
    with pytest.raises(ValueError):
        prompts.format_question(QUESTION, ["x"] * 11)


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("Answer: B, Confidence: 70", (1, 0.7)),
        ("answer: (d), confidence: 100", (3, 1.0)),
        ("Answer: B", (None, None)),
        ("Answer: J, Confidence: 50", (None, None)),  # J is not an option here
        ("Answer: A, Confidence: 250", (None, None)),
        ("I think it is B.", (None, None)),
    ],
)
def test_parse_p2(reply, expected):
    assert prompts.parse_p2(reply, n_options=4) == expected


def test_clean_options_remaps_answer_index():
    assert data.clean_options(["a", "N/A", "b", "N/A"], 2) == (["a", "b"], 1)
    with pytest.raises(ValueError):
        data.clean_options(["a", "N/A"], 1)


def test_splits_are_stratified_and_deterministic():
    questions = [
        {"qid": i, "category": "math" if i % 2 else "law"} for i in range(200)
    ]
    first = data.make_splits(questions, seed=0)
    assert first == data.make_splits(list(reversed(questions)), seed=0)
    assert first != data.make_splits(questions, seed=1)
    assert Counter(first.values()) == {"train": 120, "val": 40, "test": 40}
    math_split = Counter(first[q["qid"]] for q in questions if q["category"] == "math")
    assert math_split == {"train": 60, "val": 20, "test": 20}
