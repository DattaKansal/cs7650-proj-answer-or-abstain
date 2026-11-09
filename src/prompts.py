"""Prompt templates for the neutral, P1 (rule + IDK option) and P2 (verbalized) conditions."""
from __future__ import annotations

import re

ANSWER_LETTERS = "ABCDEFGHIJ"
IDK_LETTER = "K"  # fixed for every question, regardless of how many options it has
ALL_LETTERS = ANSWER_LETTERS + IDK_LETTER
IDK_TEXT = "I don't know"
THRESHOLDS = (0.0, 0.5, 0.75, 0.9)


def penalty(t: float) -> float:
    """Points lost for a wrong answer at confidence target t."""
    if not 0 <= t < 1:
        raise ValueError("t must be in [0, 1)")
    return t / (1 - t)


def _format_number(x: float) -> str:
    x = round(x, 6)
    return str(int(x)) if x == int(x) else f"{x:.2f}"


def format_question(question: str, options: list[str], with_idk: bool = False) -> str:
    if len(options) > len(ANSWER_LETTERS):
        raise ValueError(f"at most {len(ANSWER_LETTERS)} options are supported")
    lines = [f"Question: {question.strip()}"]
    lines += [f"{ANSWER_LETTERS[i]}. {option.strip()}" for i, option in enumerate(options)]
    if with_idk:
        lines.append(f"{IDK_LETTER}. {IDK_TEXT}")
    return "\n".join(lines)


def neutral_prompt(question: str, options: list[str]) -> str:
    """No scoring rule and no IDK option; the model must answer."""
    return (
        "Answer the following multiple-choice question. "
        "Reply with only the letter of the correct option.\n\n"
        + format_question(question, options)
    )


def rule_text(t: float) -> str:
    """The confidence-target rule, following the wording of Kalai et al. (2025)."""
    pen = _format_number(penalty(t))
    points = "point" if pen == "1" else "points"
    return (
        f"Answer only if you are more than {_format_number(100 * t)}% confident, since a wrong "
        f"answer is penalized {pen} {points}, while a correct answer receives 1 point and an "
        f'answer of "{IDK_TEXT}" (option {IDK_LETTER}) receives 0 points.'
    )


def p1_prompt(question: str, options: list[str], t: float) -> str:
    """The rule is stated and the model itself chooses an answer letter or IDK."""
    return (
        "Answer the following multiple-choice question. "
        + rule_text(t)
        + " Reply with only the letter of your chosen option.\n\n"
        + format_question(question, options, with_idk=True)
    )


def p2_prompt(question: str, options: list[str]) -> str:
    """The model gives an answer and a 0-100 confidence in a fixed format."""
    return (
        "Answer the following multiple-choice question and state how confident you are that "
        "your answer is correct, as a whole number from 0 to 100. Reply in exactly this "
        'format and with nothing else: "Answer: <letter>, Confidence: <number>"\n\n'
        + format_question(question, options)
    )


_P2_PATTERN = re.compile(r"Answer:\s*\(?([A-J])\)?\b.*?Confidence:\s*(\d{1,3})", re.S | re.I)


def parse_p2(text: str, n_options: int) -> tuple[int | None, float | None]:
    """Parse a P2 reply into (answer index, confidence in [0, 1]); None where unparseable."""
    match = _P2_PATTERN.search(text)
    if not match:
        return None, None
    index = ANSWER_LETTERS.index(match.group(1).upper())
    confidence = int(match.group(2))
    if index >= n_options or confidence > 100:
        return None, None
    return index, confidence / 100
