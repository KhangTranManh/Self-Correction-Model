"""Parse selective-revision outputs and compute objective transition rewards."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping


KEEP_RE = re.compile(r"\A\s*<decision>KEEP</decision>\s*\Z", re.DOTALL)
REVISE_RE = re.compile(
    r"\A\s*<decision>REVISE</decision>\s*<answer>\s*(.*?)\s*</answer>\s*\Z",
    re.DOTALL,
)
DECISION_SEARCH_RE = re.compile(r"<decision>(KEEP|REVISE)</decision>")
ANSWER_SEARCH_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


@dataclass(frozen=True)
class ReviewAction:
    decision: str
    revised_answer: str | None
    valid: bool


def parse_review(text: str) -> ReviewAction:
    if KEEP_RE.fullmatch(text):
        return ReviewAction("KEEP", None, True)
    match = REVISE_RE.fullmatch(text)
    if match and match.group(1).strip():
        return ReviewAction("REVISE", match.group(1).strip(), True)
    return ReviewAction("INVALID", None, False)


def parse_review_for_training(text: str) -> ReviewAction:
    """Recover a decision for reward shaping without relaxing evaluation."""
    strict = parse_review(text)
    if strict.valid:
        return strict
    decision_match = DECISION_SEARCH_RE.search(text)
    if not decision_match:
        return ReviewAction("INVALID", None, False)
    decision = decision_match.group(1)
    if decision == "KEEP":
        return ReviewAction("KEEP", None, True)
    answer_match = ANSWER_SEARCH_RE.search(text)
    answer = answer_match.group(1).strip() if answer_match else ""
    return ReviewAction("REVISE", answer, True)


def transition_name(initial_correct: bool, final_correct: bool) -> str:
    if initial_correct and final_correct:
        return "correct_to_correct"
    if initial_correct and not final_correct:
        return "correct_to_wrong"
    if not initial_correct and final_correct:
        return "wrong_to_correct"
    return "wrong_to_wrong"


def transition_reward(
    *,
    initial_correct: bool,
    final_correct: bool,
    action: ReviewAction,
    rewards: Mapping[str, float],
) -> tuple[float, str]:
    if not action.valid:
        return float(rewards["invalid_contract"]), "invalid_contract"
    if initial_correct and action.decision == "KEEP":
        return float(rewards["correct_keep"]), "correct_keep"
    if initial_correct and final_correct:
        return float(rewards["correct_safe_revision"]), "correct_safe_revision"
    if initial_correct and not final_correct:
        return float(rewards["correct_to_wrong"]), "correct_to_wrong"
    if not initial_correct and final_correct:
        return float(rewards["wrong_to_correct"]), "wrong_to_correct"
    if action.decision == "REVISE":
        return float(rewards["wrong_revise_wrong"]), "wrong_revise_wrong"
    return float(rewards["wrong_keep"]), "wrong_keep"
