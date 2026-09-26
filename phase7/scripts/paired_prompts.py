"""Exact Phase 7 paired prompt construction."""

from __future__ import annotations

from src.core.prompts import build_prompt
from src.core.schema import Problem


COMMON_SUFFIX = "\n\nGive a complete solution from scratch."
VISIBLE_HEADER = "\n\nUnverified candidate answer from an earlier attempt:\n"


def paired_prompt(problem: Problem, arm: str, initial_output: str | None = None) -> str:
    base = build_prompt(problem) + COMMON_SUFFIX
    if arm == "blind_resolve":
        return base
    if arm == "answer_visible_resolve":
        if initial_output is None:
            raise ValueError("Visible arm requires the complete initial answer")
        return base + VISIBLE_HEADER + initial_output
    raise ValueError(f"Unknown paired arm: {arm}")
