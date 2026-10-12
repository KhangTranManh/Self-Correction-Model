"""Lenient and strict final-answer reading for Phase 14.

Lenient (primary) is the exact Phase 1 extraction used by Phases 9-13:
last \\boxed{...}, else the last "Đáp số/Answer: ..." line, else the last
number in the text. Strict (sensitivity check) accepts only the first two;
an output without a boxed or labelled answer has no strict answer and is
scored wrong. Both modes compare values with the same SymPy normalization.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.data.verifiers.math import _ANSWER_PATTERNS, _to_sympy, extract_last_boxed  # noqa: E402
from phase9.scripts.answers import parse as parse_lenient, same  # noqa: E402,F401

MODES = ("lenient", "strict")


def extract_strict(text: str) -> str | None:
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed
    for pattern in _ANSWER_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return None


def parse_strict(text: str):
    extracted = extract_strict(text)
    if extracted is None:
        return None
    try:
        return _to_sympy(extracted)
    except Exception:
        return None


def parse(text: str, mode: str = "lenient"):
    if mode == "lenient":
        return parse_lenient(text)
    if mode == "strict":
        return parse_strict(text)
    raise ValueError(f"Unknown answer mode: {mode}")


def correct(text: str, reference, mode: str = "lenient") -> bool:
    """Gold comparison; ``reference`` is a parsed SymPy value."""
    return same(parse(text, mode), reference)
