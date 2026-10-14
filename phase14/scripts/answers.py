"""Lenient and strict final-answer reading for Phase 14.

Lenient (primary) is the exact Phase 1 extraction used by Phases 9-13:
last \\boxed{...}, else the last "Đáp số/Answer: ..." line, else the last
number in the text. Strict (sensitivity check) accepts only the first two;
an output without a boxed or labelled answer has no strict answer and is
scored wrong. Rescued (sensitivity check, added 2026-10-12 before any lock):
lenient, but when the lenient value is not a number, the first number on the
last "Đáp số"/"Answer" line is read instead; it recovers answers written as
sentences ("Đáp số: Mỗi trẻ nhận được 9 trứng."), which the lenient reading
turns into a non-numeric expression (paper check T4: 29/561 Qwen2.5-7B-Instruct
first attempts, 4/561 Self_Correction). It changes correctness only; judged
pairs and picks keep the lenient reading. All modes compare values with the
same SymPy normalization.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.data.verifiers.math import _ANSWER_PATTERNS, _to_sympy, extract_last_boxed  # noqa: E402
import phase9.scripts.answers as _phase9_answers  # noqa: E402
import phase13.scripts.constrained as _phase13_constrained  # noqa: E402
from phase9.scripts.answers import parse as parse_lenient  # noqa: E402

_same_uncached = _phase9_answers.same
_same_memo: dict = {}


def same(left, right) -> bool:
    """phase9 ``same`` with a memo (pure function; SymPy simplify dominated scoring time).

    Installed into phase9.scripts.answers and phase13.scripts.constrained so their
    ``vote`` and ``pick`` reuse it; results are identical to the uncached function.
    """
    try:
        key = (left, right)
        hit = _same_memo.get(key)
    except TypeError:  # unhashable value: compute directly
        return _same_uncached(left, right)
    if hit is None:
        hit = _same_memo[key] = _same_uncached(left, right)
    return hit


_phase9_answers.same = same
_phase13_constrained.same = same

MODES = ("lenient", "strict", "rescued")
_FINAL_LINE = re.compile(r"(?:Đáp số|Answer)\s*:?(.{0,120})", flags=re.IGNORECASE)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


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


def parse_rescued(text: str):
    value = parse_lenient(text)
    if value is not None and getattr(value, "is_number", False):
        return value
    lines = _FINAL_LINE.findall(text)
    match = _NUMBER.search(lines[-1]) if lines else None
    if match is None:
        return value
    try:
        return _to_sympy(match.group())  # project rule for "1,5" vs "1,200"
    except Exception:
        return value


def parse(text: str, mode: str = "lenient"):
    if mode == "lenient":
        return parse_lenient(text)
    if mode == "strict":
        return parse_strict(text)
    if mode == "rescued":
        return parse_rescued(text)
    raise ValueError(f"Unknown answer mode: {mode}")


def correct(text: str, reference, mode: str = "lenient") -> bool:
    """Gold comparison; ``reference`` is a parsed SymPy value."""
    return same(parse(text, mode), reference)
