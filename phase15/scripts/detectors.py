"""Phase 15 candidate error types, using the Phase 14 rules unchanged.

Judge types are Phase 14 classes 2, 3, 5, and 6 from
``paper/scripts/phase14_paper_tables.py``. Classes 1, 4, and 7 are not
candidate types. Solve types follow the Phase 15 preregistration:

- S-format: the final line is not ``Đáp số: <number>``, lenient parsing is not
  a number, and rescued parsing is a number.
- S-loop: the solution hit the 768-token cap and the Phase 14 truncation
  labeller says ``b`` (looping) for a solution, not a judgment.
- S-integer: proposed, not frozen. The question asks for a count and the
  parsed answer is a non-integer number. A unit word makes it a measurement,
  not this type.

A text can carry more than one solve type. A judgment has at most one class.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper/scripts"))
sys.path.insert(0, str(ROOT / "phase1"))

from phase14.scripts.answers import parse  # noqa: E402
from phase14_paper_tables import classify  # noqa: E402
import phase14_truncation as truncation  # noqa: E402


JUDGE_TYPES = {2: "J-format", 3: "J-loop", 5: "J-resolve", 6: "J-third"}
_EXACT_FINAL = re.compile(r"(?m)^\s*Đáp số:\s*-?\d+(?:[.,]\d+)?\s*$")
_COUNT = re.compile(r"\bhow many\b|\bnumber of\b|\bbao nhiêu\b", re.IGNORECASE)
_UNIT = re.compile(
    r"\b(?:meter|meters|metre|metres|kilogram|kilograms|kg|dollar|dollars|hour|hours|"
    r"minute|minutes|second|seconds|percent|cm|mm|km|mét|kilôgam|đồng|giờ|phút|giây)\b|%",
    re.IGNORECASE,
)


def _numeric(value) -> bool:
    return value is not None and bool(getattr(value, "is_number", False))


def judge_error(output: str, shown_a, shown_b, *, hit_token_cap: bool, constrained: bool = False) -> str | None:
    """One candidate judge type, or None when the Phase 14 class is not 2, 3, 5, or 6."""
    row = {"output": output, "hit_token_cap": bool(hit_token_cap), "generated_tokens": 1}
    kind, _pick = classify(row, shown_a, shown_b, constrained)
    return JUDGE_TYPES.get(kind)


def solve_format(output: str) -> bool:
    if _EXACT_FINAL.search(output):
        return False
    return not _numeric(parse(output, "lenient")) and _numeric(parse(output, "rescued"))


def solve_loop(output: str, *, hit_768_cap: bool) -> bool:
    return bool(hit_768_cap) and truncation.label(output, False) == "b"


def asks_for_count(question: str) -> bool:
    return bool(_COUNT.search(question)) and not bool(_UNIT.search(question))


def solve_integer(output: str, question: str) -> bool:
    if not asks_for_count(question):
        return False
    value = parse(output, "lenient")
    if not _numeric(value):
        value = parse(output, "rescued")
    return _numeric(value) and not bool(value.is_integer)


def solve_errors(output: str, question: str, *, hit_768_cap: bool) -> list[str]:
    found = []
    if solve_format(output):
        found.append("S-format")
    if solve_loop(output, hit_768_cap=hit_768_cap):
        found.append("S-loop")
    if solve_integer(output, question):
        found.append("S-integer")
    return found
