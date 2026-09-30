"""Gold-free final-answer comparison for voting and agreement checks.

Uses the exact Phase 1 extraction and SymPy normalization, so two outputs are
"the same answer" exactly when the verifier would score them identically
against a common reference. Unparseable outputs never agree with anything.
"""

from __future__ import annotations

from pathlib import Path
import sys

import sympy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
from src.data.verifiers.math import _extract_answer, _to_sympy  # noqa: E402


def parse(text: str):
    extracted = _extract_answer(text)
    if extracted is None:
        return None
    try:
        return _to_sympy(extracted)
    except Exception:
        return None


def same(left, right) -> bool:
    if left is None or right is None:
        return False
    try:
        return bool(sympy.simplify(left - right) == 0)
    except Exception:
        return left == right


def vote(texts: list[str], priority: list[int]) -> int:
    """Return the index of the chosen text by plurality of final answers.

    Ties (including all-distinct or all-unparseable) are broken by ``priority``:
    the first index in that order belonging to a top cluster wins.
    """
    values = [parse(text) for text in texts]
    support = [sum(same(values[i], values[j]) for j in range(len(texts))) if values[i] is not None else 0
               for i in range(len(texts))]
    best = max(support)
    for index in priority:
        if support[index] == best:
            return index
    raise RuntimeError("Priority order must cover every index")
