"""Constrained-verdict judge prompt and parsing for Phase 13 direction B.

The judge sees the Phase 9 judge prompt plus a rule: end with exactly one line
"Verdict: Solution A" or "Verdict: Solution B", and make the final answer equal
the chosen solution's final answer. A judgment's pick is the solution named by
its last verdict line; without a valid verdict line it falls back to which
solution its final answer matches, or "invented" if it matches neither.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.core.schema import Problem  # noqa: E402
from phase9.scripts.answers import parse, same  # noqa: E402
from phase9.scripts.collect_checkpoint import judge_prompt  # noqa: E402

VERDICT_RULE = (
    "\n\nAt the very end, write exactly one line: \"Verdict: Solution A\" or "
    "\"Verdict: Solution B\", naming the solution whose final answer is correct. "
    "Your final answer must equal that solution's final answer."
)
_VERDICT = re.compile(r"Verdict:\s*Solution\s*([AB])\b")


def constrained_prompt(problem: Problem, first: str, second: str) -> str:
    return judge_prompt(problem, first, second) + VERDICT_RULE


def verdict_line(position: str) -> str:
    return f"\nVerdict: Solution {position}"


def pick(text: str, shown_a, shown_b, constrained: bool) -> str:
    """Return "A", "B", or "invented" for one single-order judgment."""
    if constrained:
        found = _VERDICT.findall(text)
        if found:
            return found[-1]
    value = parse(text)
    return "A" if same(value, shown_a) else "B" if same(value, shown_b) else "invented"
