"""Pilot scoring with the Phase 1 verifier and the Phase 14 class rules.

This file is copied into the Kaggle kernel. It does not import the paper
package. Local tests check it against ``detectors.py`` on fixed fixtures.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import sys


def _install_paths() -> None:
    here = Path(__file__).resolve().parent
    candidates = [
        here.parents[1] / "phase1",
        here.parent / "vendor",
        here / "vendor",
    ]
    for path in candidates:
        if (path / "src" / "data" / "verifiers" / "math.py").exists():
            sys.path.insert(0, str(path))
            return
    raise ImportError("Phase 1 math verifier was not found for pilot scoring")


_install_paths()
from src.data.verifiers.math import _extract_answer, _to_sympy  # noqa: E402


_FINAL_LINE = re.compile(r"(?:Đáp số|Dap so|Answer)\s*:?(.{0,120})", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_EXACT_FINAL = re.compile(r"(?m)^\s*Đáp số:\s*-?\d+(?:[.,]\d+)?\s*$")
_COUNT = re.compile(r"\bhow many\b|\bnumber of\b|\bbao nhiêu\b", re.IGNORECASE)
_UNIT = re.compile(
    r"\b(?:meter|meters|metre|metres|kilogram|kilograms|kg|dollar|dollars|hour|hours|"
    r"minute|minutes|second|seconds|percent|cm|mm|km|mét|kilôgam|đồng|giờ|phút|giây)\b|%",
    re.IGNORECASE,
)
_VERDICT = re.compile(r"Verdict:\s*Solution\s*([AB])\b", re.IGNORECASE)
WORD = re.compile(r"\w+", re.UNICODE)
FOREIGN = re.compile(r"[Ѐ-ӿ֐-ۿ฀-๿぀-ヿ㐀-鿿가-힯]")
STEPS = re.compile(r"(Bước\s*\d|Step\s*\d|^\s*\d+[.)]\s)", re.MULTILINE)
SHOWN = re.compile(
    r"(Solution|solution|Lời giải|lời giải|Giải pháp|giải pháp|Cách|cách|Phương pháp|"
    r"phương pháp|Method|method|Approach|approach)\s*(A|B|1|2)\b"
)
EN_WORDS = {"the", "is", "we", "and", "of", "to", "that", "this", "for", "so", "are", "with", "can", "by",
            "find", "now", "then", "therefore", "since", "be", "it", "not", "have", "which"}
VI_WORDS = {"ta", "là", "và", "của", "có", "cho", "được", "nên", "vì", "không", "này", "các", "một", "với",
            "khi", "thì", "để", "số", "bài", "giải", "vậy", "bước", "tính", "đó"}
MATH = re.compile(r"\$[^$]*\$|\\\[.*?\\\]|\\begin\{.*?\\end\{[^}]*\}", re.DOTALL)
JUDGE_TYPES = {2: "J-format", 3: "J-loop", 5: "J-resolve", 6: "J-third"}


def parse_lenient(text: str):
    extracted = _extract_answer(text)
    if extracted is None:
        return None
    try:
        return _to_sympy(extracted)
    except Exception:
        return None


def parse_rescued(text: str):
    value = parse_lenient(text)
    if value is not None and bool(getattr(value, "is_number", False)):
        return value
    lines = _FINAL_LINE.findall(text)
    match = _NUMBER.search(lines[-1]) if lines else None
    if match is None:
        return value
    try:
        return _to_sympy(match.group())
    except Exception:
        return value


def same(left, right) -> bool:
    if left is None or right is None:
        return False
    try:
        import sympy
        return bool(sympy.simplify(left - right) == 0)
    except Exception:
        return left == right


def numeric(value) -> bool:
    return value is not None and bool(getattr(value, "is_number", False))


def correct_answer(text: str, reference: str) -> bool:
    try:
        gold = _to_sympy(reference)
    except Exception:
        return False
    return same(parse_lenient(text), gold)


def asks_for_count(question: str) -> bool:
    return bool(_COUNT.search(question)) and not bool(_UNIT.search(question))


def solve_errors(output: str, question: str, *, hit_768_cap: bool) -> list[str]:
    found = []
    if not _EXACT_FINAL.search(output) and not numeric(parse_lenient(output)) and numeric(parse_rescued(output)):
        found.append("S-format")
    if hit_768_cap and _label(output, False) == "b":
        found.append("S-loop")
    if asks_for_count(question):
        value = parse_lenient(output)
        if not numeric(value):
            value = parse_rescued(output)
        if numeric(value) and not bool(value.is_integer):
            found.append("S-integer")
    return found


def pick(text: str, shown_a, shown_b) -> str:
    value = parse_lenient(text)
    if same(value, shown_a):
        return "A"
    if same(value, shown_b):
        return "B"
    return "invented"


def judge_class(output: str, shown_a, shown_b, *, hit_token_cap: bool) -> tuple[int, str]:
    chosen = pick(output, shown_a, shown_b)
    if chosen in ("A", "B"):
        return 1, chosen
    rescued = parse_rescued(output)
    match_a, match_b = same(rescued, shown_a), same(rescued, shown_b)
    if match_a != match_b:
        return 2, "A" if match_a else "B"
    if hit_token_cap:
        return (3 if _label(output, True) == "b" else 4), "x"
    if _label(output, True) == "d":
        return 5, "x"
    if parse_lenient(output) is not None or rescued is not None:
        return 6, "x"
    return 7, "x"


def judge_error(output: str, shown_a, shown_b, *, hit_token_cap: bool) -> str | None:
    kind, _choice = judge_class(output, shown_a, shown_b, hit_token_cap=hit_token_cap)
    return JUDGE_TYPES.get(kind)


def _lang(text: str) -> str | None:
    words = [word.lower() for word in WORD.findall(MATH.sub(" ", text))]
    english = sum(word in EN_WORDS for word in words)
    vietnamese = sum(word in VI_WORDS for word in words)
    if english + vietnamese < 8:
        return None
    if english > 2 * vietnamese:
        return "en"
    if vietnamese > 2 * english:
        return "vi"
    return None


def _literal_repeat(text: str) -> bool:
    words = [word.lower() for word in WORD.findall(text)]
    grams = Counter(tuple(words[index:index + 10]) for index in range(max(0, len(words) - 9)))
    if grams and grams.most_common(1)[0][1] >= 3:
        return True
    segments = [part.strip() for part in re.split(r"[\n.]", text) if len(part.strip()) >= 25]
    return bool(segments) and Counter(segments).most_common(1)[0][1] >= 3


def _templated_repeat(text: str) -> bool:
    lines = [line.strip() for line in re.sub(r"\d+", "0", text).split("\n") if len(line.strip()) >= 15]
    return bool(lines) and Counter(lines).most_common(1)[0][1] >= 5


def _label(text: str, judgment: bool) -> str:
    third = max(1, len(text) // 3)
    first, last = _lang(text[:third]), _lang(text[-third:])
    if (re.search(r"(.)\1{29,}", text) or len(FOREIGN.findall(text)) >= 0.02 * max(1, len(text))
            or (first and last and first != last)):
        return "c"
    if _literal_repeat(text) or _templated_repeat(text):
        return "b"
    tail = text[int(0.4 * len(text)):]
    if judgment and not SHOWN.search(tail) and STEPS.search(tail):
        return "d"
    return "a"


def solve_prompt(question: str) -> str:
    return (
        "Giải bài toán sau từng bước, sau đó ghi rõ đáp số cuối cùng theo định dạng "
        f"'Đáp số: <giá trị>'.\n\nBài toán: {question}"
    )


def judge_prompt(question: str, solution_a: str, solution_b: str) -> str:
    return (
        solve_prompt(question)
        + "\n\nTwo independent solutions to this problem are shown below. Either or both "
        "may be wrong, and their final answers may differ. Check each solution step by "
        "step, then write your own complete, correct solution.\n\n"
        f"Solution A:\n{solution_a}\n\nSolution B:\n{solution_b}"
    )
