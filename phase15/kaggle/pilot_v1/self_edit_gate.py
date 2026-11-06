"""Accept, normalize, or retry a Phase 15 self-edit before any LoRA update.

The Kaggle smoke trained on every self-edit as written. Four of the eight
lessons used numbered-list markers, and all eight ended by naming the teaching
item's verdict side. Marker digits are removed before the number check, because
a marker such as ``3.`` is not a copied teaching value even when 3 is a teaching
number. A trailing single-side verdict is removed for the same reason: it names
the teaching answer rather than the behavior.

This does not call a model. ``retry`` means the normalized lesson still copies
a forbidden number or a protected probe/control phrase, or it has no rule and
template. The caller may generate one replacement and review that once.
"""

from __future__ import annotations

from fractions import Fraction
import re
import unicodedata


NGRAM = 8
_MARKER = re.compile(r"(?m)^([ \t]*)(?:step[ \t]+)?(\d{1,2})(?:[.):])[ \t]+", re.IGNORECASE)
_VERDICT = re.compile(r"Verdict:\s*Solution\s*([AB])\b", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"\w+", re.UNICODE)
_TEMPLATE = re.compile(r"ideal response template", re.IGNORECASE)


def _words(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _WORD.findall(folded)


def _canon(token: str) -> str | None:
    token = token.replace(",", "")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", token):
        return None
    if "." in token:
        value = Fraction(token)
        return str(value.numerator) if value.denominator == 1 else str(value)
    return str(int(token))


def number_values(text: str) -> set[str]:
    found = set()
    for token in _NUMBER.findall(text):
        value = _canon(token)
        if value is not None:
            found.add(value)
    return found


def normalize(text: str) -> tuple[str, int, bool]:
    """Return text, marker replacements, and whether a single-side verdict was removed."""
    cleaned, markers = _MARKER.subn(r"\1- ", text.strip())
    lines = cleaned.splitlines()
    stripped = False
    match = _VERDICT.search(lines[-1]) if lines else None
    if match and len(_VERDICT.findall(lines[-1])) == 1:
        side = match.group(1).upper()
        other = "B" if side == "A" else "A"
        body = "\n".join(lines[:-1])
        if not re.search(rf"Solution\s*{other}\b", body, re.IGNORECASE):
            lines = lines[:-1]
            stripped = True
    return "\n".join(lines).strip(), markers, stripped


def _grams(text: str) -> set[tuple[str, ...]]:
    words = _words(text)
    if len(words) < NGRAM:
        return set()
    return {tuple(words[index : index + NGRAM]) for index in range(len(words) - NGRAM + 1)}


def protected_overlap(text: str, protected_texts: list[str]) -> bool:
    grams = _grams(text)
    if not grams:
        return False
    for source in protected_texts:
        if grams & _grams(source):
            return True
    return False


def _shape_problems(text: str) -> list[str]:
    problems = []
    if not text:
        return ["empty"]
    if _VERDICT.search(text) and len({m.upper() for m in _VERDICT.findall(text)}) < 2:
        problems.append("committed_side_remains")
    parts = _TEMPLATE.split(text, maxsplit=1)
    if len(parts) == 2:
        rule, template = parts
    else:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        rule = paragraphs[0] if paragraphs else ""
        template = "\n".join(paragraphs[1:])
    if len(_words(rule)) < 8:
        problems.append("no_rule")
    if len(_words(template)) < 8:
        problems.append("no_template")
    return problems


def review(text: str, *, forbidden_numbers: set[int], protected_texts: list[str]) -> dict:
    """Review one self-edit.

    ``forbidden_numbers`` are the teaching, probe, and control numbers.
    ``protected_texts`` are the probe and control source texts. The teaching
    text is the model's input, so it is not itself a protected source.
    """
    raw = text.strip()
    chosen, markers, stripped = normalize(raw)
    reasons: list[str] = []
    if markers:
        reasons.append("list_markers")
    if stripped:
        reasons.append("committed_side")
    forbidden = set()
    for number in forbidden_numbers:
        value = _canon(str(number))
        if value is None:
            raise ValueError(f"Forbidden number is not a plain value: {number}")
        forbidden.add(value)
    leaked = number_values(chosen) & forbidden
    if leaked:
        reasons.append("forbidden_number")
    if protected_overlap(chosen, protected_texts):
        reasons.append("protected_ngram")
    reasons.extend(_shape_problems(chosen))
    hard = {"forbidden_number", "protected_ngram", "empty", "no_rule", "no_template", "committed_side_remains"}
    if hard & set(reasons):
        status, kept = "retry", None
    elif reasons:
        status, kept = "normalize", chosen
    else:
        status, kept = "accept", raw
    return {
        "status": status,
        "reasons": reasons,
        "text": kept,
        "list_markers": markers,
        "stripped_verdict": stripped,
        "ngram": NGRAM,
    }
