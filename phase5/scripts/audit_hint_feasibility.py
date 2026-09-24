"""Build and audit deterministic Phase 5 review hints without model calls.

Location/type eligibility is deliberately conservative. A wrong answer is
eligible only when its own text contains an explicitly numbered step with a
self-contained, single-operation numeric equality that is arithmetically
false. Correct controls require a similarly checkable true equality. Reference
answers and verifier details are never used to choose the location or type.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_DIR = ROOT / "phase5/data/splits/v1"
DEFAULT_OUTPUT_DIR = ROOT / "phase5/data/hints/v2"
SPLITS = ("train", "development", "protected_test")

STEP_RE = re.compile(r"(?:^|\s)(?:bước|step)\s*(\d+)\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?(?![\w])")
RHS_RE = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)")

OPERATOR_NAMES = {
    "+": "addition",
    "-": "subtraction",
    "*": "multiplication",
    "×": "multiplication",
    "/": "division",
    "÷": "division",
}


def stable_rank(source_id: str, occurrence: int) -> str:
    value = f"phase5_hint_control_v1|{source_id}|{occurrence}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def parse_number(token: str) -> Fraction | None:
    token = token.strip()
    if "," in token:
        whole, decimal = token.split(",", 1)
        # A three-digit suffix is ambiguous with a thousands separator. Reject
        # rather than silently selecting one interpretation.
        if len(decimal) == 3:
            return None
        token = f"{whole}.{decimal}"
    try:
        return Fraction(token)
    except (ValueError, ZeroDivisionError):
        return None


def operator_occurrences(lhs: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for index, char in enumerate(lhs):
        if char in "+-*×÷":
            found.append((index, char))
        elif char == "/":
            # Accept a slash only when it directly separates numeric tokens;
            # this excludes unit strings such as pound/day.
            left = lhs[:index].rstrip()
            right = lhs[index + 1:].lstrip()
            if left and right and left[-1].isdigit() and right[0].isdigit():
                found.append((index, char))
    return found


def parse_equation(line: str, step_number: int, occurrence: int) -> dict[str, Any] | None:
    # Chained equalities, percentages, approximations, LaTeX, parentheses and
    # algebraic expressions caused false positives in the rejected v1 parser.
    # Keep only literal one-operation numeric equalities.
    if line.count("=") != 1:
        return None
    if any(marker in line for marker in ("%", "≈", "\\", "(", ")", "[", "]", "{", "}")):
        return None
    if re.search(r"\d[.,]\d{3}[.,]\d{3}", line):
        return None
    lhs, rhs_text = line.split("=", 1)
    rhs_match = RHS_RE.match(rhs_text)
    if not rhs_match:
        return None
    operators = operator_occurrences(lhs)
    if len(operators) != 1:
        return None
    operator_index, operator = operators[0]
    before = list(NUMBER_RE.finditer(lhs[:operator_index]))
    after = list(NUMBER_RE.finditer(lhs[operator_index + 1:]))
    if not before or not after:
        return None
    left_token = before[-1].group(0)
    right_token = after[0].group(0)
    stated_token = rhs_match.group(1)
    if "." in stated_token and len(stated_token.rsplit(".", 1)[1]) > 6:
        return None
    if "," in stated_token and len(stated_token.rsplit(",", 1)[1]) > 6:
        return None
    left = parse_number(left_token)
    right = parse_number(right_token)
    stated = parse_number(stated_token)
    if left is None or right is None or stated is None:
        return None
    try:
        if operator == "+":
            evaluated = left + right
        elif operator == "-":
            evaluated = left - right
        elif operator in {"*", "×"}:
            evaluated = left * right
        else:
            if right == 0:
                return None
            evaluated = left / right
    except ZeroDivisionError:
        return None
    return {
        "occurrence": occurrence,
        "step_number": step_number,
        "line": line.strip(),
        "operator": OPERATOR_NAMES[operator],
        "left_operand": left_token,
        "right_operand": right_token,
        "stated_result": stated_token,
        "evaluated_result": str(evaluated),
        "is_true": evaluated == stated,
    }


def extract_checkable_equations(text: str) -> list[dict[str, Any]]:
    equations: list[dict[str, Any]] = []
    current_step: int | None = None
    for line in text.splitlines():
        step_match = STEP_RE.search(line.casefold())
        if step_match:
            current_step = int(step_match.group(1))
        if current_step is None:
            continue
        parsed = parse_equation(line, current_step, len(equations))
        if parsed is not None:
            equations.append(parsed)
    return equations


def build_hints(initial_correct: bool, annotation: dict[str, Any] | None) -> dict[str, str | None]:
    status = (
        "No error was found in the previous answer. Review it and follow the output contract."
        if initial_correct else
        "The previous answer contains an error. Review it and follow the output contract."
    )
    if annotation is None:
        return {"status": status, "location": None, "type": None}
    step = annotation["step_number"]
    prefix = (
        "No error was found in the previous answer."
        if initial_correct else
        "The previous answer contains an error."
    )
    location = (
        f"{prefix} Recheck the arithmetic in Step {step}. "
        "No corrected value is provided. Follow the output contract."
    )
    typed = (
        f"{prefix} Recheck the {annotation['operator']} arithmetic in Step {step}. "
        "No corrected value is provided. Follow the output contract."
    )
    return {"status": status, "location": location, "type": typed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite hint audit: {args.output_dir}")

    audit_rows: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for split in SPLITS:
        for row in read_jsonl(args.split_dir / f"{split}.jsonl"):
            equations = extract_checkable_equations(row["initial_output"])
            if row["initial_correct"]:
                candidates = [item for item in equations if item["is_true"]]
                annotation = min(
                    candidates,
                    key=lambda item: stable_rank(row["problem_id"], item["occurrence"]),
                    default=None,
                )
                reason = "deterministic_true_equation_control" if annotation else "no_checkable_true_equation"
            else:
                candidates = [item for item in equations if not item["is_true"]]
                annotation = candidates[0] if candidates else None
                reason = "first_self_contained_false_equation" if annotation else "no_checkable_false_equation"

            hints = build_hints(row["initial_correct"], annotation)
            label = "correct" if row["initial_correct"] else "wrong"
            counts[split][f"{label}_total"] += 1
            counts[split][f"{label}_location_type_eligible"] += int(annotation is not None)
            counts[split][f"reason:{reason}"] += 1
            if annotation:
                counts[split][f"operator:{annotation['operator']}"] += 1
            audit_rows.append({
                "problem_id": row["problem_id"],
                "split": split,
                "initial_correct": row["initial_correct"],
                "neutral_eligible": True,
                "status_eligible": True,
                "location_eligible": annotation is not None,
                "type_eligible": annotation is not None,
                "annotation_method": "strict_literal_single_operation_equation_v2",
                "eligibility_reason": reason,
                "annotation": annotation,
                "model_visible_hints": hints,
                "reference_answer_used_for_annotation": False,
                "verifier_detail_used_for_annotation": False,
                "corrected_value_in_model_visible_hint": False,
            })

    args.output_dir.mkdir(parents=True, exist_ok=False)
    audit_hash = write_jsonl(args.output_dir / "hint_audit.jsonl", audit_rows)
    summary = {
        "schema_version": "phase5_hint_feasibility_v2",
        "status": "frozen_before_review_generation",
        "method": "strict_literal_single_operation_equation_v2",
        "policy": (
            "Wrong rows require an explicitly numbered model step containing a "
            "literal arithmetically false equality with exactly one operation and "
            "one equals sign. Chained, percent, approximate, LaTeX, parenthesized, "
            "and algebraic expressions are rejected. Correct controls use the same "
            "rule and require a true equality."
        ),
        "rows": len(audit_rows),
        "counts": {split: dict(sorted(value.items())) for split, value in counts.items()},
        "global": {
            "neutral_eligible": sum(row["neutral_eligible"] for row in audit_rows),
            "status_eligible": sum(row["status_eligible"] for row in audit_rows),
            "location_type_eligible": sum(row["location_eligible"] for row in audit_rows),
            "correct_location_type_eligible": sum(
                row["location_eligible"] and row["initial_correct"] for row in audit_rows
            ),
            "wrong_location_type_eligible": sum(
                row["location_eligible"] and not row["initial_correct"] for row in audit_rows
            ),
        },
        "safeguards": {
            "reference_answer_used_for_annotation": False,
            "verifier_detail_used_for_annotation": False,
            "corrected_value_in_model_visible_hint": False,
            "location_type_is_exploratory_due_to_timing_deviation": True,
            "ineligible_rows_are_not_refilled": True,
        },
        "hint_audit_sha256": audit_hash,
    }
    payload = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    (args.output_dir / "hint_summary.json").write_bytes(payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
