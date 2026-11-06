"""Score a locked test transcript for Phase 16 C-upper-inclusive.

This module never executes a candidate program. A separate isolated runner
must provide actual values; a trusted oracle must provide expected values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VARIANTS = {"count", "sum", "select"}
KINDS = {
    "ordinary", "lower", "outside", "empty", "single_point", "duplicate",
    "upper_anchor", "upper_added",
}
REQUIRED_KINDS = {
    "ordinary", "lower", "outside", "empty", "single_point", "duplicate",
}
EXECUTION_FAILURES = {"syntax_error", "runtime_error", "timeout"}


def _same(left: Any, right: Any) -> bool:
    """Strict JSON-value comparison; bool must not count as integer 1 or 0."""
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same(left[key], right[key]) for key in left
        )
    return left == right


def _valid_value(value: Any, variant: str) -> bool:
    if variant in {"count", "sum"}:
        return type(value) is int
    return (
        type(value) is list
        and all(type(item) is str for item in value)
        and len(set(value)) == len(value)
    )


def _expected_pair_valid(anchor: dict, added: dict, variant: str) -> bool:
    first, second = anchor["expected"], added["expected"]
    if variant == "count":
        return second == first + 1
    if variant == "sum":
        value = added.get("added_value")
        return type(value) is int and value != 0 and second == first + value
    item = added.get("added_item")
    return type(item) is str and item not in first and second == first + [item]


def score(record: dict) -> dict:
    """Return exact correctness and an observable, conservative error label."""
    if not isinstance(record, dict):
        raise ValueError("Transcript must be a JSON object")
    task_id = record.get("task_id")
    variant = record.get("variant")
    status = record.get("execution_status")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("Missing task_id")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if status in EXECUTION_FAILURES:
        return {
            "task_id": task_id,
            "variant": variant,
            "full_correct": False,
            "label": status,
            "signature_pairs": 0,
            "passed_cases": 0,
            "total_cases": 0,
        }
    if status != "ok":
        raise ValueError(f"Unknown execution_status: {status}")

    raw_cases = record.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("Successful execution requires cases")
    by_id: dict[str, dict] = {}
    kinds: set[str] = set()
    pairs: dict[str, dict[str, dict]] = {}
    for case in raw_cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object")
        case_id = case.get("case_id")
        kind = case.get("kind")
        if not isinstance(case_id, str) or not case_id or case_id in by_id:
            raise ValueError("Missing or duplicate case_id")
        if kind not in KINDS:
            raise ValueError(f"Unknown case kind: {kind}")
        if "expected" not in case or "actual" not in case:
            raise ValueError(f"Missing expected/actual: {case_id}")
        if not _valid_value(case["expected"], variant):
            raise ValueError(f"Invalid oracle value: {case_id}")
        by_id[case_id] = case
        kinds.add(kind)
        if kind.startswith("upper_"):
            pair_id = case.get("pair_id")
            upper = case.get("upper")
            if not isinstance(pair_id, str) or not pair_id or type(upper) is not int:
                raise ValueError(f"Invalid boundary pair metadata: {case_id}")
            part = pairs.setdefault(pair_id, {})
            if kind in part:
                raise ValueError(f"Duplicate boundary pair part: {pair_id}")
            part[kind] = case
        elif "pair_id" in case:
            raise ValueError(f"Unexpected pair_id: {case_id}")

    if not REQUIRED_KINDS.issubset(kinds):
        raise ValueError("Required non-boundary test kind is missing")
    if len(pairs) < 2:
        raise ValueError("At least two upper-boundary pairs are required")

    uppers: set[int] = set()
    for pair_id, part in pairs.items():
        if set(part) != {"upper_anchor", "upper_added"}:
            raise ValueError(f"Incomplete boundary pair: {pair_id}")
        anchor, added = part["upper_anchor"], part["upper_added"]
        if anchor["upper"] != added["upper"]:
            raise ValueError(f"Boundary pair has different R values: {pair_id}")
        if anchor["upper"] in uppers:
            raise ValueError("Boundary pairs must use distinct R values")
        uppers.add(anchor["upper"])
        if not _expected_pair_valid(anchor, added, variant):
            raise ValueError(f"Oracle pair relation failed: {pair_id}")

    passed = {
        case_id: _same(case["actual"], case["expected"])
        for case_id, case in by_id.items()
    }
    full_correct = all(passed.values())
    nonboundary_pass = all(
        passed[case_id]
        for case_id, case in by_id.items()
        if case["kind"] in {"ordinary", "lower", "outside"}
    )
    signature_pairs = sum(
        passed[anchor["case_id"]]
        and not passed[added["case_id"]]
        and _same(anchor["actual"], added["actual"])
        for part in pairs.values()
        for anchor, added in [(part["upper_anchor"], part["upper_added"])]
    )
    signature = nonboundary_pass and signature_pairs >= 2
    return {
        "task_id": task_id,
        "variant": variant,
        "full_correct": full_correct,
        "label": (
            "correct" if full_correct else
            "upper_boundary_signature" if signature else "other_failure"
        ),
        "signature_pairs": signature_pairs,
        "passed_cases": sum(passed.values()),
        "total_cases": len(by_id),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path)
    args = parser.parse_args()
    record = json.loads(args.transcript.read_text(encoding="utf-8"))
    print(json.dumps(score(record), ensure_ascii=False, sort_keys=True, indent=2))
