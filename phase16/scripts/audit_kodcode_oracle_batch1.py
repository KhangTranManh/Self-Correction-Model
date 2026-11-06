"""Run trusted CPU oracle checks for the first clean batch-1 count task."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

try:
    from phase16.scripts.score_boundary import score
except ModuleNotFoundError:  # Direct invocation from the repository root.
    from score_boundary import score


TASK_ID = "Algorithm_6681_I"
QUESTION_SHA256 = "ff1781ec90657f3fafea297c308df7aebafac58982e8e6769cb16d18e93c6a6c"


def brute_force(values: list[int], lower: int, upper: int) -> int:
    total = 0
    for value in values:
        if value >= lower and value <= upper:
            total += 1
    return total


def reference(values: list[int], lower: int, upper: int) -> int:
    return sum(lower <= value <= upper for value in values)


def exclusive_upper(values: list[int], lower: int, upper: int) -> int:
    return sum(lower <= value < upper for value in values)


def exclusive_lower(values: list[int], lower: int, upper: int) -> int:
    return sum(lower < value <= upper for value in values)


CASES = [
    ("ordinary", "ordinary", [250, 300], 200, 400, {}),
    ("lower", "lower", [200], 200, 400, {}),
    ("outside", "outside", [100, 500], 200, 400, {}),
    ("empty", "empty", [], 200, 400, {}),
    ("single", "single_point", [5], 5, 5, {}),
    ("duplicate", "duplicate", [4, 5, 5], 1, 5, {}),
    ("p1-anchor", "upper_anchor", [2], 1, 5, {"pair_id": "p1", "upper": 5}),
    ("p1-added", "upper_added", [2, 5], 1, 5, {"pair_id": "p1", "upper": 5}),
    ("p2-anchor", "upper_anchor", [1, 3], 1, 7, {"pair_id": "p2", "upper": 7}),
    ("p2-added", "upper_added", [1, 3, 7], 1, 7, {"pair_id": "p2", "upper": 7}),
]


def digest_source(function) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def transcript(function) -> dict:
    cases = []
    for case_id, kind, values, lower, upper, metadata in CASES:
        expected = brute_force(values, lower, upper)
        actual = function(values, lower, upper)
        cases.append({"case_id": case_id, "kind": kind, "expected": expected,
                      "actual": actual, **metadata})
    return {"task_id": TASK_ID, "variant": "count", "execution_status": "ok", "cases": cases}


def build() -> dict:
    agreement = all(brute_force(values, lower, upper) == reference(values, lower, upper)
                    for _, _, values, lower, upper, _ in CASES)
    reference_score = score(transcript(reference))
    exclusive_score = score(transcript(exclusive_upper))
    lower_score = score(transcript(exclusive_lower))
    if not agreement or not reference_score["full_correct"]:
        raise RuntimeError("independent oracle/reference disagreement")
    if exclusive_score["label"] != "upper_boundary_signature":
        raise RuntimeError("exclusive-upper mutant did not produce the named signature")
    if lower_score["full_correct"]:
        raise RuntimeError("unrelated mutant was not caught")
    test_material = json.dumps(CASES, separators=(",", ":"), sort_keys=False)
    return {
        "schema_version": "p16_kodcode_oracle_audit_v1",
        "task_id": TASK_ID,
        "family_id": "kc_60463e25cc5178ff79d9",
        "question_sha256": QUESTION_SHA256,
        "variant": "count",
        "universe_mode": "explicit_collection",
        "test_count": len(CASES),
        "oracle_reference_agree": agreement,
        "reference_score": reference_score,
        "exclusive_upper_score": exclusive_score,
        "unrelated_mutant_score": lower_score,
        "oracle_sha256": digest_source(brute_force),
        "reference_sha256": digest_source(reference),
        "tests_sha256": hashlib.sha256(test_material.encode()).hexdigest(),
        "oracle_status": "validated_on_trusted_cpu_cases",
        "history_status": "pending",
        "near_duplicate_status": "pending",
        "sandbox_status": "pending",
        "fully_certified": False,
        "limitations": "Trusted local functions only; no untrusted code or sandbox was exercised.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = build()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("task_id", "test_count", "oracle_status", "fully_certified")}, sort_keys=True))


if __name__ == "__main__":
    main()
