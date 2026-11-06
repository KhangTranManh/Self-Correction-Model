"""Run trusted CPU oracle checks for the batch-2 strobogrammatic task."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

try:
    from phase16.scripts.score_boundary import score
except ModuleNotFoundError:
    from score_boundary import score


TASK_ID = "Data_Structure_8706_I"
QUESTION_SHA256 = "ee8c7230a978027c8e0dcc66cf15dcc59d4580483ede2093fd59f3a776742c52"
ROTATE = {"0": "0", "1": "1", "6": "9", "8": "8", "9": "6"}


def is_strobogrammatic(value: int) -> bool:
    text = str(value)
    return all(character in ROTATE for character in text) and \
        "".join(ROTATE[character] for character in reversed(text)) == text


def brute_force(low: str, high: str, include_high: bool = True) -> int:
    left, right = int(low), int(high)
    stop = right + 1 if include_high else right
    return sum(is_strobogrammatic(value) for value in range(left, stop))


def generated_values(max_length: int) -> set[int]:
    current = ["", "0", "1", "8"]
    values: set[int] = set()
    pairs = (("0", "0"), ("1", "1"), ("6", "9"), ("8", "8"), ("9", "6"))
    for length in range(1, max_length + 1):
        if length == 1:
            layer = ["0", "1", "8"]
        else:
            inner_length = length - 2
            inner = [text for text in current if len(text) == inner_length]
            layer = [left + middle + right for left, right in pairs for middle in inner
                     if left != "0"]
        current.extend(layer)
        values.update(int(text) for text in layer)
    return values


def reference(low: str, high: str) -> int:
    left, right = int(low), int(high)
    return sum(left <= value <= right for value in generated_values(len(high)))


def exclusive_upper(low: str, high: str) -> int:
    return brute_force(low, high, include_high=False)


def exclusive_lower(low: str, high: str) -> int:
    left, right = int(low), int(high)
    return sum(is_strobogrammatic(value) for value in range(left + 1, right + 1))


CASES = [
    ("ordinary", "ordinary", "50", "100", {}),
    ("lower", "lower", "11", "20", {}),
    ("outside", "outside", "12", "20", {}),
    ("empty", "empty", "12", "20", {}),
    ("single", "single_point", "69", "69", {}),
    ("sparse", "duplicate", "100", "110", {}),
    ("p1-anchor", "upper_anchor", "50", "68", {"pair_id": "p1", "upper": 69}),
    ("p1-added", "upper_added", "50", "69", {"pair_id": "p1", "upper": 69}),
    ("p2-anchor", "upper_anchor", "80", "87", {"pair_id": "p2", "upper": 88}),
    ("p2-added", "upper_added", "80", "88", {"pair_id": "p2", "upper": 88}),
]


def digest_source(function) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def transcript(function) -> dict:
    cases = []
    for case_id, kind, low, high, metadata in CASES:
        cases.append({"case_id": case_id, "kind": kind,
                      "expected": brute_force(low, high),
                      "actual": function(low, high), **metadata})
    return {"task_id": TASK_ID, "variant": "count", "execution_status": "ok", "cases": cases}


def build() -> dict:
    agreement = all(brute_force(low, high) == reference(low, high)
                    for _, _, low, high, _ in CASES)
    reference_score = score(transcript(reference))
    exclusive_score = score(transcript(exclusive_upper))
    lower_score = score(transcript(exclusive_lower))
    if not agreement or not reference_score["full_correct"]:
        raise RuntimeError("independent oracle/reference disagreement")
    if exclusive_score["label"] != "upper_boundary_signature":
        raise RuntimeError("exclusive-upper mutant did not produce the named signature")
    if lower_score["full_correct"]:
        raise RuntimeError("unrelated mutant was not caught")
    test_material = json.dumps(CASES, separators=(",", ":"))
    return {
        "schema_version": "p16_kodcode_oracle_audit_v1",
        "task_id": TASK_ID, "family_id": "kc_4832b255505000ec819a",
        "question_sha256": QUESTION_SHA256, "variant": "count",
        "universe_mode": "implicit_integer_range", "test_count": len(CASES),
        "oracle_reference_agree": agreement, "reference_score": reference_score,
        "exclusive_upper_score": exclusive_score, "unrelated_mutant_score": lower_score,
        "oracle_sha256": digest_source(brute_force),
        "reference_sha256": digest_source(reference),
        "tests_sha256": hashlib.sha256(test_material.encode()).hexdigest(),
        "oracle_status": "validated_on_trusted_cpu_cases",
        "history_status": "exact_prior_clean_near_duplicate_pending",
        "sandbox_status": "pending", "fully_certified": False,
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

