"""Validate the frozen P16-S0 synthetic-development pool manifest.

This validates recorded evidence and split invariants. It does not execute
candidate code or independently establish that an oracle claim is true.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


ROLE_COUNTS = {
    "screen": 160,
    "positive_probe": 40,
    "near_negative": 20,
    "unrelated_control": 20,
    "reserve": 60,
}
APPLICABLE_ROLES = {"screen", "positive_probe", "reserve"}
VARIANT_COUNTS = {
    ("screen", "count"): 54,
    ("screen", "sum"): 53,
    ("screen", "select"): 53,
    ("positive_probe", "count"): 14,
    ("positive_probe", "sum"): 13,
    ("positive_probe", "select"): 13,
    ("reserve", "count"): 20,
    ("reserve", "sum"): 20,
    ("reserve", "select"): 20,
}
HASH_FIELDS = ("prompt_sha256", "oracle_sha256", "reference_sha256", "tests_sha256")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(document: dict) -> dict:
    _require(document.get("schema_version") == "p16_synthetic_development_pool_v1",
             "unexpected schema_version")
    _require(document.get("source_revision") ==
             "97d81f0695ddd7c5c02bfe16436db9a17a4a21e5",
             "unexpected source_revision")
    tasks = document.get("tasks")
    _require(isinstance(tasks, list), "tasks must be a list")
    _require(len(tasks) == 300, "manifest must contain exactly 300 tasks")

    roles: Counter[str] = Counter()
    variants: Counter[tuple[str, str]] = Counter()
    applicable_subsets: Counter[str] = Counter()
    seen: dict[str, set[str]] = {key: set() for key in
                                 ("task_id", "family_id", "prompt_sha256")}

    for index, task in enumerate(tasks):
        _require(isinstance(task, dict), f"task {index} is not an object")
        prefix = f"task {index}"
        for key in ("task_id", "family_id", "subset"):
            value = task.get(key)
            _require(isinstance(value, str) and value, f"{prefix}: missing {key}")
        for key in seen:
            value = task.get(key)
            _require(isinstance(value, str) and value, f"{prefix}: missing {key}")
            _require(value not in seen[key], f"{prefix}: duplicate {key}")
            seen[key].add(value)
        for key in HASH_FIELDS:
            value = task.get(key)
            _require(isinstance(value, str) and len(value) == 64 and
                     all(char in "0123456789abcdef" for char in value),
                     f"{prefix}: invalid {key}")

        role = task.get("role")
        _require(role in ROLE_COUNTS, f"{prefix}: invalid role")
        roles[role] += 1
        _require(task.get("contract_reviewed") is True,
                 f"{prefix}: contract not reviewed")
        _require(task.get("history_status") == "clean",
                 f"{prefix}: history status is not clean")
        _require(task.get("near_duplicate_status") == "clean",
                 f"{prefix}: near-duplicate status is not clean")
        _require(task.get("license_status") == "accepted_for_noncommercial_research",
                 f"{prefix}: license status not accepted")
        _require(task.get("oracle_validated") is True,
                 f"{prefix}: oracle not validated")
        _require(task.get("sandbox_smoke") is True,
                 f"{prefix}: sandbox smoke missing")
        _require(task.get("deterministic_rerun") is True,
                 f"{prefix}: deterministic rerun missing")
        _require(isinstance(task.get("test_count"), int) and task["test_count"] >= 10,
                 f"{prefix}: fewer than 10 frozen tests")
        _require(task.get("reference_all_pass") is True,
                 f"{prefix}: reference does not pass all tests")
        _require(task.get("unrelated_mutant_caught") is True,
                 f"{prefix}: unrelated mutant not caught")

        applies = role in APPLICABLE_ROLES
        _require(task.get("rule_applicable") is applies,
                 f"{prefix}: role/rule_applicable mismatch")
        variant = task.get("variant")
        if applies:
            _require(variant in {"count", "sum", "select"},
                     f"{prefix}: invalid applicable variant")
            _require(task.get("universe_mode") in
                     {"explicit_collection", "implicit_integer_range"},
                     f"{prefix}: invalid universe_mode")
            _require(isinstance(task.get("upper_pair_count"), int) and
                     task["upper_pair_count"] >= 2,
                     f"{prefix}: fewer than two upper-bound pairs")
            _require(task.get("exclusive_upper_signature") is True,
                     f"{prefix}: exclusive-upper signature not established")
            variants[(role, variant)] += 1
            applicable_subsets[task["subset"]] += 1
        else:
            _require(variant == "none", f"{prefix}: control variant must be none")
            _require(task.get("universe_mode") == "not_applicable",
                     f"{prefix}: control universe_mode must be not_applicable")
            _require(task.get("exclusive_upper_signature") is False,
                     f"{prefix}: control cannot claim exclusive-upper signature")

    _require(dict(roles) == ROLE_COUNTS,
             f"role counts differ: {dict(roles)}")
    _require(dict(variants) == VARIANT_COUNTS,
             f"role/variant counts differ: {dict(variants)}")
    _require(len(applicable_subsets) >= 5,
             "applicable tasks must span at least five subsets")
    _require(max(applicable_subsets.values()) <= 60,
             "a subset contributes more than 60 applicable tasks")
    return {
        "schema_version": "p16_synthetic_development_pool_validation_v1",
        "valid": True,
        "task_count": len(tasks),
        "role_counts": dict(sorted(roles.items())),
        "applicable_count": sum(applicable_subsets.values()),
        "applicable_subset_counts": dict(sorted(applicable_subsets.items())),
        "family_count": len(seen["family_id"]),
        "limitations": "Recorded-evidence validation only; no code was executed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(json.loads(args.manifest.read_text(encoding="utf-8")))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            parser.error(f"refusing to overwrite {args.output}")
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

