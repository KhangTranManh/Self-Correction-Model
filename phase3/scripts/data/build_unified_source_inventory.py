"""Build the additive Phase 3 unified source/behavior-eligibility inventory.

This script joins metadata and references only. It does not copy prompts, tests,
model outputs, or reference solutions into the unified inventory, and it does
not construct SFT conversations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "behavior"

ORIGINAL_PATHS = {
    "inventory": ROOT / "data" / "source_inventory.jsonl",
    "source": ROOT / "data" / "source" / "problems.jsonl",
    "assignments": ROOT / "data" / "buckets" / "bucket_assignments.jsonl",
    "base_attempts": ROOT / "data" / "attempts" / "base" / "raw_attempts.jsonl",
    "v1_attempts": ROOT / "data" / "attempts" / "self_correction_v1" / "raw_attempts.jsonl",
}

APPS_PATHS = {
    "source": ROOT / "data" / "apps_pilot" / "candidates" / "code_candidates.jsonl",
    "assignments": ROOT / "data" / "apps_pilot" / "buckets" / "bucket_assignments.jsonl",
    "base_attempts": ROOT / "data" / "apps_pilot" / "attempts" / "base" / "raw_attempts.jsonl",
    "v1_attempts": ROOT / "data" / "apps_pilot" / "attempts" / "self_correction_v1" / "raw_attempts.jsonl",
}

BUCKETS = ("CC", "WW", "WC", "CW")
BEHAVIOR_MAP = {
    "CC": ["preserve_false_feedback", "preserve_neutral"],
    "WW": ["repair_true_feedback", "repair_neutral"],
    "WC": ["preserve_false_feedback", "preserve_neutral", "normal_solve"],
    "CW": ["regression_recovery", "normal_solve"],
}
TRANSITION_MAP = {
    "CC": "stable_correct",
    "WW": "stable_wrong",
    "WC": "improved",
    "CW": "regressed",
}
BEHAVIOR_ROLE_MAP = {
    "CC": ["preserve"],
    "WW": ["repair"],
    "WC": ["preserve", "normal_solve"],
    "CW": ["regression_recovery", "normal_solve"],
}
ACTION_MAP = {
    "CC": ["KEEP"],
    "WW": ["REVISE"],
    "WC": ["KEEP", "SOLVE"],
    "CW": ["REVISE", "SOLVE"],
}
EXPECTED_BUCKET_STATE = {
    "CC": (True, True),
    "WW": (False, False),
    "WC": (False, True),
    "CW": (True, False),
}
REQUIRED_FIELDS = (
    "id",
    "dataset",
    "domain",
    "bucket",
    "base_correct",
    "v1_correct",
    "eligible_behaviors",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _index(rows: list[dict[str, Any]], key: str, source: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, start=1):
        value = str(row[key])
        if value in result:
            raise ValueError(f"Duplicate {key}={value!r} in {source}:{line_number}")
        result[value] = row
    return result


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _ref(path: Path, problem_id: str) -> str:
    return f"{_relative(path)}#id={problem_id}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _states(bucket: str) -> tuple[bool, bool]:
    if bucket not in EXPECTED_BUCKET_STATE:
        raise ValueError(f"Unknown transition bucket {bucket!r}")
    return EXPECTED_BUCKET_STATE[bucket]


def _unified_row(
    *,
    source: dict[str, Any],
    assignment: dict[str, Any],
    base_attempt: dict[str, Any],
    v1_attempt: dict[str, Any],
    origin: str,
    source_path: Path,
    base_attempt_path: Path,
    v1_attempt_path: Path,
) -> dict[str, Any]:
    problem_id = str(source["id"])
    bucket = str(assignment["bucket"])
    expected_base, expected_v1 = _states(bucket)
    base_correct = bool(base_attempt["initial_correct"])
    v1_correct = bool(v1_attempt["initial_correct"])
    if (base_correct, v1_correct) != (expected_base, expected_v1):
        raise ValueError(
            f"Bucket/correctness mismatch for {problem_id}: bucket={bucket}, "
            f"base={base_correct}, v1={v1_correct}"
        )

    dataset = str(source["dataset"])
    if dataset == "apps":
        source_task_id: Any = source.get("problem_id")
    elif dataset == "mbpp":
        source_task_id = source.get("task_id")
    else:
        source_task_id = source.get("source_index")

    row: dict[str, Any] = {
        "id": problem_id,
        "dataset": dataset,
        "source_dataset": dataset,
        "domain": source["domain"],
        "split": source["split"],
        "source_split": source["split"],
        "source_index": source.get("source_index"),
        "source_task_id": source_task_id,
        "source_origin": origin,
        "source_ref": _ref(source_path, problem_id),
        "bucket": bucket,
        "base_correct": base_correct,
        "v1_correct": v1_correct,
        "base_initial_state": "correct" if base_correct else "wrong",
        "initial_state": "correct" if v1_correct else "wrong",
        "phase3_initial_state": "correct" if v1_correct else "wrong",
        "transition_type": TRANSITION_MAP[bucket],
        "eligible_behaviors": list(BEHAVIOR_MAP[bucket]),
        "behavior_role": list(BEHAVIOR_ROLE_MAP[bucket]),
        "eligible_actions": list(ACTION_MAP[bucket]),
        "base_attempt_ref": _ref(base_attempt_path, problem_id),
        "v1_attempt_ref": _ref(v1_attempt_path, problem_id),
    }
    if dataset == "apps":
        row.update(
            difficulty=source.get("difficulty"),
            source_host=source.get("source_host"),
            test_mode=source.get("test_mode"),
            test_count=source.get("test_count"),
        )
    return row


def _join_original() -> list[dict[str, Any]]:
    inventory_rows = _read_jsonl(ORIGINAL_PATHS["inventory"])
    source_rows = _read_jsonl(ORIGINAL_PATHS["source"])
    assignment_rows = _read_jsonl(ORIGINAL_PATHS["assignments"])
    base_rows = _read_jsonl(ORIGINAL_PATHS["base_attempts"])
    v1_rows = _read_jsonl(ORIGINAL_PATHS["v1_attempts"])
    inventory = _index(inventory_rows, "id", ORIGINAL_PATHS["inventory"])
    source = _index(source_rows, "id", ORIGINAL_PATHS["source"])
    assignments = _index(assignment_rows, "problem_id", ORIGINAL_PATHS["assignments"])
    base = _index(base_rows, "id", ORIGINAL_PATHS["base_attempts"])
    v1 = _index(v1_rows, "id", ORIGINAL_PATHS["v1_attempts"])
    id_sets = [set(mapping) for mapping in (inventory, source, assignments, base, v1)]
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        raise ValueError("Original Phase 3 source/inventory/assignment/attempt ID sets differ")

    unified = []
    for source_row in source_rows:
        problem_id = str(source_row["id"])
        metadata = inventory[problem_id]
        assignment = assignments[problem_id]
        if any(
            metadata[field] != assignment[field_in_assignment]
            for field, field_in_assignment in (
                ("bucket", "bucket"),
                ("base_correct", "base_initial_correct"),
                ("v1_correct", "v1_initial_correct"),
            )
        ):
            raise ValueError(f"Original inventory/assignment mismatch for {problem_id}")
        unified.append(
            _unified_row(
                source=source_row,
                assignment=assignment,
                base_attempt=base[problem_id],
                v1_attempt=v1[problem_id],
                origin="original_phase3",
                source_path=ORIGINAL_PATHS["source"],
                base_attempt_path=ORIGINAL_PATHS["base_attempts"],
                v1_attempt_path=ORIGINAL_PATHS["v1_attempts"],
            )
        )
    return unified


def _join_apps() -> list[dict[str, Any]]:
    source_rows = _read_jsonl(APPS_PATHS["source"])
    assignment_rows = _read_jsonl(APPS_PATHS["assignments"])
    base_rows = _read_jsonl(APPS_PATHS["base_attempts"])
    v1_rows = _read_jsonl(APPS_PATHS["v1_attempts"])
    source = _index(source_rows, "id", APPS_PATHS["source"])
    assignments = _index(assignment_rows, "problem_id", APPS_PATHS["assignments"])
    base = _index(base_rows, "id", APPS_PATHS["base_attempts"])
    v1 = _index(v1_rows, "id", APPS_PATHS["v1_attempts"])
    id_sets = [set(mapping) for mapping in (source, assignments, base, v1)]
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        raise ValueError("APPS source/assignment/attempt ID sets differ")
    unified = []
    for source_row in source_rows:
        problem_id = str(source_row["id"])
        unified.append(
            _unified_row(
                source=source_row,
                assignment=assignments[problem_id],
                base_attempt=base[problem_id],
                v1_attempt=v1[problem_id],
                origin="apps_pilot",
                source_path=APPS_PATHS["source"],
                base_attempt_path=APPS_PATHS["base_attempts"],
                v1_attempt_path=APPS_PATHS["v1_attempts"],
            )
        )
    return unified


def _nested_counts(rows: list[dict[str, Any]], outer: str, inner: str) -> dict[str, dict[str, int]]:
    outer_values = sorted({str(row[outer]) for row in rows})
    inner_values = BUCKETS if inner == "bucket" else sorted({str(row[inner]) for row in rows})
    return {
        outer_value: {
            inner_value: sum(
                str(row[outer]) == outer_value and str(row[inner]) == inner_value
                for row in rows
            )
            for inner_value in inner_values
        }
        for outer_value in outer_values
    }


def _behavior_cross_counts(rows: list[dict[str, Any]], dimension: str) -> dict[str, dict[str, int]]:
    dimension_values = sorted({str(row[dimension]) for row in rows})
    behaviors = sorted({behavior for row in rows for behavior in row["eligible_behaviors"]})
    return {
        behavior: {
            value: sum(
                str(row[dimension]) == value and behavior in row["eligible_behaviors"]
                for row in rows
            )
            for value in dimension_values
        }
        for behavior in behaviors
    }


def _state_coverage(rows: list[dict[str, Any]], correct: bool) -> dict[str, Any]:
    selected = [row for row in rows if row["v1_correct"] is correct]
    return {
        "total": len(selected),
        "math": sum(row["domain"] == "math" for row in selected),
        "code": sum(row["domain"] == "code" for row in selected),
        "by_dataset": dict(sorted(Counter(row["dataset"] for row in selected).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    inventory_path = output_dir / "unified_source_inventory.jsonl"
    summary_path = output_dir / "unified_source_summary.json"

    original = _join_original()
    apps = _join_apps()
    rows = original + apps
    ids = [row["id"] for row in rows]
    duplicate_ids = sorted(problem_id for problem_id, count in Counter(ids).items() if count > 1)
    inconsistent_rows = [
        row["id"]
        for row in rows
        if (row["base_correct"], row["v1_correct"]) != EXPECTED_BUCKET_STATE.get(row["bucket"])
    ]
    missing_required_fields = [
        {"id": row.get("id"), "fields": [field for field in REQUIRED_FIELDS if field not in row or row[field] is None]}
        for row in rows
        if any(field not in row or row[field] is None for field in REQUIRED_FIELDS)
    ]
    invalid_eligibility = [
        row["id"] for row in rows if row["eligible_behaviors"] != BEHAVIOR_MAP.get(row["bucket"])
    ]
    humaneval_rows = [row["id"] for row in rows if "humaneval" in row["dataset"].casefold()]
    frozen_eval_rows = [
        row["id"]
        for row in rows
        if not (
            (row["dataset"] == "gsm8k" and row["split"] == "train")
            or (row["dataset"] == "mbpp" and row["split"] in {"train", "validation", "prompt"})
            or (row["dataset"] == "apps" and row["split"] == "train")
        )
    ]

    counts_by_dataset = dict(sorted(Counter(row["dataset"] for row in rows).items()))
    counts_by_domain = dict(sorted(Counter(row["domain"] for row in rows).items()))
    counts_by_bucket = {bucket: sum(row["bucket"] == bucket for row in rows) for bucket in BUCKETS}
    dataset_bucket = _nested_counts(rows, "dataset", "bucket")
    domain_bucket = _nested_counts(rows, "domain", "bucket")
    expected_dataset_bucket = {
        "gsm8k": {"CC": 692, "WW": 85, "WC": 181, "CW": 42},
        "mbpp": {"CC": 24, "WW": 434, "WC": 5, "CW": 11},
        "apps": {"CC": 120, "WW": 132, "WC": 21, "CW": 27},
    }
    expected_combined = {"CC": 836, "WW": 651, "WC": 207, "CW": 80}
    count_discrepancies = {
        "counts_by_dataset_and_bucket": {
            dataset: {
                bucket: {"expected": expected, "actual": dataset_bucket.get(dataset, {}).get(bucket)}
                for bucket, expected in bucket_counts.items()
                if dataset_bucket.get(dataset, {}).get(bucket) != expected
            }
            for dataset, bucket_counts in expected_dataset_bucket.items()
            if any(dataset_bucket.get(dataset, {}).get(bucket) != expected for bucket, expected in bucket_counts.items())
        },
        "combined_bucket_counts": {
            bucket: {"expected": expected, "actual": counts_by_bucket.get(bucket)}
            for bucket, expected in expected_combined.items()
            if counts_by_bucket.get(bucket) != expected
        },
    }

    validations = {
        "expected_total_1774": len(rows) == 1774,
        "all_ids_unique": not duplicate_ids and len(set(ids)) == len(rows),
        "original_row_count_unchanged": len(original) == 1474,
        "apps_row_count_300": len(apps) == 300,
        "known_dataset_bucket_counts_match": not any(count_discrepancies.values()),
        "no_inconsistent_rows": not inconsistent_rows,
        "no_missing_required_fields": not missing_required_fields,
        "eligibility_mapping_exact": not invalid_eligibility,
        "no_frozen_evaluation_rows": not frozen_eval_rows,
        "no_humaneval_rows": not humaneval_rows,
    }
    all_passed = all(validations.values())
    if not all_passed:
        raise RuntimeError(
            "Unified inventory validation failed before write: "
            + json.dumps({key: value for key, value in validations.items() if not value})
        )

    _write_jsonl(inventory_path, rows)
    behavior_counts = {
        behavior: sum(behavior in row["eligible_behaviors"] for row in rows)
        for behavior in sorted({behavior for row in rows for behavior in row["eligible_behaviors"]})
    }
    keep_by_domain = {
        domain: sum("KEEP" in row["eligible_actions"] and row["domain"] == domain for row in rows)
        for domain in ("math", "code")
    }
    revise_by_domain = {
        domain: sum("REVISE" in row["eligible_actions"] and row["domain"] == domain for row in rows)
        for domain in ("math", "code")
    }
    summary = {
        "total_rows": len(rows),
        "unique_ids": len(set(ids)),
        "counts_by_dataset": counts_by_dataset,
        "counts_by_domain": counts_by_domain,
        "counts_by_bucket": counts_by_bucket,
        "counts_by_dataset_and_bucket": dataset_bucket,
        "counts_by_domain_and_bucket": domain_bucket,
        "eligible_rows_by_behavior": behavior_counts,
        "behavior_by_domain": _behavior_cross_counts(rows, "domain"),
        "behavior_by_dataset": _behavior_cross_counts(rows, "dataset"),
        "eligible_action_source_counts_by_domain": {
            "KEEP": keep_by_domain,
            "REVISE": revise_by_domain,
        },
        "v1_correct_source_pool": _state_coverage(rows, True),
        "v1_wrong_source_pool": _state_coverage(rows, False),
        "code_preserve_source_count": sum(
            row["domain"] == "code" and "preserve" in row["behavior_role"] for row in rows
        ),
        "code_repair_source_count": sum(
            row["domain"] == "code" and "repair" in row["behavior_role"] for row in rows
        ),
        "code_regression_recovery_source_count": sum(
            row["domain"] == "code" and "regression_recovery" in row["behavior_role"] for row in rows
        ),
        "code_revise_source_count_including_regression_recovery": revise_by_domain["code"],
        "inconsistent_rows": inconsistent_rows,
        "duplicate_ids": duplicate_ids,
        "missing_required_fields": missing_required_fields,
        "invalid_eligibility_rows": invalid_eligibility,
        "frozen_evaluation_rows": frozen_eval_rows,
        "humaneval_rows": humaneval_rows,
        "count_discrepancies": count_discrepancies,
        "validation": {**validations, "all_passed": all_passed},
        "output": {
            "inventory": _relative(inventory_path),
            "inventory_sha256": _sha256(inventory_path),
            "summary": _relative(summary_path),
        },
        "inputs": {
            **{
                f"original_{name}": {"path": _relative(path), "sha256": _sha256(path)}
                for name, path in ORIGINAL_PATHS.items()
            },
            **{
                f"apps_{name}": {"path": _relative(path), "sha256": _sha256(path)}
                for name, path in APPS_PATHS.items()
            },
        },
        "notes": {
            "initial_state_semantics": "initial_state and phase3_initial_state refer to Self_Correction_v1 correctness, the checkpoint entering Phase 3",
            "revise_semantics": "REVISE includes WW repair and CW regression recovery; code_repair_source_count reports WW repair only",
            "cw_constraint": "CW rows are eligible for regression_recovery and normal_solve, never ordinary repair",
            "source_content": "Prompts, tests, solutions, and raw model outputs remain in referenced source/attempt artifacts",
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    concise = {
        "output_paths": {
            "inventory": str(inventory_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
        "total_unified_rows": len(rows),
        "counts_by_dataset": counts_by_dataset,
        "counts_by_bucket": counts_by_bucket,
        "v1_correct_math": summary["v1_correct_source_pool"]["math"],
        "v1_correct_code": summary["v1_correct_source_pool"]["code"],
        "eligible_KEEP_by_domain": keep_by_domain,
        "eligible_REVISE_by_domain": revise_by_domain,
        "code_repair_only": summary["code_repair_source_count"],
        "duplicates": len(duplicate_ids),
        "inconsistencies": len(inconsistent_rows),
        "missing_required_fields": len(missing_required_fields),
        "all_validation_checks_passed": all_passed,
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
