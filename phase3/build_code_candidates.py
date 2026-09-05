"""Build the Phase 3 code-expansion candidate pool from local MBPP artifacts.

This step is intentionally inference-free.  It audits the MBPP rows already in
the repository, excludes every row used by training/source construction or by a
documented evaluation range, and writes only previously unused candidates.

The repository documents the stable MBPP task_id layout in
``phase1/src/pipeline/prepare_datasets.py``:

* prompt: task_id 1..10
* test: task_id 11..510
* validation: task_id 511..600
* train: task_id 601..974

It also documents test task_id 11..160 as legacy training data and task_id
161..510 as held-out evaluation data.  The wider historical held-out range is
used here even though the current P0 command defaults to test offset 200
(task_id 211), because narrowing the protection would contaminate earlier evals.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SELECTION_SEED = 314159
MBPP_TASK_IDS = frozenset(range(1, 975))
MBPP_SPLIT_RANGES = {
    "prompt": range(1, 11),
    "test": range(11, 511),
    "validation": range(511, 601),
    "train": range(601, 975),
}
LEGACY_TEST_TRAIN_IDS = frozenset(range(11, 161))
HISTORICAL_HELD_OUT_IDS = frozenset(range(161, 511))
CURRENT_P0_DEFAULT_IDS = frozenset(range(211, 511))

PHASE3_DIR = Path(__file__).resolve().parent
REPO_DIR = PHASE3_DIR.parent
DEFAULT_SOURCE_PROBLEMS = PHASE3_DIR / "data" / "source" / "problems.jsonl"
DEFAULT_SOURCE_INVENTORY = PHASE3_DIR / "data" / "source_inventory.jsonl"
DEFAULT_PHASE1_CODE = REPO_DIR / "phase1" / "data" / "problems" / "code.jsonl"
DEFAULT_OUTPUT_DIR = PHASE3_DIR / "data" / "expansion"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def _task_id(row: dict[str, Any]) -> int:
    raw = row.get("task_id")
    if isinstance(raw, int):
        return raw
    match = re.fullmatch(r"mbpp(?:_(?:train|validation|prompt))?_(\d+)", str(row.get("id", "")))
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot trace MBPP task_id from row id={row.get('id')!r}")


def _split_for_task_id(task_id: int) -> str:
    for split, task_range in MBPP_SPLIT_RANGES.items():
        if task_id in task_range:
            return split
    raise ValueError(f"MBPP task_id {task_id} is outside the documented 1..974 range")


def _prompt(row: dict[str, Any]) -> str:
    return str(row.get("problem") or row.get("question") or "")


def _normalized_prompt(row: dict[str, Any]) -> str:
    return " ".join(_prompt(row).split()).casefold()


def _tests_status(row: dict[str, Any]) -> tuple[bool, str | None]:
    tests = row.get("tests")
    if not isinstance(tests, list) or not tests or not all(
        isinstance(test, str) and test.strip() for test in tests
    ):
        return False, "missing_or_empty_tests"
    try:
        ast.parse("\n".join(tests))
    except SyntaxError as exc:
        return False, f"tests_syntax_error:{exc.msg}"
    return True, None


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_DIR).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_problems_path = Path(args.source_problems)
    source_inventory_path = Path(args.source_inventory)
    phase1_code_path = Path(args.phase1_code)
    output_dir = Path(args.output_dir)

    source_rows_all = _read_jsonl(source_problems_path)
    inventory_rows_all = _read_jsonl(source_inventory_path)
    phase1_rows_all = _read_jsonl(phase1_code_path)

    source_rows = [row for row in source_rows_all if row.get("dataset") == "mbpp"]
    inventory_rows = [row for row in inventory_rows_all if row.get("dataset") == "mbpp"]
    phase1_rows = [row for row in phase1_rows_all if str(row.get("id", "")).startswith("mbpp_")]

    inventory_ids = [str(row["id"]) for row in inventory_rows]
    source_ids = [str(row["id"]) for row in source_rows]
    if len(inventory_ids) != len(set(inventory_ids)):
        raise ValueError("Duplicate MBPP IDs found in Phase 3 source_inventory.jsonl")
    if set(inventory_ids) != set(source_ids):
        missing_metadata = sorted(set(source_ids) - set(inventory_ids))
        missing_source = sorted(set(inventory_ids) - set(source_ids))
        raise ValueError(
            "Phase 3 MBPP source/inventory mismatch: "
            f"missing_metadata={missing_metadata[:5]}, missing_source={missing_source[:5]}"
        )

    source_by_task = {_task_id(row): row for row in source_rows}
    inventory_task_ids = {_task_id(row) for row in source_rows if row["id"] in set(inventory_ids)}
    expected_train_side = (
        set(MBPP_SPLIT_RANGES["prompt"])
        | set(MBPP_SPLIT_RANGES["validation"])
        | set(MBPP_SPLIT_RANGES["train"])
    )
    if set(source_by_task) != expected_train_side:
        raise ValueError(
            "Phase 3 source does not contain exactly the documented MBPP "
            f"train-side task IDs; missing={sorted(expected_train_side - set(source_by_task))[:10]}, "
            f"unexpected={sorted(set(source_by_task) - expected_train_side)[:10]}"
        )

    local_views: dict[int, list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    for row in source_rows:
        local_views[_task_id(row)].append((row, source_problems_path))
    for row in phase1_rows:
        local_views[_task_id(row)].append((row, phase1_code_path))

    invalid_test_ids: dict[int, str] = {}
    missing_prompt_ids: set[int] = set()
    cross_file_conflicts: list[int] = []
    for task_id, views in local_views.items():
        prompts = {_prompt(row) for row, _ in views}
        test_lists = {tuple(row.get("tests") or []) for row, _ in views}
        if len(prompts) != 1 or len(test_lists) != 1:
            cross_file_conflicts.append(task_id)
        representative = views[0][0]
        if not _prompt(representative).strip():
            missing_prompt_ids.add(task_id)
        tests_ok, reason = _tests_status(representative)
        if not tests_ok:
            invalid_test_ids[task_id] = reason or "invalid_tests"

    if cross_file_conflicts:
        raise ValueError(
            "Conflicting prompt/tests for the same MBPP task across local files: "
            f"{sorted(cross_file_conflicts)[:10]}"
        )

    phase1_task_ids = {_task_id(row) for row in phase1_rows}
    locally_materialized_ids = set(local_views)
    legacy_materialized_ids = phase1_task_ids & LEGACY_TEST_TRAIN_IDS
    already_used_ids = inventory_task_ids | legacy_materialized_ids
    reserved_eval_ids = set(HISTORICAL_HELD_OUT_IDS)

    if already_used_ids & reserved_eval_ids:
        raise ValueError("Documented training/source and held-out MBPP task ranges overlap")
    if already_used_ids | reserved_eval_ids != set(MBPP_TASK_IDS):
        raise ValueError(
            "The local project definitions do not account for the complete MBPP universe"
        )

    # Candidate construction happens before and without any Base/V1 correctness data.
    candidate_ids = sorted(
        locally_materialized_ids
        - already_used_ids
        - reserved_eval_ids
        - set(invalid_test_ids)
        - missing_prompt_ids
    )
    candidates: list[dict[str, Any]] = []
    for task_id in candidate_ids:
        row, source_file = local_views[task_id][0]
        tests = list(row["tests"])
        candidates.append(
            {
                "id": str(row["id"]),
                "dataset": "mbpp",
                "split": _split_for_task_id(task_id),
                "domain": "code",
                "source_index": row.get("source_index", task_id),
                "task_id": task_id,
                "prompt": _prompt(row),
                "reference_answer": row.get("ground_truth"),
                "tests": tests,
                "test_code": "\n".join(tests),
                "entry_point": row.get("entry_point"),
                "source_file": _relative(source_file),
                "selection_reason": "unused_train_code_problem",
                "already_used": False,
                "reserved_for_eval": False,
                "eligible_for_expansion": True,
            }
        )

    candidate_id_values = [row["id"] for row in candidates]
    candidate_content_values = [_normalized_prompt(row) for row in candidates]
    duplicate_candidate_ids = sum(count - 1 for count in Counter(candidate_id_values).values() if count > 1)
    duplicate_candidate_content = sum(
        count - 1 for count in Counter(candidate_content_values).values() if count > 1
    )
    existing_source_overlap = len({row["task_id"] for row in candidates} & inventory_task_ids)
    frozen_eval_overlap = len({row["task_id"] for row in candidates} & reserved_eval_ids)
    missing_candidate_tests = sum(not _tests_status(row)[0] for row in candidates)
    if any(
        (
            duplicate_candidate_ids,
            duplicate_candidate_content,
            existing_source_overlap,
            frozen_eval_overlap,
            missing_candidate_tests,
        )
    ):
        raise RuntimeError("Candidate validation failed; refusing to write unsafe output")

    exclusions: list[dict[str, Any]] = []
    for task_id in sorted(MBPP_TASK_IDS):
        split = _split_for_task_id(task_id)
        views = local_views.get(task_id, [])
        if task_id in inventory_task_ids:
            reason = "already_in_phase3_source_inventory"
        elif task_id in legacy_materialized_ids:
            reason = "already_used_for_phase1_training"
        elif task_id in reserved_eval_ids:
            reason = "protected_historical_held_out_evaluation"
        elif task_id in invalid_test_ids:
            reason = invalid_test_ids[task_id]
        elif task_id in missing_prompt_ids:
            reason = "missing_prompt"
        else:
            continue
        source_files = sorted({_relative(path) for _, path in views})
        exclusions.append(
            {
                "id": str(views[0][0]["id"]) if views else f"mbpp_{task_id}",
                "dataset": "mbpp",
                "split": split,
                "domain": "code",
                "task_id": task_id,
                "source_index": task_id - MBPP_SPLIT_RANGES[split].start,
                "source_materialized_locally": bool(views),
                "source_files": source_files,
                "exclusion_reason": reason,
                "already_used": task_id in already_used_ids,
                "reserved_for_eval": task_id in reserved_eval_ids,
                "eligible_for_expansion": False,
            }
        )

    exclusion_counts = dict(sorted(Counter(row["exclusion_reason"] for row in exclusions).items()))
    cross_file_duplicate_task_ids = sorted(
        task_id for task_id, views in local_views.items() if len(views) > 1
    )
    content_groups: dict[str, set[int]] = defaultdict(set)
    for task_id, views in local_views.items():
        content_groups[_normalized_prompt(views[0][0])].add(task_id)
    local_duplicate_content_groups = [
        sorted(task_ids) for task_ids in content_groups.values() if len(task_ids) > 1
    ]

    candidate_path = output_dir / "code_candidates.jsonl"
    summary_path = output_dir / "candidate_summary.json"
    exclusions_path = output_dir / "exclusions.jsonl"
    _write_jsonl(candidate_path, candidates)
    _write_jsonl(exclusions_path, exclusions)

    summary: dict[str, Any] = {
        "source_dataset": "mbpp",
        "candidate_count": len(candidates),
        "unique_candidate_ids": len(set(candidate_id_values)),
        "existing_source_overlap": existing_source_overlap,
        "frozen_eval_overlap": frozen_eval_overlap,
        "duplicate_id_count": duplicate_candidate_ids,
        "duplicate_content_count": duplicate_candidate_content,
        "missing_tests_count": missing_candidate_tests,
        "selection_seed": SELECTION_SEED,
        "source_split": "none_available",
        "purpose": "increase unique code-correct source coverage for Phase 3 behavior training",
        "selection_used_model_correctness": False,
        "total_mbpp_rows_inspected": len(source_rows) + len(phase1_rows),
        "unique_mbpp_source_tasks_inspected": len(locally_materialized_ids),
        "source_inventory_rows_inspected": len(inventory_rows),
        "documented_mbpp_universe_size": len(MBPP_TASK_IDS),
        "rows_already_used": len(already_used_ids),
        "rows_in_current_source_inventory": len(inventory_task_ids),
        "rows_already_used_for_phase1_training_outside_current_inventory": len(legacy_materialized_ids),
        "rows_reserved_for_evaluation": len(reserved_eval_ids),
        "rows_rejected_for_invalid_or_missing_tests": len(invalid_test_ids),
        "rows_rejected_for_missing_prompt": len(missing_prompt_ids),
        "rows_remaining_as_eligible_candidates": len(candidates),
        "exclusion_counts": exclusion_counts,
        "local_audit": {
            "cross_file_duplicate_task_id_count": len(cross_file_duplicate_task_ids),
            "cross_file_duplicate_task_ids_are_identical": True,
            "duplicate_problem_content_group_count": len(local_duplicate_content_groups),
            "conflicting_cross_file_task_ids": cross_file_conflicts,
        },
        "protected_ranges": {
            "legacy_phase1_training": "MBPP full/test task_id 11..160",
            "historical_held_out_evaluation": "MBPP full/test task_id 161..510",
            "current_p0_default_subset": "MBPP full/test task_id 211..510 (offset 200, 300 rows)",
            "protection_applied": "historical task_id 161..510 (the wider range)",
        },
        "ambiguities": [
            {
                "issue": "Current P0 defaults begin at full/test offset 200, while the earlier held-out protocol begins at offset 150.",
                "resolution": "Protect the wider historical range, task_id 161..510; task_id 161..210 is not reclaimed.",
            }
        ],
        "shortage": {
            "requested_candidate_count": None,
            "available_unused_mbpp_training_rows": 0,
            "minimum_rows_needed_for_a_nonempty_expansion": 1,
            "shortage_against_nonempty_expansion": 1,
            "reason": "All 474 train-side MBPP tasks are already in the Phase 3 source inventory; test tasks 11..160 were already used for Phase 1 training and 161..510 are protected evaluation data.",
            "action": "Stopped without adding APPS, CodeContests, HumanEval, or any other dataset.",
        },
        "example_candidate_ids": candidate_id_values[:5],
        "outputs": {
            "code_candidates": _relative(candidate_path),
            "candidate_summary": _relative(summary_path),
            "exclusions": _relative(exclusions_path),
        },
        "inputs": {
            "source_problems": {"path": _relative(source_problems_path), "sha256": _sha256(source_problems_path)},
            "source_inventory": {"path": _relative(source_inventory_path), "sha256": _sha256(source_inventory_path)},
            "phase1_code": {"path": _relative(phase1_code_path), "sha256": _sha256(phase1_code_path)},
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-problems", default=str(DEFAULT_SOURCE_PROBLEMS))
    parser.add_argument("--source-inventory", default=str(DEFAULT_SOURCE_INVENTORY))
    parser.add_argument("--phase1-code", default=str(DEFAULT_PHASE1_CODE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
