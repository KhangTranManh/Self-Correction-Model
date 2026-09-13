"""Scan all usable APPS-train and non-test MBPP sources for recovery mining.

This is a source/manifest preparation step. It does not call a model, claim
that references were freshly verified, create final splits, or start training.
Previously sampled, protected, and already admitted source IDs are excluded.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase3.scripts.data.collect_apps_pilot import _prompt as build_apps_prompt
from phase3.scripts.data.collect_initial_attempts import _problem as build_problem_record
from phase3.scripts.data.collect_initial_attempts import build_prompt as build_mbpp_prompt
from phase3.scripts.data.prepare_router_recovery_apps_expansion import REVISION, parse_source
from phase3.scripts.data.prepare_router_recovery_v2 import protected_sources


SEED = 20260908
MBPP_SOURCE = ROOT / "data" / "source" / "problems.jsonl"
CURRENT_PAIRS = ROOT / "data" / "router_recovery_v2" / "current" / "verified_pairs.partial.jsonl"
RAW_CANDIDATE_GLOBS = (
    "data/router_recovery_v2/generation/*_candidates.jsonl",
    "data/router_recovery_v2/apps_expansion/apps*_candidates.jsonl",
)

# APPS contains legitimate stress tests with integers longer than Python's
# default JSON digit guard. These files are pinned dataset artifacts, not
# untrusted API payloads, and preserving numeric test values is required.
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def stable(value: str) -> str:
    return hashlib.sha256(f"{SEED}|full_code_scan|{value}".encode()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attempted_sources() -> tuple[set[str], dict[str, int]]:
    result: set[str] = set()
    counts: dict[str, int] = {}
    for pattern in RAW_CANDIDATE_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            rows = read_jsonl(path)
            ids = {str(row.get("source_id") or "") for row in rows} - {""}
            result.update(ids)
            counts[str(path.relative_to(ROOT)).replace("\\", "/")] = len(ids)
    return result, counts


def scan_apps() -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    audit: Counter[str] = Counter()
    stream = load_dataset(
        "codeparrot/apps",
        split="train",
        revision=REVISION,
        streaming=True,
        trust_remote_code=True,
    )
    for raw in stream:
        audit["scanned"] += 1
        row = parse_source(raw)
        if row is None:
            audit["structurally_invalid"] += 1
            continue
        solutions = row.pop("solutions", [])
        references = [value for value in solutions if isinstance(value, str) and value.strip()]
        if not references:
            audit["missing_reference_solution"] += 1
            continue
        row["reference_answer"] = references[0]
        row["ground_truth"] = references[0]
        row["reference_verification_status"] = "pending_fresh_verification"
        rows.append(row)
        audit["structurally_eligible"] += 1
    return rows, audit


def scan_mbpp() -> tuple[list[dict[str, Any]], Counter[str]]:
    rows = [row for row in read_jsonl(MBPP_SOURCE) if row.get("dataset") == "mbpp"]
    audit: Counter[str] = Counter(scanned=len(rows), structurally_eligible=len(rows))
    for row in rows:
        row["reference_verification_status"] = "pending_fresh_verification"
    return rows, audit


def generation_task(row: dict[str, Any], source_file: str) -> dict[str, Any]:
    source_id = str(row["id"])
    dataset = str(row["dataset"])
    prompt = (
        build_apps_prompt(row)
        if dataset == "apps"
        else build_mbpp_prompt(build_problem_record(row))
    )
    return {
        "schema_version": "phase3_router_recovery_v2_generation_task_v1",
        "task_id": f"recovery_full_scan::base::{source_id}",
        "pair_id": f"router_recovery_v2::{source_id}",
        "source_id": source_id,
        "problem_ref": f"{source_file}#id={source_id}",
        "dataset": dataset,
        "domain": "code",
        "problem_message": {"role": "user", "content": prompt},
        "existing_verified_correct_answer": None,
        "existing_correct_answer_ref": None,
        "model_origin_to_sample": "base",
        "model_id_to_sample": "Qwen/Qwen2.5-7B-Instruct",
        "desired_new_state": "one_verified_correct_and_one_plausible_wrong",
        "samples_requested": 8,
        "priority": f"full_scan_{dataset}_dual_outcome",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "router_recovery_v2" / "full_code_scan"),
    )
    parser.add_argument("--batch-sources", type=int, default=32)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()

    protected, protected_counts = protected_sources()
    current = {str(row["source_id"]) for row in read_jsonl(CURRENT_PAIRS)}
    attempted, attempted_files = attempted_sources()

    apps, apps_audit = scan_apps()
    mbpp, mbpp_audit = scan_mbpp()
    all_rows = apps + mbpp
    exclusion_counts: Counter[str] = Counter()
    excluded_rows = []
    eligible_by_dataset: dict[str, list[dict[str, Any]]] = {"apps": [], "mbpp": []}
    seen_ids: set[str] = set()
    for row in all_rows:
        sid = str(row["id"])
        reason = None
        if sid in seen_ids:
            reason = "duplicate_source_id"
        elif sid in protected:
            reason = "protected_source"
        elif sid in current:
            reason = "already_admitted"
        elif sid in attempted:
            reason = "previously_sampled"
        seen_ids.add(sid)
        if reason:
            exclusion_counts[reason] += 1
            excluded_rows.append({"source_id": sid, "dataset": row["dataset"], "reason": reason})
        else:
            eligible_by_dataset[str(row["dataset"])].append(row)

    difficulty_rank = {"introductory": 0, "interview": 1, "competition": 2, None: 3}
    eligible_by_dataset["apps"].sort(
        key=lambda row: (difficulty_rank.get(row.get("difficulty"), 4), stable(str(row["id"])))
    )
    eligible_by_dataset["mbpp"].sort(key=lambda row: stable(str(row["id"])))

    source_files = {}
    manifests = {}
    for dataset in ("apps", "mbpp"):
        source_path = output / f"{dataset}_eligible_sources.jsonl"
        write_jsonl(source_path, eligible_by_dataset[dataset])
        source_ref = str(source_path.relative_to(ROOT)).replace("\\", "/")
        source_files[dataset] = source_path
        tasks = [generation_task(row, source_ref) for row in eligible_by_dataset[dataset]]
        manifest_path = output / f"{dataset}_generation_manifest.jsonl"
        write_jsonl(manifest_path, tasks)
        manifests[dataset] = tasks

        batch_dir = output / "generation_batches" / dataset
        for start in range(0, len(tasks), args.batch_sources):
            batch = tasks[start : start + args.batch_sources]
            batch_path = batch_dir / f"{1 + start // args.batch_sources:03d}_{dataset}_base.jsonl"
            write_jsonl(batch_path, batch)

    # A round-robin launch order makes both datasets visible early while APPS
    # remains the larger pool. This is an index only; batches remain separate.
    apps_batches = sorted((output / "generation_batches" / "apps").glob("*.jsonl"))
    mbpp_batches = sorted((output / "generation_batches" / "mbpp").glob("*.jsonl"))
    launch_order = []
    limit = max(len(apps_batches), len(mbpp_batches))
    for index in range(limit):
        for dataset, paths in (("apps", apps_batches), ("mbpp", mbpp_batches)):
            if index >= len(paths):
                continue
            path = paths[index]
            rows = manifests[dataset][index * args.batch_sources : (index + 1) * args.batch_sources]
            launch_order.append({
                "sequence": len(launch_order) + 1,
                "dataset": dataset,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sources": len(rows),
                "jobs": sum(int(row["samples_requested"]) for row in rows),
                "sha256": sha256(path),
            })

    write_jsonl(output / "exclusions.jsonl", excluded_rows)
    write_json(output / "launch_order.json", launch_order)
    summary = {
        "schema_version": "phase3_router_recovery_v2_full_code_scan_v1",
        "status": "source_scan_complete_generation_not_started",
        "selected_before_model_inference": True,
        "training_started": False,
        "minimum_pair_deficit": 92,
        "apps": {
            "dataset": "codeparrot/apps",
            "revision": REVISION,
            "scan": dict(apps_audit),
            "eligible_new_sources": len(eligible_by_dataset["apps"]),
            "eligible_by_difficulty": dict(Counter(str(row.get("difficulty")) for row in eligible_by_dataset["apps"])),
            "eligible_by_test_mode": dict(Counter(str(row.get("test_mode")) for row in eligible_by_dataset["apps"])),
            "generation_batches": len(apps_batches),
        },
        "mbpp": {
            "dataset": "google-research-datasets/mbpp",
            "allowed_splits": ["train", "validation", "prompt"],
            "scan": dict(mbpp_audit),
            "eligible_new_sources": len(eligible_by_dataset["mbpp"]),
            "eligible_by_split": dict(Counter(str(row.get("split")) for row in eligible_by_dataset["mbpp"])),
            "generation_batches": len(mbpp_batches),
        },
        "exclusions": dict(exclusion_counts),
        "protected_sets": protected_counts,
        "protected_union_sources": len(protected),
        "current_admitted_sources": len(current),
        "previously_sampled_sources": len(attempted),
        "attempted_candidate_files": attempted_files,
        "batch_sources": args.batch_sources,
        "samples_per_source": 8,
        "total_eligible_sources": sum(len(rows) for rows in eligible_by_dataset.values()),
        "total_generation_jobs_if_exhausted": sum(
            int(task["samples_requested"]) for tasks in manifests.values() for task in tasks
        ),
        "run_policy": "Run one APPS batch and one MBPP batch, verify yield, then continue only while useful.",
        "reference_verification": "Pending; model-generated correct and wrong candidates must both be freshly verified before admission.",
    }
    write_json(output / "scan_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
