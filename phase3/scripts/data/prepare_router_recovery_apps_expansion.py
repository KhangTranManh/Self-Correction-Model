"""Freeze new source-disjoint APPS tasks for Router Recovery V2 dual-outcome mining."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlparse

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase3.lib.apps_verifier import verify
from phase3.scripts.data.collect_apps_pilot import _prompt as build_apps_prompt


REVISION = "21e74ddf8de1a21436da12e3e653065c5213e9d1"
SEED = 20260908


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable(value: str) -> str:
    return hashlib.sha256(f"{SEED}|apps_expansion|{value}".encode()).hexdigest()


def normal_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).casefold().encode()).hexdigest()


def parse_source(row: dict[str, Any]) -> dict[str, Any] | None:
    problem_id = row.get("problem_id")
    question = row.get("question")
    if not isinstance(problem_id, int) or not isinstance(question, str) or not question.strip():
        return None
    try:
        tests = json.loads(row.get("input_output") or "")
        solutions = json.loads(row.get("solutions") or "[]")
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(tests, dict) or not isinstance(tests.get("inputs"), list) or not tests["inputs"]:
        return None
    if len(tests["inputs"]) != len(tests.get("outputs") or []):
        return None
    if not isinstance(solutions, list) or not solutions:
        return None
    mode = "call_based" if tests.get("fn_name") else "standard_input"
    return {
        "id": f"apps_train_{problem_id}",
        "dataset": "apps",
        "split": "train",
        "domain": "code",
        "source_index": problem_id,
        "problem_id": problem_id,
        "problem": question,
        "starter_code": row.get("starter_code") or "",
        "solutions": solutions,
        "tests": tests,
        "input_output_raw": row.get("input_output") or "",
        "difficulty": row.get("difficulty"),
        "url": row.get("url") or "",
        "source_host": urlparse(str(row.get("url") or "")).netloc or "unknown",
        "test_mode": mode,
        "test_count": len(tests["inputs"]),
        "source_file": f"codeparrot/apps@{REVISION}/train",
        "selected_before_model_inference": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "router_recovery_v2" / "apps_expansion"))
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--batch-sources", type=int, default=40)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()

    excluded_ids = set()
    excluded_hashes = set()
    local_paths = [
        ROOT / "data" / "apps_pilot" / "candidates" / "code_candidates.jsonl",
        ROOT / "runs" / "representation_probe" / "probe_dataset.jsonl",
        ROOT / "data" / "two_stage_selective_repair" / "frozen_eval.jsonl",
        ROOT / "data" / "decision_only" / "decision_only_dataset.jsonl",
        ROOT / "data" / "router_calibration_v1" / "calibration_all.jsonl",
        ROOT / "data" / "router_recovery_v2" / "current" / "verified_pairs.partial.jsonl",
    ]
    for path in local_paths:
        for row in read_jsonl(path):
            sid = str(row.get("source_id", row.get("id", "")))
            if sid:
                excluded_ids.add(sid)
            problem = str(row.get("problem") or row.get("question") or "")
            if not problem and isinstance(row.get("messages"), list) and row["messages"]:
                problem = str(row["messages"][0].get("content") or "")
            if problem:
                excluded_hashes.add(normal_hash(problem))

    stream = load_dataset(
        "codeparrot/apps", split="train", revision=REVISION,
        streaming=True, trust_remote_code=True,
    )
    eligible = []
    seen = set(excluded_ids)
    for raw in stream:
        if raw.get("difficulty") != "introductory":
            continue
        row = parse_source(raw)
        if not row or row["id"] in seen or normal_hash(row["problem"]) in excluded_hashes:
            continue
        seen.add(row["id"])
        eligible.append(row)

    # Round-robin test mode and source host after deterministic within-group sort.
    groups: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(eligible, key=lambda value: stable(value["id"])):
        groups[(row["test_mode"], row["source_host"])].append(row)
    keys = sorted(groups, key=lambda key: stable("|".join(key)))
    ordered = []
    while any(groups.values()):
        for key in keys:
            if groups[key]:
                ordered.append(groups[key].popleft())

    selected = []
    reference_failures = []
    for row in ordered:
        reference = None
        detail = None
        for solution in row.pop("solutions"):
            if not isinstance(solution, str) or not solution.strip():
                continue
            result = verify(solution, row["tests"], per_test_timeout_seconds=3, memory_limit_mb=512)
            detail = result["detail"]
            if result["passed"]:
                reference = solution
                break
        if reference is None:
            reference_failures.append({"source_id": row["id"], "detail": detail})
            continue
        row["reference_answer"] = reference
        row["ground_truth"] = reference
        row["reference_fresh_verified"] = True
        selected.append(row)
        if len(selected) >= args.target:
            break
    if len(selected) < args.target:
        raise RuntimeError(f"Only {len(selected)} verified new APPS sources; expected {args.target}")

    candidate_path = output / "candidates.jsonl"
    write_jsonl(candidate_path, selected)
    tasks = []
    for row in selected:
        tasks.append({
            "schema_version": "phase3_router_recovery_v2_generation_task_v1",
            "task_id": f"recovery_apps_dual::self_correction_v1::{row['id']}",
            "pair_id": f"router_recovery_v2::{row['id']}",
            "source_id": row["id"],
            "problem_ref": f"data/router_recovery_v2/apps_expansion/candidates.jsonl#id={row['id']}",
            "dataset": "apps",
            "domain": "code",
            "problem_message": {"role": "user", "content": build_apps_prompt(row)},
            "existing_verified_correct_answer": None,
            "existing_correct_answer_ref": None,
            "model_origin_to_sample": "self_correction_v1",
            "model_id_to_sample": "Kxck/Self_Correction_v1",
            "desired_new_state": "one_verified_correct_and_one_plausible_wrong",
            "samples_requested": 8,
            "priority": "new_apps_dual_outcome",
        })
    batch_summary = []
    for start in range(0, len(tasks), args.batch_sources):
        batch = tasks[start : start + args.batch_sources]
        path = output / "generation_batches" / f"{1 + start // args.batch_sources:02d}_apps_dual.jsonl"
        write_jsonl(path, batch)
        batch_summary.append({
            "batch": path.stem,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "tasks": len(batch),
            "jobs": sum(row["samples_requested"] for row in batch),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    write_jsonl(output / "generation_manifest.jsonl", tasks)
    write_jsonl(output / "reference_failures.jsonl", reference_failures)
    summary = {
        "schema_version": "phase3_router_recovery_v2_apps_expansion_v1",
        "dataset": "codeparrot/apps",
        "revision": REVISION,
        "selected_before_model_inference": True,
        "selected_sources": len(selected),
        "reference_fresh_pass": len(selected),
        "reference_failures_before_target": len(reference_failures),
        "test_modes": dict(Counter(row["test_mode"] for row in selected)),
        "source_hosts": dict(Counter(row["source_host"] for row in selected)),
        "excluded_source_ids": len(excluded_ids),
        "batches": batch_summary,
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "training_started": False,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
