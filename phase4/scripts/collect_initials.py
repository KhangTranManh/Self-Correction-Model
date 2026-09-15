"""Collect natural initial answers and deterministic verifier results."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any

import httpx
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phase1"))

from phase4.lib.rollout import (
    as_problem,
    completion,
    read_jsonl,
    sha256,
    verify,
    write_jsonl_atomic,
)
from src.core.prompts import build_prompt


CHECKPOINT_SCHEMA_VERSION = "phase4_initial_checkpoint_v1"


def checkpoint_path(output: Path) -> Path:
    """Keep the append-only audit separate from the final, sorted output."""
    return output.with_suffix(output.suffix + ".audit.jsonl")


def _json_line(row: dict[str, Any]) -> bytes:
    return (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def append_jsonl_durable(path: Path, row: dict[str, Any]) -> None:
    """Append one recoverable audit record and force it to disk before continuing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_json_line(row))
        handle.flush()
        os.fsync(handle.fileno())


def checkpoint_provenance(args: argparse.Namespace, config: dict[str, Any], manifest_sha256: str) -> dict[str, Any]:
    rollout = config["rollout"]
    return {
        "source_manifest_sha256": manifest_sha256,
        "policy_checkpoint": args.policy_checkpoint,
        "serving_model": args.serving_model,
        "seed_offset": args.seed_offset,
        "experiment_seed": int(config["experiment"]["seed"]),
        "initial_temperature": float(rollout["initial_temperature"]),
        "initial_max_tokens": int(rollout["initial_max_tokens"]),
        # Retain both the complete parsed settings and the source file digest.  The
        # latter also catches edits which preserve YAML's parsed value.
        "config": config,
        "config_sha256": sha256(args.config),
    }


def load_checkpoint(path: Path, provenance: dict[str, Any], valid_ids: set[str]) -> dict[str, dict[str, Any]]:
    records = read_jsonl(path)
    if not records or records[0].get("record_type") != "metadata":
        raise ValueError(f"Checkpoint {path} is missing its metadata record")
    if records[0].get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema in {path}")
    if records[0].get("provenance") != provenance:
        raise ValueError(
            f"Checkpoint {path} provenance does not match this collection request"
        )

    completed: dict[str, dict[str, Any]] = {}
    for record in records[1:]:
        if record.get("record_type") != "completion" or not isinstance(record.get("row"), dict):
            raise ValueError(f"Malformed checkpoint record in {path}")
        row = record["row"]
        problem_id = str(row.get("problem_id"))
        if problem_id not in valid_ids:
            raise ValueError(f"Checkpoint {path} contains unknown problem id {problem_id!r}")
        if problem_id in completed:
            raise ValueError(f"Checkpoint {path} contains duplicate problem id {problem_id!r}")
        completed[problem_id] = row
    return completed


def open_checkpoint(path: Path, provenance: dict[str, Any], valid_ids: set[str]) -> dict[str, dict[str, Any]]:
    if path.exists():
        return load_checkpoint(path, provenance, valid_ids)
    metadata = {
        "record_type": "metadata",
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "provenance": provenance,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(_json_line(metadata))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        return load_checkpoint(path, provenance, valid_ids)
    return {}


def verify_existing_output(
    output: Path,
    completed: dict[str, dict[str, Any]],
    expected_ids: set[str],
) -> bool:
    """Only accept an existing final file when the compatible audit proves it."""
    if not output.exists():
        return False
    existing = read_jsonl(output)
    existing_by_id = {str(row.get("problem_id")): row for row in existing}
    if (
        len(existing_by_id) != len(existing)
        or set(existing_by_id) != expected_ids
        or set(completed) != expected_ids
        or any(existing_by_id[problem_id] != completed[problem_id] for problem_id in expected_ids)
    ):
        raise ValueError(
            f"Refusing to overwrite existing output {output}: it is not proven by "
            "a complete compatible checkpoint"
        )
    return True


async def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rollout = config["rollout"]
    rows = read_jsonl(args.problems)
    ids = [str(row.get("id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Problem manifest contains duplicate ids")
    manifest_sha256 = sha256(args.problems)
    provenance = checkpoint_provenance(args, config, manifest_sha256)
    audit_path = checkpoint_path(args.output)
    completed = open_checkpoint(audit_path, provenance, set(ids))
    if verify_existing_output(args.output, completed, set(ids)):
        counts = Counter(
            "correct" if row["initial_correct"] else "wrong" for row in completed.values()
        )
        print(json.dumps({"rows": len(completed), "counts": counts, "output": str(args.output)}, indent=2))
        return
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout, connect=30.0)

    async def collect(client: httpx.AsyncClient, row: dict, index: int) -> dict:
        problem = as_problem(row)
        task_prompt = build_prompt(problem)
        output = await completion(
            client,
            semaphore,
            model=args.serving_model,
            messages=[{"role": "user", "content": task_prompt}],
            temperature=float(rollout["initial_temperature"]),
            max_tokens=int(rollout["initial_max_tokens"]),
            seed=int(config["experiment"]["seed"]) + args.seed_offset + index,
        )
        result = await asyncio.to_thread(verify, problem, output)
        return {
            "schema_version": "phase4_initial_rollout_v1",
            "problem_id": problem.id,
            "domain": problem.domain,
            "policy_checkpoint": args.policy_checkpoint,
            "source_manifest_sha256": manifest_sha256,
            "task_prompt": task_prompt,
            "initial_output": output,
            "initial_correct": result["passed"],
            "initial_verifier_detail": result["detail"],
            "verifier_spec": {
                "reference_answer": problem.reference_answer,
                "entry_point": problem.entry_point,
                "tests": problem.tests,
                "calc_steps": problem.calc_steps,
            },
        }

    pending = [
        (row, index)
        for index, row in enumerate(rows)
        if str(row["id"]) not in completed
    ]
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/") + "/", timeout=timeout) as client:
        tasks = [asyncio.create_task(collect(client, row, index)) for row, index in pending]
        for task in asyncio.as_completed(tasks):
            row = await task
            problem_id = str(row["problem_id"])
            # This synchronous, fsync-backed append is deliberately before the
            # next completed task is consumed: a returned answer is never lost.
            append_jsonl_durable(audit_path, {"record_type": "completion", "row": row})
            completed[problem_id] = row

    results = [completed[problem_id] for problem_id in ids]
    results.sort(key=lambda row: row["problem_id"])
    # A different process could have created the final file while this run was
    # collecting.  Re-check immediately before the atomic replace.
    if args.output.exists():
        if verify_existing_output(args.output, completed, set(ids)):
            counts = Counter("correct" if row["initial_correct"] else "wrong" for row in results)
            print(json.dumps({"rows": len(results), "counts": counts, "output": str(args.output)}, indent=2))
            return
    write_jsonl_atomic(args.output, results)
    counts = Counter("correct" if row["initial_correct"] else "wrong" for row in results)
    print(json.dumps({"rows": len(results), "counts": counts, "output": str(args.output)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--serving-model", default="phase4-actor")
    parser.add_argument("--base-url", default="http://127.0.0.1:8999/v1")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed-offset", type=int, default=4_000_000)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/transition_rl_v1.yaml",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
