"""Build a high-temperature wrong-mining retry from sources whose prior samples all passed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase3.lib.behavior import ProvenanceResolver, VerificationEngine


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    if len(args.manifest) != len(args.candidate):
        raise RuntimeError("Pass matching --manifest and --candidate arguments")
    resolver = ProvenanceResolver()
    verifier = VerificationEngine()
    selected = []
    for manifest_value, candidate_value in zip(args.manifest, args.candidate, strict=True):
        tasks = {row["task_id"]: row for row in read_jsonl(Path(manifest_value))}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in read_jsonl(Path(candidate_value)):
            grouped.setdefault(row["task_id"], []).append(row)
        for task_id, task in tasks.items():
            source = resolver.resolve(task["problem_ref"], task["source_id"])
            candidates = grouped.get(task_id, [])
            checks = [(row, verifier.verify(source, str(row.get("raw_answer") or ""))) for row in candidates]
            if not checks or not all(check["passed"] for _, check in checks):
                continue
            correct_row, correct_check = sorted(checks, key=lambda item: item[0]["candidate_id"])[0]
            new_task = dict(task)
            new_task["task_id"] = f"recovery_adaptive_wrong::{task['model_origin_to_sample']}::{task['source_id']}"
            new_task["existing_verified_correct_answer"] = correct_row["raw_answer"]
            new_task["existing_correct_answer_ref"] = f"{Path(candidate_value).name}#candidate_id={correct_row['candidate_id']}"
            new_task["desired_new_state"] = "plausible_wrong_after_all_prior_samples_passed"
            new_task["samples_requested"] = args.samples
            new_task["priority"] = "adaptive_high_temperature_wrong_mining"
            selected.append(new_task)
    if len({row["source_id"] for row in selected}) != len(selected):
        raise RuntimeError("Adaptive manifests overlap source IDs")
    selected.sort(key=lambda row: hashlib.sha256(row["source_id"].encode()).hexdigest())
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "tasks": len(selected), "jobs": len(selected) * args.samples,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
