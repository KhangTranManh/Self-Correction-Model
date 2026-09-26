"""Freeze Phase 7 development/protected IDs after the initial-answer quota."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase7.scripts.collect_initials_4bit import read_jsonl, sha256, write_atomic

CONFIG = ROOT / "phase7/configs/blind_resolve_v1.yaml"


def rank(seed: int, problem_id: str) -> str:
    return hashlib.sha256(f"phase7-split-v1|{seed}|{problem_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-dir", type=Path,
                        default=ROOT / "outputs/phase7_initials_v1")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "phase7/data/split_v1")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    source = config["source_plan"]
    candidates_path = ROOT / source["candidate_manifest"]
    if sha256(candidates_path) != source["candidate_manifest_sha256"]:
        raise ValueError("Candidate manifest hash mismatch")
    candidates = read_jsonl(candidates_path)
    summary_path = args.initial_dir / "summary.json"
    audit_path = args.initial_dir / "initial_rollouts.audit.jsonl"
    rollout_path = args.initial_dir / "initial_rollouts.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] != "quota_met" or sha256(audit_path) != summary["audit_sha256"]:
        raise ValueError("Initial audit is incomplete or changed")
    if sha256(rollout_path) != summary["rollouts_sha256"]:
        raise ValueError("Initial rollout hash mismatch")
    rollouts = read_jsonl(rollout_path)
    audit = read_jsonl(audit_path)
    if len(rollouts) != summary["completed"] or len(audit) != len(rollouts) + 1:
        raise ValueError("Initial audit length mismatch")
    for index, row in enumerate(rollouts):
        if row != audit[index + 1]["row"] or row["problem_id"] != candidates[index]["id"]:
            raise ValueError("Initial audit is not the frozen ordered prefix")
    if len(rollouts) > int(source["initial_generation_ceiling"]):
        raise ValueError("Initial prefix exceeds ceiling")
    quota = int(source["stop_at_first_prefix_with_each_class_at_least"])
    correct = sum(bool(row["initial_correct"]) for row in rollouts)
    wrong = len(rollouts) - correct
    if min(correct, wrong) < quota or min(correct - bool(rollouts[-1]["initial_correct"]),
                                           wrong - (not bool(rollouts[-1]["initial_correct"]))) >= quota:
        raise ValueError("Initial run did not stop at first quota prefix")
    by_class = {True: [], False: []}
    for row in rollouts:
        by_class[bool(row["initial_correct"])].append(row["problem_id"])
    seed = int(config["seed"])
    counts = source["development"], source["protected"]
    selected = {"development": [], "protected": []}
    for label, key in ((True, "initially_correct"), (False, "initially_wrong")):
        ordered = sorted(by_class[label], key=lambda pid: rank(seed, pid))
        dev_n, protected_n = (int(counts[0][key]), int(counts[1][key]))
        if len(ordered) < dev_n + protected_n:
            raise ValueError(f"Not enough {key} sources")
        selected["development"].extend(ordered[:dev_n])
        selected["protected"].extend(ordered[dev_n:dev_n + protected_n])
    selected = {name: sorted(ids, key=lambda pid: rank(seed, pid))
                for name, ids in selected.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in selected.items():
        path = args.output_dir / f"{name}_ids.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != ids:
                raise ValueError(f"Existing {name} split differs")
        else:
            write_atomic(path, json.dumps(ids, indent=2) + "\n")
    report = {
        "schema_version": "phase7_split_v1", "rank_rule": "sha256(phase7-split-v1|seed|problem_id)",
        "seed": seed, "candidate_manifest_sha256": sha256(candidates_path),
        "initial_audit_sha256": sha256(audit_path),
        "initial_rollouts_sha256": sha256(rollout_path),
        "prefix_length": len(rollouts), "prefix_correct": correct, "prefix_wrong": wrong,
        "development_count": len(selected["development"]),
        "protected_count": len(selected["protected"]),
        "development_ids_sha256": sha256(args.output_dir / "development_ids.json"),
        "protected_ids_sha256": sha256(args.output_dir / "protected_ids.json"),
    }
    write_atomic(args.output_dir / "split_report.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
