"""Read-only audit of source boundaries and the recoverability of Phase 5 probes."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P5 = ROOT / "phase5/data/splits/v1"
CHECKPOINTS = ("original_solver", "warmstart_v2", "correction_sft_v3")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((P5 / "manifest.json").read_text(encoding="utf-8"))
    split_rows = {}
    for split in ("train", "development", "protected_test"):
        path = P5 / f"{split}.jsonl"
        expected = manifest["output_sha256"][path.name]
        observed = lf_sha256(path)
        if observed != expected:
            raise RuntimeError(f"Frozen {split} hash mismatch: {observed} != {expected}")
        split_rows[split] = rows(path)
        expected_count = manifest["split_counts"][split]
        if len(split_rows[split]) != expected_count["rows"]:
            raise RuntimeError(f"Frozen {split} row count mismatch")
        correct = sum(row["initial_correct"] is True for row in split_rows[split])
        wrong = sum(row["initial_correct"] is False for row in split_rows[split])
        if (correct, wrong) != (expected_count["correct"], expected_count["wrong"]):
            raise RuntimeError(f"Frozen {split} class count mismatch")
        if any(not row.get("initial_output") for row in split_rows[split]):
            raise RuntimeError(f"Missing initial answer in {split}")

    source_sets = {
        split: {row["question_sha256"] for row in values}
        for split, values in split_rows.items()
    }
    for left in source_sets:
        for right in source_sets:
            if left < right and source_sets[left] & source_sets[right]:
                raise RuntimeError(f"Source overlap: {left}/{right}")

    other_paths = {
        "phase6_holdout": ROOT / "phase6/data/harness_confirmation_holdout_v1.jsonl",
        "phase7_candidates": ROOT / "phase7/data/candidates_v1/candidate_problems.jsonl",
    }
    other = {name: rows(path) for name, path in other_paths.items()}
    for name, values in other.items():
        hashes = {row["question_sha256"] for row in values}
        for split, own in source_sets.items():
            if hashes & own:
                raise RuntimeError(f"Phase 5 {split} overlaps {name}")
    if ({row["question_sha256"] for row in other["phase6_holdout"]}
            & {row["question_sha256"] for row in other["phase7_candidates"]}):
        raise RuntimeError("Phase 6 holdout overlaps Phase 7 candidates")

    lock = json.loads((ROOT / "phase5/data/protocol/protected_opening_v1_lock.json").read_text(encoding="utf-8"))
    locked_files = {}
    for name, item in lock["files"].items():
        path = ROOT / item["path"]
        status = "missing" if not path.exists() else "verified" if sha256(path) == item["sha256"] else "lf_only" if lf_sha256(path) == item["sha256"] else "hash_mismatch"
        locked_files[name] = {"path": item["path"], "status": status}
    probe_files = {}
    for checkpoint in CHECKPOINTS:
        base = ROOT / "outputs/phase5_gpu_vllm/probe_v1"
        probe_files[checkpoint] = {
            "model": (base / "selection" / f"{checkpoint}_probe.joblib").exists(),
            "train_activations": (base / "activations" / f"{checkpoint}_train.npz").exists(),
            "development_activations": (base / "activations" / f"{checkpoint}_development.npz").exists(),
        }
    report = {
        "schema_version": "phase5_probe_data_recovery_audit_v1",
        "phase5_splits_verified": {name: len(values) for name, values in split_rows.items()},
        "phase6_holdout_rows": len(other["phase6_holdout"]),
        "phase7_candidate_rows": len(other["phase7_candidates"]),
        "question_hash_overlap": 0,
        "locked_files": locked_files,
        "probe_files": probe_files,
        "recovery_status": "exact_probe_available" if all(
            item["model"] for item in probe_files.values()) else "exact_probe_missing",
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
