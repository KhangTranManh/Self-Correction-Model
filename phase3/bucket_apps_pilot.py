"""Pair APPS Base/V1 attempts, create CC/WW/WC/CW buckets, and apply pilot gate."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "apps_pilot.yaml"
BUCKETS = ("CC", "WW", "WC", "CW")


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_unique(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            problem_id = row["id"]
            if problem_id in rows:
                raise RuntimeError(f"Duplicate {problem_id!r} at {path}:{line_number}")
            order.append(problem_id)
            rows[problem_id] = row
    return order, rows


def _bucket(base_correct: bool, v1_correct: bool) -> str:
    if base_correct and v1_correct:
        return "CC"
    if not base_correct and not v1_correct:
        return "WW"
    if not base_correct and v1_correct:
        return "WC"
    return "CW"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    attempts_root = _resolve(config["paths"]["attempts_dir"])
    bucket_root = _resolve(config["paths"]["buckets_dir"])
    candidate_path = _resolve(config["paths"]["candidates"])
    candidate_summary_path = _resolve(config["paths"]["candidate_summary"])
    base_path = attempts_root / "base" / "raw_attempts.jsonl"
    v1_path = attempts_root / "self_correction_v1" / "raw_attempts.jsonl"
    candidate_order, candidates = _read_unique(candidate_path)
    base_order, base = _read_unique(base_path)
    v1_order, v1 = _read_unique(v1_path)
    if candidate_order != base_order or candidate_order != v1_order:
        raise RuntimeError("Candidate/Base/V1 IDs or order differ")

    candidate_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    for model_name, rows in (("base", base), ("self_correction_v1", v1)):
        wrong_hashes = [key for key, row in rows.items() if row["candidate_sha256"] != candidate_hash]
        if wrong_hashes:
            raise RuntimeError(f"{model_name} contains rows from another candidate manifest")

    grouped: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in BUCKETS}
    assignments: list[dict[str, Any]] = []
    for problem_id in candidate_order:
        candidate = candidates[problem_id]
        base_row = base[problem_id]
        v1_row = v1[problem_id]
        paired_fields = ("problem", "source_index", "problem_id", "test_mode", "test_count")
        if any(base_row[field] != v1_row[field] for field in paired_fields):
            raise RuntimeError(f"Base/V1 source mismatch for {problem_id}")
        bucket = _bucket(base_row["initial_correct"], v1_row["initial_correct"])
        assignment = {
            "problem_id": problem_id,
            "dataset": "apps",
            "split": "train",
            "difficulty": candidate["difficulty"],
            "domain": "code",
            "bucket": bucket,
            "source_index": candidate["source_index"],
            "source_host": candidate["source_host"],
            "test_mode": candidate["test_mode"],
            "test_count": candidate["test_count"],
            "base_correct": base_row["initial_correct"],
            "v1_correct": v1_row["initial_correct"],
            "base_verifier_detail": base_row["verifier_detail"],
            "v1_verifier_detail": v1_row["verifier_detail"],
        }
        assignments.append(assignment)
        grouped[bucket].append(
            {
                **assignment,
                "problem": candidate["problem"],
                "tests": candidate["tests"],
                "base_initial_output": base_row["initial_output"],
                "v1_initial_output": v1_row["initial_output"],
            }
        )

    hashes: dict[str, str] = {}
    for bucket in BUCKETS:
        path = bucket_root / f"{bucket}.jsonl"
        hashes[path.relative_to(ROOT).as_posix()] = _write_jsonl(path, grouped[bucket])
    assignments_path = bucket_root / "bucket_assignments.jsonl"
    hashes[assignments_path.relative_to(ROOT).as_posix()] = _write_jsonl(assignments_path, assignments)

    counts = Counter(row["bucket"] for row in assignments)
    total = len(assignments)
    v1_correct = counts["CC"] + counts["WC"]
    base_correct = counts["CC"] + counts["CW"]
    candidate_summary = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
    source_counts = candidate_summary["counts_by_source_host"]
    mode_counts = candidate_summary["counts_by_test_mode"]
    gate_cfg = config["decision_gate"]
    gate_checks = {
        "v1_correct_count": v1_correct >= int(gate_cfg["minimum_v1_correct_count"]),
        "v1_correct_rate": (v1_correct / total) >= float(gate_cfg["minimum_v1_correct_rate"]),
        "both_test_modes": (not gate_cfg["require_both_test_modes"]) or len(mode_counts) >= 2,
        "minimum_source_hosts": len(source_counts) >= int(gate_cfg["minimum_source_hosts"]),
        "maximum_single_source_fraction": max(source_counts.values()) / total
        <= float(gate_cfg["maximum_single_source_fraction"]),
    }
    scale_recommended = all(gate_checks.values())
    summary = {
        "definitions": {
            "CC": "Base correct, Self_Correction_v1 correct",
            "WW": "Base wrong, Self_Correction_v1 wrong",
            "WC": "Base wrong, Self_Correction_v1 correct",
            "CW": "Base correct, Self_Correction_v1 wrong",
        },
        "complete": total == int(config["dataset"]["pilot_size"]),
        "total": total,
        "counts": {bucket: counts[bucket] for bucket in BUCKETS},
        "base_correct": base_correct,
        "base_correct_rate": base_correct / total,
        "v1_correct": v1_correct,
        "v1_correct_rate": v1_correct / total,
        "v1_net_correct_gain": counts["WC"] - counts["CW"],
        "counts_by_test_mode": {
            mode: {bucket: sum(row["test_mode"] == mode and row["bucket"] == bucket for row in assignments) for bucket in BUCKETS}
            for mode in sorted(mode_counts)
        },
        "counts_by_source_host": {
            host: {bucket: sum(row["source_host"] == host and row["bucket"] == bucket for row in assignments) for bucket in BUCKETS}
            for host in sorted(source_counts)
        },
        "decision_gate": {
            "thresholds": gate_cfg,
            "checks": gate_checks,
            "scale_recommended": scale_recommended,
            "decision": "APPS introductory is suitable to scale for Phase 3 code-correct coverage"
            if scale_recommended
            else "Do not scale APPS from this pilot under the predefined gate",
        },
        "candidate_sha256": candidate_hash,
        "input_sha256": {
            "base_attempts": hashlib.sha256(base_path.read_bytes()).hexdigest(),
            "self_correction_v1_attempts": hashlib.sha256(v1_path.read_bytes()).hexdigest(),
        },
        "output_sha256": hashes,
    }
    report_path = _resolve(config["paths"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
