"""Freeze the balanced Phase 5 train/development/protected source split.

This is a deterministic CPU-only operation over the already collected initial
answers. It does not load a model, generate text, inspect review outcomes, or
open any protected review/probe result.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "phase5/configs/experiments.yaml"
DEFAULT_CANDIDATES = ROOT / "phase5/data/candidates_v1/candidate_problems.jsonl"
DEFAULT_ROLLOUTS = ROOT / "outputs/phase5_remote_v100/initial_rollouts.jsonl"
DEFAULT_OUTPUT = ROOT / "phase5/data/splits/v1"
SPLIT_COUNTS_PER_CLASS = {"train": 120, "development": 40, "protected_test": 80}


def canonical_sha256(path: Path) -> str:
    """Hash text evidence using the LF representation recorded on the GPU."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rank(seed: int, purpose: str, source_id: str) -> str:
    value = f"phase5_split_v1|{seed}|{purpose}|{source_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--rollouts", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite frozen split: {args.output_dir}")

    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    candidates_hash = canonical_sha256(args.candidates)
    rollouts_hash = canonical_sha256(args.rollouts)
    if candidates_hash != registry["candidate_sources"]["candidate_problems_sha256"]:
        raise ValueError("Candidate manifest hash differs from the registry")
    if rollouts_hash != registry["initial_collection"]["initial_rollouts_sha256"]:
        raise ValueError("Initial rollout hash differs from the registry")

    candidates = read_jsonl(args.candidates)
    rollouts = read_jsonl(args.rollouts)
    candidate_by_id = {str(row["id"]): row for row in candidates}
    if len(candidate_by_id) != len(candidates):
        raise ValueError("Candidate IDs are not unique")
    if len(rollouts) != registry["initial_collection"]["completed_rows"]:
        raise ValueError("Initial rollout count differs from the registry")
    rollout_ids = [str(row["problem_id"]) for row in rollouts]
    if len(set(rollout_ids)) != len(rollout_ids):
        raise ValueError("Initial rollout IDs are not unique")
    if rollout_ids != [str(row["id"]) for row in candidates[:len(rollouts)]]:
        raise ValueError("Initial rollouts are not the frozen candidate prefix")

    by_class = {
        True: [row for row in rollouts if row["initial_correct"] is True],
        False: [row for row in rollouts if row["initial_correct"] is False],
    }
    required = sum(SPLIT_COUNTS_PER_CLASS.values())
    if any(len(rows) < required for rows in by_class.values()):
        raise ValueError("At least 240 natural examples are required per class")

    selected: dict[bool, list[dict[str, Any]]] = {}
    for label, rows in by_class.items():
        selected[label] = sorted(
            rows,
            key=lambda row: rank(args.seed, "balanced_selection", str(row["problem_id"])),
        )[:required]

    outputs: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_COUNTS_PER_CLASS}
    assignments: dict[str, str] = {}
    for label, rows in selected.items():
        ordered = sorted(
            rows,
            key=lambda row: rank(args.seed, "split_assignment", str(row["problem_id"])),
        )
        offset = 0
        for split, count in SPLIT_COUNTS_PER_CLASS.items():
            for rollout in ordered[offset:offset + count]:
                source_id = str(rollout["problem_id"])
                source = candidate_by_id[source_id]
                record = {
                    **source,
                    **rollout,
                    "split": split,
                    "selection_rank_sha256": rank(args.seed, "balanced_selection", source_id),
                    "assignment_rank_sha256": rank(args.seed, "split_assignment", source_id),
                    "row_order_sha256": rank(args.seed, f"row_order:{split}", source_id),
                }
                outputs[split].append(record)
                assignments[source_id] = split
            offset += count

    expected_totals = {name: count * 2 for name, count in SPLIT_COUNTS_PER_CLASS.items()}
    all_ids: list[str] = []
    for split, rows in outputs.items():
        rows.sort(key=lambda row: row["row_order_sha256"])
        labels = Counter(row["initial_correct"] for row in rows)
        if len(rows) != expected_totals[split] or labels != {True: len(rows) // 2, False: len(rows) // 2}:
            raise AssertionError(f"Split balance invariant failed: {split}")
        all_ids.extend(str(row["problem_id"]) for row in rows)
    if len(all_ids) != 480 or len(set(all_ids)) != 480:
        raise AssertionError("Frozen splits are not source-disjoint")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_hashes = {
        f"{split}.jsonl": write_jsonl(args.output_dir / f"{split}.jsonl", rows)
        for split, rows in outputs.items()
    }
    assignment_rows = [
        {"problem_id": source_id, "split": split}
        for source_id, split in sorted(assignments.items())
    ]
    output_hashes["source_assignments.jsonl"] = write_jsonl(
        args.output_dir / "source_assignments.jsonl", assignment_rows
    )

    manifest = {
        "schema_version": "phase5_balanced_split_v1",
        "status": "frozen_before_review_or_probe_outcomes",
        "seed": args.seed,
        "selection_rule": (
            "Rank each initial-correctness class by SHA256 of "
            "phase5_split_v1|seed|balanced_selection|problem_id; take 240 per class. "
            "Rank selected rows within class by the split_assignment namespace, then "
            "allocate 120 train, 40 development, and 80 protected_test per class."
        ),
        "inputs": {
            "candidates": str(args.candidates.relative_to(ROOT)).replace("\\", "/"),
            "candidates_sha256_lf": candidates_hash,
            "initial_rollouts": str(args.rollouts.relative_to(ROOT)).replace("\\", "/"),
            "initial_rollouts_sha256_lf": rollouts_hash,
        },
        "available_initial_rows": len(rollouts),
        "available_by_class": {
            "correct": len(by_class[True]), "wrong": len(by_class[False]),
        },
        "split_counts": {
            split: {
                "rows": len(rows),
                "correct": sum(row["initial_correct"] is True for row in rows),
                "wrong": sum(row["initial_correct"] is False for row in rows),
            }
            for split, rows in outputs.items()
        },
        "total_selected": len(all_ids),
        "source_disjoint": True,
        "review_outcomes_seen": False,
        "probe_outcomes_seen": False,
        "output_sha256": output_hashes,
    }
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    (args.output_dir / "manifest.json").write_bytes(manifest_payload)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
