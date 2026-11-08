"""Build the fail-closed P16-R0 CPU admission inventory from frozen A0.

The inventory accounts for every dataset-side failure, but it does not create
an independent oracle or admit any correction.  Only deterministic
``substantive_content`` candidates are queued for later human-reviewed,
task-specific oracle work.  Execution/scaffold failures and nondeterministic
reruns are retained as explicit rejections.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from colab_a0_discovery import load_inputs, load_lock, sha256_file, write_new_json
from rescore_a0_discovery import (
    load_json,
    verified_completions,
    verify_manifest,
)


SCHEMA_VERSION = "p16_a0_admission_inventory_v1"
EXPECTED_RUN_COUNTS = {
    "correct": 156,
    "runtime_error": 57,
    "syntax_or_interface_error": 18,
    "wrong_answer": 169,
}
FAILURE_CLASS_BY_LABEL = {
    "wrong_answer": "substantive_content",
    "runtime_error": "execution_or_scaffold",
    "syntax_or_interface_error": "execution_or_scaffold",
}


def canonical_sha256(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_rescore(path: Path, manifest_path: Path, lock: dict,
                   source_hashes: dict[str, str]) -> dict:
    report = load_json(path)
    _require(report.get("schema_version") == "p16_a0_discovery_rescore_v1",
             "unexpected rescore schema")
    claimed_hash = report.get("report_sha256")
    without_hash = copy.deepcopy(report)
    without_hash.pop("report_sha256", None)
    _require(claimed_hash == canonical_sha256(without_hash),
             "rescore report hash mismatch")
    _require(report.get("source_manifest_sha256") == sha256_file(manifest_path),
             "rescore source manifest hash mismatch")
    _require(report.get("source_lock_sha256") == lock.get("lock_sha256"),
             "rescore source lock mismatch")
    _require(report.get("source_file_sha256") == dict(sorted(source_hashes.items())),
             "rescore source file hashes mismatch")
    _require(report.get("rescored_counts") == EXPECTED_RUN_COUNTS,
             "rescore aggregate differs from frozen A0 result")
    _require(report.get("completed") == sum(EXPECTED_RUN_COUNTS.values()),
             "rescore completion count mismatch")
    _require(report.get("dataset_oracle_only") is True
             and report.get("independent_oracle_validated") is False
             and report.get("gpu_training_authorized") is False,
             "rescore claims exceed the P16-G0 boundary")
    return report


def verify_nondeterminism(path: Path, rescore_path: Path,
                          rescore: dict) -> dict[int, dict]:
    evidence = load_json(path)
    _require(evidence.get("schema_version") ==
             "p16_a0_rescore_nondeterminism_evidence_v1",
             "unexpected nondeterminism-evidence schema")
    _require(evidence.get("source_rescore_file_sha256") ==
             sha256_file(rescore_path),
             "nondeterminism evidence points to a different rescore file")
    _require(evidence.get("r0_admission_eligible") is False
             and evidence.get("independent_oracle_validated") is False
             and evidence.get("rejection_reason") ==
             "deterministic_rerun_failed",
             "nondeterminism evidence does not fail admission closed")
    disagreements = rescore.get("disagreements")
    _require(isinstance(disagreements, list) and disagreements,
             "rescore has no recorded disagreement")
    by_position = {evidence.get("position"): evidence}
    _require(set(by_position) == {row.get("position") for row in disagreements},
             "nondeterminism evidence does not cover every disagreement")
    for row in disagreements:
        item = by_position[row["position"]]
        _require(item.get("item_id") == row.get("item_id"),
                 "nondeterminism item identity mismatch")
    return by_position


def build_inventory(bank: dict, completions: list[dict], rescore: dict,
                    nondeterministic: dict[int, dict], *,
                    source: dict[str, str] | None = None) -> dict:
    """Build an inventory from already integrity-checked in-memory inputs."""

    tasks = bank.get("tasks")
    _require(isinstance(tasks, list), "bank tasks must be a list")
    by_id = {task.get("item_id"): task for task in tasks}
    _require(len(by_id) == len(tasks), "bank item IDs are not unique")
    positions = [row.get("position") for row in completions]
    _require(len(positions) == len(set(positions)),
             "completion positions are not unique")
    item_ids = [row.get("item_id") for row in completions]
    _require(len(item_ids) == len(set(item_ids)),
             "completion item IDs are not unique")

    observed_counts = Counter(row.get("verifier_label") for row in completions)
    _require(dict(sorted(observed_counts.items())) ==
             dict(sorted(rescore.get("rescored_counts", {}).items())),
             "saved completion counts differ from rescore")

    records: list[dict] = []
    for row in completions:
        label = row.get("verifier_label")
        if label == "correct":
            continue
        expected_class = FAILURE_CLASS_BY_LABEL.get(label)
        _require(expected_class is not None, f"unknown failure label: {label}")
        _require(row.get("pre_intervention_class") == expected_class,
                 f"failure class mismatch at position {row.get('position')}")
        task = by_id.get(row.get("item_id"))
        _require(task is not None,
                 f"bank task missing at position {row.get('position')}")
        for key in ("family_id", "prompt_sha256", "tests_sha256"):
            _require(row.get(key) == task.get(key),
                     f"bank identity mismatch for {key} at position "
                     f"{row.get('position')}")

        position = row["position"]
        rejection_reasons: list[str] = []
        class_gate = "passed"
        if expected_class != "substantive_content":
            class_gate = "failed"
            rejection_reasons.append("execution_or_scaffold_not_correction_data")
        rerun_gate = "passed"
        if position in nondeterministic:
            rerun_gate = "failed"
            rejection_reasons.append("deterministic_rerun_failed")

        if rejection_reasons:
            admission_status = "rejected_before_independent_oracle_review"
            oracle_gate = "not_reviewed_ineligible"
        else:
            admission_status = "pending_independent_oracle_review"
            oracle_gate = "pending"

        verification = row.get("verification")
        _require(isinstance(verification, dict),
                 f"verification missing at position {position}")
        record = {
            "position": position,
            "item_id": row["item_id"],
            "family_id": row["family_id"],
            "source_id_sha256": task.get("source_id_sha256"),
            "source_site": task.get("source_site"),
            "function_name": task.get("function_name"),
            "prompt_sha256": row["prompt_sha256"],
            "published_tests_sha256": row["tests_sha256"],
            "recovered_output_sha256": row.get("recovered_output_sha256"),
            "verifier_label": label,
            "pre_intervention_class": expected_class,
            "published_test_count": verification.get("test_count"),
            "published_match_count": verification.get("match_count"),
            "dataset_oracle_only": True,
            "source_history_exact_clean": task.get("history_exact_clean"),
            "source_near_duplicate_component_size":
                task.get("near_duplicate_component_size"),
            "gates": {
                "failure_class": class_gate,
                "deterministic_rerun": rerun_gate,
                "independent_oracle": oracle_gate,
                "contract_review": "pending" if not rejection_reasons else
                                   "not_reviewed_ineligible",
                "full_history_review": "pending" if not rejection_reasons else
                                       "not_reviewed_ineligible",
                "near_duplicate_review": "pending" if not rejection_reasons else
                                         "not_reviewed_ineligible",
                "human_approval": "pending" if not rejection_reasons else
                                  "not_reviewed_ineligible",
            },
            "admission_status": admission_status,
            "rejection_reasons": rejection_reasons,
            "correction_admitted": False,
        }
        record["record_sha256"] = canonical_sha256(record)
        records.append(record)

    records.sort(key=lambda value: value["position"])
    status_counts = Counter(row["admission_status"] for row in records)
    class_counts = Counter(row["pre_intervention_class"] for row in records)
    rejection_counts = Counter(
        reason for row in records for reason in row["rejection_reasons"])
    document = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "cpu_independent_oracle_admission_worklist",
        "source": source or {},
        "counts": {
            "a0_completed": len(completions),
            "dataset_side_failures": len(records),
            "pre_intervention_class": dict(sorted(class_counts.items())),
            "admission_status": dict(sorted(status_counts.items())),
            "rejection_reason": dict(sorted(rejection_counts.items())),
            "correction_admitted": 0,
        },
        "records": records,
        "claims": {
            "dataset_oracle_only": True,
            "independent_oracle_validated": False,
            "correction_pool_certified": False,
            "gpu_training_authorized": False,
            "protected_data_opening_authorized": False,
        },
    }
    document["inventory_sha256"] = canonical_sha256(document)
    return document


def run(root: Path, run_dir: Path, manifest_path: Path, rescore_path: Path,
        nondeterminism_path: Path, output: Path) -> dict:
    manifest = verify_manifest(run_dir, manifest_path)
    lock = load_lock(run_dir)
    _require(manifest.get("lock_sha256") == lock.get("lock_sha256"),
             "manifest and lock differ")
    completions, source_hashes = verified_completions(run_dir, lock)
    rescore = verify_rescore(rescore_path, manifest_path, lock, source_hashes)
    nondeterministic = verify_nondeterminism(
        nondeterminism_path, rescore_path, rescore)
    bank, _ = load_inputs(root)
    source = {
        "run_manifest_sha256": sha256_file(manifest_path),
        "lock_sha256": lock["lock_sha256"],
        "bank_sha256": bank["bank_sha256"],
        "rescore_file_sha256": sha256_file(rescore_path),
        "nondeterminism_file_sha256": sha256_file(nondeterminism_path),
    }
    document = build_inventory(
        bank, completions, rescore, nondeterministic, source=source)
    _require(document["counts"]["dataset_side_failures"] == 244,
             "frozen A0 failure total changed")
    _require(document["counts"]["admission_status"] == {
        "pending_independent_oracle_review": 168,
        "rejected_before_independent_oracle_review": 76,
    }, "frozen A0 admission accounting changed")
    write_new_json(output, document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Directory containing candidate_bank.json and scripts/")
    parser.add_argument("--run", type=Path, required=True,
                        help="Frozen a0_discovery_run_v2 directory")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rescore", type=Path, required=True)
    parser.add_argument("--nondeterminism", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = run(args.root, args.run, args.manifest, args.rescore,
                   args.nondeterminism, args.output)
    print(json.dumps(document["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
