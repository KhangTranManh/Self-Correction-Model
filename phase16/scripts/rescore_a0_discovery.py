"""Re-execute a frozen P16-G0 run without loading model weights.

This is a deterministic dataset-test rescore, not independent-oracle
admission. It verifies the downloaded manifest, lock, shard summaries and
completion identities before executing each saved recovered output in the
same networkless sandbox used by generation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from colab_a0_discovery import (
    SHARD_COUNT,
    SHARD_SIZE,
    canonical_sha256,
    classify,
    load_inputs,
    load_lock,
    result_label,
    sandbox_execute,
    sandbox_preflight,
    sha256_file,
    write_new_json,
)


SCHEMA_VERSION = "p16_a0_discovery_rescore_v1"


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"expected JSON objects in {path}")
    return rows


def verify_manifest(run: Path, manifest_path: Path) -> dict:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != "p16_a0_discovery_run_manifest_v1":
        raise RuntimeError("unexpected run manifest schema")
    if manifest.get("completed") != SHARD_COUNT * SHARD_SIZE:
        raise RuntimeError("run manifest completion count mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("run manifest has no file hashes")
    for name, expected in files.items():
        if (not isinstance(name, str) or not isinstance(expected, str)
                or len(expected) != 64):
            raise RuntimeError("malformed run manifest file entry")
        path = run / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"run manifest hash mismatch for {name}")
    return manifest


def verified_completions(run: Path, lock: dict) -> tuple[list[dict], dict[str, str]]:
    completions: list[dict] = []
    source_hashes: dict[str, str] = {}
    for shard in range(SHARD_COUNT):
        audit_path = run / f"a0_shard_{shard:02d}.jsonl"
        summary_path = run / f"a0_shard_{shard:02d}_summary.json"
        rows = load_jsonl(audit_path)
        summary = load_json(summary_path)
        claimed_summary = summary.get("summary_sha256")
        if claimed_summary != canonical_sha256({
                key: value for key, value in summary.items()
                if key != "summary_sha256"}):
            raise RuntimeError(f"shard {shard} summary hash mismatch")
        if (summary.get("shard_index") != shard
                or summary.get("completed") != SHARD_SIZE
                or summary.get("lock_sha256") != lock["lock_sha256"]
                or summary.get("audit_sha256") != sha256_file(audit_path)):
            raise RuntimeError(f"shard {shard} summary state mismatch")
        if any(row.get("kind") == "failure" for row in rows):
            raise RuntimeError(f"shard {shard} contains a failure record")
        shard_completions = [row for row in rows if row.get("kind") == "completion"]
        if len(shard_completions) != SHARD_SIZE:
            raise RuntimeError(f"shard {shard} completion count mismatch")
        expected_positions = list(range(shard * SHARD_SIZE,
                                        (shard + 1) * SHARD_SIZE))
        if [row.get("position") for row in shard_completions] != expected_positions:
            raise RuntimeError(f"shard {shard} position sequence mismatch")
        label_counts = Counter(row.get("verifier_label") for row in shard_completions)
        for label, count in label_counts.items():
            if summary.get("counts", {}).get(label) != count:
                raise RuntimeError(f"shard {shard} stored label count mismatch")
        for row in shard_completions:
            locked = lock["selected"][row["position"]]
            if (row.get("item_id") != locked["item_id"]
                    or row.get("family_id") != locked["family_id"]
                    or row.get("prompt_sha256") != locked["prompt_sha256"]
                    or row.get("tests_sha256") != locked["tests_sha256"]):
                raise RuntimeError(f"completion identity mismatch at {row['position']}")
            raw = row.get("raw_output")
            recovered = row.get("recovered_output")
            if not isinstance(raw, str) or not isinstance(recovered, str):
                raise RuntimeError(f"completion output type mismatch at {row['position']}")
            if hashlib.sha256(raw.encode()).hexdigest() != row.get("raw_output_sha256"):
                raise RuntimeError(f"raw output hash mismatch at {row['position']}")
            if (hashlib.sha256(recovered.encode()).hexdigest()
                    != row.get("recovered_output_sha256")):
                raise RuntimeError(f"recovered output hash mismatch at {row['position']}")
        completions.extend(shard_completions)
        source_hashes[audit_path.name] = sha256_file(audit_path)
        source_hashes[summary_path.name] = sha256_file(summary_path)
    return completions, source_hashes


def run_rescore(root: Path, run: Path, manifest_path: Path, output: Path) -> dict:
    manifest = verify_manifest(run, manifest_path)
    lock = load_lock(run)
    if manifest.get("lock_sha256") != lock["lock_sha256"]:
        raise RuntimeError("manifest and lock differ")
    completions, source_hashes = verified_completions(run, lock)
    bank, _ = load_inputs(root)
    by_id = {task["item_id"]: task for task in bank["tasks"]}
    sandbox_preflight()

    rescored_counts: Counter[str] = Counter()
    disagreements: list[dict] = []
    for row in completions:
        position = row["position"]
        task = by_id.get(row["item_id"])
        if task is None:
            raise RuntimeError(f"bank task missing at position {position}")
        if (task["family_id"] != row["family_id"]
                or task["prompt_sha256"] != row["prompt_sha256"]
                or task["tests_sha256"] != row["tests_sha256"]):
            raise RuntimeError(f"bank identity mismatch at position {position}")
        rescored = sandbox_execute(row["recovered_output"], task)
        label = result_label(rescored)
        rescored_counts[label] += 1
        rescored_class = classify(row["raw_status"], label, row["recovery_status"])
        if (label != row["verifier_label"]
                or rescored != row["verification"]
                or rescored_class != row["pre_intervention_class"]):
            disagreements.append({
                "position": position,
                "item_id": row["item_id"],
                "stored_label": row["verifier_label"],
                "rescored_label": label,
                "stored_class": row["pre_intervention_class"],
                "rescored_class": rescored_class,
                "stored_verification_sha256": canonical_sha256(row["verification"]),
                "rescored_verification_sha256": canonical_sha256(rescored),
            })

    report = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_lock_sha256": lock["lock_sha256"],
        "source_file_sha256": dict(sorted(source_hashes.items())),
        "completed": len(completions),
        "rescored_counts": dict(sorted(rescored_counts.items())),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "all_saved_verdicts_reproduced": not disagreements,
        "model_generation": False,
        "dataset_oracle_only": True,
        "independent_oracle_validated": False,
        "gpu_training_authorized": False,
    }
    report["report_sha256"] = canonical_sha256(report)
    write_new_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Directory containing candidate_bank.json and scripts/")
    parser.add_argument("--run", type=Path, required=True,
                        help="Frozen a0_discovery_run_v2 directory")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run / "a0_rescore_v1.json"
    report = run_rescore(args.root, args.run, args.manifest, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["disagreement_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
