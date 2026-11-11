"""Verify frozen A0 inputs and create a CPU sandbox execution plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from close_a0_cpu_sandbox_v2 import canonical_sha256, prepare
from rescore_a0_discovery import load_lock, verified_completions, verify_manifest
from track_a0_cpu_admission import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--previous-closure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    reviews = [json.loads(path.read_text(encoding="utf-8")) for path in args.reviews]
    previous = json.loads(args.previous_closure.read_text(encoding="utf-8"))
    manifest = verify_manifest(args.run, args.manifest)
    lock = load_lock(args.run)
    if manifest.get("lock_sha256") != lock["lock_sha256"]:
        raise RuntimeError("manifest and lock differ")
    completions, _ = verified_completions(args.run, lock)
    tracker, prepared = prepare(inventory, reviews, completions, previous)
    positions = [row[0]["position"] for row in prepared]
    paths = [args.inventory, args.manifest, args.previous_closure, *args.reviews]
    report = {
        "schema_version": "p16_a0_cpu_sandbox_plan_v2",
        "source": {"inventory_sha256": inventory["inventory_sha256"],
                   "review_chain_tip_sha256": tracker["source"]["review_chain_tip_sha256"],
                   "previous_closure_sha256": previous["closure_sha256"],
                   "source_file_sha256": {p.name: sha256_file(p) for p in paths}},
        "oracle_ready": len(tracker["positions"]["oracle_ready"]),
        "previous_closure_records": 24,
        "remaining_count": len(positions),
        "batches": [{"start": start, "count": len(positions[start:start + 10]),
                     "positions": positions[start:start + 10]}
                    for start in range(0, len(positions), 10)],
        "claims": {"sandbox_execution_performed": False,
                   "correction_pool_certified": False,
                   "gpu_training_authorized": False},
    }
    report["plan_sha256"] = canonical_sha256(report)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(f"Verified {len(positions)} remaining records in {len(report['batches'])} batches")
    print(f"plan_sha256: {report['plan_sha256']}")


if __name__ == "__main__":
    main()
