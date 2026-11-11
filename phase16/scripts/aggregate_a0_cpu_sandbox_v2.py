"""Verify and summarize all exact-comparison CPU sandbox closures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from track_a0_cpu_admission import canonical_sha256, sha256_file
from verify_a0_cpu_sandbox_v2_batch import verify


def aggregate(plan, tracker, previous, batches, *, source_files):
    tracker_body = dict(tracker)
    if tracker_body.pop("tracker_sha256", None) != canonical_sha256(tracker_body):
        raise RuntimeError("tracker hash mismatch")
    prior_body = dict(previous)
    if prior_body.pop("rescore_sha256", None) != canonical_sha256(prior_body):
        raise RuntimeError("first-24 exact rescore hash mismatch")
    if (previous.get("schema_version") != "p16_a0_cpu_sandbox_v1_exact_rescore_v1" or
            previous["source"]["previous_closure_sha256"] != plan["source"]["previous_closure_sha256"]):
        raise RuntimeError("first-24 rescore source mismatch")
    if tracker["source"]["review_chain_tip_sha256"] != plan["source"]["review_chain_tip_sha256"]:
        raise RuntimeError("tracker review chain mismatch")
    if len(batches) != len(plan["batches"]):
        raise RuntimeError("batch count mismatch")
    batches = sorted(batches, key=lambda report: report["offset"])
    if [b["offset"] for b in batches] != [b["start"] for b in plan["batches"]]:
        raise RuntimeError("batch offsets are incomplete or duplicated")
    for batch in batches:
        verify(plan, batch)
    records = previous["records"] + [row for batch in batches for row in batch["records"]]
    if [row["position"] for row in records] != tracker["positions"]["oracle_ready"]:
        raise RuntimeError("exact rescores do not cover the oracle-ready queue")
    names = ("substantive_residual_reproduced", "residual_not_reproduced", "execution_or_scaffold_unresolved")
    if any(row.get("verdict") not in names or row.get("correction_admitted") is not False for row in records):
        raise RuntimeError("invalid record disposition")
    counts = {name: sum(row["verdict"] == name for row in records) for name in names}
    if previous["counts"] != {name: sum(row["verdict"] == name for row in previous["records"]) for name in names}:
        raise RuntimeError("previous rescore counts mismatch")
    report = {"schema_version": "p16_a0_cpu_sandbox_exact_summary_v1",
              "source": {"plan_sha256": plan["plan_sha256"],
                         "tracker_sha256": tracker["tracker_sha256"],
                         "previous_exact_rescore_sha256": previous["rescore_sha256"],
                         "source_file_sha256": dict(sorted(source_files.items()))},
              "counts": {"oracle_ready_checked": len(records), **counts, "correction_admitted": 0},
              "positions_by_verdict": {name: [row["position"] for row in records if row["verdict"] == name]
                                       for name in names},
              "claims": {"model_generation": False, "gpu_used": False,
                         "correction_pool_certified": False, "gpu_training_authorized": False}}
    report["summary_sha256"] = canonical_sha256(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--previous-exact", type=Path, required=True)
    parser.add_argument("--batches", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    paths = [args.plan, args.tracker, args.previous_exact, *args.batches]
    report = aggregate(read(args.plan), read(args.tracker), read(args.previous_exact),
                       [read(p) for p in args.batches],
                       source_files={p.name: sha256_file(p) for p in paths})
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(report["counts"], indent=2))
    print(f"summary_sha256: {report['summary_sha256']}")


if __name__ == "__main__":
    main()
