"""Verify one sandbox v2 artifact against its frozen execution plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from track_a0_cpu_admission import canonical_sha256


def verify(plan, report):
    plan_body = dict(plan)
    plan_hash = plan_body.pop("plan_sha256", None)
    if plan_hash != canonical_sha256(plan_body):
        raise RuntimeError("plan hash mismatch")
    body = dict(report)
    report_hash = body.pop("closure_sha256", None)
    if report_hash != canonical_sha256(body):
        raise RuntimeError("batch hash mismatch")
    if report.get("schema_version") != "p16_a0_cpu_sandbox_closure_batch_v2":
        raise RuntimeError("unexpected batch schema")
    source = report.get("source", {})
    if source != plan.get("source"):
        raise RuntimeError("batch sources differ from plan")
    matches = [batch for batch in plan["batches"] if batch["start"] == report.get("offset")]
    if len(matches) != 1:
        raise RuntimeError("batch offset is not in plan")
    records = report["records"]
    if [row["position"] for row in records] != matches[0]["positions"]:
        raise RuntimeError("batch positions differ from plan")
    if report["total_remaining"] != plan["remaining_count"]:
        raise RuntimeError("remaining count differs from plan")
    names = ("substantive_residual_reproduced", "residual_not_reproduced",
             "execution_or_scaffold_unresolved")
    if any(row.get("verdict") not in names or row.get("correction_admitted") is not False
           for row in records):
        raise RuntimeError("invalid verdict or admitted correction")
    if report["counts"] != {name: sum(row["verdict"] == name for row in records)
                            for name in names}:
        raise RuntimeError("batch verdict counts differ")
    if any(report["claims"].get(key) is not False for key in
           ("model_generation", "gpu_used", "correction_pool_certified", "gpu_training_authorized")):
        raise RuntimeError("unexpected batch claim")
    return report_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    report = json.loads(args.batch.read_text(encoding="utf-8"))
    print(f"verified offset {report['offset']}: {verify(plan, report)}")


if __name__ == "__main__":
    main()
