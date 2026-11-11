"""Bind existing exact-history metadata to the A0 follow-up worklist.

This reads the prior scan and its source metadata, not historical task text.
It cannot establish semantic or complete historical non-overlap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from track_a0_cpu_admission import canonical_sha256, sha256_file


def build(worklist, bank, overlap, metadata_path, *, source_files):
    body = dict(worklist)
    if body.pop("worklist_sha256", None) != canonical_sha256(body):
        raise RuntimeError("worklist hash mismatch")
    bank_body = dict(bank)
    if bank_body.pop("bank_sha256", None) != canonical_sha256(bank_body):
        raise RuntimeError("bank hash mismatch")
    if overlap.get("schema_version") != "p16_taco_exact_overlap_v1":
        raise RuntimeError("unexpected historical overlap schema")
    if overlap["metadata_sha256"] != sha256_file(metadata_path):
        raise RuntimeError("historical metadata hash mismatch")
    if overlap.get("parse_errors"):
        raise RuntimeError("historical scan contained parse errors")
    metadata = {}
    with metadata_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in metadata:
                raise RuntimeError("duplicate source metadata id")
            metadata[row["id"]] = row
    hits = {row["id"] for row in overlap["historical_exact_hits"]}
    tasks = {row["item_id"]: row for row in bank["tasks"]}
    records = []
    for item in worklist["records"]:
        task = tasks[item["item_id"]]
        source = metadata[item["item_id"]]
        if task["normalized_question_sha256"] != source["normalized_question_sha256"]:
            raise RuntimeError(f"question identity mismatch at {item['position']}")
        if task["prompt_sha256"] != item["prompt_sha256"]:
            raise RuntimeError(f"prompt identity mismatch at {item['position']}")
        records.append({"position": item["position"], "item_id": item["item_id"],
                        "historical_exact_hit": item["item_id"] in hits,
                        "historical_semantic_review": "pending",
                        "manual_cross_history_review": "pending"})
    report = {"schema_version": "p16_a0_existing_history_evidence_v1",
              "source": {"worklist_sha256": worklist["worklist_sha256"],
                         "bank_sha256": bank["bank_sha256"],
                         "metadata_sha256": overlap["metadata_sha256"],
                         "source_file_sha256": dict(sorted(source_files.items()))},
              "scope": {"historical_files_scanned_previously": overlap["historical_files_scanned"],
                        "historical_scan_parse_errors": len(overlap["parse_errors"]),
                        "historical_scan_limitations": overlap["limits"]},
              "counts": {"records_checked": len(records),
                         "exact_hits": sum(row["historical_exact_hit"] for row in records),
                         "semantic_review_pending": len(records),
                         "manual_cross_history_pending": len(records)},
              "records": records,
              "claims": {"historical_text_opened_in_this_audit": False,
                         "full_semantic_history_complete": False,
                         "near_duplicate_review_complete": False,
                         "gpu_training_authorized": False}}
    report["audit_sha256"] = canonical_sha256(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worklist", "bank", "overlap", "metadata", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    paths = [args.worklist, args.bank, args.overlap, args.metadata]
    report = build(read(args.worklist), read(args.bank), read(args.overlap), args.metadata,
                   source_files={p.name: sha256_file(p) for p in paths})
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(report["counts"], indent=2))
    print(f"audit_sha256: {report['audit_sha256']}")


if __name__ == "__main__":
    main()
