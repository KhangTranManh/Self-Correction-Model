"""Check position 392's currency symbol against the pinned TACO train shard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as parquet

from track_a0_cpu_admission import canonical_sha256, sha256_file, verify_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "inventory", "metadata-summary", "shard-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    bank, inventory, summary = read(args.bank), read(args.inventory), read(args.metadata_summary)
    verify_inventory(inventory)
    bank_body = dict(bank)
    if bank_body.pop("bank_sha256", None) != canonical_sha256(bank_body):
        raise RuntimeError("bank hash mismatch")
    frozen = next(row for row in inventory["records"] if row["position"] == 392)
    task = next(row for row in bank["tasks"] if row["item_id"] == frozen["item_id"])
    if task["prompt_sha256"] != frozen["prompt_sha256"]:
        raise RuntimeError("prompt identity mismatch")
    index = task["row_index"]
    question = None
    for entry in summary["shards"]:
        path = args.shard_dir / entry["name"]
        source = parquet.ParquetFile(path)
        if index >= source.metadata.num_rows:
            index -= source.metadata.num_rows
            continue
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError("pinned shard hash mismatch")
        for group in range(source.metadata.num_row_groups):
            count = source.metadata.row_group(group).num_rows
            if index >= count:
                index -= count
                continue
            question = source.read_row_group(group, columns=["question"]).column("question")[index].as_py()
            break
        shard = entry
        break
    if question is None or question != task["question"]:
        raise RuntimeError("pinned question differs from frozen bank")
    non_ascii = sorted({f"U+{ord(char):04X}" for char in question if ord(char) > 127})
    report = {"schema_version": "p16_a0_position392_source_audit_v1",
              "position": 392, "item_id": frozen["item_id"],
              "source": {"bank_sha256": bank["bank_sha256"],
                         "taco_revision": summary["revision"], "shard": shard["name"],
                         "shard_sha256": shard["sha256"],
                         "question_sha256": hashlib.sha256(question.encode()).hexdigest()},
              "non_ascii_codepoints": non_ascii,
              "replacement_character_count": question.count("\ufffd"),
              "currency_symbol": "U+00A3" if "£" in question else None,
              "claim": "source identity and code points only; output formatting remains a separate contract question"}
    report["audit_sha256"] = canonical_sha256(report)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({"non_ascii_codepoints": non_ascii,
                      "replacement_character_count": report["replacement_character_count"],
                      "audit_sha256": report["audit_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
