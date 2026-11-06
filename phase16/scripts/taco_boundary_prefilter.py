"""Source-text-only, high-recall C-boundary prefilter; never an eligibility label."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import pyarrow.parquet as pq

from taco_inventory import sha256_file


MARKERS = {
    "inclusive": re.compile(r"\binclusiv(?:e|ely)\b", re.I),
    "at_most": re.compile(r"\bat\s+most\b", re.I),
    "less_or_equal": re.compile(r"\b(?:less|smaller)\s+than\s+or\s+equal\b", re.I),
    "not_greater": re.compile(r"\bnot\s+(?:greater|more|larger)\s+than\b", re.I),
    "between": re.compile(r"\bbetween\b", re.I),
    "closed_range": re.compile(r"\[\s*[A-Za-z][\w]*\s*,\s*[A-Za-z][\w]*\s*\]"),
}


def run(raw_dir: Path, metadata_path: Path, overlap_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise SystemExit(f"output already exists: {output_path}")
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = [json.loads(line) for line in stream]
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    historical_hits = {item["id"] for item in overlap["historical_exact_hits"]}
    duplicate_later = {
        row_id for group in overlap["within_taco_exact_text_groups"]
        for row_id in group[1:]
    }
    kept = []
    marker_counts, source_counts = Counter(), Counter()
    row_index = 0
    for shard in sorted((raw_dir / "ALL").glob("train-*.parquet")):
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=256, columns=["question"]):
            for row in batch.to_pylist():
                meta = metadata[row_index]
                row_index += 1
                if (meta["io_mode"] != "call_based" or meta["id"] in historical_hits
                        or meta["id"] in duplicate_later or meta["picture_flag"]):
                    continue
                question = row.get("question") or ""
                markers = [name for name, pattern in MARKERS.items()
                           if pattern.search(question)]
                if markers:
                    marker_counts.update(markers)
                    source_counts[meta["source"]] += 1
                    kept.append({"id": meta["id"], "markers": markers})
    if row_index != len(metadata):
        raise SystemExit(f"row mismatch: {row_index} vs {len(metadata)}")
    result = {
        "schema_version": "p16_taco_boundary_prefilter_v1",
        "metadata_sha256": sha256_file(metadata_path),
        "overlap_sha256": sha256_file(overlap_path),
        "rows_considered": row_index,
        "candidate_count": len(kept),
        "marker_counts": dict(sorted(marker_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "candidate_ids": kept,
        "interpretation": "Source-only lexical candidates, NOT verified inclusive-boundary tasks.",
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"candidate_count": len(kept),
                      "marker_counts": result["marker_counts"],
                      "source_counts": result["source_counts"]}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("metadata_path", type=Path)
    parser.add_argument("overlap_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    run(args.raw_dir, args.metadata_path, args.overlap_path, args.output_path)
