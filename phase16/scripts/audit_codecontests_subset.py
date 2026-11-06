"""Metadata-only capacity check for downloaded CodeContests train shards.

This deliberately does not execute submissions or certify tests as valid.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taco_inventory import normalized_text, sha256_file  # noqa: E402


def paired_count(tests: dict | None) -> tuple[int, bool]:
    tests = tests or {}
    inputs = tests.get("input") or []
    outputs = tests.get("output") or []
    return min(len(inputs), len(outputs)), len(inputs) == len(outputs)


def python3_available(solutions: dict | None) -> bool:
    solutions = solutions or {}
    languages = solutions.get("language") or []
    texts = solutions.get("solution") or []
    return len(languages) == len(texts) and any(
        language == 3 and bool(text) for language, text in zip(languages, texts)
    )


def audit(workspace: Path, shards: list[Path], output: Path) -> None:
    if output.exists():
        raise SystemExit(f"output exists: {output}")
    taco_hashes = {
        json.loads(line)["normalized_question_sha256"]
        for line in (workspace / "phase16/data/taco_inventory_v2/train_metadata.jsonl")
        .read_text(encoding="utf-8").splitlines()
    }
    counts = Counter()
    unique_description_hashes: set[str] = set()
    candidate_hashes: set[str] = set()
    shard_rows = []
    columns = ["description", "public_tests", "private_tests", "generated_tests",
               "solutions", "incorrect_solutions", "input_file", "output_file",
               "cf_contest_id"]
    for shard in sorted(shards):
        local = Counter()
        for batch in pq.ParquetFile(shard).iter_batches(columns=columns):
            for row in batch.to_pylist():
                counts["rows"] += 1
                local["rows"] += 1
                digest = hashlib.sha256(
                    normalized_text(row.get("description") or "").encode()
                ).hexdigest()
                if digest in unique_description_hashes:
                    counts["cross_shard_or_within_shard_duplicate"] += 1
                    local["duplicate"] += 1
                unique_description_hashes.add(digest)
                if digest in taco_hashes:
                    counts["exact_taco_overlap"] += 1
                    local["exact_taco_overlap"] += 1
                test_counts = [paired_count(row.get(f"{kind}_tests"))
                               for kind in ("public", "private", "generated")]
                raw_test_gate = all(paired for _, paired in test_counts) and sum(
                    n for n, _ in test_counts
                ) >= 5
                if raw_test_gate:
                    counts["five_paired_raw_tests"] += 1
                    local["five_paired_raw_tests"] += 1
                if (
                    raw_test_gate
                    and digest not in taco_hashes
                    and not (row.get("input_file") or row.get("output_file"))
                    and python3_available(row.get("solutions"))
                    and python3_available(row.get("incorrect_solutions"))
                ):
                    candidate_hashes.add(digest)
                    counts["candidate_rows_before_dedup"] += 1
                    local["candidate_rows_before_dedup"] += 1
        shard_rows.append({"name": shard.name, "sha256": sha256_file(shard),
                           "counts": dict(local)})
    result = {
        "schema_version": "p16_codecontests_subset_metadata_v1",
        "source_revision": "802411c3010cb00d1b05bad57ca77365a3c699d6",
        "counts": dict(counts),
        "unique_descriptions": len(unique_description_hashes),
        "unique_metadata_candidates": len(candidate_hashes),
        "shards": shard_rows,
        "limitations": [
            "Downloaded train shards only, not the full dataset or a random sample.",
            "Raw paired test count and source labels are not oracle validation.",
            "Only exact normalized-text overlap with TACO train is excluded.",
            "Near duplicates, other historical phases, and family splits remain unchecked.",
        ],
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({"counts": result["counts"],
                      "unique_metadata_candidates": len(candidate_hashes)},
                     sort_keys=True))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    audit(args.workspace, args.shards, args.output)
