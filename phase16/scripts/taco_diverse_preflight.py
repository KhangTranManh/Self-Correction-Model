"""Source-only structural preflight for a diverse, non-boundary TACO branch.

No model output or dataset-provided solution is executed. Sample rows are
engineering/exploration only and permanently excluded from later study pools.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from taco_inventory import sha256_file


SALT = "P16-D1-source-preflight-2026-11-03"


def rank(row_id: str) -> str:
    return hashlib.sha256((SALT + "|" + row_id).encode()).hexdigest()


def choose(metadata_path: Path, overlap_path: Path) -> list[dict]:
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    historical = {item["id"] for item in overlap["historical_exact_hits"]}
    later_duplicates = {
        row_id for group in overlap["within_taco_exact_text_groups"]
        for row_id in group[1:]
    }
    strata = defaultdict(list)
    with metadata_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (row["io_mode"] != "call_based" or row["id"] in historical
                    or row["id"] in later_duplicates or row["picture_flag"]):
                continue
            strata[(row["source"], row["difficulty"])].append(row)
    picked = []
    for key in sorted(strata):
        picked.extend(sorted(strata[key], key=lambda r: rank(r["id"]))[:2])
    return sorted(picked, key=lambda r: int(r["id"].split("_")[-1]))


def inspect(row: dict, meta: dict) -> dict:
    flags = []
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        flags.append("missing_question")
        question = ""
    try:
        io = json.loads(row.get("input_output") or "", parse_int=str)
    except (json.JSONDecodeError, TypeError, ValueError):
        io = None
        flags.append("invalid_io_json")
    if not isinstance(io, dict):
        io = {}
    inputs, outputs = io.get("inputs"), io.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        flags.append("invalid_test_lists")
        inputs, outputs = [], []
    elif not inputs or len(inputs) != len(outputs):
        flags.append("empty_or_unpaired_tests")
    fn = io.get("fn_name")
    if not isinstance(fn, str) or not fn.isidentifier():
        flags.append("invalid_fn_name")
    if len(inputs) < 3:
        flags.append("fewer_than_3_tests")
    if any(not isinstance(x, (list, dict)) for x in inputs):
        flags.append("non_sequence_call_input")
    if isinstance(fn, str) and fn not in question and fn not in (row.get("starter_code") or ""):
        flags.append("fn_name_not_in_public_contract")
    try:
        solutions = json.loads(row.get("solutions") or "[]")
    except (json.JSONDecodeError, TypeError, ValueError):
        solutions = None
    if not isinstance(solutions, list) or not solutions:
        flags.append("no_parseable_reference_solution")
    return {
        "id": meta["id"],
        "source": meta["source"],
        "difficulty": meta["difficulty"],
        "skills": meta["skills"],
        "test_count": len(inputs),
        "solution_count": len(solutions) if isinstance(solutions, list) else None,
        "input_shapes": dict(Counter(type(x).__name__ for x in inputs)),
        "flags": flags,
    }


def run(raw_dir: Path, metadata_path: Path, overlap_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise SystemExit(f"output already exists: {output_path}")
    selected = choose(metadata_path, overlap_path)
    by_index = {int(item["id"].split("_")[-1]): item for item in selected}
    inspected = []
    index = 0
    for shard in sorted((raw_dir / "ALL").glob("train-*.parquet")):
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(
            batch_size=256,
            columns=["question", "input_output", "starter_code", "solutions"],
        ):
            for row in batch.to_pylist():
                if index in by_index:
                    inspected.append(inspect(row, by_index[index]))
                index += 1
    if len(inspected) != len(selected):
        raise SystemExit("selected rows missing from raw train shards")
    flags = Counter(flag for row in inspected for flag in row["flags"])
    result = {
        "schema_version": "p16_taco_diverse_source_preflight_v1",
        "sample_salt": SALT,
        "metadata_sha256": sha256_file(metadata_path),
        "overlap_sha256": sha256_file(overlap_path),
        "sample_count": len(inspected),
        "source_counts": dict(Counter(row["source"] for row in inspected)),
        "difficulty_counts": dict(Counter(row["difficulty"] for row in inspected)),
        "flag_counts": dict(flags),
        "rows": inspected,
        "interpretation": "Structural source-only engineering sample; no solution or model code executed.",
        "later_study_exclusion": [row["id"] for row in inspected],
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "sample_count": len(inspected),
        "source_counts": result["source_counts"],
        "difficulty_counts": result["difficulty_counts"],
        "flag_counts": result["flag_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("metadata_path", type=Path)
    parser.add_argument("overlap_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    run(args.raw_dir, args.metadata_path, args.overlap_path, args.output_path)
