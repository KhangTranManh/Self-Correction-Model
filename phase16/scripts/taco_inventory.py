"""Inventory a pinned TACO train snapshot without executing dataset content.

This is a metadata/provenance pass, not an eligibility decision or dedup audit.
It deliberately never opens TACO test and never writes raw questions/solutions.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit


REVISION = "d593ed0a2becbbc952230bb89be09189bf1056dc"
COLUMNS = [
    "question", "input_output", "starter_code", "difficulty", "source",
    "url", "name", "tags", "skill_types", "picture_num", "solutions",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def canonical_url(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parts = urlsplit(value.strip())
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            return None
        path = parts.path.rstrip("/") or "/"
        return urlunsplit(("https", parts.netloc.lower(), path, "", ""))
    except ValueError:
        return None


def metadata_list(value: object) -> list[str] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            return None
    if not isinstance(parsed, list):
        return None
    return [str(item) for item in parsed if isinstance(item, str)]


def io_metadata(value: object) -> tuple[str, int, str | None]:
    if not isinstance(value, str) or not value.strip():
        return "missing", 0, None
    try:
        # Some contest outputs contain integers with thousands of digits.
        # We only inspect shape/counts here, so keep JSON numbers as text.
        obj = json.loads(value, parse_int=str)
    except (json.JSONDecodeError, TypeError):
        return "malformed", 0, None
    if not isinstance(obj, dict):
        return "malformed", 0, None
    inputs, outputs = obj.get("inputs"), obj.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return "malformed", 0, None
    if len(inputs) != len(outputs) or not inputs:
        return "unpaired_or_empty", min(len(inputs), len(outputs)), None
    fn = obj.get("fn_name")
    if fn is not None and not isinstance(fn, str):
        return "malformed", len(inputs), None
    return ("call_based" if fn else "standard_input"), len(inputs), fn or None


def inventory(raw_dir: Path, output_dir: Path) -> None:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit("pyarrow is required only for reading TACO parquet shards") from exc

    shards = sorted((raw_dir / "ALL").glob("train-*.parquet"))
    if len(shards) != 9:
        raise SystemExit(f"expected 9 train shards; found {len(shards)}")
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    counts = Counter()
    source_counts, difficulty_counts, io_counts = Counter(), Counter(), Counter()
    skill_counts = Counter()
    shard_records = []
    metadata_path = output_dir / "train_metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8", newline="\n") as out:
        for shard in shards:
            shard_records.append({
                "name": shard.name, "bytes": shard.stat().st_size,
                "sha256": sha256_file(shard),
            })
            parquet = pq.ParquetFile(shard)
            absent = set(COLUMNS) - set(parquet.schema_arrow.names)
            if absent:
                raise SystemExit(f"{shard.name}: missing columns {sorted(absent)}")
            for batch in parquet.iter_batches(batch_size=256, columns=COLUMNS):
                for row in batch.to_pylist():
                    counts["rows"] += 1
                    question = row.get("question")
                    if not isinstance(question, str) or not question.strip():
                        counts["missing_question"] += 1
                        question = ""
                    source = str(row.get("source") or "<missing>")
                    difficulty = str(row.get("difficulty") or "<missing>")
                    source_counts[source] += 1
                    difficulty_counts[difficulty] += 1
                    io_mode, test_count, fn_name = io_metadata(row.get("input_output"))
                    io_counts[io_mode] += 1
                    skills = metadata_list(row.get("skill_types"))
                    if skills is None:
                        counts["missing_or_unparseable_skills"] += 1
                    else:
                        skill_counts.update(set(skills))
                    solutions = metadata_list(row.get("solutions"))
                    if solutions is None:
                        counts["missing_or_unparseable_solutions"] += 1
                    url = canonical_url(row.get("url"))
                    if url is None:
                        counts["missing_or_invalid_url"] += 1
                    picture = row.get("picture_num")
                    has_picture = str(picture or "0").strip() not in {"0", "0.0", ""}
                    if has_picture:
                        counts["picture_flag"] += 1
                    record = {
                        "id": f"taco_train_{counts['rows'] - 1:05d}",
                        "source": source,
                        "difficulty": difficulty,
                        "name": row.get("name"),
                        "canonical_url": url,
                        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                        "normalized_question_sha256": hashlib.sha256(
                            normalized_text(question).encode("utf-8")
                        ).hexdigest(),
                        "question_chars": len(question),
                        "io_mode": io_mode,
                        "test_count": test_count,
                        "fn_name": fn_name,
                        "solution_count": len(solutions) if solutions is not None else None,
                        "skills": skills,
                        "picture_flag": has_picture,
                    }
                    out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": "p16_taco_train_inventory_v1",
        "source_repo": "BAAI/TACO",
        "revision": REVISION,
        "split_opened": "train",
        "test_opened": False,
        "shards": shard_records,
        "metadata_sha256": sha256_file(metadata_path),
        "counts": dict(counts),
        "source_counts": dict(sorted(source_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "io_counts": dict(sorted(io_counts.items())),
        "skill_counts": dict(sorted(skill_counts.items())),
        "eligibility_assessed": False,
        "historical_overlap_assessed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "rows": counts["rows"], "io_counts": dict(io_counts),
        "metadata_sha256": summary["metadata_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    inventory(args.raw_dir, args.output_dir)
