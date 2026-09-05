"""Pair Base/V1 initial attempts and split both models into CC/WW/WC/CW buckets."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ATTEMPTS_ROOT = ROOT / "data" / "attempts"
OUTPUT_ROOT = ROOT / "data" / "buckets"
MODEL_FILES = {
    "base": ATTEMPTS_ROOT / "base" / "raw_attempts.jsonl",
    "self_correction_v1": (
        ATTEMPTS_ROOT / "self_correction_v1" / "raw_attempts.jsonl"
    ),
}
BUCKETS = ("CC", "WW", "WC", "CW")


def _read_unique(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            problem_id = row["id"]
            if problem_id in rows:
                raise RuntimeError(f"Duplicate ID {problem_id!r} in {path}:{line_number}")
            rows[problem_id] = row
    return rows


def _bucket(base_correct: bool, v1_correct: bool) -> str:
    if base_correct and v1_correct:
        return "CC"
    if not base_correct and not v1_correct:
        return "WW"
    if not base_correct and v1_correct:
        return "WC"
    return "CW"


def _derived_row(row: dict, bucket: str) -> dict:
    return {
        "problem_id": row["id"],
        "bucket": bucket,
        **{key: value for key, value in row.items() if key != "id"},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    base = _read_unique(MODEL_FILES["base"])
    v1 = _read_unique(MODEL_FILES["self_correction_v1"])
    if set(base) != set(v1):
        raise RuntimeError(
            f"Base/V1 ID mismatch: base_only={len(set(base) - set(v1))}, "
            f"v1_only={len(set(v1) - set(base))}"
        )

    paired_fields = ("dataset", "problem", "ground_truth")
    grouped: dict[str, dict[str, list[dict]]] = {
        bucket: {model: [] for model in MODEL_FILES} for bucket in BUCKETS
    }
    assignments: list[dict] = []

    # Preserve the source order from the Base raw artifact.
    for problem_id, base_row in base.items():
        v1_row = v1[problem_id]
        if any(base_row[field] != v1_row[field] for field in paired_fields):
            raise RuntimeError(f"Source mismatch between Base and V1 for {problem_id}")
        bucket = _bucket(base_row["initial_correct"], v1_row["initial_correct"])
        grouped[bucket]["base"].append(_derived_row(base_row, bucket))
        grouped[bucket]["self_correction_v1"].append(_derived_row(v1_row, bucket))
        assignments.append(
            {
                "problem_id": problem_id,
                "dataset": base_row["dataset"],
                "bucket": bucket,
                "base_initial_correct": base_row["initial_correct"],
                "v1_initial_correct": v1_row["initial_correct"],
            }
        )

    hashes: dict[str, str] = {}
    files: dict[str, dict[str, str]] = {}
    for bucket in BUCKETS:
        files[bucket] = {}
        expected = len(grouped[bucket]["base"])
        if len(grouped[bucket]["self_correction_v1"]) != expected:
            raise RuntimeError(f"Unpaired rows in bucket {bucket}")
        for model, rows in grouped[bucket].items():
            path = OUTPUT_ROOT / bucket / f"{model}.jsonl"
            relative_path = path.relative_to(ROOT).as_posix()
            hashes[relative_path] = _write_jsonl(path, rows)
            files[bucket][model] = relative_path

    assignments_path = OUTPUT_ROOT / "bucket_assignments.jsonl"
    assignments_relative = assignments_path.relative_to(ROOT).as_posix()
    hashes[assignments_relative] = _write_jsonl(
        assignments_path, assignments
    )

    counts = Counter(row["bucket"] for row in assignments)
    dataset_counts = {
        dataset: dict(
            Counter(
                row["bucket"] for row in assignments if row["dataset"] == dataset
            )
        )
        for dataset in sorted({row["dataset"] for row in assignments})
    }
    if sum(counts.values()) != len(base):
        raise RuntimeError("Bucket counts do not partition the paired source IDs")

    summary = {
        "definitions": {
            "CC": "Base correct, Self_Correction_v1 correct",
            "WW": "Base wrong, Self_Correction_v1 wrong",
            "WC": "Base wrong, Self_Correction_v1 correct",
            "CW": "Base correct, Self_Correction_v1 wrong",
        },
        "total_paired_problems": len(base),
        "counts": {bucket: counts[bucket] for bucket in BUCKETS},
        "counts_by_dataset": {
            dataset: {bucket: values.get(bucket, 0) for bucket in BUCKETS}
            for dataset, values in dataset_counts.items()
        },
        "files": files,
        "assignments_file": assignments_relative,
        "sha256": hashes,
    }
    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
