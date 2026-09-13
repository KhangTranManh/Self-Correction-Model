"""Convert the protected frozen behavior benchmark into a probe-only manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


NEUTRAL_REVIEW = "Review your previous answer carefully and decide whether it should be kept or revised."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", default="phase3/data/two_stage_selective_repair/frozen_eval.jsonl")
    parser.add_argument("--training-probe", default="phase3/runs/representation_probe/probe_dataset.jsonl")
    parser.add_argument(
        "--output-dir",
        default="phase3/data/frozen_200_probe",
    )
    args = parser.parse_args()

    frozen_path = Path(args.frozen).resolve()
    training_path = Path(args.training_probe).resolve()
    output_dir = Path(args.output_dir).resolve()
    frozen = read_jsonl(frozen_path)
    training = read_jsonl(training_path)
    training_ids = {str(row["source_id"]) for row in training}

    rows = []
    for source in frozen:
        label = str(source["expected_decision"])
        answer = str(source["initial_answer"])
        rows.append({
            "schema_version": "phase3_frozen_200_probe_v1",
            "source_id": str(source["source_id"]),
            "dataset": str(source["dataset"]),
            "domain": str(source["domain"]),
            "label": label,
            "class_id": 0 if label == "KEEP" else 1,
            "split": "external_test",
            "feedback_type": "neutral_review",
            "messages": [
                {"role": "user", "content": str(source["task_prompt"])},
                {"role": "assistant", "content": answer},
                {"role": "user", "content": NEUTRAL_REVIEW},
            ],
            "answer_length_chars": len(answer),
            "answer_length_words": len(answer.split()),
            "source_ref": str(source["source_ref"]),
            "initial_generation_source": str(source["initial_generation_source"]),
            "initial_correct": bool(source["initial_correct"]),
        })

    ids = [row["source_id"] for row in rows]
    overlap = sorted(set(ids) & training_ids)
    errors = []
    if len(rows) != 200 or len(set(ids)) != 200:
        errors.append("Expected 200 unique frozen sources")
    if Counter(row["label"] for row in rows) != Counter({"KEEP": 100, "REVISE": 100}):
        errors.append("Expected exactly 100 KEEP and 100 REVISE rows")
    if overlap:
        errors.append(f"Training-probe source overlap: {len(overlap)}")
    if any(row["initial_correct"] != (row["label"] == "KEEP") for row in rows):
        errors.append("Correctness/decision label mismatch")
    if errors:
        raise RuntimeError("; ".join(errors))

    dataset_path = output_dir / "probe_dataset.jsonl"
    write_jsonl(dataset_path, rows)
    summary = {
        "schema_version": "phase3_frozen_200_probe_summary_v1",
        "rows": len(rows),
        "unique_sources": len(set(ids)),
        "training_probe_overlap": len(overlap),
        "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
        "domains": dict(sorted(Counter(row["domain"] for row in rows).items())),
        "datasets": dict(sorted(Counter(row["dataset"] for row in rows).items())),
        "source_sha256": sha256(frozen_path),
        "training_probe_sha256": sha256(training_path),
        "dataset_sha256": sha256(dataset_path),
        "inference_contract": "same three-message neutral-review prompt as frozen behavioral benchmark",
        "usage": "external probe test only; never fit or tune on these labels",
        "validation": "PASS",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
