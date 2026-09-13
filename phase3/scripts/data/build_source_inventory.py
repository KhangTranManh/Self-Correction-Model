"""Build one canonical Phase 3 inventory row for every paired source problem."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "phase3.yaml"
ASSIGNMENTS_PATH = ROOT / "data" / "buckets" / "bucket_assignments.jsonl"
SUMMARY_PATH = ROOT / "data" / "source_inventory_summary.json"

ELIGIBLE_BEHAVIORS = {
    "CC": ["preserve_false_feedback", "preserve_neutral"],
    "WW": ["repair_true_feedback", "repair_neutral"],
    # V1 is the Phase 3 starting policy. A WC row is currently correct under V1,
    # so its initial answer can support the same preservation conditions as CC.
    "WC": ["preserve_false_feedback", "preserve_neutral"],
    "CW": ["normal_solve", "regression_recovery"],
}

EXPECTED_CORRECTNESS = {
    "CC": (True, True),
    "WW": (False, False),
    "WC": (False, True),
    "CW": (True, False),
}


def _read_unique(path: Path, id_key: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row[id_key]
            if row_id in rows:
                raise RuntimeError(f"Duplicate {id_key}={row_id!r} at {path}:{line_number}")
            rows[row_id] = row
    return rows


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    source_path = ROOT / config["paths"]["source_problems"]
    output_path = ROOT / config["paths"]["source_inventory"]

    sources = _read_unique(source_path, "id")
    assignments = _read_unique(ASSIGNMENTS_PATH, "problem_id")
    if set(sources) != set(assignments):
        raise RuntimeError(
            f"Source/assignment mismatch: "
            f"source_only={len(set(sources) - set(assignments))}, "
            f"assignment_only={len(set(assignments) - set(sources))}"
        )

    inventory: list[dict] = []
    for problem_id, source in sources.items():
        assignment = assignments[problem_id]
        if source["dataset"] != assignment["dataset"]:
            raise RuntimeError(f"Dataset mismatch for {problem_id}")
        bucket = assignment["bucket"]
        base_correct = assignment["base_initial_correct"]
        v1_correct = assignment["v1_initial_correct"]
        if (base_correct, v1_correct) != EXPECTED_CORRECTNESS[bucket]:
            raise RuntimeError(f"Incorrect truth flags for bucket {bucket}: {problem_id}")
        inventory.append(
            {
                "id": problem_id,
                "dataset": source["dataset"],
                "domain": source["domain"],
                "bucket": bucket,
                "base_correct": base_correct,
                "v1_correct": v1_correct,
                "eligible_behaviors": ELIGIBLE_BEHAVIORS[bucket],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in inventory
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(output_path)

    bucket_counts = Counter(row["bucket"] for row in inventory)
    behavior_counts = Counter(
        behavior for row in inventory for behavior in row["eligible_behaviors"]
    )
    summary = {
        "total": len(inventory),
        "unique_ids": len({row["id"] for row in inventory}),
        "counts_by_bucket": {
            bucket: bucket_counts[bucket] for bucket in ELIGIBLE_BEHAVIORS
        },
        "counts_by_dataset_and_bucket": {
            dataset: {
                bucket: sum(
                    row["dataset"] == dataset and row["bucket"] == bucket
                    for row in inventory
                )
                for bucket in ELIGIBLE_BEHAVIORS
            }
            for dataset in sorted({row["dataset"] for row in inventory})
        },
        "eligible_rows_by_behavior": dict(sorted(behavior_counts.items())),
        "behavior_mapping": ELIGIBLE_BEHAVIORS,
        "inventory_file": output_path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
