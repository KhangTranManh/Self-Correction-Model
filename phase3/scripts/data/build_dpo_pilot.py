"""Build decision-token DPO preferences from verified contrastive behavior rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "contrastive_pairs" / "contrastive_behavior_rows.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "dpo_pilot"
KEEP = "<decision>KEEP</decision>"
REVISE = "<decision>REVISE</decision>"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = read_jsonl(input_path)
    preferences: list[dict[str, Any]] = []
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for source in source_rows:
        decision = source["decision"]
        messages = source["messages"]
        expected = KEEP if decision == "KEEP" else REVISE
        if decision not in {"KEEP", "REVISE"}:
            raise RuntimeError(f"Unexpected decision: {decision}")
        if len(messages) != 4 or messages[-1] != {"role": "assistant", "content": expected}:
            raise RuntimeError(f"Invalid decision conversation: {source['construction_id']}")
        row = {
            "preference_id": f"dpo_{source['construction_id']}",
            "pair_id": source["pair_id"],
            "source_id": source["source_id"],
            "dataset": source["dataset"],
            "domain": source["domain"],
            "answer_state": "verified_correct" if decision == "KEEP" else "verified_wrong",
            "neutral_template_id": source["neutral_template_id"],
            "prompt_messages": messages[:-1],
            "chosen": expected,
            "rejected": REVISE if decision == "KEEP" else KEEP,
            "source_ref": source["source_ref"],
            "answer_ref": source["answer_ref"],
        }
        preferences.append(row)
        by_pair[row["pair_id"]].append(row)

    errors: list[str] = []
    if len(preferences) != 240 or len(by_pair) != 120:
        errors.append(f"expected 240 rows/120 pairs, found {len(preferences)}/{len(by_pair)}")
    if len({row["preference_id"] for row in preferences}) != len(preferences):
        errors.append("duplicate preference_id")
    for pair_id, pair_rows in by_pair.items():
        if len(pair_rows) != 2:
            errors.append(f"{pair_id}: expected two rows")
            continue
        if {row["answer_state"] for row in pair_rows} != {"verified_correct", "verified_wrong"}:
            errors.append(f"{pair_id}: missing correct/wrong state")
        if len({row["neutral_template_id"] for row in pair_rows}) != 1:
            errors.append(f"{pair_id}: paired templates differ")
        if len({row["source_id"] for row in pair_rows}) != 1:
            errors.append(f"{pair_id}: paired source IDs differ")
    if errors:
        raise RuntimeError("; ".join(errors[:20]))

    output_path = output_dir / "dpo_preferences.jsonl"
    write_jsonl(output_path, preferences)
    summary = {
        "schema_version": "phase3_dpo_preferences_v1",
        "source": {"path": str(input_path), "sha256": file_hash(input_path)},
        "output": {"path": str(output_path), "sha256": file_hash(output_path)},
        "preference_rows": len(preferences),
        "same_problem_pairs": len(by_pair),
        "answer_state_distribution": dict(sorted(Counter(row["answer_state"] for row in preferences).items())),
        "dataset_distribution": dict(sorted(Counter(row["dataset"] for row in preferences).items())),
        "domain_distribution": dict(sorted(Counter(row["domain"] for row in preferences).items())),
        "template_by_answer_state": {
            state: dict(sorted(Counter(row["neutral_template_id"] for row in preferences if row["answer_state"] == state).items()))
            for state in ("verified_correct", "verified_wrong")
        },
        "validation": {
            "all_passed": True,
            "chosen_matches_verified_state": True,
            "rejected_is_opposite_decision": True,
            "exactly_two_preferences_per_pair": True,
            "same_template_within_pair": True,
            "source_verification_inherited_from_validated_contrastive_rows": True,
            "synthetic_answers": False,
            "training_started": False,
        },
    }
    (output_dir / "dpo_preferences_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
