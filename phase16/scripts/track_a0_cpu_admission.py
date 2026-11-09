"""Build a model-free CPU tracker for P16-R0 A0 admission progress.

The tracker verifies the frozen inventory and the ordered review hash chain,
then derives a fail-closed funnel and descriptive capacity projection.  It
does not execute model code, construct an oracle, admit a correction, or
authorize training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "p16_a0_cpu_admission_tracker_v1"
INVENTORY_SCHEMA = "p16_a0_admission_inventory_v1"
REVIEW_SCHEMA = "p16_a0_cpu_review_batch_v1"
TARGET_CORRECTIONS = 64
MILESTONES = [24, 48, 80, 120, 168]


def canonical_sha256(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _without_hash(document: dict, key: str) -> dict:
    value = copy.deepcopy(document)
    value.pop(key, None)
    return value


def verify_inventory(inventory: dict) -> dict[int, dict]:
    _require(inventory.get("schema_version") == INVENTORY_SCHEMA,
             "unexpected inventory schema")
    claimed = inventory.get("inventory_sha256")
    _require(claimed == canonical_sha256(
        _without_hash(inventory, "inventory_sha256")),
        "inventory canonical hash mismatch")
    records = inventory.get("records")
    _require(isinstance(records, list), "inventory records must be a list")
    by_position = {row.get("position"): row for row in records}
    _require(len(by_position) == len(records),
             "inventory positions are not unique")
    for position, row in by_position.items():
        _require(isinstance(position, int), "inventory position is not integer")
        claimed_record = row.get("record_sha256")
        _require(claimed_record == canonical_sha256(
            _without_hash(row, "record_sha256")),
            f"inventory record hash mismatch at position {position}")
    counts = inventory.get("counts", {})
    statuses = counts.get("admission_status", {})
    _require(statuses.get("pending_independent_oracle_review") == 168,
             "frozen review queue is not 168")
    _require(statuses.get("rejected_before_independent_oracle_review") == 76,
             "frozen pre-review rejection count is not 76")
    _require(counts.get("correction_admitted") == 0,
             "frozen inventory unexpectedly admits corrections")
    return by_position


def verify_reviews(inventory: dict, inventory_by_position: dict[int, dict],
                   reviews: list[dict]) -> tuple[dict[int, dict], list[dict]]:
    _require(reviews, "at least one review batch is required")
    latest: dict[int, dict] = {}
    chain: list[dict] = []
    previous_hash: str | None = None
    inventory_hash = inventory["inventory_sha256"]
    for index, review in enumerate(reviews, start=1):
        _require(review.get("schema_version") == REVIEW_SCHEMA,
                 f"unexpected review schema at batch {index}")
        claimed = review.get("review_sha256")
        _require(claimed == canonical_sha256(
            _without_hash(review, "review_sha256")),
            f"review canonical hash mismatch at batch {index}")
        source = review.get("source", {})
        _require(source.get("inventory_sha256") == inventory_hash,
                 f"review inventory mismatch at batch {index}")
        if index == 1:
            _require("previous_review_sha256" not in source,
                     "first review unexpectedly has a predecessor")
        else:
            _require(source.get("previous_review_sha256") == previous_hash,
                     f"review hash chain mismatch at batch {index}")
        rows = review.get("rows")
        positions = review.get("positions")
        _require(isinstance(rows, list) and rows,
                 f"review rows missing at batch {index}")
        _require(positions == [row.get("position") for row in rows],
                 f"review positions differ from rows at batch {index}")
        _require(len(positions) == len(set(positions)),
                 f"duplicate position inside batch {index}")
        for row in rows:
            position = row.get("position")
            frozen = inventory_by_position.get(position)
            _require(frozen is not None,
                     f"review position {position} absent from inventory")
            _require(frozen.get("admission_status") ==
                     "pending_independent_oracle_review",
                     f"review position {position} was pre-rejected")
            for key in ("item_id", "family_id", "prompt_sha256",
                        "recovered_output_sha256"):
                _require(row.get(key) == frozen.get(key),
                         f"review identity mismatch for {key} at {position}")
            if position in latest:
                _require(latest[position].get("review_disposition") ==
                         "deferred_in_frozen_order",
                         f"position {position} was reviewed twice without deferral")
            disposition = row.get("review_disposition")
            _require(isinstance(disposition, str) and disposition,
                     f"review disposition missing at position {position}")
            _require(row.get("correction_admitted") is False,
                     "current v1 review chain must not admit corrections")
            latest[position] = row
        chain.append({
            "batch_index": index,
            "review_sha256": claimed,
            "positions": positions,
        })
        previous_hash = claimed
    return latest, chain


def wilson_interval(successes: int, total: int,
                    z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive total")
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _gate_pending(row: dict, gate: str) -> bool:
    value = row.get(gate)
    if gate == "history_review" and isinstance(value, dict):
        return value.get("full_semantic_history_status") != "passed"
    if gate == "near_duplicate_review" and isinstance(value, dict):
        return value.get("manual_cross_history_status") != "passed"
    if isinstance(value, str):
        return value not in {"passed", "approved"}
    return True


def build_tracker(inventory: dict, reviews: list[dict], *,
                  source_files: dict[str, str] | None = None) -> dict:
    inventory_by_position = verify_inventory(inventory)
    latest, chain = verify_reviews(inventory, inventory_by_position, reviews)
    dispositions = {
        position: row["review_disposition"] for position, row in latest.items()
    }
    oracle_ready_positions = sorted(
        position for position, disposition in dispositions.items()
        if disposition == "oracle_ready_remaining_gates_pending"
    )
    rejected_positions = sorted(
        position for position, disposition in dispositions.items()
        if disposition.startswith("rejected_")
    )
    deferred_positions = sorted(
        position for position, disposition in dispositions.items()
        if disposition == "deferred_in_frozen_order"
    )
    admitted_positions = sorted(
        position for position, row in latest.items()
        if row.get("correction_admitted") is True
    )
    reviewed_positions = sorted(latest)
    resolved = len(oracle_ready_positions) + len(rejected_positions)
    _require(resolved + len(deferred_positions) == len(reviewed_positions),
             "unknown latest review disposition")
    point_rate = len(oracle_ready_positions) / resolved
    lower, upper = wilson_interval(len(oracle_ready_positions), resolved)

    def reviews_needed(rate: float) -> int | None:
        return math.ceil(TARGET_CORRECTIONS / rate) if rate > 0 else None

    queue_size = 168
    next_position = reviews[-1].get("next_position")
    if next_position is None:
        next_position = reviews[-1].get("next_position_blocked")
    report = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "model_free_cpu_admission_progress_tracking",
        "source": {
            "inventory_sha256": inventory["inventory_sha256"],
            "review_chain_tip_sha256": reviews[-1]["review_sha256"],
            "review_batch_count": len(reviews),
            "source_file_sha256": dict(sorted((source_files or {}).items())),
        },
        "funnel": {
            "a0_dataset_side_failures": 244,
            "rejected_before_oracle_review": 76,
            "independent_oracle_review_queue": queue_size,
            "unique_positions_reviewed": len(reviewed_positions),
            "contract_oracle_resolved": resolved,
            "oracle_ready_remaining_gates": len(oracle_ready_positions),
            "contract_rejected": len(rejected_positions),
            "deferred_latest": len(deferred_positions),
            "unreviewed_queue": queue_size - len(reviewed_positions),
            "sandbox_pending": sum(
                _gate_pending(latest[position], "sandbox_review")
                for position in oracle_ready_positions),
            "full_semantic_history_pending": sum(
                _gate_pending(latest[position], "history_review")
                for position in oracle_ready_positions),
            "manual_cross_history_pending": sum(
                _gate_pending(latest[position], "near_duplicate_review")
                for position in oracle_ready_positions),
            "human_approval_pending": sum(
                _gate_pending(latest[position], "human_approval")
                for position in oracle_ready_positions),
            "correction_admitted": len(admitted_positions),
        },
        "positions": {
            "reviewed": reviewed_positions,
            "oracle_ready": oracle_ready_positions,
            "contract_rejected": rejected_positions,
            "deferred_latest": deferred_positions,
            "admitted": admitted_positions,
            "next_frozen_position": next_position,
        },
        "review_chain": chain,
        "capacity_projection": {
            "basis": "oracle_ready_among_contract_oracle_resolved",
            "observed_ready": len(oracle_ready_positions),
            "observed_resolved": resolved,
            "point_rate": point_rate,
            "wilson_95_lower": lower,
            "wilson_95_upper": upper,
            "estimated_total_reviews_for_64_at_point_rate":
                reviews_needed(point_rate),
            "estimated_total_reviews_for_64_at_wilson_lower":
                reviews_needed(lower),
            "estimated_total_reviews_for_64_at_wilson_upper":
                reviews_needed(upper),
            "projected_oracle_ready_if_all_168_reviewed_at_point_rate":
                round(queue_size * point_rate),
            "small_sample_descriptive_only": True,
        },
        "admission_projection": {
            "status": "unavailable_no_completed_admission_decisions",
            "gate_complete_decisions": 0,
            "admission_rate": None,
        },
        "milestones": [{
            "reviewed_target": value,
            "remaining_from_current": max(0, value - len(reviewed_positions)),
            "reached": len(reviewed_positions) >= value,
        } for value in MILESTONES],
        "claims": {
            "model_generation": False,
            "untrusted_model_code_executed": False,
            "independent_oracle_admission_complete": False,
            "correction_pool_certified": False,
            "gpu_training_authorized": False,
            "protected_data_opening_authorized": False,
        },
    }
    report["tracker_sha256"] = canonical_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    reviews = [json.loads(path.read_text(encoding="utf-8"))
               for path in args.reviews]
    source_files = {args.inventory.name: sha256_file(args.inventory)}
    source_files.update({path.name: sha256_file(path) for path in args.reviews})
    report = build_tracker(inventory, reviews, source_files=source_files)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps({
        "tracker_sha256": report["tracker_sha256"],
        "funnel": report["funnel"],
        "capacity_projection": report["capacity_projection"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
