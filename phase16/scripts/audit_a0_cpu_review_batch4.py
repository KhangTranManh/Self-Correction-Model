"""Continue P16-R0 CPU review at frozen positions 23, 28 and 29."""

from __future__ import annotations

import argparse
import copy
from decimal import Decimal, ROUND_HALF_EVEN
from fractions import Fraction
import json
from pathlib import Path

from audit_a0_cpu_review_batch1 import (
    _near_duplicate_evidence,
    _require,
    _verify_bank,
    _verify_inventory,
    canonical_sha256,
)
from audit_a0_cpu_review_batch2 import _evidence


SCHEMA_VERSION = "p16_a0_cpu_review_batch_v1"
EXPECTED_BATCH3_REVIEW_SHA256 = (
    "fac1983720608dea56277157ea806b3240d84948ada264093805757910a11c73"
)
FROZEN_POSITIONS = [23, 28, 29]


def calculate_scrap_oracle(scraps: list[int], robots: int) -> int:
    numerator = 50 * robots * (100 ** len(scraps))
    denominator = 1
    for scrap in scraps:
        denominator *= 100 - scrap
    return (numerator + denominator - 1) // denominator


def calculate_scrap_reference(scraps: list[int], robots: int) -> int:
    target = 50 * robots * (100 ** len(scraps))
    retained = 1
    for scrap in scraps:
        retained *= 100 - scrap
    low, high = 0, 50 * robots
    while high * retained < target:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if middle * retained >= target:
            high = middle
        else:
            low = middle + 1
    return low


def calculate_scrap_mutant(scraps: list[int], robots: int) -> int:
    remaining = 50 * robots
    for scrap in scraps:
        remaining /= 1 - scrap / 100
    return int(remaining)


SCRAP_CASES = [
    ("no_effect", ([0], 1)),
    ("single_fractional", ([10], 1)),
    ("single_exact", ([50], 1)),
    ("contract_example", ([10, 10, 10], 90)),
    ("small_loss", ([1], 1)),
    ("mixed_losses", ([5, 20, 33], 7)),
    ("large_loss", ([99], 2)),
]


def race_oracle(first_speed: int, second_speed: int,
                lead: int) -> list[int] | None:
    if first_speed >= second_speed:
        return None
    total_seconds = lead * 3600 // (second_speed - first_speed)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return [hours, minutes, seconds]


def race_reference(first_speed: int, second_speed: int,
                   lead: int) -> list[int] | None:
    if not second_speed > first_speed:
        return None
    duration = Fraction(lead, second_speed - first_speed)
    whole_seconds = duration.numerator * 3600 // duration.denominator
    hours = whole_seconds // 3600
    minutes = (whole_seconds - hours * 3600) // 60
    seconds = whole_seconds - hours * 3600 - minutes * 60
    return [hours, minutes, seconds]


def race_mutant(first_speed: int, second_speed: int,
                lead: int) -> list[int] | None:
    if first_speed >= second_speed:
        return None
    time_hours = lead / (second_speed - first_speed)
    time_minutes = int(time_hours * 60)
    time_seconds = int((time_hours - int(time_hours)) * 60)
    return [int(time_hours), time_minutes, time_seconds]


RACE_CASES = [
    ("contract_first", (720, 850, 70)),
    ("contract_second", (80, 91, 37)),
    ("equal_speed", (10, 10, 5)),
    ("slower_second", (12, 10, 5)),
    ("exact_hour", (10, 15, 5)),
    ("seconds_only", (10, 3610, 1)),
    ("multi_hour", (1, 2, 5)),
]


def _round_half_even_ratio(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder * 2
    if doubled < denominator:
        return quotient
    if doubled > denominator:
        return quotient + 1
    return quotient if quotient % 2 == 0 else quotient + 1


def day_plan_oracle(hours: int, tasks: int,
                    duration: int) -> list[int] | str:
    available = hours * 60
    work = tasks * duration
    if work > available:
        return "You're not sleeping tonight!"
    if tasks == 1:
        return [duration]
    break_duration = _round_half_even_ratio(available - work, tasks - 1)
    schedule = []
    for index in range(tasks):
        schedule.append(duration)
        if index + 1 < tasks:
            schedule.append(break_duration)
    return schedule


def day_plan_reference(hours: int, tasks: int,
                       duration: int) -> list[int] | str:
    remaining = hours * 60 - tasks * duration
    if remaining < 0:
        return "You're not sleeping tonight!"
    if tasks == 1:
        return [duration]
    per_break = int((Decimal(remaining) / Decimal(tasks - 1)).quantize(
        Decimal("1"), rounding=ROUND_HALF_EVEN))
    return [duration if index % 2 == 0 else per_break
            for index in range(tasks * 2 - 1)]


def day_plan_mutant(hours: int, tasks: int,
                    duration: int) -> list[int] | str:
    if tasks * duration > hours * 60:
        return "You're not sleeping tonight!"
    schedule = []
    for index in range(tasks):
        schedule.append(duration)
        if index + 1 < tasks:
            schedule.append(10)
    return schedule


DAY_PLAN_CASES = [
    ("contract_schedule", (8, 5, 30)),
    ("contract_impossible", (3, 5, 60)),
    ("integer_break", (1, 3, 10)),
    ("fractional_break", (2, 4, 20)),
    ("single_task", (1, 1, 30)),
    ("single_task_impossible", (1, 1, 61)),
    ("one_break", (1, 2, 15)),
]


def _verify_batch3(batch3: dict) -> None:
    _require(batch3.get("schema_version") == SCHEMA_VERSION,
             "unexpected batch-3 schema")
    claimed = batch3.get("review_sha256")
    without_hash = copy.deepcopy(batch3)
    without_hash.pop("review_sha256", None)
    _require(claimed == canonical_sha256(without_hash),
             "batch-3 canonical hash mismatch")
    _require(claimed == EXPECTED_BATCH3_REVIEW_SHA256,
             "batch 4 points to a different batch 3")
    _require(batch3.get("next_position") == 23,
             "batch 3 did not stop before position 23")


def build(inventory: dict, bank: dict, batch3: dict) -> dict:
    _verify_batch3(batch3)
    by_position = _verify_inventory(inventory)
    by_id = _verify_bank(bank)
    selected = [by_position[position] for position in FROZEN_POSITIONS]
    for row in selected:
        task = by_id.get(row["item_id"])
        _require(task is not None, f"bank task missing for {row['item_id']}")
        for key in ("family_id", "function_name", "prompt_sha256"):
            _require(task.get(key) == row.get(key),
                     f"bank/inventory {key} mismatch for {row['item_id']}")

    definitions = [
        (selected[0], SCRAP_CASES, calculate_scrap_oracle,
         calculate_scrap_reference, calculate_scrap_mutant,
         "sequential percentage loss and whole-kilogram purchase are explicit"),
        (selected[1], RACE_CASES, race_oracle,
         race_reference, race_mutant,
         "relative speed, floor-to-second and output components are explicit"),
        (selected[2], DAY_PLAN_CASES, day_plan_oracle,
         day_plan_reference, day_plan_mutant,
         "remaining minutes are evenly allocated across intervening breaks"),
    ]
    rows = []
    for record, cases, oracle, reference, mutant, reason in definitions:
        task = by_id[record["item_id"]]
        rows.append({
            "position": record["position"],
            "item_id": record["item_id"],
            "family_id": record["family_id"],
            "source_record_sha256": record["record_sha256"],
            "prompt_sha256": record["prompt_sha256"],
            "recovered_output_sha256": record["recovered_output_sha256"],
            "contract_review": {"status": "passed", "reason": reason},
            "independent_oracle": _evidence(
                cases, oracle, reference, mutant),
            "history_review": {
                "frozen_phase_1_15_exact_status": "passed",
                "full_semantic_history_status": "pending",
            },
            "near_duplicate_review": _near_duplicate_evidence(bank, task),
            "sandbox_review": "pending_no_local_untrusted_code_sandbox",
            "human_approval": "pending",
            "review_disposition": "oracle_ready_remaining_gates_pending",
            "correction_admitted": False,
        })

    document = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "inventory_sha256": inventory["inventory_sha256"],
            "bank_sha256": bank["bank_sha256"],
            "previous_review_sha256": batch3["review_sha256"],
        },
        "frozen_review_rule": "review pending positions 23, 28 and 29",
        "positions": FROZEN_POSITIONS,
        "counts": {
            "newly_examined": 3,
            "contract_passed": 3,
            "trusted_oracle_validated": 3,
            "correction_admitted": 0,
            "cumulative_positions_examined": 11,
            "cumulative_contract_rejected": 2,
            "cumulative_oracle_ready": 9,
        },
        "rows": rows,
        "claims": {
            "untrusted_model_code_executed": False,
            "independent_oracle_admission_complete": False,
            "correction_pool_certified": False,
            "gpu_training_authorized": False,
            "protected_data_opening_authorized": False,
        },
        "next_position": 30,
    }
    document["review_sha256"] = canonical_sha256(document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--previous-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    document = build(
        json.loads(args.inventory.read_text(encoding="utf-8")),
        json.loads(args.bank.read_text(encoding="utf-8")),
        json.loads(args.previous_review.read_text(encoding="utf-8")),
    )
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False)
                           + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(document["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
