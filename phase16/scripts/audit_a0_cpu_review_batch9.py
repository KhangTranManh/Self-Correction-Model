"""Build P16-R0 A0 CPU contract/oracle review batch 9.

Trusted cases come only from task contracts. Recovered model code is inspected
as text to define faithful mutants but is never executed by this review step.
No correction is admitted and no GPU work is authorized.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import string

from audit_a0_cpu_review_batch8 import (
    EXPECTED_BANK_SHA256,
    EXPECTED_INVENTORY_SHA256,
    SCHEMA_VERSION,
    _base_row,
    _evidence,
    _near_duplicate_evidence,
    _require,
    _verify_hashed,
    canonical_sha256,
)


FROZEN_POSITIONS = [62, 64, 66, 68, 70]
EXPECTED_FUNCTIONS = [
    "rank_of_element", "pick_peaks", "blocks", "calc_ms", "find_pattern",
]


def rank_of_element_oracle(arr: list[int], i: int) -> int:
    target = arr[i]
    return (sum(value <= target for value in arr[:i])
            + sum(value < target for value in arr[i + 1:]))


def rank_of_element_reference(arr: list[int], i: int) -> int:
    target = arr[i]
    strictly_smaller = sum(value < target for index, value in enumerate(arr)
                           if index != i)
    equal_before = sum(value == target for value in arr[:i])
    return strictly_smaller + equal_before


def rank_of_element_mutant(arr: list[int], i: int) -> int:
    n = len(arr)
    if i == 0:
        return sum(1 for j in range(i + 1, n) if arr[j] <= arr[i])
    if i == n - 1:
        return sum(1 for j in range(n - 1) if arr[j] <= arr[i])
    left_count = sum(1 for j in range(i) if arr[j] <= arr[i])
    right_count = sum(1 for j in range(i + 1, n) if arr[j] < arr[i])
    return left_count + right_count


RANK_CASES = [
    ("contract_example", ([2, 1, 2, 1, 2], 2)),
    ("first_with_equal_after", ([2, 1, 2], 0)),
    ("last_with_equal_before", ([2, 1, 2], 2)),
    ("all_equal_middle", ([5, 5, 5, 5], 2)),
    ("negative_values", ([-2, -5, -2, 0], 2)),
    ("unique_middle", ([9, 1, 6, 3, 8], 2)),
]


def pick_peaks_oracle(arr: list[int]) -> dict[str, list[int]]:
    positions: list[int] = []
    peaks: list[int] = []
    rising_at = None
    for index in range(1, len(arr)):
        if arr[index] > arr[index - 1]:
            rising_at = index
        elif arr[index] < arr[index - 1] and rising_at is not None:
            positions.append(rising_at)
            peaks.append(arr[rising_at])
            rising_at = None
    return {"pos": positions, "peaks": peaks}


def pick_peaks_reference(arr: list[int]) -> dict[str, list[int]]:
    positions: list[int] = []
    peaks: list[int] = []
    index = 1
    while index < len(arr) - 1:
        if arr[index] <= arr[index - 1]:
            index += 1
            continue
        plateau_end = index
        while (plateau_end + 1 < len(arr)
               and arr[plateau_end + 1] == arr[index]):
            plateau_end += 1
        if plateau_end + 1 < len(arr) and arr[plateau_end + 1] < arr[index]:
            positions.append(index)
            peaks.append(arr[index])
        index = plateau_end + 1
    return {"pos": positions, "peaks": peaks}


def pick_peaks_mutant(arr: list[int]) -> dict[str, list[int]]:
    positions: list[int] = []
    peaks: list[int] = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] >= arr[i + 1]:
            positions.append(i)
            peaks.append(arr[i])
        elif arr[i] == arr[i - 1] and arr[i] > arr[i + 1]:
            j = i + 1
            while j < len(arr) and arr[j] == arr[i]:
                j += 1
            if j < len(arr) and arr[j] < arr[i]:
                positions.append(i)
                peaks.append(arr[i])
    return {"pos": positions, "peaks": peaks}


PICK_PEAKS_CASES = [
    ("single_peak", ([0, 1, 2, 5, 1, 0],)),
    ("contract_multiple", ([3, 2, 3, 6, 4, 1, 2, 3, 2, 1, 2, 3],)),
    ("plateau_peak", ([1, 2, 2, 2, 1],)),
    ("plateau_rises", ([1, 2, 2, 2, 3],)),
    ("end_not_peak", ([1, 2, 3],)),
    ("start_not_peak", ([3, 2, 1],)),
    ("empty", ([],)),
    ("two_plateaus", ([0, 2, 2, 1, 3, 3, 0],)),
]


BLOCK_ORDER = string.ascii_lowercase + string.ascii_uppercase + string.digits


def blocks_oracle(value: str) -> str:
    counts = Counter(value)
    layers = []
    for layer in range(max(counts.values(), default=0)):
        layers.append("".join(character for character in BLOCK_ORDER
                              if counts[character] > layer))
    return "-".join(layers)


def blocks_reference(value: str) -> str:
    remaining = Counter(value)
    layers = []
    while remaining:
        layer = []
        for character in BLOCK_ORDER:
            if remaining.get(character, 0):
                layer.append(character)
                remaining[character] -= 1
                if remaining[character] == 0:
                    del remaining[character]
        layers.append("".join(layer))
    return "-".join(layers)


def blocks_mutant(value: str) -> str:
    groups = {0: [], 1: [], 2: []}
    for character in value:
        if character.islower():
            groups[0].append(character)
        elif character.isupper():
            groups[1].append(character)
        else:
            groups[2].append(character)
    return "".join(character for group in range(3)
                   for character in sorted(groups[group]))


BLOCK_CASES = [
    ("unique", ("21AxBz",)),
    ("repeated_lower", ("abacad",)),
    ("empty", ("",)),
    ("repeated_all_groups", ("Aa1Aa1",)),
    ("hierarchy", ("9zZ0aA",)),
    ("contract_complex", ("hbh420sUUW222IWOxndjn93cdop69NICEep832",)),
]


def calc_ms_oracle(n: int) -> int:
    return 20 ** n


def calc_ms_reference(n: int) -> int:
    cubes = 1
    for _ in range(n):
        cubes *= 20
    return cubes


def calc_ms_mutant(n: int) -> int:
    if n == 0:
        return 1
    return 20 * calc_ms_mutant(n - 1) ** 3 // 27


CALC_MS_CASES = [
    ("level_zero", (0,)),
    ("level_one", (1,)),
    ("level_two", (2,)),
    ("level_three", (3,)),
    ("level_six", (6,)),
]


DEFINITIONS = [
    (RANK_CASES, rank_of_element_oracle, rank_of_element_reference,
     rank_of_element_mutant,
     "rank counts less-or-equal values before and strictly smaller values after the selected index"),
    (PICK_PEAKS_CASES, pick_peaks_oracle, pick_peaks_reference,
     pick_peaks_mutant,
     "strict and plateau peaks, plateau start positions and excluded endpoints are explicit"),
    (BLOCK_CASES, blocks_oracle, blocks_reference, blocks_mutant,
     "first-suitable-block placement and lowercase/uppercase/digit ordering are explicit"),
    (CALC_MS_CASES, calc_ms_oracle, calc_ms_reference, calc_ms_mutant,
     "each Menger iteration retains twenty copies of every prior cube"),
]


def build(inventory: dict, bank: dict, batch8: dict) -> dict:
    _verify_hashed(inventory, "inventory_sha256",
                   "p16_a0_admission_inventory_v1", "inventory")
    _require(inventory["inventory_sha256"] == EXPECTED_INVENTORY_SHA256,
             "batch 9 points to a different frozen inventory")
    _verify_hashed(bank, "bank_sha256", "p16_a0_candidate_bank_v1", "bank")
    _require(bank["bank_sha256"] == EXPECTED_BANK_SHA256,
             "batch 9 points to a different frozen bank")
    _verify_hashed(batch8, "review_sha256", SCHEMA_VERSION, "batch 8")
    _require(batch8.get("next_position") == 62,
             "batch 8 does not hand off at position 62")

    by_position = {record["position"]: record for record in inventory["records"]}
    by_id = {task["item_id"]: task for task in bank["tasks"]}
    pending = [record["position"] for record in inventory["records"]
               if record["position"] >= batch8["next_position"]
               and record["admission_status"] == "pending_independent_oracle_review"]
    _require(pending[:5] == FROZEN_POSITIONS,
             "batch 9 positions are not the next five pending records")
    selected = [by_position[position] for position in FROZEN_POSITIONS]
    for record, expected_name in zip(selected, EXPECTED_FUNCTIONS):
        task = by_id.get(record["item_id"])
        _require(task is not None, "task absent from frozen bank")
        for key in ("family_id", "prompt_sha256"):
            _require(task.get(key) == record.get(key),
                     f"bank/inventory {key} mismatch at position {record['position']}")
        _require(task["function_name"] == expected_name,
                 f"unexpected function at position {record['position']}")

    rows = []
    for record, definition in zip(selected[:4], DEFINITIONS):
        task = by_id[record["item_id"]]
        _require(record["source_history_exact_clean"] is True,
                 f"exact history not clean at position {record['position']}")
        cases, oracle, reference, mutant, reason = definition
        row = _base_row(record)
        row.update({
            "contract_review": {"status": "passed", "reason": reason},
            "independent_oracle": _evidence(cases, oracle, reference, mutant),
            "history_review": {
                "frozen_phase_1_15_exact_status": "passed",
                "full_semantic_history_status": "pending",
            },
            "near_duplicate_review": _near_duplicate_evidence(bank, task),
            "sandbox_review": "pending_cpu_sandbox_closure",
            "human_approval": "pending",
            "review_disposition": "oracle_ready_remaining_gates_pending",
            "correction_admitted": False,
        })
        rows.append(row)

    rejected = _base_row(selected[4])
    rejected.update({
        "contract_review": {
            "status": "failed",
            "reason": "the text delegates pattern semantics to unavailable test cases and does not define empty or singleton sequences",
        },
        "independent_oracle": {"status": "not_constructed_contract_failed"},
        "history_review": "not_reviewed_ineligible",
        "near_duplicate_review": "not_reviewed_ineligible",
        "sandbox_review": "not_reviewed_ineligible",
        "human_approval": "not_applicable_rejected",
        "review_disposition": "rejected_contract_under_specified",
        "rejection_reasons": ["contract_under_specified_pattern_semantics"],
        "correction_admitted": False,
    })
    rows.append(rejected)

    document = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "inventory_sha256": inventory["inventory_sha256"],
            "bank_sha256": bank["bank_sha256"],
            "previous_review_sha256": batch8["review_sha256"],
        },
        "frozen_review_rule": "review next five pending positions from 62",
        "positions": FROZEN_POSITIONS,
        "counts": {
            "newly_examined": 5,
            "contract_passed": 4,
            "contract_rejected": 1,
            "trusted_oracle_validated": 4,
            "correction_admitted": 0,
            "cumulative_positions_examined": 34,
            "cumulative_contract_rejected": 6,
            "cumulative_oracle_ready": 28,
        },
        "rows": rows,
        "claims": {
            "untrusted_model_code_executed": False,
            "independent_oracle_admission_complete": False,
            "correction_pool_certified": False,
            "gpu_training_authorized": False,
            "protected_data_opening_authorized": False,
        },
        "next_position": pending[5],
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
    args.output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps(document["counts"], indent=2, sort_keys=True))
    print(f"next_position: {document['next_position']}")


if __name__ == "__main__":
    main()
