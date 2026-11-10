"""Build P16-R0 A0 CPU contract/oracle review batch 10."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from audit_a0_cpu_review_batch9 import (
    EXPECTED_BANK_SHA256, EXPECTED_INVENTORY_SHA256, SCHEMA_VERSION,
    _base_row, _evidence, _near_duplicate_evidence, _require,
    _verify_hashed, canonical_sha256,
)


FROZEN_POSITIONS = [71, 73, 74, 76, 77]
EXPECTED_FUNCTIONS = ["remove", "bits_war", "solve", "button_sequences",
                      "numberOfSteps"]


def remove_oracle(value: str) -> str:
    return re.sub(r"!+(?= |$)", "", value)


def remove_reference(value: str) -> str:
    return " ".join(word.rstrip("!") for word in value.split(" "))


def remove_mutant(value: str) -> str:
    words = value.split()
    return " ".join(word[:-1] if word.endswith("!") else word
                    for word in words)


REMOVE_CASES = [
    ("single", ("Hi!",)), ("multiple", ("Hi!!!",)),
    ("leading", ("!Hi",)), ("both", ("!Hi!",)),
    ("sentence", ("Hi! Hi!",)),
    ("mixed", ("!!!Hi !!hi!!! !hi",)),
    ("no_marks", ("hello world",)),
]


def _signed_bits(number: int) -> int:
    return (-1 if number < 0 else 1) * abs(number).bit_count()


def bits_war_oracle(numbers: list[int]) -> str:
    odd = sum(_signed_bits(number) for number in numbers if number % 2)
    even = sum(_signed_bits(number) for number in numbers if number % 2 == 0)
    return "odds win" if odd > even else "evens win" if even > odd else "tie"


def bits_war_reference(numbers: list[int]) -> str:
    scores = {"odd": 0, "even": 0}
    for number in numbers:
        side = "odd" if abs(number) % 2 else "even"
        contribution = bin(abs(number)).count("1")
        scores[side] += contribution if number >= 0 else -contribution
    if scores["odd"] == scores["even"]:
        return "tie"
    return "odds win" if scores["odd"] > scores["even"] else "evens win"


def bits_war_mutant(numbers: list[int]) -> str:
    odd = sum(int(abs(number).bit_count() % 2 == 1) for number in numbers)
    even = sum(int(abs(number).bit_count() % 2 == 0) for number in numbers)
    return "odds win" if odd > even else "evens win" if even > odd else "tie"


BITS_CASES = [
    ("odds", ([1, 5, 12],)), ("evens", ([7, -3, 20],)),
    ("tie", ([7, -3, -2, 6],)), ("negative_odd", ([-7, 2],)),
    ("negative_even", ([3, -6],)), ("empty", ([],)),
    ("bit_weight", ([15, 2],)),
]


def number_steps_oracle(steps: int, multiple: int) -> int:
    minimum = (steps + 1) // 2
    candidate = ((minimum + multiple - 1) // multiple) * multiple
    return candidate if candidate <= steps else -1


def number_steps_reference(steps: int, multiple: int) -> int:
    for moves in range((steps + 1) // 2, steps + 1):
        if moves % multiple == 0:
            return moves
    return -1


def number_steps_mutant(steps: int, multiple: int) -> int:
    if steps < multiple:
        return -1
    best = [0] + [10 ** 9] * steps
    for height in range(1, steps + 1):
        best[height] = best[height - 1] + 1
        if height >= 2:
            best[height] = min(best[height], best[height - 2] + 1)
    for moves in best:
        if moves % multiple == 0:
            return moves
    return -1


NUMBER_STEPS_CASES = [
    ("contract_success", (10, 2)), ("contract_impossible", (3, 5)),
    ("exact_minimum", (8, 2)), ("round_up", (5, 2)),
    ("only_all_singles", (3, 3)), ("large", (10_000, 10)),
]


DEFINITIONS = {
    71: (REMOVE_CASES, remove_oracle, remove_reference, remove_mutant,
         "remove every trailing exclamation mark from each space-delimited word"),
    73: (BITS_CASES, bits_war_oracle, bits_war_reference, bits_war_mutant,
         "number parity selects the side and signed set-bit counts determine its score"),
    77: (NUMBER_STEPS_CASES, number_steps_oracle, number_steps_reference,
         number_steps_mutant,
         "valid move counts span ceil(steps/2) through steps; choose the first multiple"),
}

REJECTIONS = {
    74: ("the contract does not define a result for base factors 2 or 5 or how to resolve a missing divisibility rule",
         "contract_under_specified_domain_and_no_solution"),
    76: ("the transition rules do not define precedence when one button is released as the other is pressed in the same tick",
         "contract_under_specified_transition_precedence"),
}


def build(inventory: dict, bank: dict, batch9: dict) -> dict:
    _verify_hashed(inventory, "inventory_sha256",
                   "p16_a0_admission_inventory_v1", "inventory")
    _require(inventory["inventory_sha256"] == EXPECTED_INVENTORY_SHA256,
             "batch 10 points to a different frozen inventory")
    _verify_hashed(bank, "bank_sha256", "p16_a0_candidate_bank_v1", "bank")
    _require(bank["bank_sha256"] == EXPECTED_BANK_SHA256,
             "batch 10 points to a different frozen bank")
    _verify_hashed(batch9, "review_sha256", SCHEMA_VERSION, "batch 9")
    _require(batch9.get("next_position") == 71,
             "batch 9 does not hand off at position 71")
    by_position = {row["position"]: row for row in inventory["records"]}
    by_id = {task["item_id"]: task for task in bank["tasks"]}
    pending = [row["position"] for row in inventory["records"]
               if row["position"] >= 71
               and row["admission_status"] == "pending_independent_oracle_review"]
    _require(pending[:5] == FROZEN_POSITIONS,
             "batch 10 positions are not the next five pending records")
    rows = []
    for position, expected_name in zip(FROZEN_POSITIONS, EXPECTED_FUNCTIONS):
        record = by_position[position]
        task = by_id[record["item_id"]]
        _require(task["function_name"] == expected_name,
                 f"unexpected function at position {position}")
        for key in ("family_id", "prompt_sha256"):
            _require(task[key] == record[key],
                     f"bank/inventory {key} mismatch at position {position}")
        row = _base_row(record)
        if position in DEFINITIONS:
            cases, oracle, reference, mutant, reason = DEFINITIONS[position]
            row.update({
                "contract_review": {"status": "passed", "reason": reason},
                "independent_oracle": _evidence(cases, oracle, reference, mutant),
                "history_review": {"frozen_phase_1_15_exact_status": "passed",
                                   "full_semantic_history_status": "pending"},
                "near_duplicate_review": _near_duplicate_evidence(bank, task),
                "sandbox_review": "pending_cpu_sandbox_closure",
                "human_approval": "pending",
                "review_disposition": "oracle_ready_remaining_gates_pending",
                "correction_admitted": False,
            })
        else:
            reason, code = REJECTIONS[position]
            row.update({
                "contract_review": {"status": "failed", "reason": reason},
                "independent_oracle": {"status": "not_constructed_contract_failed"},
                "history_review": "not_reviewed_ineligible",
                "near_duplicate_review": "not_reviewed_ineligible",
                "sandbox_review": "not_reviewed_ineligible",
                "human_approval": "not_applicable_rejected",
                "review_disposition": "rejected_contract_under_specified",
                "rejection_reasons": [code], "correction_admitted": False,
            })
        rows.append(row)
    document = {
        "schema_version": SCHEMA_VERSION,
        "source": {"inventory_sha256": inventory["inventory_sha256"],
                   "bank_sha256": bank["bank_sha256"],
                   "previous_review_sha256": batch9["review_sha256"]},
        "frozen_review_rule": "review next five pending positions from 71",
        "positions": FROZEN_POSITIONS,
        "counts": {"newly_examined": 5, "contract_passed": 3,
                   "contract_rejected": 2, "trusted_oracle_validated": 3,
                   "correction_admitted": 0,
                   "cumulative_positions_examined": 39,
                   "cumulative_contract_rejected": 8,
                   "cumulative_oracle_ready": 31},
        "rows": rows,
        "claims": {"untrusted_model_code_executed": False,
                   "independent_oracle_admission_complete": False,
                   "correction_pool_certified": False,
                   "gpu_training_authorized": False,
                   "protected_data_opening_authorized": False},
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
    report = build(*[json.loads(path.read_text(encoding="utf-8")) for path in
                     (args.inventory, args.bank, args.previous_review)])
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8", newline="\n")
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    print(f"next_position: {report['next_position']}")


if __name__ == "__main__":
    main()
