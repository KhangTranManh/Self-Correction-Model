"""Build P16-R0 A0 CPU contract/oracle review batch 8 without model execution."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from audit_a0_cpu_review_batch7 import (
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

FROZEN_POSITIONS = [53, 56, 58, 60, 61]
EXPECTED_FUNCTIONS = ["morse_converter", "uncollapse", "string_merge",
                      "checkchoose", "custom_fib"]
WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def uncollapse_oracle(digits: str) -> str:
    words = []
    offset = 0
    while offset < len(digits):
        for word in WORDS:
            if digits.startswith(word, offset):
                words.append(word)
                offset += len(word)
                break
        else:
            raise ValueError("input must consist of collapsed English digits")
    return " ".join(words)


def uncollapse_reference(digits: str) -> str:
    matches = re.findall("|".join(WORDS), digits)
    if "".join(matches) != digits:
        raise ValueError("input must consist of collapsed English digits")
    return " ".join(matches)


def uncollapse_mutant(digits: str) -> str:
    for word, number in zip(WORDS, "0123456789"):
        digits = digits.replace(word, number)
    return digits


UNCOLLAPSE_CASES = [
    ("single", ("three",)),
    ("two", ("eightsix",)),
    ("three", ("fivefourseven",)),
    ("four", ("ninethreesixthree",)),
    ("long", ("fivethreefivesixthreenineonesevenoneeight",)),
    ("repeat", ("oneoneone",)),
]


def string_merge_oracle(string1: str, string2: str, letter: str) -> str:
    return string1[:string1.index(letter)] + string2[string2.index(letter):]


def string_merge_reference(string1: str, string2: str, letter: str) -> str:
    prefix = ""
    for character in string1:
        if character == letter:
            break
        prefix += character
    suffix = ""
    found = False
    for character in string2:
        if character == letter and not found:
            found = True
        if found:
            suffix += character
    return prefix + suffix


def string_merge_mutant(string1: str, string2: str, letter: str) -> str:
    part1 = string1.split(letter)[0] + letter
    part2 = letter + string2.split(letter)[1]
    return part1 + part2


STRING_MERGE_CASES = [
    ("hello_world", ("hello", "world", "l")),
    ("coding_anywhere", ("coding", "anywhere", "n")),
    ("jason_samson", ("jason", "samson", "s")),
    ("wonderful_people", ("wonderful", "people", "e")),
    ("initial_letter", ("abc", "adef", "a")),
]


def checkchoose_oracle(m: int, n: int) -> int:
    for count in range(n + 1):
        if math.comb(n, count) == m:
            return count
    return -1


def checkchoose_reference(m: int, n: int) -> int:
    for count in range(n + 1):
        value = math.factorial(n) // (math.factorial(count) * math.factorial(n - count))
        if value == m:
            return count
    return -1


def checkchoose_mutant(m: int, n: int) -> int:
    if m == 0:
        return 0
    for count in range(1, n + 1):
        if math.comb(n, count) == m:
            return count
    return -1


CHECKCHOOSE_CASES = [
    ("six_four", (6, 4)),
    ("four_four", (4, 4)),
    ("four_two", (4, 2)),
    ("thirtyfive_seven", (35, 7)),
    ("thirtysix_seven", (36, 7)),
    ("one_five", (1, 5)),
    ("zero_five", (0, 5)),
    ("one_one", (1, 1)),
]


def custom_fib_oracle(signature: list[int], indexes: list[int], n: int) -> int:
    sequence = list(signature)
    length = len(signature)
    while len(sequence) <= n:
        window = sequence[-length:]
        sequence.append(sum(window[index] for index in indexes))
    return sequence[n]


def custom_fib_reference(signature: list[int], indexes: list[int], n: int) -> int:
    sequence = list(signature)
    length = len(signature)
    for step in range(length, n + 1):
        total = 0
        for index in indexes:
            total += sequence[step - length + index]
        sequence.append(total)
    return sequence[n]


def custom_fib_mutant(signature: list[int], indexes: list[int], n: int) -> int:
    if n < len(signature):
        return signature[n]
    sequence = signature[:]
    for _ in range(len(signature), n + 1):
        next_value = sum(sequence[index] for index in indexes)
        sequence.append(next_value)
    return sequence[n]


CUSTOM_FIB_CASES = [
    ("fib_initial", ([1, 1], [0, 1], 0)),
    ("fib_second_next", ([1, 1], [0, 1], 2)),
    ("fib_third_next", ([1, 1], [0, 1], 3)),
    ("fib_fourth_next", ([1, 1], [0, 1], 4)),
    ("tribonacci", ([3, 5, 2], [0, 1, 2], 4)),
    ("repeated_index", ([7, 3, 4, 1], [1, 1], 6)),
]


DEFINITIONS = [
    (UNCOLLAPSE_CASES, uncollapse_oracle, uncollapse_reference,
     uncollapse_mutant, "collapsed English digit words are returned as space-separated words"),
    (STRING_MERGE_CASES, string_merge_oracle, string_merge_reference,
     string_merge_mutant, "both first letter occurrences and the single shared dividing letter are specified"),
    (CHECKCHOOSE_CASES, checkchoose_oracle, checkchoose_reference,
     checkchoose_mutant, "binomial equation, smallest nonnegative x and no-solution sentinel are specified"),
    (CUSTOM_FIB_CASES, custom_fib_oracle, custom_fib_reference,
     custom_fib_mutant, "indexes refer to the sliding window of the last signature-length elements"),
]


def build(inventory: dict, bank: dict, batch7: dict) -> dict:
    _verify_hashed(inventory, "inventory_sha256", "p16_a0_admission_inventory_v1", "inventory")
    _require(inventory["inventory_sha256"] == EXPECTED_INVENTORY_SHA256,
             "batch 8 points to a different frozen inventory")
    _verify_hashed(bank, "bank_sha256", "p16_a0_candidate_bank_v1", "bank")
    _require(bank["bank_sha256"] == EXPECTED_BANK_SHA256,
             "batch 8 points to a different frozen bank")
    _verify_hashed(batch7, "review_sha256", SCHEMA_VERSION, "batch 7")
    _require(batch7.get("next_position") == 53, "batch 7 does not hand off at position 53")
    by_position = {record["position"]: record for record in inventory["records"]}
    by_id = {task["item_id"]: task for task in bank["tasks"]}
    pending = [record["position"] for record in inventory["records"]
               if record["position"] >= batch7["next_position"]
               and record["admission_status"] == "pending_independent_oracle_review"]
    _require(pending[:5] == FROZEN_POSITIONS,
             "batch 8 positions are not the next five pending records")
    selected = [by_position[position] for position in FROZEN_POSITIONS]
    for record, expected_name in zip(selected, EXPECTED_FUNCTIONS):
        task = by_id.get(record["item_id"])
        _require(task is not None, "task absent from frozen bank")
        for key in ("family_id", "prompt_sha256"):
            _require(task.get(key) == record.get(key),
                     f"bank/inventory {key} mismatch at position {record['position']}")
        _require(task["function_name"] == expected_name,
                 f"unexpected function at position {record['position']}")
    rejected = _base_row(selected[0])
    rejected.update({
        "contract_review": {
            "status": "failed",
            "reason": "the text lists digit-to-Morse mappings but does not specify multi-digit input separators or exact return type/serialization",
        },
        "independent_oracle": {"status": "not_constructed_contract_failed"},
        "history_review": "not_reviewed_ineligible",
        "near_duplicate_review": "not_reviewed_ineligible",
        "sandbox_review": "not_reviewed_ineligible",
        "human_approval": "not_applicable_rejected",
        "review_disposition": "rejected_contract_under_specified",
        "rejection_reasons": ["contract_under_specified_input_output_serialization"],
        "correction_admitted": False,
    })
    rows = [rejected]
    for record, definition in zip(selected[1:], DEFINITIONS):
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
            "sandbox_review": "pending_no_local_untrusted_code_sandbox",
            "human_approval": "pending",
            "review_disposition": "oracle_ready_remaining_gates_pending",
            "correction_admitted": False,
        })
        rows.append(row)
    document = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "inventory_sha256": inventory["inventory_sha256"],
            "bank_sha256": bank["bank_sha256"],
            "previous_review_sha256": batch7["review_sha256"],
        },
        "frozen_review_rule": "review next five pending positions from 53",
        "positions": FROZEN_POSITIONS,
        "counts": {
            "newly_examined": 5,
            "contract_passed": 4,
            "contract_rejected": 1,
            "trusted_oracle_validated": 4,
            "correction_admitted": 0,
            "cumulative_positions_examined": 29,
            "cumulative_contract_rejected": 5,
            "cumulative_oracle_ready": 24,
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
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False)
                           + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(document["counts"], indent=2, sort_keys=True))
    print(f"next_position: {document['next_position']}")


if __name__ == "__main__":
    main()
