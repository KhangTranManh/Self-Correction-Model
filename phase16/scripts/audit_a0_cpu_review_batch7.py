"""Build P16-R0 A0 CPU contract/oracle review batch 7.

Trusted cases come from task contracts, never published tests or reference
solutions. Recovered model code is not executed. No correction is admitted.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from audit_a0_cpu_review_batch6 import (
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

FROZEN_POSITIONS = [43, 44, 47, 48, 51]
EXPECTED_FUNCTIONS = [
    "base_finder", "search_substr", "archers_ready", "combs", "wheres_wally",
]


def base_finder_oracle(seq: list[str]) -> int:
    symbols: set[str] = set()
    for number in seq:
        for symbol in number:
            symbols.add(symbol)
    return len(symbols)


def base_finder_reference(seq: list[str]) -> int:
    highest_digit = max(int(character) for number in seq for character in number)
    return highest_digit + 1


def base_finder_mutant(seq: list[str]) -> int:
    integers = [int(number) for number in seq]
    return max(integers) - min(integers) + 1


BASE_FINDER_CASES = [
    ("decimal_one_to_ten", ([str(i) for i in range(1, 11)],)),
    ("seven_one_to_ten", (["1", "2", "3", "4", "5", "6", "10", "11", "12", "13"],)),
    ("binary_zero_to_nine", ([format(i, "b") for i in range(10)],)),
    ("ternary_zero_to_nine", (["0", "1", "2", "10", "11", "12", "20", "21", "22", "100"],)),
    ("octal_zero_to_nine", (["0", "1", "2", "3", "4", "5", "6", "7", "10", "11"],)),
]


def search_substr_oracle(full_text: str, search_text: str,
                         allow_overlap: bool = True) -> int:
    if not search_text:
        return 0
    count = 0
    position = 0
    while position <= len(full_text) - len(search_text):
        if full_text[position:position + len(search_text)] == search_text:
            count += 1
            position += 1 if allow_overlap else len(search_text)
        else:
            position += 1
    return count


def search_substr_reference(full_text: str, search_text: str,
                            allow_overlap: bool = True) -> int:
    if search_text == "":
        return 0
    count = 0
    start = 0
    while (found := full_text.find(search_text, start)) != -1:
        count += 1
        start = found + (1 if allow_overlap else len(search_text))
    return count


def search_substr_mutant(full_text: str, search_text: str,
                         allow_overlap: bool = True) -> int:
    if not search_text:
        return 0
    count = start = 0
    while True:
        start = full_text.find(search_text, start)
        if start == -1:
            break
        count += 1
        start += 1 if not allow_overlap else len(search_text)
    return count


SEARCH_SUBSTR_CASES = [
    ("separate", ("aa_bb_cc_dd_bb_e", "bb")),
    ("single", ("aaabbbcccc", "bbb")),
    ("overlap_default", ("aaa", "aa")),
    ("empty_search", ("aaa", "")),
    ("no_overlap", ("aaa", "aa", False)),
    ("repeated_overlap", ("aaaa", "aa", True)),
    ("repeated_no_overlap", ("aaaa", "aa", False)),
]


def archers_ready_oracle(archers: list[int]) -> bool:
    return bool(archers) and all(arrows >= 5 for arrows in archers)


def archers_ready_reference(archers: list[int]) -> bool:
    if len(archers) == 0:
        return False
    for arrows in archers:
        if arrows < 5:
            return False
    return True


def archers_ready_mutant(archers: list[int]) -> bool:
    return all(arrows >= 5 for arrows in archers)


ARCHERS_READY_CASES = [
    ("mixed", ([1, 2, 3, 4, 5, 6, 7, 8],)),
    ("enough", ([5, 6, 7, 8],)),
    ("no_archers", ([],)),
    ("one_below", ([4],)),
    ("one_exactly_five", ([5],)),
    ("all_exactly_five", ([5, 5, 5],)),
]


def combs_oracle(comb1: str, comb2: str) -> int:
    best = len(comb1) + len(comb2)
    for offset in range(-len(comb2) + 1, len(comb1)):
        if any(0 <= offset + j < len(comb1)
               and comb1[offset + j] == "*" and character == "*"
               for j, character in enumerate(comb2)):
            continue
        best = min(best, max(len(comb1), offset + len(comb2)) - min(0, offset))
    return best


def combs_reference(comb1: str, comb2: str) -> int:
    teeth1 = {index for index, character in enumerate(comb1) if character == "*"}
    teeth2 = {index for index, character in enumerate(comb2) if character == "*"}
    for purse_length in range(max(len(comb1), len(comb2)),
                              len(comb1) + len(comb2) + 1):
        for start1 in range(purse_length - len(comb1) + 1):
            occupied1 = {start1 + index for index in teeth1}
            for start2 in range(purse_length - len(comb2) + 1):
                if occupied1.isdisjoint({start2 + index for index in teeth2}):
                    return purse_length
    raise AssertionError("side-by-side combs must fit")  # pragma: no cover


def combs_mutant(comb1: str, comb2: str) -> int:
    max_gap = 0
    i = j = 0
    while i < len(comb1) and j < len(comb2):
        if comb1[i] == "*" and comb2[j] == "*":
            max_gap = max(max_gap, i + 1, j + 1)
            i += 1
            j += 1
        elif comb1[i] == ".":
            i += 1
        elif comb2[j] == ".":
            j += 1
    return max_gap + min(len(comb1), len(comb2))


COMBS_CASES = [
    ("contract_example", ("*..*", "*.*")),
    ("one_tooth_each", ("*", "*")),
    ("both_full", ("***", "***")),
    ("interleave", ("*.*", "*.*.*")),
    ("long_gap", ("*...*", "*")),
    ("unequal_lengths", ("*..*", "*...*")),
]


def wheres_wally_oracle(value: str) -> int:
    match = re.search(r"(?:^| )Wally(?=[^a-zA-Z0-9]|$)", value)
    if match is None:
        return -1
    return match.start() + int(value[match.start()] == " ")


def wheres_wally_reference(value: str) -> int:
    start = 0
    while (index := value.find("Wally", start)) != -1:
        end = index + 5
        if (index == 0 or value[index - 1] == " ") and (
            end == len(value) or not value[end].isalnum()
        ):
            return index
        start = index + 1
    return -1


def wheres_wally_mutant(string: str) -> int:
    if (" Wally" in string or "Wally." in string or "Wally," in string
            or "Wally;" in string or "Wally:" in string):
        return string.find("Wally")
    return -1


WHERES_WALLY_CASES = [
    ("just_wally", ("Wally",)),
    ("space_before", ("Where's Wally",)),
    ("no_match", ("Where's Waldo",)),
    ("attached_both_sides", ("DWally Wallyd .Wally",)),
    ("period_after", ("Hi Wally.",)),
    ("apostrophe_after", ("It's Wally's.",)),
    ("first_of_two", ("Wally Wally",)),
    ("apostrophe_before", ("'Wally Wally",)),
    ("mixed_case_not_match", ("wally",)),
]


DEFINITIONS = [
    (BASE_FINDER_CASES, base_finder_oracle, base_finder_reference,
     base_finder_mutant, "base is the number of unique digits across ten consecutive base-encoded numbers"),
    (SEARCH_SUBSTR_CASES, search_substr_oracle, search_substr_reference,
     search_substr_mutant, "overlap default, non-overlap option and empty-search behavior are explicit"),
    (ARCHERS_READY_CASES, archers_ready_oracle, archers_ready_reference,
     archers_ready_mutant, "a nonempty group and at least five arrows per archer are required"),
    (COMBS_CASES, combs_oracle, combs_reference,
     combs_mutant, "teeth may interleave without collision; no comb reversal and minimum purse length are explicit"),
    (WHERES_WALLY_CASES, wheres_wally_oracle, wheres_wally_reference,
     wheres_wally_mutant, "the examples constrain leading space/start, following punctuation and embedded words"),
]


def build(inventory: dict, bank: dict, batch6: dict) -> dict:
    _verify_hashed(inventory, "inventory_sha256",
                   "p16_a0_admission_inventory_v1", "inventory")
    _require(inventory["inventory_sha256"] == EXPECTED_INVENTORY_SHA256,
             "batch 7 points to a different frozen inventory")
    _verify_hashed(bank, "bank_sha256", "p16_a0_candidate_bank_v1", "bank")
    _require(bank["bank_sha256"] == EXPECTED_BANK_SHA256,
             "batch 7 points to a different frozen bank")
    _verify_hashed(batch6, "review_sha256", SCHEMA_VERSION, "batch 6")
    _require(batch6.get("next_position") == 43,
             "batch 6 does not hand off at position 43")
    by_position = {row["position"]: row for row in inventory["records"]}
    by_id = {task["item_id"]: task for task in bank["tasks"]}
    pending = [row["position"] for row in inventory["records"]
               if row["position"] >= batch6["next_position"]
               and row["admission_status"] == "pending_independent_oracle_review"]
    _require(pending[:5] == FROZEN_POSITIONS,
             "batch 7 positions are not the next five pending records")
    rows = []
    for position, function_name, definition in zip(
            FROZEN_POSITIONS, EXPECTED_FUNCTIONS, DEFINITIONS):
        record = by_position[position]
        task = by_id.get(record["item_id"])
        _require(task is not None, f"bank task missing for {record['item_id']}")
        for key in ("family_id", "prompt_sha256"):
            _require(task.get(key) == record.get(key),
                     f"bank/inventory {key} mismatch for {record['item_id']}")
        _require(task.get("function_name") == function_name,
                 f"unexpected function at position {position}")
        _require(record["source_history_exact_clean"] is True,
                 f"exact history not clean at position {position}")
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
            "previous_review_sha256": batch6["review_sha256"],
        },
        "frozen_review_rule": "review next five pending positions from 43",
        "positions": FROZEN_POSITIONS,
        "counts": {
            "newly_examined": 5,
            "contract_passed": 5,
            "contract_rejected": 0,
            "trusted_oracle_validated": 5,
            "correction_admitted": 0,
            "cumulative_positions_examined": 24,
            "cumulative_contract_rejected": 4,
            "cumulative_oracle_ready": 20,
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
