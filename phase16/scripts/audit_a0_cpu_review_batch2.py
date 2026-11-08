"""Continue P16-R0 CPU review at frozen positions 6, 10 and 12.

The artifact resolves the specialized trusted oracle deferred at position 6,
then examines the next two pending records.  Oracle cases come only from the
task contracts.  Dataset expected values and reference solutions are not used
to derive any verdict, and exact recovered model code is not executed locally.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

from audit_a0_cpu_review_batch1 import (
    _near_duplicate_evidence,
    _require,
    _verify_bank,
    _verify_inventory,
    canonical_sha256,
    source_sha256,
)


SCHEMA_VERSION = "p16_a0_cpu_review_batch_v1"
EXPECTED_BATCH1_REVIEW_SHA256 = (
    "4186bc1a8fe1b339187f2301aea4222b5477de628bbec6749cf95833e04875a2"
)
FROZEN_POSITIONS = [6, 10, 12]


def _validate_tab(tab: list[str]) -> int:
    if not isinstance(tab, list) or len(tab) != 6:
        raise ValueError("tab must contain six strings")
    if not all(isinstance(line, str) for line in tab):
        raise ValueError("tab lines must be strings")
    widths = {len(line) for line in tab}
    if len(widths) != 1:
        raise ValueError("tab lines must have equal display width")
    return len(tab[0])


def transpose_grid_oracle(amount: int, tab: list[str]) -> list[str] | str:
    """Transpose a physical tab grid by reconstructing its logical columns."""

    width = _validate_tab(tab)
    groups: dict[int, dict[int, tuple[int, int]]] = {}
    for row_index, line in enumerate(tab):
        for match in re.finditer(r"\d+", line):
            groups.setdefault(match.start(), {})[row_index] = (
                int(match.group()), len(match.group()))

    output = [[] for _ in tab]
    column = 0
    while column < width:
        events = groups.get(column)
        if not events:
            for row_index, line in enumerate(tab):
                output[row_index].append(line[column])
            column += 1
            continue

        old_width = max(event[1] for event in events.values())
        transformed: dict[int, str] = {}
        for row_index, (fret, _) in events.items():
            new_fret = fret + amount
            if new_fret < 0 or new_fret > 22:
                return "Out of frets!"
            transformed[row_index] = str(new_fret)
        new_width = max(len(value) for value in transformed.values())

        for row_index, line in enumerate(tab):
            segment = line[column:column + old_width]
            if row_index in events:
                original_width = events[row_index][1]
                _require(segment[original_width:] ==
                         "-" * (old_width - original_width),
                         "non-padding content overlaps a fret cell")
                cell = transformed[row_index].ljust(new_width, "-")
            else:
                _require(segment == "-" * old_width,
                         "non-note row does not contain fret-cell padding")
                cell = "-" * new_width
            output[row_index].append(cell)
        column += old_width
    return ["".join(parts) for parts in output]


def transpose_grid_reference(amount: int, tab: list[str]) -> list[str] | str:
    """Independent right-to-left slice replacement for tab fret cells."""

    width = _validate_tab(tab)
    events_by_start: dict[int, list[tuple[int, int, int]]] = {}
    for row_index in range(6):
        cursor = 0
        while cursor < width:
            if tab[row_index][cursor].isdigit():
                end = cursor + 1
                while end < width and tab[row_index][end].isdigit():
                    end += 1
                events_by_start.setdefault(cursor, []).append(
                    (row_index, int(tab[row_index][cursor:end]), end - cursor))
                cursor = end
            else:
                cursor += 1

    replacements: list[tuple[int, int, dict[int, str]]] = []
    for start, events in events_by_start.items():
        old_width = max(event[2] for event in events)
        new_values: dict[int, str] = {}
        for row_index, fret, _ in events:
            new_fret = fret + amount
            if new_fret < 0 or new_fret > 22:
                return "Out of frets!"
            new_values[row_index] = str(new_fret)
        new_width = max(len(value) for value in new_values.values())
        replacements.append((start, old_width, {
            row_index: value.ljust(new_width, "-")
            for row_index, value in new_values.items()
        }))

    result = list(tab)
    for start, old_width, values in sorted(replacements, reverse=True):
        replacement_width = len(next(iter(values.values())))
        for row_index in range(6):
            original = tab[row_index][start:start + old_width]
            if row_index in values:
                match = re.match(r"\d+", original)
                _require(match is not None and
                         original[match.end():] == "-" * (old_width - match.end()),
                         "invalid simultaneous-note cell")
                replacement = values[row_index]
            else:
                _require(original == "-" * old_width,
                         "invalid non-note fret-cell padding")
                replacement = "-" * replacement_width
            result[row_index] = (result[row_index][:start] + replacement
                                 + result[row_index][start + old_width:])
    return result


def transpose_linewise_mutant(amount: int, tab: list[str]) -> list[str] | str:
    """Trusted equivalent of the linewise model strategy, not model execution."""

    transposed_tab = []
    for string in tab:
        transposed_string = ""
        for note in string.split("-"):
            if note.isdigit():
                fret = int(note) + amount
                if 0 <= fret <= 22:
                    transposed_string += str(fret)
                else:
                    return "Out of frets!"
            else:
                transposed_string += note
            transposed_string += "-"
        transposed_tab.append(transposed_string.rstrip("-"))
    return transposed_tab


TAB_CASES = [
    ("same_width", (2, ["e|-5-|", "B|---|", "G|---|", "D|---|", "A|---|", "E|---|"])),
    ("expand_one", (2, ["e|-8-7-|", "B|-----|", "G|-----|", "D|-----|", "A|-----|", "E|-----|"])),
    ("contract_one", (-2, ["e|-10-9-|", "B|------|", "G|------|", "D|------|", "A|------|", "E|------|"])),
    ("simultaneous_mixed_width", (1, ["e|-9--|", "B|-10-|", "G|----|", "D|----|", "A|----|", "E|----|"])),
    ("slide_marker", (2, ["e|--/8-7-|", "B|-------|", "G|-------|", "D|-------|", "A|-------|", "E|-------|"])),
    ("lower_bound", (-1, ["e|-0-|", "B|---|", "G|---|", "D|---|", "A|---|", "E|---|"])),
    ("upper_bound", (1, ["e|-22-|", "B|----|", "G|----|", "D|----|", "A|----|", "E|----|"])),
]


WEAPONS = {
    "Laval": "Shado Valious",
    "Cragger": "Vengdualize",
    "Lagravis": "Blazeprowlor",
    "Crominus": "Grandorius",
    "Tormak": "Tygafyre",
    "LiElla": "Roarburn",
}


def identify_weapon_oracle(character: str) -> str:
    weapon = WEAPONS.get(character)
    return f"{character}-{weapon}" if weapon is not None else "Not a character"


def identify_weapon_reference(character: str) -> str:
    for name, weapon in (
        ("Laval", "Shado Valious"),
        ("Cragger", "Vengdualize"),
        ("Lagravis", "Blazeprowlor"),
        ("Crominus", "Grandorius"),
        ("Tormak", "Tygafyre"),
        ("LiElla", "Roarburn"),
    ):
        if character == name:
            return name + "-" + weapon
    return "Not a character"


def identify_weapon_mutant(character: str) -> str:
    return f"{character}-{WEAPONS.get(character, 'Not a character')}"


WEAPON_CASES = [
    ("valid_laval", "Laval"),
    ("valid_cragger", "Cragger"),
    ("valid_lagravis", "Lagravis"),
    ("valid_crominus", "Crominus"),
    ("valid_tormak", "Tormak"),
    ("valid_liella", "LiElla"),
    ("invalid_unknown", "Unknown"),
    ("invalid_empty", ""),
    ("invalid_case", "laval"),
]


def _is_square(value: int) -> bool:
    root = math.isqrt(value)
    return root * root == value


def closest_pair_oracle(upper_limit: int) -> tuple[int, int] | None:
    for first in range(upper_limit - 1, 0, -1):
        for second in range(first - 1, 0, -1):
            if (_is_square(first + second)
                    and _is_square(first - second)):
                return first, second
    return None


def closest_pair_reference(upper_limit: int) -> tuple[int, int] | None:
    best = None
    maximum_root = math.isqrt(2 * max(upper_limit - 1, 0)) + 1
    for sum_root in range(1, maximum_root + 1):
        for difference_root in range(1, sum_root):
            total = sum_root * sum_root + difference_root * difference_root
            delta = sum_root * sum_root - difference_root * difference_root
            if total % 2 or delta % 2:
                continue
            candidate = total // 2, delta // 2
            if (0 < candidate[1] < candidate[0] < upper_limit
                    and (best is None or candidate > best)):
                best = candidate
    return best


def closest_pair_mutant(upper_limit: int) -> tuple[int, int] | None:
    for first in range(upper_limit - 1, 10, -1):
        for second in range(first - 1, 0, -1):
            if (_is_square(first + second)
                    and _is_square(first - second)):
                return first, second
    return None


PAIR_CASES = [
    ("minimum_pair", 6),
    ("contract_ten", 10),
    ("just_above_ten", 11),
    ("contract_thirty", 30),
    ("contract_fifty", 50),
    ("above_example", 51),
    ("round_hundred", 100),
    ("non_round", 123),
    ("upper_range", 999),
]


def _evidence(cases: list[tuple[str, Any]], oracle: Callable,
              reference: Callable, mutant: Callable) -> dict:
    rows = []
    for case_id, argument in cases:
        if isinstance(argument, tuple):
            expected = oracle(*argument)
            independent = reference(*argument)
            mutant_value = mutant(*argument)
        else:
            expected = oracle(argument)
            independent = reference(argument)
            mutant_value = mutant(argument)
        _require(expected == independent,
                 f"independent oracle disagreement on {case_id}")
        rows.append({
            "case_id": case_id,
            "input_sha256": canonical_sha256(argument),
            "expected_sha256": canonical_sha256(expected),
            "model_equivalent_mutant_matches": mutant_value == expected,
        })
    caught = [row["case_id"] for row in rows
              if not row["model_equivalent_mutant_matches"]]
    _require(caught, "trusted model-equivalent mutant was not detected")
    return {
        "status": "validated_on_trusted_cpu_cases",
        "derivation_boundary": "task_contract_only",
        "published_test_values_used": False,
        "dataset_reference_solutions_used": False,
        "test_count": len(rows),
        "oracle_reference_agree": True,
        "model_equivalent_mutant_caught": True,
        "failing_case_ids": caught,
        "oracle_source_sha256": source_sha256(oracle),
        "reference_source_sha256": source_sha256(reference),
        "mutant_source_sha256": source_sha256(mutant),
        "cases_sha256": canonical_sha256(cases),
        "transcript_sha256": canonical_sha256(rows),
    }


def _verify_batch1(batch1: dict) -> None:
    _require(batch1.get("schema_version") == SCHEMA_VERSION,
             "unexpected batch-1 schema")
    claimed = batch1.get("review_sha256")
    without_hash = copy.deepcopy(batch1)
    without_hash.pop("review_sha256", None)
    _require(claimed == canonical_sha256(without_hash),
             "batch-1 canonical hash mismatch")
    _require(claimed == EXPECTED_BATCH1_REVIEW_SHA256,
             "batch 2 points to a different batch 1")
    _require(batch1.get("next_position_blocked") == 6,
             "batch 1 did not stop at position 6")


def build(inventory: dict, bank: dict, batch1: dict) -> dict:
    _verify_batch1(batch1)
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
        (selected[0], TAB_CASES, transpose_grid_oracle,
         transpose_grid_reference, transpose_linewise_mutant,
         "guitar-tab physical grid and fret bounds are explicit"),
        (selected[1], WEAPON_CASES, identify_weapon_oracle,
         identify_weapon_reference, identify_weapon_mutant,
         "the six mappings and invalid-input result are explicit"),
        (selected[2], PAIR_CASES, closest_pair_oracle,
         closest_pair_reference, closest_pair_mutant,
         "largest lexicographic pair and strict upper limit are explicit"),
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
            "previous_review_sha256": batch1["review_sha256"],
        },
        "frozen_review_rule": (
            "resolve blocked position 6, then review pending positions 10 and 12"
        ),
        "positions": FROZEN_POSITIONS,
        "counts": {
            "resumed": 1,
            "newly_examined": 2,
            "contract_passed": 3,
            "trusted_oracle_validated": 3,
            "correction_admitted": 0,
            "cumulative_positions_examined": 5,
            "cumulative_contract_rejected": 1,
            "cumulative_oracle_ready": 4,
        },
        "rows": rows,
        "claims": {
            "untrusted_model_code_executed": False,
            "independent_oracle_admission_complete": False,
            "correction_pool_certified": False,
            "gpu_training_authorized": False,
            "protected_data_opening_authorized": False,
        },
        "next_position": 16,
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
