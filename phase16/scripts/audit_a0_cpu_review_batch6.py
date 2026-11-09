"""Build P16-R0 A0 CPU contract/oracle review batch 6.

The batch reviews the next five pending records in frozen A0 order. Trusted
cases and oracle implementations are derived from task contracts only. It
does not use published expected values or reference solutions, execute the
recovered model output, admit corrections, or authorize training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import re
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "p16_a0_cpu_review_batch_v1"
EXPECTED_INVENTORY_SHA256 = (
    "00743b1561a4c0e632ca2020c91e7a1c891e6d9e75af28b3ff04ad7edc6be377"
)
EXPECTED_BANK_SHA256 = (
    "1a5334e86e8904a35aa4d04234480057523c801104ce991b6b1c1b237e19a352"
)
FROZEN_POSITIONS = [37, 39, 40, 41, 42]
EXPECTED_FUNCTIONS = [
    "parse_fen", "diamond", "gordon", "finding_k", "change_case",
]


def canonical_sha256(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def source_sha256(function: Callable[..., Any]) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _without_hash(document: dict, key: str) -> dict:
    value = copy.deepcopy(document)
    value.pop(key, None)
    return value


def diamond_oracle(size: int) -> str | None:
    if size <= 0 or size % 2 == 0:
        return None
    widths = list(range(1, size + 1, 2))
    widths += widths[-2::-1]
    return "".join(" " * ((size - width) // 2) + "*" * width + "\n"
                   for width in widths)


def diamond_reference(size: int) -> str | None:
    if size <= 0 or not size & 1:
        return None
    middle = size // 2
    rows = []
    for row in range(size):
        distance = abs(middle - row)
        rows.append(" " * distance + "*" * (size - 2 * distance))
    return "\n".join(rows) + "\n"


def diamond_mutant(size: int) -> str | None:
    if size <= 0 or size % 2 == 0:
        return None
    half = size // 2
    result = []
    for index in range(size):
        stars = abs(half - index) * 2 + 1
        spaces = max(0, half - abs(half - index))
        result.append(" " * spaces + "*" * stars + "\n")
    return "".join(result)


DIAMOND_CASES = [
    ("one", (1,)), ("three", (3,)), ("five", (5,)),
    ("even", (4,)), ("zero", (0,)), ("negative", (-3,)),
]


VOWELS = frozenset("aeiouAEIOU")


def gordon_oracle(value: str) -> str:
    words = []
    for word in value.split():
        transformed = "".join(
            "@" if character.lower() == "a"
            else "*" if character in VOWELS
            else character.upper()
            for character in word
        )
        words.append(transformed + "!!!!")
    return " ".join(words)


def gordon_reference(value: str) -> str:
    def transform(match: re.Match[str]) -> str:
        word = match.group(0).upper()
        word = word.replace("A", "@")
        return re.sub(r"[EIOU]", "*", word) + "!!!!"
    return " ".join(transform(match) for match in re.finditer(r"\S+", value))


def gordon_mutant(value: str) -> str:
    vowels = {"a": "@", "e": "*", "i": "*", "o": "*", "u": "*",
              "A": "@", "E": "*", "I": "*", "O": "*", "U": "*"}
    return "".join(
        vowels.get(word, word).upper() + "!!!!"
        if word in vowels else word.upper() + "!!!!"
        for word in value.split()
    )


GORDON_CASES = [
    ("four_words", ("What are you doing",)),
    ("all_vowels", ("aeiou aeiou aeiou aeiou",)),
    ("mixed_case", ("Amazing Eels In Orbit",)),
    ("single_letter_words", ("a e i o",)),
]


def finding_k_oracle(values: list[int]) -> int:
    differences = [abs(value - values[0]) for value in values[1:]]
    divisor = 0
    for difference in differences:
        divisor = math.gcd(divisor, difference)
    return divisor if divisor else -1


def finding_k_reference(values: list[int]) -> int:
    spread = max(values) - min(values)
    if spread == 0:
        return -1
    valid = [candidate for candidate in range(1, spread + 1)
             if len({value % candidate for value in values}) == 1]
    return max(valid)


def finding_k_mutant(values: list[int]) -> int:
    counts: dict[int, int] = {}
    maximum = -1
    for value in values:
        counts[value] = counts.get(value, 0) + 1
        if len(counts) > 2:
            return -1
        maximum = max(maximum, value % min(counts))
    return maximum if len(counts) == 2 else -1


FINDING_K_CASES = [
    ("consecutive", ([1, 2, 3],)),
    ("all_equal", ([1, 1, 1],)),
    ("example_three", ([5, 2, 8],)),
    ("two_values", ([4, 10],)),
    ("shared_difference_gcd", ([7, 19, 31, 43],)),
    ("unordered", ([100, 10, 70, 40],)),
]


def _manual_words(identifier: str) -> list[str] | None:
    if identifier == "":
        return []
    if not identifier.isascii() or not identifier.isalpha() and not any(
            separator in identifier for separator in "_-"):
        return None
    separators = {character for character in identifier if character in "_-"}
    if len(separators) > 1:
        return None
    if separators:
        separator = next(iter(separators))
        parts = identifier.split(separator)
        if any(not part or not part.isalpha() or not part.islower()
               for part in parts):
            return None
        return parts
    if not identifier.isalpha() or not identifier[0].islower():
        return None
    words: list[str] = []
    start = 0
    for index, character in enumerate(identifier):
        if character.isupper():
            words.append(identifier[start:index].lower())
            start = index
    words.append(identifier[start:].lower())
    return words


def change_case_oracle(identifier: str, target: str) -> str | None:
    if target not in {"camel", "snake", "kebab"}:
        return None
    words = _manual_words(identifier)
    if words is None:
        return None
    if not words:
        return ""
    if target == "camel":
        return words[0] + "".join(word.capitalize() for word in words[1:])
    return ("_" if target == "snake" else "-").join(words)


def change_case_reference(identifier: str, target: str) -> str | None:
    if target not in {"camel", "snake", "kebab"}:
        return None
    if identifier == "":
        words: list[str] = []
    elif re.fullmatch(r"[a-z]+(?:_[a-z]+)*", identifier):
        words = identifier.split("_")
    elif re.fullmatch(r"[a-z]+(?:-[a-z]+)*", identifier):
        words = identifier.split("-")
    elif re.fullmatch(r"[a-z]+(?:[A-Z][a-z]*)*", identifier):
        words = [match.group(0).lower() for match in
                 re.finditer(r"[a-z]+|[A-Z][a-z]*", identifier)]
    else:
        return None
    if target == "camel":
        return "" if not words else words[0] + "".join(
            word.title() for word in words[1:])
    return ("_" if target == "snake" else "-").join(words)


def change_case_mutant(identifier: str, target: str) -> str | None:
    if target == "snake":
        return "".join(f"_{char.lower()}" if char.isupper() else char
                       for char in identifier).lstrip("_")
    if target == "camel":
        return "".join(word.capitalize() for word in identifier.split("_"))
    if target == "kebab":
        return "-".join(word.lower() for word in identifier.split("_"))
    return None


CHANGE_CASE_CASES = [
    ("camel_to_snake", ("snakeCase", "snake")),
    ("kebab_to_camel", ("some-lisp-name", "camel")),
    ("snake_to_kebab", ("map_to_all", "kebab")),
    ("acronym_to_kebab", ("doHTMLRequest", "kebab")),
    ("mixed_invalid", ("invalid-inPut_bad", "kebab")),
    ("target_invalid", ("valid-input", "huh???")),
    ("empty", ("", "camel")),
    ("leading_separator", ("_invalid", "camel")),
    ("double_separator", ("bad__name", "snake")),
]


def _evidence(cases: list[tuple[str, tuple]], oracle: Callable,
              reference: Callable, mutant: Callable) -> dict:
    transcript = []
    for case_id, arguments in cases:
        expected = oracle(*copy.deepcopy(arguments))
        independently_expected = reference(*copy.deepcopy(arguments))
        _require(expected == independently_expected,
                 f"independent oracle disagreement on {case_id}")
        mutated = mutant(*copy.deepcopy(arguments))
        transcript.append({
            "case_id": case_id,
            "input_sha256": canonical_sha256(arguments),
            "expected_sha256": canonical_sha256(expected),
            "model_equivalent_mutant_matches": mutated == expected,
        })
    caught = [row["case_id"] for row in transcript
              if not row["model_equivalent_mutant_matches"]]
    _require(caught, "trusted model-equivalent mutant was not detected")
    return {
        "status": "validated_on_trusted_cpu_cases",
        "derivation_boundary": "task_contract_only",
        "published_test_values_used": False,
        "dataset_reference_solutions_used": False,
        "test_count": len(cases),
        "oracle_reference_agree": True,
        "model_equivalent_mutant_caught": True,
        "failing_case_ids": caught,
        "oracle_source_sha256": source_sha256(oracle),
        "reference_source_sha256": source_sha256(reference),
        "mutant_source_sha256": source_sha256(mutant),
        "cases_sha256": canonical_sha256(cases),
        "transcript_sha256": canonical_sha256(transcript),
    }


def five_grams(value: str) -> frozenset[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    if not tokens:
        return frozenset()
    if len(tokens) < 5:
        return frozenset({tuple(tokens)})
    return frozenset(tuple(tokens[index:index + 5])
                     for index in range(len(tokens) - 4))


def _near_duplicate_evidence(bank: dict, target: dict) -> dict:
    target_grams = five_grams(target["question"])
    similarities = []
    for task in bank["tasks"]:
        if task["item_id"] == target["item_id"]:
            continue
        other = five_grams(task["question"])
        union = target_grams | other
        score = len(target_grams & other) / len(union) if union else 0.0
        similarities.append((score, task["item_id"], task["prompt_sha256"]))
    similarities.sort(key=lambda row: (-row[0], row[1]))
    top = [{"jaccard": score, "item_id": item_id, "prompt_sha256": prompt}
           for score, item_id, prompt in similarities[:10]]
    _require(target.get("near_duplicate_component_size") == 1,
             "target is no longer a singleton frozen component")
    _require(top and top[0]["jaccard"] < 0.80,
             "unexpected bank-local near duplicate")
    return {
        "frozen_component_size": 1,
        "method": "token_5gram_jaccard",
        "threshold": 0.80,
        "maximum_other_similarity": top[0]["jaccard"],
        "top_ten_sha256": canonical_sha256(top),
        "bank_local_automated_status": "passed",
        "manual_cross_history_status": "pending",
    }


def _verify_hashed(document: dict, hash_key: str, schema: str,
                   label: str) -> None:
    _require(document.get("schema_version") == schema,
             f"unexpected {label} schema")
    _require(document.get(hash_key) == canonical_sha256(
        _without_hash(document, hash_key)), f"{label} canonical hash mismatch")


def _base_row(record: dict) -> dict:
    return {
        "position": record["position"],
        "item_id": record["item_id"],
        "family_id": record["family_id"],
        "source_record_sha256": record["record_sha256"],
        "prompt_sha256": record["prompt_sha256"],
        "recovered_output_sha256": record["recovered_output_sha256"],
    }


def build(inventory: dict, bank: dict, batch5: dict) -> dict:
    _verify_hashed(inventory, "inventory_sha256",
                   "p16_a0_admission_inventory_v1", "inventory")
    _require(inventory["inventory_sha256"] == EXPECTED_INVENTORY_SHA256,
             "batch 6 points to a different frozen inventory")
    _verify_hashed(bank, "bank_sha256", "p16_a0_candidate_bank_v1", "bank")
    _require(bank["bank_sha256"] == EXPECTED_BANK_SHA256,
             "batch 6 points to a different frozen bank")
    _verify_hashed(batch5, "review_sha256", SCHEMA_VERSION, "batch 5")
    _require(batch5.get("next_position") == 37,
             "batch 5 does not hand off at position 37")

    by_position = {row["position"]: row for row in inventory["records"]}
    by_id = {task["item_id"]: task for task in bank["tasks"]}
    pending = [row["position"] for row in inventory["records"]
               if row["position"] >= batch5["next_position"]
               and row["admission_status"] ==
               "pending_independent_oracle_review"]
    _require(pending[:5] == FROZEN_POSITIONS,
             "batch 6 positions are not the next five pending records")
    selected = [by_position[position] for position in FROZEN_POSITIONS]
    for record, function_name in zip(selected, EXPECTED_FUNCTIONS):
        task = by_id.get(record["item_id"])
        _require(task is not None, f"bank task missing for {record['item_id']}")
        for key in ("family_id", "prompt_sha256"):
            _require(task.get(key) == record.get(key),
                     f"bank/inventory {key} mismatch for {record['item_id']}")
        _require(task.get("function_name") == function_name,
                 f"unexpected function at position {record['position']}")

    rejected = _base_row(selected[0])
    rejected.update({
        "contract_review": {
            "status": "failed",
            "reason": (
                "the contract depicts eight visual rows but does not uniquely "
                "specify the returned row delimiter or trailing-newline policy"
            ),
        },
        "independent_oracle": {"status": "not_constructed_contract_failed"},
        "history_review": "not_reviewed_ineligible",
        "near_duplicate_review": "not_reviewed_ineligible",
        "sandbox_review": "not_reviewed_ineligible",
        "human_approval": "not_applicable_rejected",
        "review_disposition": "rejected_contract_under_specified",
        "rejection_reasons": ["contract_under_specified_output_serialization"],
        "correction_admitted": False,
    })
    rows = [rejected]
    definitions = [
        (selected[1], DIAMOND_CASES, diamond_oracle, diamond_reference,
         diamond_mutant, "odd-size geometry, invalid sizes and newline policy are explicit"),
        (selected[2], GORDON_CASES, gordon_oracle, gordon_reference,
         gordon_mutant, "word boundaries, vowel mapping, capitals and suffixes are explicit"),
        (selected[3], FINDING_K_CASES, finding_k_oracle, finding_k_reference,
         finding_k_mutant, "maximal common remainder modulus and all-equal sentinel are explicit"),
        (selected[4], CHANGE_CASE_CASES, change_case_oracle,
         change_case_reference, change_case_mutant,
         "three valid identifier grammars, invalid mixing and targets are explicit"),
    ]
    for record, cases, oracle, reference, mutant, reason in definitions:
        task = by_id[record["item_id"]]
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
            "previous_review_sha256": batch5["review_sha256"],
        },
        "frozen_review_rule": "review next five pending positions from 37",
        "positions": FROZEN_POSITIONS,
        "counts": {
            "newly_examined": 5,
            "contract_passed": 4,
            "contract_rejected": 1,
            "trusted_oracle_validated": 4,
            "correction_admitted": 0,
            "cumulative_positions_examined": 19,
            "cumulative_contract_rejected": 4,
            "cumulative_oracle_ready": 15,
        },
        "rows": rows,
        "claims": {
            "untrusted_model_code_executed": False,
            "independent_oracle_admission_complete": False,
            "correction_pool_certified": False,
            "gpu_training_authorized": False,
            "protected_data_opening_authorized": False,
        },
        "next_position": 43,
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
