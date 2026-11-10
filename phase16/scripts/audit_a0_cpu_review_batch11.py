"""Build P16-R0 A0 CPU contract/oracle review batch 11."""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path

from a0_cpu_review_common import build_review_batch


POSITIONS = [80, 81, 82, 85, 86]
FUNCTIONS = ["get_new_notes", "sum_of_a_beach", "is_valid_HK_phone_number",
             "alpha_seq", "egged"]


def notes_oracle(salary, bills):
    return max(0, (salary - sum(bills)) // 5)


def notes_reference(salary, bills):
    disposable = salary
    for bill in bills:
        disposable -= bill
    return 0 if disposable < 5 else disposable // 5


def notes_mutant(salary, bills):
    return (salary - sum(bills)) // 5


NOTES = [("exact", (100, [20, 30])), ("remainder", (99, [20, 30])),
         ("under_five", (53, [50])), ("negative", (40, [30, 20])),
         ("no_bills", (25, []))]


WORDS = ("sand", "water", "fish", "sun")


def beach_oracle(beach):
    lowered = beach.lower()
    return sum(lowered.count(word) for word in WORDS)


def beach_reference(beach):
    lowered = beach.casefold()
    total = 0
    for word in WORDS:
        total += len(re.findall(re.escape(word), lowered))
    return total


def beach_mutant(beach):
    return sum(len(beach.split(word)) - 1 for word in ("Sand", "Water", "Fish", "Sun"))


BEACH = [("mixed_case", ("WAtErSlIde",)),
         ("three", ("GolDeNSanDyWateRyBeaChSuNN",)),
         ("four", ("gOfIshsunesunFiSh",)), ("none", ("cItYTowNcARShoW",)),
         ("repeated", ("sandsandSUNsun",)), ("empty", ("",))]


def phone_oracle(number):
    return re.fullmatch(r"[0-9]{4} [0-9]{4}", number) is not None


def phone_reference(number):
    return (len(number) == 9 and number[4] == " "
            and all("0" <= c <= "9" for c in number[:4] + number[5:]))


def phone_mutant(number):
    return len("".join(filter(str.isdigit, number))) == 8


PHONE = [("valid", ("1234 5678",)), ("no_space", ("12345678",)),
         ("hyphen", ("1234-5678",)), ("extra", ("x1234 5678",)),
         ("two_spaces", ("1234  5678",)), ("unicode", ("１２３４ ５６７８",))]


def alpha_oracle(value):
    return ",".join(c.upper() + c.lower() * (ord(c.lower()) - 97)
                    for c in sorted(value, key=str.lower))


def alpha_reference(value):
    pieces = []
    for c in sorted(value.lower()):
        position = string.ascii_lowercase.index(c) + 1
        pieces.append(c.upper() + c * (position - 1))
    return ",".join(pieces)


def alpha_mutant(value):
    return ",".join(c.upper() + "a" * (string.ascii_lowercase.index(c.lower()))
                    for c in sorted(value))


ALPHA = [("simple", ("abc",)), ("mixed", ("ZpglnRxqenU",)),
         ("case", ("bA",)), ("duplicates", ("nN",)), ("single", ("z",))]


def _production(age):
    eggs = 300
    for _ in range(1, age):
        eggs = eggs * 80 // 100
    return eggs


def egged_oracle(year, span):
    if year == 0:
        return "No chickens yet!"
    return 3 * sum(_production(age) for age in range(1, min(year, span) + 1))


def egged_reference(year, span):
    cohorts = []
    for current_year in range(1, year + 1):
        cohorts.append({"age": 1, "count": 3})
        if current_year == year:
            return sum(cohort["count"] * _production(cohort["age"])
                       for cohort in cohorts if cohort["age"] <= span)
        for cohort in cohorts:
            cohort["age"] += 1
    return "No chickens yet!"


def egged_mutant(year, span):
    if year == 0:
        return "No chickens yet!"
    total = 0
    for i in range(year):
        total += 3 * (i + 1) * 300 * (0.8 ** (span - 1))
    return int(total)


EGGED = [("zero", (0, 3)), ("first", (1, 3)), ("second", (2, 3)),
         ("lifespan_one", (4, 1)), ("dies", (5, 3)), ("rounding", (4, 5))]


DEFINITIONS = {
    80: (NOTES, notes_oracle, notes_reference, notes_mutant,
         "disposable income buys whole five-pound notes and negative income yields zero"),
    81: (BEACH, beach_oracle, beach_reference, beach_mutant,
         "count non-overlapping sand, water, fish and sun substrings without case"),
    82: (PHONE, phone_oracle, phone_reference, phone_mutant,
         "the complete value must be four ASCII digits, one space and four ASCII digits"),
    85: (ALPHA, alpha_oracle, alpha_reference, alpha_mutant,
         "case-insensitive alphabetical order and each letter's own lowercase repetition are explicit"),
    86: (EGGED, egged_oracle, egged_reference, egged_mutant,
         "three yearly chickens contribute age-specific individually rounded production until lifespan"),
}


def build(inventory, bank, previous):
    return build_review_batch(
        inventory, bank, previous, batch_number=11, positions=POSITIONS,
        functions=FUNCTIONS, definitions=DEFINITIONS, rejections={},
        cumulative_examined=44, cumulative_ready=36, cumulative_rejected=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--previous-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    report = build(*[json.loads(p.read_text(encoding="utf-8")) for p in
                     (args.inventory, args.bank, args.previous_review)])
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8", newline="\n")
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    print(f"next_position: {report['next_position']}")


if __name__ == "__main__":
    main()
