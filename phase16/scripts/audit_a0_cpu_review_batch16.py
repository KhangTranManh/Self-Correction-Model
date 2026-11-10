"""Build P16-R0 A0 CPU contract/oracle review batch 16."""

from __future__ import annotations

from itertools import combinations

from a0_cpu_review_common import build_review_batch, run_review_cli


POSITIONS = [144, 146, 147, 150, 151]
FUNCTIONS = ["solve", "mark_spot", "find_longest", "acronym_buster", "find_a_b"]


def reversals_oracle(value):
    if len(value) % 2:
        return -1
    opens = closes = 0
    for character in value:
        if character == "(":
            opens += 1
        elif opens:
            opens -= 1
        else:
            closes += 1
    return (opens + 1) // 2 + (closes + 1) // 2


def reversals_reference(value):
    if len(value) % 2:
        return -1
    for flips in range(len(value) + 1):
        for selected in combinations(range(len(value)), flips):
            changed = list(value)
            for index in selected:
                changed[index] = ")" if changed[index] == "(" else "("
            balance = 0
            valid = True
            for character in changed:
                balance += 1 if character == "(" else -1
                if balance < 0:
                    valid = False
                    break
            if valid and balance == 0:
                return flips
    raise AssertionError("even parenthesis strings are always repairable")


def reversals_mutant(value):
    stack = []
    reversals = 0
    for character in value:
        if character == "(":
            stack.append(character)
        elif character == ")" and stack:
            stack.pop()
        elif character == ")" and not stack:
            reversals += 1
        elif character == "(" and stack and stack[-1] == "(":
            stack.pop()
            reversals += 1
    if len(stack) % 2 != 0:
        return -1
    return len(stack) // 2 - reversals


REVERSALS = [
    ("example_two", (")(",)), ("example_one", ("(((())",)),
    ("odd", ("(((",)), ("empty", ("",)), ("balanced", ("(())",)),
    ("two_blocks", ("))((",)), ("crossed", ("())(",)),
    ("all_open", ("(((",)), ("all_closed", ("))))",)),
]


def longest_oracle(values):
    return max(values, key=lambda number: len(str(abs(number))))


def _digit_count(number):
    number = abs(number)
    if number == 0:
        return 1
    count = 0
    while number:
        count += 1
        number //= 10
    return count


def longest_reference(values):
    best = values[0]
    for value in values[1:]:
        if _digit_count(value) > _digit_count(best):
            best = value
    return best


def longest_mutant(values):
    return min(values, key=lambda number: len(str(number)))


LONGEST = [
    ("example", ([1, 10, 100],)), ("first_tie", ([100, 999, 10],)),
    ("negative_digits", ([-999, 10, 8],)), ("sign_not_digit", ([-9, 10],)),
    ("zero", ([0, 99, 7],)), ("single", ([12345],)),
]


DEFINITIONS = {
    144: (REVERSALS, reversals_oracle, reversals_reference, reversals_mutant,
          "minimum individual parenthesis flips needed for a balanced string; odd lengths return -1"),
    147: (LONGEST, longest_oracle, longest_reference, longest_mutant,
          "select the first number having the greatest count of decimal digits, excluding a minus sign"),
}

REJECTIONS = {
    146: ("the visual X example is absent, so internal row geometry and indentation are not specified",
          "contract_under_specified_output_geometry"),
    150: ("sentence boundaries, punctuation attachment and exact capitalization after multiword replacements are not specified",
          "contract_under_specified_tokenization_and_capitalization"),
    151: ("distinct-index reuse, no-solution return and first-pair ordering with negatives or duplicates are not specified",
          "contract_under_specified_pair_selection"),
}


def build(inventory, bank, previous):
    return build_review_batch(
        inventory, bank, previous, batch_number=16, positions=POSITIONS,
        functions=FUNCTIONS, definitions=DEFINITIONS, rejections=REJECTIONS,
        cumulative_examined=69, cumulative_ready=54, cumulative_rejected=15)


if __name__ == "__main__":
    run_review_cli(build)
