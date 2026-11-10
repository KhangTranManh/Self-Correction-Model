"""Trusted contract-only review definitions for frozen A0 group C."""

from __future__ import annotations

import math
import re
from collections import deque
from itertools import product

from a0_cpu_review_common import build_review_batch


def pattern_oracle(n):
    if n <= 0:
        return ""
    digits = "".join(str(value % 10) for value in range(1, n + 1))
    return "\n".join(" " * (n - row - 1) + digits + " " * row
                      for row in range(n))


def pattern_reference(n):
    if n < 1:
        return ""
    core = "1234567890" * (n // 10) + "123456789"[:n % 10]
    width = 2 * n - 1
    rows = []
    for row in range(n):
        rows.append((" " * (n - row - 1) + core).ljust(width))
    return "\n".join(rows)


def pattern_mutant(n):
    if n <= 0:
        return ""
    result = []
    for i in range(n):
        spaces = " " * (n - i - 1)
        numbers = ''.join(str(j % 10) for j in range(1, n + 1))
        result.append(spaces + numbers + numbers[:-1][::-1])
    return '\n'.join(result)


PATTERN = [("zero", (0,)), ("negative", (-2,)), ("one", (1,)),
           ("five", (5,)), ("wrap_digits", (12,))]


def band_oracle(name):
    title = name.capitalize()
    return title + name[1:] if name[0].lower() == name[-1].lower() else "The " + title


def band_reference(name):
    if name[:1].casefold() == name[-1:].casefold():
        return name[0].upper() + name[1:] + name[1:]
    return f"The {name[0].upper()}{name[1:].lower()}"


def band_mutant(name):
    if name[0].lower() == name[-1].lower():
        return name.capitalize() * 2
    return f"The {name.capitalize()}"


BAND = [("normal", ("dolphin",)), ("overlap", ("alaska",)),
        ("second_overlap", ("agenda",)), ("single", ("x",))]


def _ascii_value(value):
    if value is None or any(not ("A" <= c <= "Z" or "a" <= c <= "z") for c in value):
        return 0
    return sum(ord(c.upper()) for c in value)


def compare_oracle(left, right):
    return _ascii_value(left) == _ascii_value(right)


def compare_reference(left, right):
    def value(text):
        if text is None or re.fullmatch("[A-Za-z]*", text) is None:
            return 0
        return sum(map(ord, text.upper()))
    return value(left) == value(right)


def compare_mutant(s1, s2):
    def sanitize_input(s):
        if s is None:
            return ""
        return ''.join(c for c in s.upper() if c.isalpha())
    return sum(ord(c) for c in sanitize_input(s1)) == sum(ord(c) for c in sanitize_input(s2))


COMPARE = [("equal_sum", ("AD", "BC")), ("case", ("gf", "FG")),
           ("invalid_whole", ("A1", "A")), ("both_empty", (None, "")),
           ("different", ("kl", "lz"))]


def change_palindrome_oracle(value):
    mismatches = sum(value[i] != value[-i - 1] for i in range(len(value) // 2))
    return mismatches == 1 or (mismatches == 0 and len(value) % 2 == 1)


def change_palindrome_reference(value):
    for index, original in enumerate(value):
        replacement = "a" if original != "a" else "b"
        candidate = value[:index] + replacement + value[index + 1:]
        if candidate == candidate[::-1]:
            return True
    return False


def change_palindrome_mutant(s):
    count = 0
    for i in range(len(s) // 2):
        if s[i] != s[-i - 1]:
            count += 1
        if count > 1:
            return False
    return True


CHANGE_PAL = [("one_mismatch", ("abbx",)), ("even_pal", ("abba",)),
              ("odd_pal", ("abcba",)), ("two_equal", ("aa",)),
              ("two_different", ("ab",)), ("two_mismatch", ("abcaab",))]


def triple_oracle(num1, num2):
    first, second = str(num1), str(num2)
    return int(any(digit * 3 in first and digit * 2 in second for digit in "0123456789"))


def triple_reference(num1, num2):
    def runs(number, length):
        text = str(number)
        return {text[i] for i in range(len(text) - length + 1)
                if len(set(text[i:i + length])) == 1}
    return int(bool(runs(num1, 3) & runs(num2, 2)))


def triple_mutant(num1, num2):
    def check(n1, n2, target):
        count1 = 0
        for char in str(n1):
            count1 = count1 + 1 if char == target else 0
            if count1 == 3:
                break
        count2 = 0
        for char in str(n2):
            count2 = count2 + 1 if char == target else 0
            if count2 == 2:
                return True
        return False
    return int(any(check(num1, num2, digit) for digit in "0123456789"))


TRIPLE = [("yes", (451999277, 41177722899)), ("no", (1222345, 12345)),
          ("false_positive", (12345, 12234)), ("six", (666789, 12345667))]


def evenator_oracle(value):
    cleaned = value.translate(str.maketrans("", "", ".,?!_"))
    return " ".join(word + word[-1] if len(word) % 2 else word for word in cleaned.split())


def evenator_reference(value):
    words = re.sub(r"[.,?!_]", "", value).split()
    for index, word in enumerate(words):
        if len(word) & 1:
            words[index] += word[-1]
    return " ".join(words)


def evenator_mutant(value):
    value = re.sub(r'[.,?!_]', '', value)
    words = value.split()
    return ' '.join(word[:-1] + word[-1] * (len(word) % 2 == 1) for word in words)


EVENATOR = [("example", ("How did we end up here? We go?",)),
            ("punctuation", ("a_b, cc!",)), ("spacing", ("odd   even",))]


def graceful_oracle(bill):
    tipped = bill * 1.15
    if tipped < 10:
        step = 1
    else:
        step = 5 * 10 ** (int(math.log10(tipped)) - 1)
    return math.ceil((tipped - 1e-12) / step) * step


def graceful_reference(bill):
    target = bill + bill * 15 / 100
    magnitude = 1
    while target >= magnitude * 10:
        magnitude *= 10
    unit = 1 if target < 10 else magnitude / 2
    result = unit
    while result + 1e-12 < target:
        result += unit
    return int(result) if float(result).is_integer() else result


def graceful_mutant(bill):
    tip = bill * 1.15
    if bill < 10:
        return round(tip)
    for i in [500, 50, 5, 0.5]:
        if tip % i == 0:
            return round(tip, 1)
        if tip < i + 0.5:
            return round(tip - (tip % i) + i, 1)


GRACEFUL = [("one", (1,)), ("seven", (7,)), ("twelve", (12,)),
            ("eighty_six", (86,)), ("hundreds", (999,))]


def _shootout_ended(a_kicks, b_kicks, a_score, b_score):
    if a_kicks <= 5 and b_score + max(0, 5 - b_kicks) < a_score:
        return True
    if b_kicks <= 5 and a_score + max(0, 5 - a_kicks) < b_score:
        return True
    return a_kicks == b_kicks and a_kicks >= 5 and a_score != b_score


def penalty_oracle(shots, score):
    start = (shots, shots, score[0], score[1], 0)
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (ak, bk, a, b, turn), depth = queue.popleft()
        if _shootout_ended(ak, bk, a, b):
            return depth
        for scored in (0, 1):
            nxt = (ak + 1, bk, a + scored, b, 1) if turn == 0 else (ak, bk + 1, a, b + scored, 0)
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, depth + 1))
    raise AssertionError("shootout must terminate")


def penalty_reference(shots, score):
    limit = max(2, 2 * max(0, 5 - shots) + 2)
    for length in range(limit + 1):
        for outcomes in product((0, 1), repeat=length):
            ak = bk = shots
            a, b = score
            ended_at = None
            if _shootout_ended(ak, bk, a, b):
                ended_at = 0
            for index, scored in enumerate(outcomes):
                if index % 2 == 0:
                    ak += 1
                    a += scored
                else:
                    bk += 1
                    b += scored
                if _shootout_ended(ak, bk, a, b):
                    ended_at = index + 1
                    break
            if ended_at == length:
                return length
    raise AssertionError("bounded enumeration failed")


def penalty_mutant(shots, score):
    score_diff = abs(score[0] - score[1])
    if score_diff > shots - max(score):
        return 1
    return 2


PENALTY = [("example", (2, [1, 2])), ("sudden_death", (10, [10, 10])),
           ("start", (0, [0, 0])), ("late_initial", (4, [4, 2]))]


def remove_pal_oracle(value):
    if value == value[::-1]:
        return "OK"
    return "remove one" if any(value[:i] + value[i + 1:] == (value[:i] + value[i + 1:])[::-1]
                               for i in range(len(value))) else "not possible"


def remove_pal_reference(value):
    left, right = 0, len(value) - 1
    while left < right and value[left] == value[right]:
        left += 1
        right -= 1
    if left >= right:
        return "OK"
    a = value[left + 1:right + 1]
    b = value[left:right]
    return "remove one" if a == a[::-1] or b == b[::-1] else "not possible"


def remove_pal_mutant(s):
    length = len(s)
    for i in range(length // 2):
        if s[i] != s[-(i + 1)]:
            return "remove one" if s[i:-i] == s[i:-i][::-1] or s[i + 1:-i + 1] == s[i + 1:-i + 1][::-1] else "not possible"
    return "OK"


REMOVE_PAL = [("pal", ("abba",)), ("edge", ("abbaa",)),
              ("impossible", ("abbaab",)), ("left", ("cabacx",)),
              ("short", ("ab",))]


def _owned(years, increment):
    if years < 15:
        return 0
    if years < 24:
        return 1
    return 2 + (years - 24) // increment


def owned_oracle(cat_years, dog_years):
    return [_owned(cat_years, 4), _owned(dog_years, 5)]


def owned_reference(cat_years, dog_years):
    def count(value, later):
        thresholds = [15, 24]
        while thresholds[-1] + later <= value:
            thresholds.append(thresholds[-1] + later)
        return sum(value >= threshold for threshold in thresholds)
    return [count(cat_years, 4), count(dog_years, 5)]


def owned_mutant(cat_years, dog_years):
    def calculate(age, base, second, subsequent):
        if age <= 1:
            return math.floor(base / 15)
        if age == 2:
            return math.floor((base + second) / 15)
        return math.floor(((age - 2) * subsequent + base + second) / 15)
    return [calculate(cat_years, 15, 9, 4), calculate(dog_years, 15, 9, 5)]


OWNED = [("babies", (0, 0)), ("first", (15, 15)), ("second", (24, 24)),
         ("later", (28, 29)), ("partial", (27, 28)), ("old", (56, 64))]


MIXES = {frozenset(): "black", frozenset({0}): "red", frozenset({1}): "green",
         frozenset({2}): "blue", frozenset({0, 1}): "yellow",
         frozenset({0, 2}): "magenta", frozenset({1, 2}): "cyan",
         frozenset({0, 1, 2}): "white"}


def hex_oracle(codes):
    if not codes:
        return "black"
    values = list(map(int, codes.split()))
    maximum = max(values)
    return MIXES[frozenset() if maximum == 0 else frozenset(i for i, x in enumerate(values) if x == maximum)]


def hex_reference(codes):
    if codes.strip() == "":
        return "black"
    red, green, blue = map(int, codes.split())
    if red == green == blue:
        return "black" if red == 0 else "white"
    if red == green > blue:
        return "yellow"
    if red == blue > green:
        return "magenta"
    if green == blue > red:
        return "cyan"
    return ("red", "green", "blue")[(red, green, blue).index(max(red, green, blue))]


def hex_mutant(codes):
    if not codes or codes == '0 0 0':
        return 'black'
    r, g, b = map(int, codes.split())
    if r > g and r > b:
        return 'white' if g + b == r else ('yellow' if g > b else 'red')
    if g > r and g > b:
        return 'white' if r + b == g else ('yellow' if r > b else 'green')
    return 'white' if r + g == b else ('magenta' if r > g else 'cyan')


HEX = [("green", ("087 255 054",)), ("yellow", ("181 181 170",)),
       ("black", ("000 000 000",)), ("white", ("001 001 001",)),
       ("red_not_sum", ("100 50 50",)), ("empty", ("",))]


def digits_oracle(number):
    digits = list(map(int, str(number)))
    while len(digits) > 1:
        digits = [(a + b + 1) // 2 for a, b in zip(digits, digits[1:])]
    return digits[0]


def digits_reference(number):
    text = str(number)
    while len(text) > 1:
        next_text = ""
        for index in range(len(text) - 1):
            total = int(text[index]) + int(text[index + 1])
            next_text += str(total // 2 + total % 2)
        text = next_text
    return int(text)


def digits_mutant(input):
    while input >= 10:
        next_input = 0
        while input > 0:
            next_input, input = next_input + (input % 10 + (input // 10 % 10)) // 2, input // 100
        input = next_input + (input % 10 + (input // 10 % 10)) // 2
    return input


DIGITS = [("example", (246,)), ("round_up", (89,)), ("single", (7,)),
          ("long", (12345,)), ("zeros", (1001,))]


MONTHS = ("January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December")


def challenge_oracle(pledged, current, month):
    if current >= pledged:
        return "Challenge is completed."
    completed = MONTHS.index(month)
    base, extra = divmod(pledged, 12)
    expected = completed * base + min(completed, extra)
    difference = current - expected
    if difference == 0:
        return "You are on track."
    direction = "ahead of" if difference > 0 else "behind"
    return f"You are {abs(difference)} {direction} schedule."


def challenge_reference(pledged, current, month):
    if current == pledged:
        return "Challenge is completed."
    monthly = [pledged // 12] * 12
    for index in range(pledged % 12):
        monthly[index] += 1
    target = sum(monthly[:MONTHS.index(month)])
    return ("You are on track." if current == target else
            f"You are {target - current} behind schedule." if current < target else
            f"You are {current - target} ahead of schedule.")


def challenge_mutant(pledged, current, month):
    month_index = MONTHS.index(month)
    if month_index == 0:
        return "You are on track."
    expected = pledged * (month_index // 4) + (pledged if month_index % 4 == 0 else pledged * (month_index % 4))
    if current == expected:
        return "You are on track."
    if current < expected:
        return f"You are {expected - current} behind schedule."
    if current > expected:
        return f"You are {current - expected} ahead of schedule."
    return "Challenge is completed."


CHALLENGE = [("february", (12, 1, "February")), ("behind", (12, 1, "March")),
             ("ahead", (12, 5, "March")), ("complete", (12, 12, "September")),
             ("uneven", (100, 17, "March")), ("january", (100, 0, "January"))]


def triangle_oracle(n):
    total = even = 0
    for row in range(1, n + 1):
        values = [row * value for value in range(1, row + 1)]
        values += values[-2::-1]
        total += sum(values)
        even += sum(value for value in values if value % 2 == 0)
    return [total, even, total - even]


def triangle_reference(n):
    rows = [[i * min(column, 2 * i - column)
             for column in range(1, 2 * i)] for i in range(1, n + 1)]
    flat = [value for row in rows for value in row]
    return [sum(flat), sum(value for value in flat if not value & 1),
            sum(value for value in flat if value & 1)]


def triangle_mutant(n):
    total = even = odd = 0
    for i in range(1, n + 1):
        for j in range(i):
            value = (i * (i - 1) // 2 + j) ** 2
            total += value
            if value % 2 == 0:
                even += value
            else:
                odd += value
    return [total, even, odd]


TRIANGLE = [("one", (1,)), ("two", (2,)), ("five", (5,)), ("ten", (10,))]


def _math_tokens(value):
    parsed = []
    for order, token in enumerate(value.split()):
        letter = next(c for c in token if c.isalpha())
        parsed.append((letter, order, int(token.replace(letter, ""))))
    return [number for _, _, number in sorted(parsed, key=lambda row: (row[0], row[1]))]


def do_math_oracle(value):
    numbers = _math_tokens(value)
    result = numbers[0]
    operations = (lambda a, b: a + b, lambda a, b: a - b,
                  lambda a, b: a * b, lambda a, b: a / b)
    for index, number in enumerate(numbers[1:]):
        result = operations[index % 4](result, number)
    return round(result)


def do_math_reference(value):
    terms = []
    for token in value.split():
        position = next(i for i, char in enumerate(token) if char.isalpha())
        terms.append((token[position], int(token[:position] + token[position + 1:])))
    terms.sort(key=lambda row: row[0])
    answer = terms[0][1]
    for index, (_, number) in enumerate(terms[1:], 1):
        operation = ("+", "-", "*", "/")[(index - 1) % 4]
        answer = {"+": answer + number, "-": answer - number,
                  "*": answer * number, "/": answer / number}[operation]
    return round(answer)


def do_math_mutant(s):
    numbers = {}
    for token in s.split():
        for i, char in enumerate(token):
            if char.isalpha():
                numbers[char] = int(token[:i] + token[i + 1:])
                break
    sorted_numbers = sorted(numbers.values())
    result = sorted_numbers[0]
    for i in range(1, len(sorted_numbers)):
        if i % 5 == 1:
            result += sorted_numbers[i]
        elif i % 5 == 2:
            result -= sorted_numbers[i]
        elif i % 5 == 3:
            result *= sorted_numbers[i]
        elif i % 5 == 4:
            result /= sorted_numbers[i]
    return round(result)


DO_MATH = [("example", ("24z6 1x23 y369 89a 900b",)),
           ("duplicate", ("24z6 1z23 y369 89z 900b",)),
           ("cycle", ("10a 90x 14b 78u 45a 7b 34y",))]


DEFINITIONS = {
    152: (PATTERN, pattern_oracle, pattern_reference, pattern_mutant, "n fixed-width rows shift the repeated 1..n digit sequence left while preserving trailing spaces"),
    156: (BAND, band_oracle, band_reference, band_mutant, "equal endpoint letters use one-letter-overlapped noun repetition; other nouns receive The"),
    157: (COMPARE, compare_oracle, compare_reference, compare_mutant, "ASCII-letter-only strings compare by uppercase ASCII sum; an invalid whole string has value zero"),
    159: (CHANGE_PAL, change_palindrome_oracle, change_palindrome_reference, change_palindrome_mutant, "exactly one substitution can repair one mirrored mismatch or change an odd palindrome's centre"),
    160: (TRIPLE, triple_oracle, triple_reference, triple_mutant, "the same decimal digit must occur in a consecutive triple and consecutive double"),
    168: (EVENATOR, evenator_oracle, evenator_reference, evenator_mutant, "remove the five named punctuation characters then duplicate each odd word's last character"),
    176: (GRACEFUL, graceful_oracle, graceful_reference, graceful_mutant, "add at least fifteen percent then round upward to the magnitude-dependent elegant multiple"),
    178: (PENALTY, penalty_oracle, penalty_reference, penalty_mutant, "minimize additional alternating kicks under initial-five early termination and complete sudden-death rounds"),
    179: (REMOVE_PAL, remove_pal_oracle, remove_pal_reference, remove_pal_mutant, "distinguish an existing palindrome from one repairable by exactly one deletion"),
    180: (OWNED, owned_oracle, owned_reference, owned_mutant, "invert the stated first-, second- and later-year pet-age increments using truncation"),
    183: (HEX, hex_oracle, hex_reference, hex_mutant, "return the named primary or mixture selected by all channels tied at the maximum; all zero is black"),
    186: (DIGITS, digits_oracle, digits_reference, digits_mutant, "replace every adjacent digit pair by its ceiling mean until one digit remains"),
    187: (CHALLENGE, challenge_oracle, challenge_reference, challenge_mutant, "spread the pledge as evenly as possible with surplus in earliest months and count completed months only"),
    194: (TRIANGLE, triangle_oracle, triangle_reference, triangle_mutant, "row n is n times 1..n..1 and totals partition exactly by parity"),
    198: (DO_MATH, do_math_oracle, do_math_reference, do_math_mutant, "stable letter order precedes left-to-right cycling +,-,*,/ operations and final rounding"),
}


REJECTIONS = {
    158: ("rounding a half-slice is required by examples but the tie-breaking rule is not specified", "contract_under_specified_rounding_ties"),
    161: ("the keyboard diagram is absent and the text does not enumerate the complete 88-key black/white pattern", "contract_under_specified_key_pattern"),
    170: ("the treatment of duplicate magnitudes and unmatched negative or zero elements is not specified", "contract_under_specified_pair_multiplicity"),
    192: ("boundaries for consecutive capitals and mixed existing underscores are not specified by the examples", "contract_under_specified_case_boundaries"),
    193: ("the missing rider's exact travel budget and substitute handoff behavior outside the single example are not specified", "contract_under_specified_missing_rider_state"),
}


def build_group_c_batch(inventory, bank, previous, *, batch_number,
                        positions, functions, cumulative_examined,
                        cumulative_ready, cumulative_rejected):
    selected = set(positions)
    return build_review_batch(
        inventory, bank, previous, batch_number=batch_number,
        positions=list(positions), functions=list(functions),
        definitions={p: v for p, v in DEFINITIONS.items() if p in selected},
        rejections={p: v for p, v in REJECTIONS.items() if p in selected},
        cumulative_examined=cumulative_examined,
        cumulative_ready=cumulative_ready,
        cumulative_rejected=cumulative_rejected)
