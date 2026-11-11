"""Trusted contract-only review definitions for frozen A0 group E."""

from __future__ import annotations

import itertools
import math
from collections import Counter

from a0_cpu_review_common import build_review_batch


def bus_oracle(current_time):
    hour, minute = map(int, current_time.split(":"))
    now = hour * 60 + minute
    leave_times = [departure - 5 for departure in range(360, 1441, 15)]
    future = [leave for leave in leave_times if leave >= now]
    return min(future) - now if future else 355 + 1440 - now


def bus_reference(current_time):
    hour, minute = map(int, current_time.split(":"))
    now = 60 * hour + minute
    if now <= 355:
        return 355 - now
    if now > 1435:
        return 1795 - now
    return (-now - 5) % 15


def bus_mutant(current_time):
    hour, minute = map(int, current_time.split(":"))
    now = hour * 60 + minute
    next_mark = now + 15 - minute % 15 - 5
    if next_mark < now:
        next_mark += 15
    return next_mark - now


BUS = [("before_service", ("05:00",)), ("on_hour", ("10:00",)),
       ("leave_now", ("12:10",)), ("missed", ("12:11",)),
       ("after_last_leave", ("23:56",)), ("midnight", ("00:00",))]


def lucky_oracle(number):
    return sum(map(int, str(abs(number)))) % 9 == 0


def lucky_reference(number):
    total, remaining = 0, abs(number)
    while remaining:
        total += remaining % 10
        remaining //= 10
    return total == 0 or total % 9 == 0


def lucky_mutant(number):
    if number == 0:
        return False
    return sum(int(digit) for digit in str(number)) % 9 == 0


LUCKY = [("zero", (0,)), ("example", (1892376,)), ("nine", (9,)),
         ("unlucky", (1234,))]


def capacity_oracle(length, width, height):
    return (length * 12 // 16) * (width * 12 // 16) * (height * 12 // 16)


def capacity_reference(length, width, height):
    result = 1
    for feet in (length, width, height):
        result *= math.floor(feet * 3 / 4)
    return result


def capacity_mutant(length, width, height):
    return (length * 12 // 16) * (width * 12 // 16) * (height // 16)


CAPACITY = [("example", (32, 64, 16)), ("cube", (4, 4, 4)),
            ("short_height", (10, 10, 2)), ("fractional_fit", (3, 5, 7))]


def palindrome_oracle(number):
    if type(number) is not int or number < 0:
        return "Not valid"
    text = str(number)
    return any(text[start:end] == text[start:end][::-1]
               for start in range(len(text)) for end in range(start + 2, len(text) + 1))


def palindrome_reference(number):
    if not isinstance(number, int) or isinstance(number, bool) or number < 0:
        return "Not valid"
    digits = str(number)
    for center in range(2 * len(digits) - 1):
        left, right = center // 2, (center + 1) // 2
        while left >= 0 and right < len(digits) and digits[left] == digits[right]:
            if right - left + 1 >= 2:
                return True
            left -= 1
            right += 1
    return False


def palindrome_mutant(number):
    if not isinstance(number, int) or number < 0:
        return "Not valid"
    text = str(number)
    for index in range(len(text) // 2):
        if text[index] != text[-index - 1]:
            return False
    return True


PALINDROME = [("single", (5,)), ("whole", (1221,)), ("inside", (141221001,)),
              ("prefix", (1215,)), ("none", (1294,)), ("invalid", ("109982",))]


def women_oracle(age):
    shown = 20 + age % 2
    return f"{age}? That's just {shown}, in base {age // 2}!"


def women_reference(age):
    base = age // 2
    ones = age - 2 * base
    return "{}? That's just 2{}, in base {}!".format(age, ones, base)


def women_mutant(age):
    for base in range(22, age):
        if age == base * (base + 1) // 2:
            return f"{age}? That's just {base}, in base {base}!"
    return None


WOMEN = [("even", (32,)), ("odd", (39,)), ("minimum_even", (22,)),
         ("minimum_odd", (23,)), ("large", (100,))]


MOVES = {"^": (0, 1), "v": (0, -1), "<": (-1, 0), ">": (1, 0)}


def simplify_oracle(path):
    positions = [(0, 0)]
    directions = []
    index_by_position = {(0, 0): 0}
    for direction in path:
        dx, dy = MOVES[direction]
        position = (positions[-1][0] + dx, positions[-1][1] + dy)
        if position in index_by_position:
            keep = index_by_position[position]
            for removed in positions[keep + 1:]:
                index_by_position.pop(removed)
            positions = positions[:keep + 1]
            directions = directions[:keep]
        else:
            directions.append(direction)
            positions.append(position)
            index_by_position[position] = len(positions) - 1
    return "".join(directions)


def simplify_reference(path):
    current = path
    while True:
        x = y = 0
        seen = {(0, 0): 0}
        changed = False
        for index, direction in enumerate(current, 1):
            dx, dy = MOVES[direction]
            x, y = x + dx, y + dy
            if (x, y) in seen:
                current = current[:seen[(x, y)]] + current[index:]
                changed = True
                break
            seen[(x, y)] = index
        if not changed:
            return current


def simplify_mutant(path):
    visited = {}
    x = y = 0
    current = []
    for direction in path:
        dx, dy = MOVES[direction]
        x, y = x + dx, y + dy
        if (x, y) in visited:
            break
        visited[(x, y)] = True
        current.append(direction)
    return "".join(current)


SIMPLIFY = [("simple", ("<>>",)), ("example", ("<^^>v<^^^",)),
            ("empty", ("",)), ("return_origin", ("^>v<^^",)),
            ("nested", ("^^v>><<^",))]


CN_DIGITS = "零一二三四五六七八九"
CN_UNITS = ("", "十", "百", "千", "万")


def _chinese_integer(number):
    if number == 0:
        return "零"
    text = str(number)
    output = []
    zero_pending = False
    for index, digit_text in enumerate(text):
        digit = int(digit_text)
        place = len(text) - index - 1
        if digit == 0:
            zero_pending = bool(output)
            continue
        if zero_pending:
            output.append("零")
            zero_pending = False
        if not (10 <= number <= 19 and index == 0 and digit == 1):
            output.append(CN_DIGITS[digit])
        output.append(CN_UNITS[place])
    return "".join(output)


def _number_parts(number):
    rendered = format(abs(number), ".8f").rstrip("0").rstrip(".")
    whole, dot, fraction = rendered.partition(".")
    return int(whole), fraction if dot else ""


def chinese_oracle(number):
    whole, fraction = _number_parts(number)
    result = _chinese_integer(whole)
    if fraction:
        result += "点" + "".join(CN_DIGITS[int(digit)] for digit in fraction)
    return ("负" if number < 0 else "") + result


def chinese_reference(number):
    whole, fraction = _number_parts(number)
    if whole == 0:
        integer = "零"
    else:
        pieces, pending = [], False
        powers = [10000, 1000, 100, 10, 1]
        for power in powers:
            digit, whole = divmod(whole, power)
            if digit:
                if pending and pieces:
                    pieces.append("零")
                if not (power == 10 and not pieces and digit == 1):
                    pieces.append(CN_DIGITS[digit])
                pieces.append(CN_UNITS[int(math.log10(power))] if power > 1 else "")
                pending = False
            elif pieces and whole:
                pending = True
        integer = "".join(pieces)
    decimal = "点" + "".join(CN_DIGITS[int(d)] for d in fraction) if fraction else ""
    return ("负" if number < 0 else "") + integer + decimal


def chinese_mutant(number):
    if number < 0:
        return "负" + chinese_mutant(-number)
    if number == 0:
        return "零"
    text = str(number).replace(".", "")
    result = ""
    for digit in reversed(text):
        result += CN_DIGITS[int(digit)]
    return result[::-1].strip("零")


CHINESE = [("zero", (0,)), ("teen", (11,)), ("hundred", (104,)),
           ("ten_thousand", (10004,)), ("large", (24681,)),
           ("decimal", (123.45,)), ("negative", (-10.08,))]


def binary_oracle(number):
    if number < 0:
        number &= 0xFFFFFFFF
    return format(number, "b")


def binary_reference(number):
    if number < 0:
        number = 2 ** 32 + number
    if number == 0:
        return "0"
    digits = []
    while number:
        digits.append(str(number % 2))
        number //= 2
    return "".join(reversed(digits))


def binary_mutant(number):
    if number >= 0:
        return bin(number)[2:].zfill(32)[-32:]
    return bin((1 << 32) + number)[2:].zfill(32)[-32:]


BINARY = [("three", (3,)), ("negative", (-3,)), ("zero", (0,)),
          ("large_positive", (2 ** 31,))]


def split_oracle(spending):
    average = sum(spending.values()) / len(spending)
    return {name: round(value - average, 2) for name, value in spending.items()}


def split_reference(spending):
    equal_share = sum(spending.values()) / len(spending)
    result = {}
    for name in spending:
        difference = spending[name] - equal_share
        result[name] = round(difference * 100) / 100
    return result


def split_mutant(spending):
    average = round(sum(spending.values()) / len(spending), 2)
    return {name: round(average - value, 2) for name, value in spending.items()}


SPLIT = [("example", ({"A": 20, "B": 15, "C": 10},)),
         ("equal", ({"A": 5, "B": 5},)),
         ("decimal", ({"A": 10, "B": 11, "C": 13},))]


def newspaper_oracle(source, target):
    if target == "":
        return 0
    if any(char not in source for char in target):
        return -1
    copies, position = 1, 0
    for char in target:
        found = source.find(char, position)
        if found < 0:
            copies += 1
            found = source.find(char)
        position = found + 1
    return copies


def newspaper_reference(source, target):
    if not target:
        return 0
    for copies in range(1, len(target) + 1):
        iterator = iter(source * copies)
        if all(any(candidate == wanted for candidate in iterator) for wanted in target):
            return copies
    return -1


def newspaper_mutant(source, target):
    n, m = len(source), len(target)
    dp = [[-1] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            if i > 0 and dp[i - 1][j] != -1 and source[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j]
            if dp[i][j] == -1 and j > 0:
                for k in range(i):
                    if dp[k][j - 1] != -1 and source[i - 1] == target[j - 1]:
                        dp[i][j] = dp[k][j - 1] + 1
                        break
    return dp[n][m]


NEWSPAPER = [("example", ("abc", "bcac")), ("impossible", ("abc", "xyz")),
             ("one", ("abc", "ac")), ("wraps", ("ab", "baba")),
             ("repeated", ("abc", "cccc"))]


def grader_oracle(score):
    if score > 1 or score < .6:
        return "F"
    for threshold, grade in ((.9, "A"), (.8, "B"), (.7, "C"), (.6, "D")):
        if score >= threshold:
            return grade


def grader_reference(score):
    return ("F" if not .6 <= score <= 1 else "A" if score >= .9 else
            "B" if score >= .8 else "C" if score >= .7 else "D")


def grader_mutant(score):
    if score < .6:
        return "F"
    if score >= .9:
        return "A"
    if score >= .8:
        return "B"
    if score >= .7:
        return "C"
    if score >= .6:
        return "D"


GRADER = [("low", (0,)), ("too_high", (1.1,)), ("a", (.9,)),
          ("b", (.8,)), ("c", (.7,)), ("d", (.6,)), ("one", (1,))]


def _compress_groups(values):
    output = []
    for value in values:
        if output and output[-1] % 2 == value % 2:
            output[-1] += value
        else:
            output.append(value)
    return output


def groups_oracle(values):
    current = values[:]
    while True:
        compressed = _compress_groups(current)
        if compressed == current:
            return len(current)
        current = compressed


def groups_reference(values):
    current = values[:]
    changed = True
    while changed:
        changed = False
        index = 0
        while index + 1 < len(current):
            if current[index] % 2 == current[index + 1] % 2:
                current[index:index + 2] = [current[index] + current[index + 1]]
                changed = True
            else:
                index += 1
    return len(current)


def groups_mutant(values):
    values = values[:]
    i = 0
    while i < len(values) - 1:
        if values[i] % 2 == values[i + 1] % 2:
            values[i] += values[i + 1]
            del values[i + 1]
        else:
            i += 1
    return len(values)


GROUPS = [("example", ([2, 1, 2, 2, 6, 5, 0, 2, 0, 5, 5, 7, 7, 4, 3, 3, 9],)),
          ("new_left_match", ([1, 1, 2],)), ("all_even", ([2, 4, 6],)),
          ("alternating", ([1, 2, 3, 4],))]


def score_combinations_oracle(number):
    digits = str(number)
    values = set()
    for length in range(1, len(digits) + 1):
        for selected in itertools.combinations(range(len(digits)), length):
            chosen = [digits[index] for index in selected]
            for permutation in set(itertools.permutations(chosen)):
                if permutation[0] != "0":
                    values.add(int("".join(permutation)))
    return sum(values)


def score_combinations_reference(number):
    digits = list(str(number))
    found = set()
    def visit(prefix, remaining):
        if prefix:
            found.add(int(prefix))
        for index, digit in enumerate(remaining):
            if not prefix and digit == "0":
                continue
            visit(prefix + digit, remaining[:index] + remaining[index + 1:])
    visit("", digits)
    return sum(found)


def score_combinations_mutant(number):
    if number == 0:
        return 0
    digits = [int(digit) for digit in str(number)]
    return sum(int("".join(map(str, permutation)))
               for permutation in itertools.permutations(digits))


SCORE_COMBINATIONS = [("example", (348,)), ("zero_digit", (340,)),
                      ("duplicates", (333,)), ("single", (6,)), ("zero", (0,))]


def example_sort_oracle(values, order):
    rank = {value: index for index, value in enumerate(order)}
    return sorted(values, key=rank.__getitem__)


def example_sort_reference(values, order):
    counts = Counter(values)
    return [value for catalogued in order for value in [catalogued] * counts[catalogued]]


def example_sort_mutant(values, order):
    counts = Counter(order)
    result = []
    for value in values:
        if value in counts:
            result.append(value)
            counts[value] -= 1
            if counts[value] == 0:
                del counts[value]
    return result


EXAMPLE_SORT = [("example", ([1, 3, 4, 4, 4, 4, 5], [4, 1, 2, 3, 5])),
                ("subset", ([3, 1, 3], [1, 2, 3])),
                ("strings", (["b", "a", "b"], ["b", "a"]))]


def bit_score_oracle(number):
    return 0 if number == 0 else (1 << number.bit_length()) - 1


def bit_score_reference(number):
    result = 0
    for value in range(number + 1):
        result |= value
    return result


def bit_score_mutant(number):
    return number | (number - 1)


BIT_SCORE = [("zero", (0,)), ("one", (1,)), ("gap", (40,)),
             ("forty_nine", (49,)), ("power", (64,))]


def nested_oracle(values, depth=1):
    return sum(nested_oracle(value, depth + 1) if isinstance(value, list) else value ** depth
               for value in values)


def nested_reference(values, depth=1):
    total = 0
    stack = [(values, depth)]
    while stack:
        current, level = stack.pop()
        for value in current:
            if isinstance(value, list):
                stack.append((value, level + 1))
            else:
                total += pow(value, level)
    return total


def nested_mutant(values, depth=1):
    total = 0
    for value in values:
        if isinstance(value, list):
            total += nested_mutant(value, depth + 1)
        else:
            total += value * (depth ** depth)
    return total


NESTED = [("described", ([1, [2], 3, [4, [5]]],)),
          ("flat", ([1, 2, 3],)), ("deep", ([[[2]]],)),
          ("zeros", ([0, [0]],))]


def powers_oracle(number):
    rows = [1 << exponent for exponent in range(number + 1)]
    return rows + [sum(rows)]


def powers_reference(number):
    rows, value, total = [], 1, 0
    for _ in range(number + 1):
        rows.append(value)
        total += value
        value *= 2
    rows.append(total)
    return rows


def powers_mutant(number):
    values = [1] * (number + 2)
    for i in range(number + 1):
        values[i + 1] += values[i]
        for j in range(i + 1, number + 1):
            values[j + 1] += values[j]
    return values


POWERS = [("zero", (0,)), ("one", (1,)), ("three", (3,)), ("ten", (10,))]


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def lexical_oracle(source):
    count = 0
    for letters in itertools.product(ALPHABET, repeat=len(source)):
        candidate = "".join(letters)
        count += candidate > source and candidate[::-1] > source[::-1]
    return count % 1000000007


def lexical_reference(source):
    count = 0
    def visit(prefix):
        nonlocal count
        if len(prefix) == len(source):
            if prefix > source and tuple(reversed(prefix)) > tuple(reversed(source)):
                count += 1
            return
        for character in ALPHABET:
            visit(prefix + character)
    visit("")
    return count % 1000000007


def lexical_mutant(source):
    count = 0
    for i in range(len(source) - 1, -1, -1):
        current = source[i]
        if current == "Z":
            continue
        for code in range(ord(current), ord("Z") + 1):
            candidate = source[:i] + chr(code) + "A" * (len(source) - i - 1)
            if candidate > source and candidate[::-1] > source[::-1]:
                count += 1
        if i > 0 and source[i - 1] < current:
            break
    return count % 1000000007


LEXICAL = [("one", ("X",)), ("example", ("XYZ",)),
           ("two", ("AZ",)), ("all_z", ("ZZ",))]


def reindeer_oracle(presents):
    if presents > 180:
        raise ValueError("Too many presents")
    return 2 + math.ceil(presents / 30)


def reindeer_reference(presents):
    if presents > 6 * 30:
        raise ValueError("Too many presents")
    carriers = (presents + 29) // 30
    return 2 + carriers


def reindeer_mutant(presents):
    if presents > 180 * 30:
        raise ValueError("Too many presents")
    return max(2, presents // 30 + 2)


REINDEER = [("zero", (0,)), ("one", (1,)), ("thirty", (30,)),
            ("thirty_one", (31,)), ("maximum", (180,))]


DEFINITIONS = {
    260: (BUS, bus_oracle, bus_reference, bus_mutant, "return minutes until the five-minute-before departure point within the 06:00-through-00:00 service window"),
    263: (LUCKY, lucky_oracle, lucky_reference, lucky_mutant, "zero and numbers whose decimal digit sum is divisible by nine are lucky"),
    268: (CAPACITY, capacity_oracle, capacity_reference, capacity_mutant, "convert every warehouse dimension from feet to inches and multiply whole 16-inch crate counts"),
    271: (PALINDROME, palindrome_oracle, palindrome_reference, palindrome_mutant, "detect any contiguous numerical palindrome of at least two digits without permutation"),
    278: (WOMEN, women_oracle, women_reference, women_mutant, "even ages render as base*2+0 and odd ages as base*2+1"),
    281: (SIMPLIFY, simplify_oracle, simplify_reference, simplify_mutant, "on revisiting a coordinate remove the entire intervening loop and continue from that earlier coordinate"),
    282: (CHINESE, chinese_oracle, chinese_reference, chinese_mutant, "render positional simplified Chinese integers, grouped interior zeroes, digitwise decimals and a negative prefix"),
    285: (BINARY, binary_oracle, binary_reference, binary_mutant, "positive binary omits leading zeroes while negatives use exactly 32-bit two's complement"),
    288: (SPLIT, split_oracle, split_reference, split_mutant, "each balance is spending minus the group mean, positive for receivers and rounded to two places"),
    290: (NEWSPAPER, newspaper_oracle, newspaper_reference, newspaper_mutant, "find the fewest repeated headings whose concatenation contains the target as an ordered subsequence"),
    291: (GRADER, grader_oracle, grader_reference, grader_mutant, "scores outside 0.6 through 1 fail before applying the descending grade thresholds"),
    295: (GROUPS, groups_oracle, groups_reference, groups_mutant, "repeatedly merge adjacent equal-parity groups because odd-group sums can create new parity adjacencies"),
    297: (SCORE_COMBINATIONS, score_combinations_oracle, score_combinations_reference, score_combinations_mutant, "sum each distinct non-leading-zero number from every nonempty digit subset permutation exactly once"),
    301: (EXAMPLE_SORT, example_sort_oracle, example_sort_reference, example_sort_mutant, "preserve input multiplicities while ordering all elements by their catalog position"),
    302: (BIT_SCORE, bit_score_oracle, bit_score_reference, bit_score_mutant, "OR every integer from zero through n, setting every bit through n's most significant bit"),
    304: (NESTED, nested_oracle, nested_reference, nested_mutant, "raise each number itself to its one-based array nesting depth before summing"),
    307: (POWERS, powers_oracle, powers_reference, powers_mutant, "binomial coefficient sums are powers of two from exponent zero through n followed by their total"),
    317: (LEXICAL, lexical_oracle, lexical_reference, lexical_mutant, "count equal-length uppercase strings greater in both forward and reversed lexicographic order"),
    318: (REINDEER, reindeer_oracle, reindeer_reference, reindeer_mutant, "two fixed reindeer plus one per started block of thirty presents, with capacity capped at 180"),
}


REJECTIONS = {
    276: ("on the stated positive-integer domain the frozen output already equals the defined sum, while the Examples section supplies no missing behavior from which to establish a residual",
          "no_contract_derived_model_failure"),
}


def build_group_e_batch(inventory, bank, previous, *, batch_number,
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
