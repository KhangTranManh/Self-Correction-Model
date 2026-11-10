"""Trusted contract-only review definitions for frozen A0 group D."""

from __future__ import annotations

import math
import re
from collections import Counter
from itertools import combinations

from a0_cpu_review_common import build_review_batch


def bad_apples_oracle(apples):
    output, pending = [], None
    for package in apples:
        good = [value for value in package if value]
        if len(good) == 2:
            output.append(package[:])
        elif len(good) == 1:
            if pending is None:
                pending = (len(output), good[0])
                output.append(None)
            else:
                output[pending[0]] = [pending[1], good[0]]
                pending = None
    return [package for package in output if package is not None]


def bad_apples_reference(apples):
    complete = []
    spare = []
    for left, right in apples:
        if left and right:
            complete.append(("whole", [left, right]))
        elif left or right:
            spare.append((len(complete), left or right))
            if len(spare) == 2:
                insertion = spare[0][0]
                complete.insert(insertion, ("repacked", [spare[0][1], spare[1][1]]))
                spare.clear()
    return [package for _, package in complete]


def bad_apples_mutant(fruit_box):
    fruit_box = [package[:] for package in fruit_box]
    def replace(package):
        if package == [0, 0]:
            return None
        if package == [0, 2] or package == [2, 0]:
            for i, candidate in enumerate(fruit_box):
                if candidate != [0, 0] and candidate != package:
                    if candidate[0] > 0:
                        fruit_box[i] = [0, 0]
                        return [candidate[0], package[1]]
                    if candidate[1] > 0:
                        fruit_box[i] = [0, 0]
                        return [package[0], candidate[1]]
            return None
        return package
    return [new for package in fruit_box if (new := replace(package)) is not None]


BAD_APPLES = [
    ("example", ([[1, 3], [7, 6], [7, 2], [1, 3], [0, 2], [4, 5], [0, 3], [7, 6]],)),
    ("two_left", ([[2, 0], [3, 0]],)), ("unpaired", ([[0, 4]],)),
    ("rotten", ([[0, 0], [1, 1]],)), ("separated", ([[0, 5], [1, 2], [6, 0]],)),
]


def step_oracle(value):
    return any(left == right for left, right in zip(value, value[1:]))


def step_reference(value):
    return re.search(r"(.)\1", value, re.DOTALL) is not None


def step_mutant(value):
    return value == "green glass door"


STEP = [("moon", ("moon",)), ("sun", ("sun",)), ("slippers", ("slippers",)),
        ("speed", ("speed",)), ("phrase", ("green glass door",)), ("empty", ("",))]


UNITS = ((604800, "week", "a"), (86400, "day", "a"),
         (3600, "hour", "an"), (60, "minute", "a"), (1, "second", "a"))


def pretty_oracle(seconds):
    if seconds == 0:
        return "just now"
    for size, name, article in UNITS:
        if seconds >= size:
            amount = seconds // size
            return f"{article} {name} ago" if amount == 1 else f"{amount} {name}s ago"


def pretty_reference(seconds):
    if seconds < 1:
        return "just now"
    boundaries = [(1, "second"), (60, "minute"), (3600, "hour"),
                  (86400, "day"), (604800, "week")]
    size, name = max((pair for pair in boundaries if seconds >= pair[0]), key=lambda pair: pair[0])
    count = int(seconds / size)
    if count == 1:
        return ("an" if name == "hour" else "a") + f" {name} ago"
    return f"{count} {name}s ago"


def pretty_mutant(seconds):
    units = [(604800, "weeks"), (86400, "days"), (3600, "hours"),
             (60, "minutes"), (1, "seconds")]
    if seconds <= 0:
        return "just now"
    for size, name in units:
        if seconds >= size:
            amount = seconds // size
            return f"a {name[:-1]} ago" if amount == 1 else f"{amount} {name[:-1]}s ago"
    return "never"


PRETTY = [("now", (0,)), ("second", (1,)), ("minute", (60,)),
          ("hour_article", (3600,)), ("floor", (40000,)), ("weeks", (52 * 604800,))]


def one_down_oracle(text):
    if not isinstance(text, str):
        return "Input is not a string"
    output = []
    for char in text:
        if "A" <= char <= "Z":
            output.append(chr((ord(char) - ord("A") - 1) % 26 + ord("A")))
        elif "a" <= char <= "z":
            output.append(chr((ord(char) - ord("a") - 1) % 26 + ord("a")))
        else:
            output.append(char)
    return "".join(output)


def one_down_reference(text):
    if type(text) is not str:
        return "Input is not a string"
    before = "ZABCDEFGHIJKLMNOPQRSTUVWXYzabcdefghijklmnopqrstuvwxy"
    source = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return text.translate(str.maketrans(source, before))


def one_down_mutant(text):
    if not isinstance(text, str):
        return "Input is not a string"
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    shifted = alphabet[1:] + alphabet[:1]
    return "".join(shifted[alphabet.index(char)] if char in alphabet else char for char in text)


ONE_DOWN = [("note", ("Uif usjdl",)), ("wrap", ("AaZz",)),
            ("spaces", ("Ifmmp Xpsme",)), ("non_string", (123,))]


def missing_oracle(value):
    counts = Counter(value)
    sets = max(counts.values())
    return "".join(char * (sets - counts[char]) for char in "abcdefghijklmnopqrstuvwxyz")


def missing_reference(value):
    sets = max(value.count(char) for char in "abcdefghijklmnopqrstuvwxyz")
    output = []
    for char in "abcdefghijklmnopqrstuvwxyz":
        output.extend([char] * (sets - value.count(char)))
    return "".join(output)


def missing_mutant(value):
    alphabet, missing = "abcdefghijklmnopqrstuvwxyz", ""
    for char in alphabet:
        if char not in value:
            missing += char * (alphabet.index(char) + 1 - alphabet.index(missing[-1]) if missing else 1)
    return missing


MISSING = [("one", ("abcdefghijklmnopqrstuvwxy",)),
           ("two", ("aabbccddeeffgghhiijjkkllmmnnooppqqrrssttuuvvwwxxyy",)),
           ("uneven", ("abbccddeeffgghhiijjkkllmmnnooppqqrrssttuuvvwwxxy",)),
           ("word", ("codewars",)), ("complete", ("abcdefghijklmnopqrstuvwxyz",))]


def subsequences_oracle(needle, haystack):
    counts = [1] + [0] * len(needle)
    for char in haystack:
        for index in range(len(needle) - 1, -1, -1):
            if needle[index] == char:
                counts[index + 1] = (counts[index + 1] + counts[index]) % 100000000
    return counts[-1]


def subsequences_reference(needle, haystack):
    total = sum("".join(haystack[i] for i in selected) == needle
                for selected in combinations(range(len(haystack)), len(needle)))
    return total % 100000000


def subsequences_mutant(needle, haystack):
    n, h = len(needle), len(haystack)
    dp = [[0] * (h + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = 1
    for i in range(1, n + 1):
        for j in range(1, h + 1):
            dp[i][j] = ((dp[i - 1][j - 1] if needle[i - 1] == haystack[j - 1] else 0) + dp[i][j - 1]) % 100000000
    return dp[n][h]


SUBSEQUENCES = [("simple", ("abc", "aabbcc")), ("repeat", ("aa", "aaaa")),
                ("spaces", ("a b", "aa  bb")), ("none", ("z", "abc")),
                ("empty_haystack", ("a", ""))]


def decrypt_oracle(text, n):
    if text is None or not text or n <= 0:
        return text
    for _ in range(n):
        split = len(text) // 2
        odd, even = text[:split], text[split:]
        text = "".join(even[i:i + 1] + odd[i:i + 1] for i in range(len(even)))
    return text


def decrypt_reference(text, n):
    if text is None:
        return None
    for _ in range(max(0, n)):
        if text == "":
            break
        middle = len(text) // 2
        first, second = text[:middle], text[middle:]
        restored = [None] * len(text)
        restored[1::2] = first
        restored[0::2] = second
        text = "".join(restored)
    return text


def decrypt_mutant(text, n):
    if text is None or len(text) == 0 or n <= 0:
        return text
    length, half = len(text), len(text) // 2
    for _ in range(n):
        first, second, result = text[:half], text[half:], []
        for i in range(half):
            result.append(second[i])
            if i < half - 1:
                result.append(first[i])
        if length % 2 != 0:
            result.append(first[-1])
        text = "".join(result)
    return text


DECRYPT = [("example_one", ("hsi  etTi sats!", 1)),
           ("example_two", ("s eT ashi tist!", 2)), ("odd", ("bac", 1)),
           ("none", (None, 3)), ("zero_rounds", ("abc", 0))]


def seven_oracle(value):
    return "".join(char for index, char in enumerate(value)
                   if not (char == "9" and index > 0 and index + 1 < len(value)
                           and value[index - 1] == value[index + 1] == "7"))


def seven_reference(value):
    while "797" in value:
        value = value.replace("797", "77")
    return value


def seven_mutant(value):
    result = []
    for i in range(len(value) - 2):
        if value[i:i + 3] == "797":
            continue
        result.append(value[i])
    result.append(value[-1])
    return "".join(result)


SEVEN = [("example", ("79712312",)), ("overlap", ("79797",)),
         ("short", ("7",)), ("none", ("123",)), ("edge", ("9797",))]


def union_oracle(n):
    if isinstance(n, bool) or not isinstance(n, (int, float)):
        return False
    size = max(7, math.ceil(n))
    middle = {size // 2} if size % 2 else {size // 2 - 1, size // 2}
    rows = []
    for row in range(size):
        rows.append("".join("X" if (row == col or row + col == size - 1
                                     or row in middle or col in middle) else "-"
                            for col in range(size)))
    return "\n".join(rows)


def union_reference(n):
    if not isinstance(n, (int, float)) or isinstance(n, bool):
        return False
    size = max(7, int(math.ceil(n)))
    grid = [["-"] * size for _ in range(size)]
    centers = range((size - 1) // 2, size // 2 + 1)
    for i in range(size):
        for j in (i, size - i - 1):
            grid[i][j] = "X"
        for center in centers:
            grid[i][center] = grid[center][i] = "X"
    return "\n".join(map("".join, grid))


def union_mutant(n):
    if not isinstance(n, int):
        return False
    n = max(7, int(round(n)))
    flag = []
    for i in range(n):
        row = ""
        for j in range(n):
            row += "X" if i in [0, n - 1] or j in [0, n - 1] else "-"
        flag.append(row)
    return "\n".join(flag)


UNION = [("odd", (9,)), ("even", (10,)), ("minimum", (1,)),
         ("decimal", (7.2,)), ("invalid", ("string",))]


RESIGNATION = "Naah, too much...here's my resignation."


def wall_oracle(x=0, y=0):
    if type(x) is not int or type(y) is not int or x < 1 or y < 1:
        return None
    if x * y > 10000:
        return RESIGNATION
    full = "|".join(["■■"] * y)
    shifted = "|".join(["■"] + ["■■"] * (y - 1) + ["■"])
    return "\n".join(shifted if (x - row) % 2 == 0 else full for row in range(x))


def wall_reference(x=0, y=0):
    if not isinstance(x, int) or isinstance(x, bool) or not isinstance(y, int) or isinstance(y, bool) or min(x, y) <= 0:
        return None
    if x * y > 10000:
        return RESIGNATION
    rows = []
    for distance_from_bottom in range(x - 1, -1, -1):
        bricks = ["■■"] * y if distance_from_bottom % 2 == 0 else ["■"] + ["■■"] * (y - 1) + ["■"]
        rows.append("|".join(bricks))
    return "\n".join(rows)


def wall_mutant(x, y):
    if not (isinstance(x, int) and isinstance(y, int) and x > 0 and y > 0):
        return None
    if x * y > 10000:
        return RESIGNATION
    def brick(row, col):
        if col == 0 or col == x - 1:
            return "■"
        return "■■" if row % 2 == 0 else "■"
    wall = []
    for i in range(y):
        wall.append("|".join(brick(i, j) for j in range(x)))
        if i < y - 1:
            wall.append(" " * (2 * x + 1))
    return "\n".join(wall)


WALL = [("five", (5, 5)), ("even_rows", (10, 7)), ("one", (1, 1)),
        ("invalid", (12, -4)), ("resign", (123, 987))]


WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def average_oracle(value):
    tokens = value.split()
    if not tokens or any(token not in WORDS for token in tokens):
        return "n/a"
    return WORDS[sum(WORDS.index(token) for token in tokens) // len(tokens)]


def average_reference(value):
    mapping = dict(zip(WORDS, range(10)))
    try:
        numbers = [mapping[token] for token in value.split()]
        return WORDS[sum(numbers) // len(numbers)]
    except (KeyError, ZeroDivisionError):
        return "n/a"


def average_mutant(value):
    if not value:
        return "n/a"
    mapping = dict(zip(WORDS, range(10)))
    numbers, current = [], ""
    for char in value + " ":
        if char.isdigit():
            if current in mapping:
                numbers.append(mapping[current])
            current = ""
        else:
            current += char
    if current in mapping:
        numbers.append(mapping[current])
    if not numbers:
        return "n/a"
    return WORDS[int(sum(numbers) / len(numbers))]


AVERAGE = [("example", ("zero nine five two",)), ("empty", ("",)),
           ("invalid", ("one ten",)), ("single", ("nine",)),
           ("floor", ("one two",))]


def road_oracle(address, n):
    return 2 * n + 1 - address


def road_reference(address, n):
    if address % 2:
        row_from_top = (address + 1) // 2
        return 2 * (n - row_from_top + 1)
    row_from_bottom = address // 2
    return 2 * (n - row_from_bottom) + 1


def road_mutant(address, n):
    return n * 2 - address


ROAD = [("one", (1, 3)), ("three", (3, 3)), ("even", (2, 3)),
        ("five", (3, 5)), ("large", (999999, 500000000000))]


def _divisor_sum(number):
    if number == 1:
        return 0
    total = 1
    root = math.isqrt(number)
    for divisor in range(2, root + 1):
        if number % divisor == 0:
            total += divisor
            partner = number // divisor
            if partner != divisor:
                total += partner
    return total


def abundant_oracle(limit):
    for number in range(limit, 0, -1):
        abundance = _divisor_sum(number) - number
        if abundance > 0:
            return [[number], [abundance]]
    return [[], []]


def abundant_reference(limit):
    for number in reversed(range(1, limit + 1)):
        proper = sum(divisor for divisor in range(1, number) if number % divisor == 0)
        if proper > number:
            return [[number], [proper - number]]
    return [[], []]


def abundant_mutant(limit):
    best_number, best_abundance = None, -1
    for number in range(limit, 1, -1):
        abundance = _divisor_sum(number) - number
        if abundance > best_abundance and abundance > 0:
            best_number, best_abundance = number, abundance
    return [[best_number], [best_abundance]]


ABUNDANT = [("fifteen", (15,)), ("nineteen", (19,)), ("hundred", (100,)),
            ("large", (999,))]


def pyramid_oracle(n):
    if n <= 0:
        return ""
    width = 2 * n - 1
    rows = []
    for height in range(1, n + 1):
        rising = "".join(str(value % 10) for value in range(1, height + 1))
        core = rising + rising[-2::-1]
        rows.append(core.center(width))
    return "\n".join(rows)


def pyramid_reference(n):
    if n < 1:
        return ""
    rows = []
    for row in range(n):
        digits = [str((index + 1) % 10) for index in range(row + 1)]
        body = "".join(digits + digits[-2::-1])
        rows.append(" " * (n - row - 1) + body + " " * (n - row - 1))
    return "\n".join(rows)


def pyramid_mutant(n):
    if n <= 0:
        return ""
    rendered = []
    for i in range(1, n + 1):
        line = ''.join(str(j % 10) for j in range(1, i + 1)) + ''.join(str(k % 10) for k in range(i - 1, 0, -1))
        rendered.append(' ' * (n - i) + line)
    # The frozen callable prints these rows and implicitly returns None.
    return None


PYRAMID = [("zero", (0,)), ("one", (1,)), ("five", (5,)), ("wrap", (12,))]


def elimination_oracle(n):
    values, left = list(range(1, n + 1)), True
    while len(values) > 1:
        values = values[1::2] if left else list(reversed(list(reversed(values))[1::2]))
        left = not left
    return values[0]


def elimination_reference(n):
    head, step, remaining, left = 1, 1, n, True
    while remaining > 1:
        if left or remaining % 2 == 1:
            head += step
        remaining //= 2
        step *= 2
        left = not left
    return head


def elimination_mutant(n):
    values = list(range(1, n + 1))
    while len(values) > 1:
        values = values[::2] if len(values) % 2 == 0 else values[1::2]
    return values[0]


ELIMINATION = [("one", (1,)), ("nine", (9,)), ("even", (8,)),
               ("ten", (10,)), ("larger", (100,))]


RELATED = {"Partridge", "PearTree", "Chat", "Dan", "Toblerone", "Lynn", "AlphaPapa", "Nomad"}


def part_oracle(values):
    count = sum(value in RELATED for value in values)
    return "Mine's a Pint" + "!" * count if count else "Lynn, I've pierced my foot on a spike!!"


def part_reference(values):
    matches = [value for value in values if value in RELATED]
    if not matches:
        return "Lynn, I've pierced my foot on a spike!!"
    return f"Mine's a Pint{'!' * len(matches)}"


def part_mutant(values):
    count = sum(value in RELATED for value in values)
    return f"Mine's a {'!' * count} pint!" if count else "Lynn, I've pierced my foot on a spike!!"


PART = [("none", (["Apple", "London"],)), ("one", (["Partridge"],)),
        ("three", (["Dan", "Lynn", "Nomad"],)), ("duplicate", (["Chat", "Chat"],))]


def angle_oracle(h, a, o):
    if h == 0:
        angle = math.degrees(math.atan2(o, a))
    elif a == 0:
        angle = math.degrees(math.asin(o / h))
    else:
        angle = math.degrees(math.acos(a / h))
    return round(angle)


def angle_reference(h, a, o):
    if h == 0:
        ratio = o / a
        return round(math.degrees(math.atan(ratio)))
    if a == 0:
        return round(math.degrees(math.asin(o / h)))
    return round(math.degrees(math.acos(a / h)))


def angle_mutant(h, a, o):
    if h == 0:
        theta = math.degrees(math.atan(a / o))
    elif a == 0:
        theta = math.degrees(math.atan(o / h))
    else:
        theta = 90
    return round(theta)


ANGLE = [("missing_h", (0, 3, 4)), ("missing_a", (5, 0, 4)),
         ("missing_o", (5, 4, 0)), ("non_integer", (13, 12, 0))]


DRINKS = {"jabroni": "Patron Tequila", "school counselor": "Anything with Alcohol",
          "programmer": "Hipster Craft Beer", "bike gang member": "Moonshine",
          "politician": "Your tax dollars", "rapper": "Cristal"}


def drink_oracle(value):
    return DRINKS.get(value.lower(), "Beer")


def drink_reference(value):
    folded = value.casefold()
    for profession, drink in DRINKS.items():
        if folded == profession:
            return drink
    return "Beer"


def drink_mutant(value):
    return DRINKS.get(value.lower(), "Beer").lower()


DRINK = [("jabroni", ("Jabroni",)), ("mixed", ("pOLitiCIaN",)),
         ("programmer", ("Programmer",)), ("default", ("Astronaut",))]


def lowest_oracle(number, k):
    stack = []
    for digit in str(number):
        while k and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)
    if k:
        del stack[-k:]
    return "".join(stack)


def lowest_reference(number, k):
    text = str(number)
    length = len(text) - k
    return min("".join(text[index] for index in chosen)
               for chosen in combinations(range(len(text)), length))


def lowest_mutant(number, k):
    stack = []
    for digit in str(number):
        while k > 0 and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)
    result = ''.join(stack[:-k if k else None])
    return str(int(result)) if result else '0'


LOWEST = [("leading_zero", (123056, 4)), ("middle", (1284569, 2)),
          ("tail", (12345, 2)), ("duplicates", (11121, 2)), ("all_but_one", (1000, 3))]


DEFINITIONS = {
    202: (BAD_APPLES, bad_apples_oracle, bad_apples_reference, bad_apples_mutant, "pair one-good-apple packages in encounter order, preserve complete-package order and discard rotten or unmatched packages"),
    204: (STEP, step_oracle, step_reference, step_mutant, "a value passes the green-glass-door rule exactly when it contains adjacent repeated characters"),
    207: (PRETTY, pretty_oracle, pretty_reference, pretty_mutant, "use the greatest whole time unit, floor its count and apply the specified singular article"),
    209: (ONE_DOWN, one_down_oracle, one_down_reference, one_down_mutant, "decode each ASCII letter to its case-preserving alphabet predecessor with wraparound"),
    211: (MISSING, missing_oracle, missing_reference, missing_mutant, "the maximum letter multiplicity determines the alphabet-set count and each deficit is emitted in alphabet order"),
    212: (SUBSEQUENCES, subsequences_oracle, subsequences_reference, subsequences_mutant, "count index-distinct ordered subsequences including spaces and retain the last eight decimal digits"),
    215: (DECRYPT, decrypt_oracle, decrypt_reference, decrypt_mutant, "invert each alternating split by interleaving the even-index half before the odd-index half"),
    222: (SEVEN, seven_oracle, seven_reference, seven_mutant, "remove every 9 whose immediate neighbors are both 7, including overlapping occurrences"),
    225: (UNION, union_oracle, union_reference, union_mutant, "ceil numeric dimensions to at least seven and draw both diagonals plus the parity-width central cross"),
    226: (WALL, wall_oracle, wall_reference, wall_mutant, "render x rows of y brick-widths with a full bottom row and alternating half-brick offsets"),
    235: (AVERAGE, average_oracle, average_reference, average_mutant, "all whitespace-delimited tokens must be zero-through-nine words before flooring their numeric mean"),
    237: (ROAD, road_oracle, road_reference, road_mutant, "opposite house numbers pair to the constant 2n+1"),
    240: (ABUNDANT, abundant_oracle, abundant_reference, abundant_mutant, "scan downward for the greatest abundant number not exceeding h and return its divisor-sum excess"),
    241: (PYRAMID, pyramid_oracle, pyramid_reference, pyramid_mutant, "each fixed-width row is a centered ascending then descending digit palindrome with trailing spaces preserved"),
    243: (ELIMINATION, elimination_oracle, elimination_reference, elimination_mutant, "remove the first encountered and every alternate survivor while alternating left and right passes"),
    244: (PART, part_oracle, part_reference, part_mutant, "each exact related term contributes one exclamation mark after the fixed Mine's a Pint text"),
    245: (ANGLE, angle_oracle, angle_reference, angle_mutant, "the zero argument identifies the missing side and the other two sides determine theta by right-triangle ratios"),
    252: (DRINK, drink_oracle, drink_reference, drink_mutant, "case-insensitive profession lookup returns the table's exact output capitalization or Beer"),
    255: (LOWEST, lowest_oracle, lowest_reference, lowest_mutant, "choose the lexicographically lowest length-preserving digit subsequence and retain leading zeroes"),
}


REJECTIONS = {
    230: ("case-insensitive duplicate elimination does not specify which original capitalization is retained or how mixed-case outputs are sorted",
          "contract_under_specified_case_preservation"),
}


def build_group_d_batch(inventory, bank, previous, *, batch_number,
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
