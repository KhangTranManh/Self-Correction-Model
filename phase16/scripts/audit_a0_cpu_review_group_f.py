"""Trusted contract-only review definitions for frozen A0 group F."""

from __future__ import annotations

import math
import re
from collections import Counter

from a0_cpu_review_common import build_review_batch


def cooking_oracle(needed_power, minutes, seconds, power):
    needed = int(needed_power[:-1])
    actual = int(power[:-1])
    adjusted = math.ceil((minutes * 60 + seconds) * needed / actual)
    return f"{adjusted // 60} minutes {adjusted % 60} seconds"


def cooking_reference(needed_power, minutes, seconds, power):
    package_watts = int(needed_power.rstrip("W"))
    oven_watts = int(power.rstrip("W"))
    numerator = (minutes * 60 + seconds) * package_watts
    cooked_seconds = (numerator + oven_watts - 1) // oven_watts
    cooked_minutes, cooked_remainder = divmod(cooked_seconds, 60)
    return "{} minutes {} seconds".format(cooked_minutes, cooked_remainder)


def cooking_mutant(needed_power, minutes, seconds, power):
    needed = int(needed_power[:-2])
    actual = int(power[:-2])
    adjusted = (minutes * 60 + seconds) * actual / needed
    cooked_minutes = int(adjusted // 60)
    cooked_seconds = int(adjusted % 60)
    if cooked_seconds >= 30:
        cooked_minutes += 1
        cooked_seconds = 0
    return f"{cooked_minutes} minute{'s' if cooked_minutes != 1 else ''} {cooked_seconds} second{'s' if cooked_seconds != 1 else ''}"


COOKING = [("example", ("600W", 4, 20, "800W")),
           ("round_up", ("600W", 0, 79, "800W")),
           ("same", ("900W", 2, 5, "900W")),
           ("slower", ("800W", 1, 0, "600W"))]


VALID = "In with a chance"
INVALID = "Plenty more fish in the sea"


def validate_oracle(value):
    compact = value.replace("-", "")
    return VALID if re.fullmatch(r"(?:07\d{9}|\+447\d{9})", compact) else INVALID


def validate_reference(value):
    digits = "".join(char for char in value if char != "-")
    local = len(digits) == 11 and digits.startswith("07") and digits.isdigit()
    international = (len(digits) == 13 and digits.startswith("+447")
                     and digits[1:].isdigit())
    return VALID if local or international else INVALID


def validate_mutant(value):
    pattern = r'^\+447\d{8}$|^\+44-\d{2}-\d{3}-\d{3}-\d{2}$|^\+44\s\d{2}\s\d{3}\s\d{3}\s\d{2}$|^\+44(\s|\-)?7(\d{2}(\s|\-)\d{3}(\s|\-)\d{3}(\s|\-)\d{2}(\s|\-))$|^\+44(\s|\-)?7\d{9}$'
    return VALID if value and re.match(pattern, value) else INVALID


VALIDATE = [("local", ("07454876120",)), ("international", ("+447454876120",)),
            ("dashes", ("+44--745---487-6120",)),
            ("outer_dashes", ("-074-54-87-61-20-",)),
            ("empty", ("",)), ("wrong_prefix", ("06454876120",))]


def withdraw_oracle(amount):
    best = min(((a + b + c, a, b, c)
                for a in range(amount // 100 + 1)
                for b in range(amount // 50 + 1)
                for c in range(amount // 20 + 1)
                if 100 * a + 50 * b + 20 * c == amount))
    return list(best[1:])


def withdraw_reference(amount):
    hundreds = amount // 100
    while hundreds >= 0:
        remainder = amount - 100 * hundreds
        fifties = remainder // 50
        while fifties >= 0:
            last = remainder - 50 * fifties
            if last % 20 == 0:
                return [hundreds, fifties, last // 20]
            fifties -= 1
        hundreds -= 1
    raise AssertionError("contract promises exchangeability")


def withdraw_mutant(amount):
    hundreds = amount // 100
    remainder = amount % 100
    fifties = remainder // 50
    remainder %= 50
    return [hundreds, fifties, remainder // 20]


WITHDRAW = [("two_fifty", (250,)), ("two_sixty", (260,)),
            ("three_seventy", (370,)), ("greedy_trap", (110,)),
            ("forty", (40,))]


DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def ant_oracle(moves):
    black = set()
    x = y = direction = 0
    for _ in range(moves):
        point = (x, y)
        if point in black:
            black.remove(point)
            direction = (direction - 1) % 4
        else:
            black.add(point)
            direction = (direction + 1) % 4
        dx, dy = DIRECTIONS[direction]
        x, y = x + dx, y + dy
    return len(black)


def ant_reference(moves):
    cells = {}
    position = [0, 0]
    heading = 0
    for _ in range(moves):
        key = tuple(position)
        was_black = cells.get(key, False)
        cells[key] = not was_black
        heading = (heading + (-1 if was_black else 1)) % 4
        position[0] += DIRECTIONS[heading][0]
        position[1] += DIRECTIONS[heading][1]
    return sum(cells.values())


def ant_mutant(moves):
    if moves <= 0:
        return 0
    side = math.ceil(math.sqrt(moves + 1))
    grid = [[False] * side for _ in range(side)]
    x = y = side // 2
    direction = 0
    for _ in range(moves):
        grid[y][x] = not grid[y][x]
        direction = (direction + (1 if grid[y][x] else -1)) % 4
        dx, dy = DIRECTIONS[direction]
        x, y = (x + dx) % side, (y + dy) % side
    return sum(map(sum, grid))


ANT = [("zero", (0,)), ("one", (1,)), ("ten", (10,)),
       ("hundred", (100,)), ("unbounded_highway", (20000,))]


def zonk_oracle(dice):
    counts = Counter(dice)
    if len(dice) == 6 and set(dice) == set(range(1, 7)):
        return 1000
    if len(dice) == 6 and sorted(counts.values()) == [2, 2, 2]:
        return 750
    score = 0
    for face, count in counts.items():
        if count >= 3:
            base = 1000 if face == 1 else face * 100
            score += (count - 2) * base
        else:
            score += count * (100 if face == 1 else 50 if face == 5 else 0)
    return score or "Zonk"


def zonk_reference(dice):
    counts = Counter(dice)
    if sorted(dice) == [1, 2, 3, 4, 5, 6]:
        return 1000
    if len(dice) == 6 and sum(count == 2 for count in counts.values()) == 3:
        return 750
    total = 0
    for face in range(1, 7):
        count = counts[face]
        if count > 2:
            total += (count - 2) * (1000 if face == 1 else face * 100)
        elif face in (1, 5):
            total += count * (100 if face == 1 else 50)
    return total if total else "Zonk"


def zonk_mutant(dice):
    counts = Counter(dice)
    scores = [1000 if sorted(dice) == list(range(1, 7)) else 0]
    for face in range(1, 7):
        scores.append(({1: 1000, 2: 200, 3: 300, 4: 400, 5: 500, 6: 600}[face]
                       * (counts[face] // 3)) if counts[face] >= 3 else 0)
    scores.append(50 * sum(value // 2 for value in counts.values() if value % 2 == 0))
    scores.append(100 * (counts[1] // 3) + 50 * (counts[5] // 3))
    return max(scores) if any(scores) else "Zonk"


ZONK = [("one", ([1, 2, 3],)), ("singles", ([3, 4, 1, 1, 5],)),
        ("two_triples", ([2, 3, 2, 3, 3, 2],)),
        ("five_kind", ([1, 1, 1, 1, 1, 5],)),
        ("zonk", ([2, 3, 4, 3, 6, 6],)),
        ("four_kind", ([2, 2, 6, 6, 2, 2],)),
        ("pairs", ([1, 3, 4, 3, 4, 1],))]


def middle_oracle(value):
    return any(abs(index - (len(value) - index - 3)) <= 1
               for index in range(len(value) - 2) if value[index:index + 3] == "abc")


def middle_reference(value):
    centers = {max(0, (len(value) - 3) // 2), max(0, (len(value) - 2) // 2)}
    return any(value[index:index + 3] == "abc" for index in centers)


def middle_mutant(value):
    index = value.find("abc")
    return index >= 0 and abs(index - (len(value) - index - 3)) <= 1


MIDDLE = [("even_balance", ("AAabcBB",)), ("odd_balance", ("AabcBB",)),
          ("off", ("AabcBBB",)), ("later_center", ("abcabcXXX",)),
          ("none", ("abx",))]


def _bird(name):
    words = name.replace("-", " ").split()
    selectors = {1: (4,), 2: (2, 2), 3: (1, 1, 2), 4: (1, 1, 1, 1)}[len(words)]
    return "".join(word[:count] for word, count in zip(words, selectors)).upper()


def bird_oracle(values):
    return [_bird(value) for value in values]


def bird_reference(values):
    output = []
    for name in values:
        words = re.split(r"[- ]+", name)
        if len(words) == 1:
            code = words[0][:4]
        elif len(words) == 2:
            code = words[0][:2] + words[1][:2]
        elif len(words) == 3:
            code = words[0][0] + words[1][0] + words[2][:2]
        else:
            code = "".join(word[0] for word in words)
        output.append(code.upper())
    return output


def bird_mutant(values):
    output = []
    for name in values:
        words = name.replace("-", " ").split()
        if len(words) == 1:
            code = words[0][:4].upper()
        elif len(words) == 2:
            code = words[0][0] + words[1][0] + words[1][1].upper()
        elif len(words) == 3:
            code = words[0][0] + words[1][0] + words[2][:2].upper()
        else:
            code = "".join(word[0] for word in words).upper()
        output.append(code)
    return output


BIRD = [("example", (["Black-Capped Chickadee", "Common Tern"],)),
        ("all_lengths", (["Mallard", "Snow Goose", "Great Blue Heron", "Black and White Warbler"],))]


def longest_oracle(value):
    counts = Counter(char for char in value.lower() if char.isalnum())
    paired = sum(count // 2 * 2 for count in counts.values())
    return paired + int(any(count % 2 for count in counts.values()))


def longest_reference(value):
    filtered = "".join(char.lower() for char in value if char.isalnum())
    pairs = 0
    odd = False
    for char in set(filtered):
        count = filtered.count(char)
        pairs += count - count % 2
        odd |= bool(count % 2)
    return pairs + int(odd)


def longest_mutant(value):
    counts = Counter(value.lower())
    length, odd = 0, False
    for count in counts.values():
        length += count if count % 2 == 0 else count - 1
        odd |= bool(count % 2)
    return length + int(odd)


LONGEST = [("name", ("Hannah",)), ("punctuation", ("aabbcc_yYx_",)),
           ("none", ("___",)), ("mixed", ("Aa11!!b",))]


EXCLUDED = {"a", "the", "on", "at", "of", "upon", "in", "as"}


def words_oracle(value):
    return sum(word.lower() not in EXCLUDED for word in re.findall(r"[A-Za-z]+", value))


def words_reference(value):
    count = 0
    current = []
    for char in value + " ":
        if "A" <= char <= "Z" or "a" <= char <= "z":
            current.append(char)
        elif current:
            count += "".join(current).lower() not in EXCLUDED
            current = []
    return count


def words_mutant(value):
    words = "".join(filter(str.isalpha, value)).split()
    return sum(word.lower() not in EXCLUDED for word in words)


WORDS = [("short", ("Hello there, little user5453 374 ())$.",)),
         ("hyphen", ("Slow-moving target and K4mp",)),
         ("excluded", ("A cat in the hat upon a stool",)),
         ("empty", ("",))]


def mesh_oracle(values):
    result = []
    for left, right in zip(values, values[1:]):
        overlap = next((left[-size:] for size in range(min(len(left), len(right)), 0, -1)
                        if left[-size:] == right[:size]), None)
        if overlap is None:
            return "failed to mesh"
        result.append(overlap)
    return "".join(result)


def mesh_reference(values):
    pieces = []
    for index in range(len(values) - 1):
        left, right = values[index], values[index + 1]
        matches = [size for size in range(1, min(len(left), len(right)) + 1)
                   if left[len(left) - size:] == right[:size]]
        if not matches:
            return "failed to mesh"
        pieces.append(right[:max(matches)])
    return "".join(pieces)


def mesh_mutant(values):
    j = 0
    for i in range(len(values) - 1):
        left, right = values[i], values[i + 1]
        j = 0
        while j < min(len(left), len(right)) and left[-j - 1] == right[j]:
            j += 1
        if j == 0:
            return "failed to mesh"
    return "".join(word[-j:] for word in values)


MESH = [("example", (["allow", "lowering", "ringmaster", "terror"],)),
        ("failed", (["kingdom", "dominator", "notorious", "usual", "allegory"],)),
        ("two", (["apply", "plywood"],)), ("different_sizes", (["apple", "each", "choice"],))]


def triangle_oracle(row):
    while len(row) > 1:
        row = "".join(left if left == right else ({"R", "G", "B"} - {left, right}).pop()
                      for left, right in zip(row, row[1:]))
    return row


def triangle_reference(row):
    values = {"R": 0, "G": 1, "B": 2}
    inverse = "RGB"
    current = [values[char] for char in row]
    while len(current) > 1:
        current = [left if left == right else 3 - left - right
                   for left, right in zip(current, current[1:])]
    return inverse[current[0]]


def triangle_mutant(row):
    while len(row) > 1:
        new = ""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1]:
                new += row[i]
            else:
                new += {"RB": "G", "RG": "B", "GB": "R"}.get(row[i:i + 2], row[i])
        row = new
    return row


TRIANGLE = [("single", ("R",)), ("example", ("RRGBRGBB",)),
            ("reverse_pair", ("GR",)), ("mixed", ("BGRR",))]


TOTAL_DISTANCE = 2.4 + 112 + 26.2


def tri_oracle(distance):
    if distance == 0:
        return "Starting Line... Good Luck!"
    remaining = round(TOTAL_DISTANCE - distance, 2)
    if remaining <= 0:
        return "You're done! Stop running!"
    if distance < 2.4:
        return {"Swim": remaining}
    if distance < 114.4:
        return {"Bike": remaining}
    return {"Run": "Nearly there!" if remaining <= 10 else remaining}


def tri_reference(distance):
    if not distance:
        return "Starting Line... Good Luck!"
    left = round(140.6 - distance, 2)
    if left <= 0:
        return "You're done! Stop running!"
    if left <= 10:
        return {"Run": "Nearly there!"}
    if distance >= 114.4:
        return {"Run": left}
    if distance >= 2.4:
        return {"Bike": left}
    return {"Swim": left}


def tri_mutant(distance):
    remaining = 135.62 - distance
    if remaining <= 0:
        return "You're done! Stop running!"
    if remaining <= 10:
        return {"Run": "Nearly there!"}
    if remaining <= 26.2:
        return {"Run": round(remaining, 2)}
    if remaining <= 114.62:
        return {"Bike": round(remaining, 2)}
    return {"Swim": round(remaining - 114.62, 2)}


TRI = [("start", (0,)), ("swim", (1,)), ("bike", (10,)),
       ("run", (120,)), ("near", (135,)), ("done", (141,))]


def x_oracle(size):
    return [[int(row == col or row + col == size - 1) for col in range(size)]
            for row in range(size)]


def x_reference(size):
    grid = [[0 for _ in range(size)] for _ in range(size)]
    for index in range(size):
        grid[index][index] = 1
        grid[index][-index - 1] = 1
    return grid


def x_mutant(size):
    matrix = [[0] * size for _ in range(size)]
    for i in range(size):
        matrix[i][0] = matrix[i][size - 1] = matrix[0][i] = matrix[size - 1][i] = 1
        if i > 0:
            matrix[i][i] = matrix[i][size - 1 - i] = 1
            matrix[size - 1 - i][i] = matrix[size - 1 - i][size - 1 - i] = 1
    return matrix


X_CASES = [("one", (1,)), ("five", (5,)), ("six", (6,)), ("three", (3,))]


def bucket_oracle(*values):
    counts = Counter(values)
    output = [None] * (len(values) + 1)
    for frequency in set(counts.values()):
        output[frequency] = sorted(value for value, count in counts.items() if count == frequency)
    return output


def bucket_reference(*values):
    result = [None for _ in range(len(values) + 1)]
    for value in sorted(set(values)):
        count = values.count(value)
        if result[count] is None:
            result[count] = []
        result[count].append(value)
    return result


def bucket_mutant(*values):
    frequencies = Counter()
    appearances = {}
    for value in values:
        frequencies[value] += 1
        appearances.setdefault(frequencies[value], []).append(value)
    result = [None] * (max(frequencies.values()) + 1)
    for count, numbers in appearances.items():
        result[count] = numbers
    return result


BUCKET = [("example", (1, 2, 3, 4, 4, 5, 5, 5)),
          ("pairs", (2, 2, 4, 4, 6, 6, 9, 9, 9, 9)),
          ("one_kind", (5, 5, 5, 5, 5)), ("distinct", (16, 7, 5, 3, 6, 23))]


def rabbits_oracle(months, births):
    immature, adult = 1, 0
    for _ in range(months):
        immature, adult = adult * births, adult + immature
    return adult


def rabbits_reference(months, births):
    if months == 0:
        return 0
    sequence = [0, 1]
    for _ in range(2, months + 1):
        sequence.append(sequence[-1] + births * sequence[-2])
    return sequence[months]


def rabbits_mutant(months, births):
    if months == 0:
        return 1
    immature, adult = [0] * (months + 1), [0] * (months + 1)
    immature[0] = 1
    for month in range(1, months + 1):
        immature[month] = adult[month - 1] * births
        adult[month] = immature[month - 1] + adult[month - 1]
    return adult[months]


RABBITS = [("zero", (0, 3)), ("one", (1, 3)), ("example", (5, 3)),
           ("binary", (6, 1))]


def tree_oracle(matrix):
    if not matrix:
        return False
    visited = set()
    def visit(node, parent):
        if node in visited:
            return False
        visited.add(node)
        return all(neighbor == parent or visit(neighbor, node) for neighbor in matrix[node])
    return visit(0, -1) and len(visited) == len(matrix)


def tree_reference(matrix):
    if not matrix:
        return False
    edge_count = sum(map(len, matrix)) // 2
    reachable, stack = set(), [0]
    while stack:
        node = stack.pop()
        if node not in reachable:
            reachable.add(node)
            stack.extend(matrix[node])
    return len(reachable) == len(matrix) and edge_count == len(matrix) - 1


def tree_mutant(matrix):
    def dfs(node, visited, graph):
        if node in visited:
            return False
        visited.add(node)
        return all(dfs(neighbor, visited, graph) for neighbor in graph[node])
    graph = {i: [] for i in range(len(matrix))}
    for i in range(len(matrix)):
        for j in matrix[i]:
            graph[i].append(j)
            graph[j].append(i)
    visited = set()
    if not dfs(0, visited, graph) or len(visited) != len(matrix):
        return False
    return all(dfs(node, visited, graph) for node in graph)


TREE = [("tree", ([[1], [0, 2], [1]],)),
        ("cycle", ([[1, 2], [0, 2], [0, 1]],)),
        ("disconnected", ([[1], [0], []],)), ("single", ([[]],))]


def complete_oracle(sequence):
    if len(set(sequence)) != len(sequence):
        return [0]
    return list(range(max(sequence) + 1)) if sequence else []


def complete_reference(sequence):
    seen = set()
    for value in sequence:
        if value in seen:
            return [0]
        seen.add(value)
    if not seen:
        return []
    return [value for value in range(0, max(seen) + 1)]


def complete_mutant(sequence):
    if len(sequence) == 0:
        return []
    unique = sorted(set(sequence))
    maximum = unique[-1]
    if len(unique) > 1:
        return list(range(maximum + 1))
    return [0] if unique[0] == 0 else unique


COMPLETE = [("order", ([2, 1],)), ("duplicate", ([1, 4, 4, 6],)),
            ("range", ([1, 4, 6],)), ("zero_duplicate", ([0, 1, 0],)),
            ("single", ([3],)), ("empty", ([],))]


def keith_oracle(number):
    if number < 10:
        return False
    window = [int(digit) for digit in str(number)]
    iteration = 0
    while True:
        value = sum(window)
        iteration += 1
        if value == number:
            return iteration
        if value > number:
            return False
        window = window[1:] + [value]


def keith_reference(number):
    digits = list(map(int, str(number)))
    if len(digits) == 1:
        return False
    width = len(digits)
    generated = digits[:]
    while generated[-1] < number:
        generated.append(sum(generated[-width:]))
        if generated[-1] == number:
            return len(generated) - width
    return False


def keith_mutant(number):
    if number < 10:
        return False
    digits = [int(digit) for digit in str(number)]
    length = len(digits)
    while True:
        value = sum(digits[-length:])
        if value == number:
            return length
        if value > number:
            return False
        digits.append(value)


KEITH = [("example", (197,)), ("not", (196,)), ("two_digit", (14,)),
         ("one_digit", (7,)), ("later", (742,))]


DEFINITIONS = {
    321: (COOKING, cooking_oracle, cooking_reference, cooking_mutant, "scale time by required watts over actual watts, ceil total seconds and use the exact output template"),
    322: (VALIDATE, validate_oracle, validate_reference, validate_mutant, "remove arbitrary dashes then accept exactly the local 07 or replacement +447 prefix followed by nine digits"),
    323: (WITHDRAW, withdraw_oracle, withdraw_reference, withdraw_mutant, "minimize total 100, 50 and 20 bills rather than applying a greedy choice that can leave ten dollars"),
    324: (ANT, ant_oracle, ant_reference, ant_mutant, "simulate an unbounded initially white grid with the specified turn, flip and move order"),
    326: (ZONK, zonk_oracle, zonk_reference, zonk_mutant, "apply exclusive straight/three-pair scores or additive kind and leftover one/five scores using each die once"),
    327: (MIDDLE, middle_oracle, middle_reference, middle_mutant, "accept any abc occurrence whose left and right character counts differ by at most one"),
    329: (BIRD, bird_oracle, bird_reference, bird_mutant, "hyphens split words and the one-through-four-word rules select 4, 2+2, 1+1+2 or 1+1+1+1 uppercase letters"),
    332: (LONGEST, longest_oracle, longest_reference, longest_mutant, "ignore non-alphanumeric characters and use all possible case-insensitive pairs plus at most one odd centre"),
    335: (WORDS, words_oracle, words_reference, words_mutant, "count maximal ASCII-letter runs except the eight case-insensitive excluded words"),
    336: (MESH, mesh_oracle, mesh_reference, mesh_mutant, "concatenate the longest nonempty suffix-prefix overlap for every adjacent pair or fail"),
    338: (TRIANGLE, triangle_oracle, triangle_reference, triangle_mutant, "equal adjacent colours persist and unequal pairs yield the third colour regardless of pair order"),
    339: (TRI, tri_oracle, tri_reference, tri_mutant, "classify the 140.6-mile race by completed distance and report total remaining distance to two places"),
    340: (X_CASES, x_oracle, x_reference, x_mutant, "set only the two diagonals of the n by n array, including the even four-cell centre"),
    343: (BUCKET, bucket_oracle, bucket_reference, bucket_mutant, "return len(input)+1 slots and place each distinct sorted value only in its final frequency bucket"),
    345: (RABBITS, rabbits_oracle, rabbits_reference, rabbits_mutant, "month zero has zero mature pairs and each month matures the prior immature cohort while adults produce b offspring"),
    346: (TREE, tree_oracle, tree_reference, tree_mutant, "an undirected graph is a tree exactly when connected and acyclic, ignoring only each DFS parent edge"),
    353: (COMPLETE, complete_oracle, complete_reference, complete_mutant, "any repeated input value returns [0]; otherwise return the complete zero-through-maximum series"),
    356: (KEITH, keith_oracle, keith_reference, keith_mutant, "return the actual count of sliding digit-window sums needed to reach the multi-digit input"),
}


REJECTIONS = {
    341: ("the list representation does not specify whether the taco has one shell marker or two, nor their exact positions", "contract_under_specified_shell_serialization"),
    342: ("the statement says strings are case sensitive but does not define whether uppercase letters are converted, preserved, or assigned separate positions", "contract_under_specified_uppercase_encoding"),
}


def build_group_f_batch(inventory, bank, previous, *, batch_number,
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
