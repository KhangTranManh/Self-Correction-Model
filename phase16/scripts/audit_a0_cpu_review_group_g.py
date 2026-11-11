"""Trusted contract-only review definitions for frozen A0 group G."""

from __future__ import annotations

import itertools
import math
import re

from a0_cpu_review_common import build_review_batch


def memesorting_oracle(text):
    endings = []
    for priority, (key, owner) in enumerate((("bug", "Roma"), ("boom", "Maxim"),
                                             ("edits", "Danik"))):
        offset = 0
        for index, char in enumerate(text.lower()):
            if char == key[offset]:
                offset += 1
                if offset == len(key):
                    endings.append((index, priority, owner))
                    break
    return min(endings)[2] if endings else "Vlad"


def memesorting_reference(text):
    candidates = []
    for priority, (key, owner) in enumerate((("bug", "Roma"), ("boom", "Maxim"),
                                             ("edits", "Danik"))):
        for positions in itertools.combinations(range(len(text)), len(key)):
            if "".join(text[i].lower() for i in positions) == key:
                candidates.append((positions[-1], priority, owner))
                break
    return min(candidates)[2] if candidates else "Vlad"


def memesorting_mutant(text):
    found = [(text.lower().find(key), owner) for key, owner in
             (("bug", "Roma"), ("boom", "Maxim"), ("edits", "Danik"))]
    found = [(position, owner) for position, owner in found if position >= 0]
    return min(found)[1] if found else "Vlad"


MEMESORTING = [("spaced_bug", ("bxxuxg",)), ("earliest_completion", ("booxmbug",)),
               ("spaced_edits", ("e-d-i-t-s",)), ("none", ("ordinary",))]


def any_odd_oracle(value):
    return bool(value & 0xAAAAAAAA)


def any_odd_reference(value):
    return any(value & (1 << index) for index in range(1, 32, 2))


def any_odd_mutant(value):
    return bool(value & 0x55555555)


ANY_ODD = [("odd_bit", (2,)), ("even_only", (5,)), ("alternating", (170,)),
           ("lowest", (1,))]


def odd_heavy_oracle(values):
    odds = [value for value in values if value % 2]
    evens = [value for value in values if value % 2 == 0]
    return bool(odds) and (not evens or min(odds) > max(evens))


def odd_heavy_reference(values):
    return any(value % 2 for value in values) and all(
        odd > even for odd in values if odd % 2 for even in values if even % 2 == 0)


def odd_heavy_mutant(values):
    odds = [value for value in values if value % 2]
    evens = [value for value in values if value % 2 == 0]
    return bool(odds) and (not evens or all(even > max(odds) for even in evens))


ODD_HEAVY = [("heavy", ([11, 4, 9, 2],)), ("interleaved", ([9, 10, 11],)),
             ("odds_only", ([3, 1],)), ("no_odds", ([2, 4],))]


def substitution_oracle(left, right):
    if len(left) != len(right):
        return False
    forward, reverse = {}, {}
    for source, target in zip(left, right):
        if forward.setdefault(source, target) != target:
            return False
        if reverse.setdefault(target, source) != source:
            return False
    return True


def substitution_reference(left, right):
    def pattern(value):
        ids = {}
        return tuple(ids.setdefault(char, len(ids)) for char in value)
    return len(left) == len(right) and pattern(left) == pattern(right)


def substitution_mutant(left, right):
    mapping = {}
    return len(left) == len(right) and all(
        mapping.setdefault(a, b) == b for a, b in zip(left, right))


SUBSTITUTION = [("bijection", ("paper", "title")), ("collision", ("ab", "aa")),
                ("forward_conflict", ("aa", "ab")), ("length", ("a", "aa"))]


def max_ball_oracle(speed):
    return round((speed / 3.6) / 9.81 * 10)


def max_ball_reference(speed):
    velocity = speed / 3.6
    best_height, best_tick = -1.0, 0
    for tick in range(10000):
        time = tick / 10
        height = velocity * time - 9.81 * time * time / 2
        if height < best_height:
            return best_tick
        best_height, best_tick = height, tick
    raise AssertionError("apex search did not terminate")


def max_ball_mutant(speed):
    velocity = speed / 3.6
    return int(velocity / 9.81)


MAX_BALL = [("fifteen", (15,)), ("twenty_five", (25,)), ("fast", (100,))]


def shift_left_oracle(left, right):
    overlap = 0
    for a, b in zip(reversed(left), reversed(right)):
        if a != b:
            break
        overlap += 1
    return len(left) + len(right) - 2 * overlap


def shift_left_reference(left, right):
    common = max((size for size in range(min(len(left), len(right)) + 1)
                  if left[len(left) - size:] == right[len(right) - size:]), default=0)
    return len(left) + len(right) - 2 * common


def shift_left_mutant(left, right):
    common = 0
    for a, b in zip(left, right):
        if a != b:
            break
        common += 1
    return max(len(left), len(right)) - common


SHIFT_LEFT = [("suffix", ("test", "west")), ("none", ("abc", "xyz")),
              ("contained", ("tower", "wer")), ("same", ("same", "same"))]


def elections_oracle(votes, voters):
    if voters == 0:
        maximum = max(votes)
        return int(votes.count(maximum) == 1)
    maximum = max(votes)
    return sum(vote + voters > maximum for vote in votes)


def elections_reference(votes, voters):
    return sum(all(index == other or vote + voters > rival
                   for other, rival in enumerate(votes))
               for index, vote in enumerate(votes))


def elections_mutant(votes, voters):
    maximum = max(votes)
    winners = votes.count(maximum)
    return winners + int(any(vote != maximum and vote + voters > maximum for vote in votes))


ELECTIONS = [("several", ([2, 3, 5, 2], 3)), ("unique_no_voters", ([1, 3, 2], 0)),
             ("tie_no_voters", ([3, 3, 2], 0)), ("none", ([1, 10], 2))]


def find_oracle(value):
    runs = list(re.finditer(r"!+|\?+", value))
    best = ""
    for left, right in zip(runs, runs[1:]):
        if left.end() == right.start() and left.group()[0] != right.group()[0]:
            candidate = left.group() + right.group()
            if len(candidate) > len(best):
                best = candidate
    return best


def find_reference(value):
    best = ""
    for boundary in range(1, len(value)):
        if {value[boundary - 1], value[boundary]} != {"!", "?"}:
            continue
        start, end = boundary - 1, boundary
        while start and value[start - 1] == value[boundary - 1]:
            start -= 1
        while end + 1 < len(value) and value[end + 1] == value[boundary]:
            end += 1
        candidate = value[start:end + 1]
        if len(candidate) > len(best):
            best = candidate
    return best


def find_mutant(value):
    matches = re.findall(r"!+\?+", value)
    return max(matches, default="")


FIND = [("forward", ("!!???",)), ("reverse", ("???!!!!",)),
        ("leftmost_tie", ("!!??xx??!!",)), ("none", ("!!!!",))]


DENSITY = {"O": 0, "A": 1, "W": 2, "H": 3}


def liquids_oracle(glass):
    ordered = iter(sorted((cell for row in glass for cell in row), key=DENSITY.get))
    return [[next(ordered) for _ in row] for row in glass]


def liquids_reference(glass):
    counts = {cell: sum(row.count(cell) for row in glass) for cell in DENSITY}
    stream = iter("".join(cell * counts[cell] for cell in "OAWH"))
    return [[next(stream) for _ in row] for row in glass]


def liquids_mutant(glass):
    return [sorted(row, key=DENSITY.get) for row in glass]


LIQUIDS = [("cross_rows", ([list("HW"), list("OA")],)),
           ("rectangular", ([list("WHO"), list("AAW")],)), ("empty", ([],))]


def reindeer_oracle(names):
    return sorted(names, key=lambda name: name.split()[-1])


def reindeer_reference(names):
    return [name for _, name in sorted(enumerate(names),
            key=lambda item: (item[1].rsplit(" ", 1)[-1], item[0]))]


def reindeer_mutant(names):
    return sorted(names, key=lambda name: (name.split()[-1], name))


REINDEER = [("ordinary", (["Dasher Tonoyan", "Dancer Moore", "Prancer Chua"],)),
            ("stable_tie", (["Zed Smith", "Amy Smith", "Bob Adams"],))]


def spinning_oracle(inner_max, outer_max):
    inner = outer = 0
    for spins in itertools.count(1):
        inner = inner_max if inner == 0 else inner - 1
        outer = 0 if outer == outer_max else outer + 1
        if inner == outer:
            return spins


def spinning_reference(inner_max, outer_max):
    for spins in itertools.count(1):
        if (inner_max - (spins - 1) % (inner_max + 1)) % (inner_max + 1) == spins % (outer_max + 1):
            return spins


def spinning_mutant(inner_max, outer_max):
    inner, outer = inner_max, outer_max
    for spins in itertools.count(1):
        inner = inner_max if inner == 0 else inner - 1
        outer = 0 if outer == outer_max else outer + 1
        if inner == outer:
            return spins


SPINNING = [("small", (2, 3)), ("equal", (3, 3)), ("asymmetric", (5, 2))]


def largest_power_oracle(limit):
    if limit == 1:
        return (0, -1)
    representations = {}
    for base in range(2, math.isqrt(max(0, limit - 1)) + 1):
        power, exponent = base * base, 2
        while power < limit:
            representations.setdefault(power, set()).add((base, exponent))
            power *= base
            exponent += 1
    if not representations:
        return (1, -1)
    largest = max(representations)
    return (largest, len(representations[largest]))


def largest_power_reference(limit):
    found = []
    for value in range(limit - 1, 0, -1):
        pairs = []
        for base in range(2, math.isqrt(value) + 1):
            power, exponent = base * base, 2
            while power <= value:
                if power == value:
                    pairs.append((base, exponent))
                power *= base
                exponent += 1
        if pairs:
            return (value, len(set(pairs)))
    return (0, -1) if limit == 1 else (1, -1)


def largest_power_mutant(limit):
    if limit <= 4:
        return (1, -1)
    return (int(math.sqrt(limit - 1)) ** 2, 1)


LARGEST_POWER = [("one", (1,)), ("small", (4,)), ("sixteen_boundary", (16,)),
                 ("multiple_representations", (65,)), ("ordinary", (30,))]


def cheese_oracle(wheels):
    return math.ceil(wheels * 6 / 10 / 60) * 8.75 * 4


def cheese_reference(wheels):
    paid_hours_each = math.ceil((wheels / 10 * 6) / 60)
    return 4 * paid_hours_each * 8.75


def cheese_mutant(wheels):
    return math.ceil(wheels * 6 / 10 / 60) * 8.75


CHEESE = [("one_hour", (100,)), ("round_up", (101,)), ("two_hours", (200,))]


def pattern_oracle(size):
    start = size % 10
    return "\n".join("".join(str((start - offset) % 10) for offset in range(row + 1))
                      + str((start - row) % 10) * (size - row - 1)
                      for row in range(size))


def pattern_reference(size):
    rows = []
    for row in range(size):
        prefix = [(size - column) % 10 for column in range(row + 1)]
        rows.append("".join(map(str, prefix + [prefix[-1]] * (size - row - 1))))
    return "\n".join(rows)


def pattern_mutant(size):
    current = size - 1
    rows = []
    for row in range(size):
        rows.append("".join(str((current - col) % 10) for col in range(row + 1)).ljust(size, str(current % 10)))
        current -= 1
    return "\n".join(rows)


PATTERN = [("small", (3,)), ("eight", (8,)), ("wrap", (17,))]


def _operator_value(operators):
    total, current = 0, 1
    for digit, operator in zip(range(2, 10), operators):
        if operator == "":
            current = current * 10 + digit if current >= 0 else current * 10 - digit
        else:
            total += current
            current = digit if operator == "+" else -digit
    return total + current


def operator_oracle(target):
    counts = [sum(operator != "" for operator in operators)
              for operators in itertools.product(("", "+", "-"), repeat=8)
              if _operator_value(operators) == target]
    return min(counts) if counts else None


def operator_reference(target):
    best = None
    def visit(next_digit, total, current, count):
        nonlocal best
        if best is not None and count >= best:
            return
        if next_digit == 10:
            if total + current == target:
                best = count
            return
        visit(next_digit + 1, total,
              current * 10 + next_digit if current >= 0 else current * 10 - next_digit,
              count)
        visit(next_digit + 1, total + current, next_digit, count + 1)
        visit(next_digit + 1, total + current, -next_digit, count + 1)
    visit(2, 0, 1, 0)
    return best


def operator_mutant(target):
    counts = []
    for prefix in itertools.product(("", "+", "-"), repeat=7):
        operators = prefix + ("",)
        if _operator_value(operators) == target:
            counts.append(sum(operator != "" for operator in prefix))
    return min(counts) if counts else None


OPERATOR = [("all_concat", (123456789,)), ("requires_last_gap", (3,)),
            ("classic", (100,)), ("zero", (0,))]


def _prime_factors(value):
    factors = []
    divisor = 2
    while divisor * divisor <= value:
        while value % divisor == 0:
            factors.append(divisor)
            value //= divisor
        divisor += 1
    if value > 1:
        factors.append(value)
    return factors


def primefac_oracle(value):
    if isinstance(value, float) and not value.is_integer():
        return []
    number = abs(int(value))
    factors = _prime_factors(number)
    if len(factors) < 2:
        return []
    return [number // min(factors), max(factors)]


def primefac_reference(value):
    if value != int(value):
        return []
    number = abs(int(value))
    divisors = [candidate for candidate in range(2, number)
                if number % candidate == 0]
    primes = [candidate for candidate in range(2, number + 1)
              if number % candidate == 0 and all(candidate % d for d in range(2, math.isqrt(candidate) + 1))]
    return [max(divisors), max(primes)] if divisors and primes else []


def primefac_mutant(value):
    if isinstance(value, float) and not value.is_integer():
        return []
    number = abs(int(value))
    factors = _prime_factors(number)
    return [number - 1, max(factors)] if len(factors) > 1 else []


PRIMEFAC = [("composite", (12,)), ("square", (49,)), ("negative", (-18,)),
            ("integral_float", (20.0,)), ("fraction", (20.5,)), ("prime", (13,))]


def word_wrap_oracle(text, limit):
    words = text.split()
    lines, current = [], ""
    for original in words:
        word = original
        if current and len(current) + 1 + len(word) <= limit:
            current += " " + word
            continue
        if current and len(word) > limit:
            available = limit - len(current) - 1
            if available > 0:
                current += " " + word[:available]
                word = word[available:]
            lines.append(current)
            current = ""
        elif current:
            lines.append(current)
            current = ""
        while len(word) > limit:
            lines.append(word[:limit])
            word = word[limit:]
        current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def word_wrap_reference(text, limit):
    lines = []
    for word in text.split():
        if not lines or len(lines[-1]) == limit:
            lines.append("")
        while word:
            line = lines[-1]
            separator = " " if line else ""
            capacity = limit - len(line) - len(separator)
            if len(word) <= capacity:
                lines[-1] = line + separator + word
                break
            if len(word) <= limit:
                lines.append("")
                continue
            if capacity:
                lines[-1] = line + separator + word[:capacity]
                word = word[capacity:]
            lines.append("")
    return "\n".join(lines)


def word_wrap_mutant(text, limit):
    lines, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > limit:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        lines.append(current)
    return "--".join(lines)


WORD_WRAP = [("ordinary", ("The quick brown fox", 10)),
             ("split_and_fill", ("a longword", 6)),
             ("long_first", ("abcdefgh ij", 4)), ("empty", ("", 5))]


DEFINITIONS = {
    357: (MEMESORTING, memesorting_oracle, memesorting_reference, memesorting_mutant, "select the receiver whose keyword first completes as an ordered subsequence"),
    358: (ANY_ODD, any_odd_oracle, any_odd_reference, any_odd_mutant, "test zero-based odd bit positions with the 0xAAAAAAAA mask"),
    360: (ODD_HEAVY, odd_heavy_oracle, odd_heavy_reference, odd_heavy_mutant, "require at least one odd and every odd value greater than every even value"),
    361: (SUBSTITUTION, substitution_oracle, substitution_reference, substitution_mutant, "require equal-length strings with a bijection between their character patterns"),
    364: (MAX_BALL, max_ball_oracle, max_ball_reference, max_ball_mutant, "convert km/h to m/s and return the nearest tenth-second sample at the ballistic apex"),
    366: (SHIFT_LEFT, shift_left_oracle, shift_left_reference, shift_left_mutant, "delete from the left of both strings until their longest common suffix remains"),
    373: (ELECTIONS, elections_oracle, elections_reference, elections_mutant, "count candidates who can become the unique maximum after receiving all remaining votes"),
    374: (FIND, find_oracle, find_reference, find_mutant, "return the leftmost longest adjacent pair of maximal alternating punctuation runs"),
    377: (LIQUIDS, liquids_oracle, liquids_reference, liquids_mutant, "sort every liquid cell globally by density and refill the original rectangular shape row-major"),
    378: (REINDEER, reindeer_oracle, reindeer_reference, reindeer_mutant, "perform a stable ascending sort using only the final name token"),
    381: (SPINNING, spinning_oracle, spinning_reference, spinning_mutant, "start both rings at zero, then decrement/wrap inner and increment/wrap outer before testing equality"),
    387: (LARGEST_POWER, largest_power_oracle, largest_power_reference, largest_power_mutant, "find the largest perfect power below n and count distinct base/exponent representations, with stated small-n sentinels"),
    392: (CHEESE, cheese_oracle, cheese_reference, cheese_mutant, "convert team output to whole paid hours and pay each of the four staff at 8.75 per hour"),
    396: (PATTERN, pattern_oracle, pattern_reference, pattern_mutant, "start at the units digit of n, extend the descending prefix each row and repeat its last digit to width n"),
    397: (OPERATOR, operator_oracle, operator_reference, operator_mutant, "search all eight gaps between digits 1 through 9 and minimize non-concatenation operators"),
    398: (PRIMEFAC, primefac_oracle, primefac_reference, primefac_mutant, "reject fractional inputs and return the largest proper divisor plus largest prime factor for composite absolute integers"),
    399: (WORD_WRAP, word_wrap_oracle, word_wrap_reference, word_wrap_mutant, "greedily fill lines, split overlong words to use available capacity and emit real newline characters"),
}


REJECTIONS = {
    382: ("the statement delegates the function's behavior to unavailable test cases and gives no input/output rule", "contract_delegates_semantics_to_unavailable_tests"),
    395: ("the statement refers to a damage formula but the captured contract contains no formula", "contract_under_specified_damage_formula"),
}


def build_group_g_batch(inventory, bank, previous, *, batch_number,
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
