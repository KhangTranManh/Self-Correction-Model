"""Independent contract-only A0 review definitions for frozen positions in group B.

Only locally authored trusted functions are executable. Recovered output was read
for mutant understanding but is never imported or executed. No artifact is written.
"""

from __future__ import annotations

import math
import re

from a0_cpu_review_common import build_review_batch
from audit_a0_cpu_review_batch6 import _evidence

POSITIONS = (87, 90, 99, 101, 104, 105, 106, 107, 110, 111,
             113, 116, 117, 127, 128, 132, 133, 137, 138, 142)


def short_form_oracle(s):
    return ''.join(c for i, c in enumerate(s)
                   if i in (0, len(s) - 1) or c.lower() not in 'aeiou')


def short_form_reference(s):
    if len(s) < 3:
        return s
    middle = s[1:-1]
    return s[0] + re.sub('[aeiou]', '', middle, flags=re.I) + s[-1]


def short_form_mutant(s):
    vowels = set('aeiouAEIOU')
    result = []
    for i, char in enumerate(s):
        if char not in vowels and (i != 0 and i != len(s) - 1):
            continue
        result.append(char)
    return ''.join(result)


def drop_cap_oracle(s):
    return re.sub(r'[^ ]+', lambda m: m.group().capitalize()
                  if len(m.group()) > 2 else m.group(), s)


def drop_cap_reference(s):
    out = []
    for part in re.split(r'( +)', s):
        out.append(part[:1].upper() + part[1:].lower()
                   if len(part) > 2 and not part.isspace() else part)
    return ''.join(out)


def drop_cap_mutant(s):
    return ' '.join(w.capitalize() if len(w) > 2 else w for w in s.split())


def palin_oracle(a, b):
    half = str(10 ** ((a - 1) // 2) + b - 1)
    return int(half + (half[::-1] if a % 2 == 0 else half[-2::-1]))


def palin_reference(a, b):
    first = 10 ** ((a - 1) // 2)
    seed = first + b - 1
    digits = list(str(seed))
    reflected = digits[:a // 2][::-1]
    return int(''.join(digits + reflected))


def palin_mutant(a, b):
    length = 10 ** (a // 2)
    base = 10 ** (a % 2)
    seed = length * (b - 1) + base
    return int(f'{seed}{str(seed)[::-1]}')


def binary_cleaner_oracle(seq):
    return ([x for x in seq if x < 2], [i for i, x in enumerate(seq) if x > 1])


def binary_cleaner_reference(seq):
    bits, positions = [], []
    for index in range(len(seq)):
        if seq[index] == 0 or seq[index] == 1:
            bits.append(seq[index])
        else:
            positions.append(index)
    return bits, positions


def binary_cleaner_mutant(seq):
    cleaned, indices = [], []
    for i, x in enumerate(seq):
        if x < 2:
            cleaned.append(0)
        else:
            cleaned.append(1)
            indices.append(i)
    return cleaned, indices


def sumin_oracle(n):
    return sum(k * k for k in range(1, n + 1))


def sumin_reference(n):
    return n * (n + 1) * (2 * n + 1) // 6


def sumin_mutant(n):
    return n * n * (n + 1) // 2


def kontti_oracle(sentence):
    out = []
    for word in sentence.split(' '):
        hit = next((i for i, c in enumerate(word) if c in 'aeiouy'), None)
        out.append(word if hit is None else 'ko' + word[hit + 1:] + '-' + word[:hit + 1] + 'ntti')
    return ' '.join(out)


def kontti_reference(sentence):
    def transform(word):
        match = re.search('[aeiouy]', word)
        if match is None:
            return word
        prefix, rest = word[:match.end()], word[match.end():]
        return 'ko' + rest + '-' + prefix + 'ntti'
    return ' '.join(map(transform, sentence.split(' ')))


def kontti_mutant(s):
    def swap_until_vowel(word):
        for i, char in enumerate(word):
            if char.lower() in 'aeiouy':
                return word[i:] + word[:i]
        return word
    return ' '.join(swap_until_vowel(word + '-kontti') for word in s.split())


def hamster_me_oracle(code, message):
    seeds = sorted(set(code))
    columns = {c: [] for c in seeds}
    current = seeds[-1]
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    ordered = alphabet[alphabet.index(seeds[0]):] + alphabet[:alphabet.index(seeds[0])]
    for letter in ordered:
        if letter in columns:
            current = letter
        else:
            columns[current].append(letter)
    lookup = {letter: seed + str(i)
              for seed, run in columns.items()
              for i, letter in enumerate([seed] + run, 1)}
    return ''.join(lookup[c] for c in message)


def hamster_me_reference(code, message):
    first = sorted(set(code))
    table = {}
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        prior = [c for c in first if c <= letter]
        root = prior[-1] if prior else first[-1]
        if letter in first:
            distance = 1
        elif prior:
            distance = sum(root < chr(i) <= letter and chr(i) not in first
                           for i in range(ord('a'), ord('z') + 1)) + 1
        else:
            distance = sum(c > root and c not in first for c in 'abcdefghijklmnopqrstuvwxyz') + ord(letter) - ord('a') + 2
        table[letter] = root + str(distance)
    return ''.join(table[c] for c in message)


def hamster_me_mutant(code, message):
    lookup = {}
    row = col = 1
    for char in code + ''.join(c for c in 'abcdefghijklmnopqrstuvwxyz' if c not in code):
        lookup[char] = str(row) + str(col)
        col += 1
        if col > 7:
            col, row = 1, row + 1
    return ''.join(lookup[c] for c in message)


def subsequence_oracle(a, b):
    counts = [1] + [0] * len(a)
    for char in b:
        for i in range(len(a) - 1, -1, -1):
            if a[i] == char:
                counts[i + 1] += counts[i]
    return counts[-1]


def subsequence_reference(a, b):
    def count(prefix, start):
        if prefix == len(a):
            return 1
        return sum(count(prefix + 1, j + 1)
                   for j in range(start, len(b)) if b[j] == a[prefix])
    return count(0, 0)


def subsequence_mutant(a, b):
    return sum(b[i:i + len(a)] == a for i in range(len(b) - len(a) + 1))


def reverse_words_oracle(text):
    return re.sub(r'[^ ]+', lambda m: m.group()[::-1], text)


def reverse_words_reference(text):
    runs, run, out = [], [], []
    for c in text:
        if c == ' ':
            out.extend(reversed(run))
            run = []
            out.append(c)
        else:
            run.append(c)
    out.extend(reversed(run))
    return ''.join(out)


def reverse_words_mutant(text):
    return ' '.join(word[::-1] for word in text.split())


def could_be_oracle(original, another):
    if not original or not another or not another.strip():
        return False
    source, candidate = original.split(), another.split()
    return bool(candidate) and all(candidate.count(w) <= source.count(w) for w in set(candidate))


def could_be_reference(original, another):
    if not original or not another or not another.strip():
        return False
    remaining = original.split()
    for word in another.split():
        if word not in remaining:
            return False
        remaining.remove(word)
    return True


def could_be_mutant(original, another):
    if not original or not another:
        return False
    return all(w in set(original.lower().split()) for w in another.lower().split())


def char_concat_oracle(word):
    return ''.join(word[i] + word[-i - 1] + str(i + 1)
                   for i in range(len(word) // 2))


def char_concat_reference(word):
    left, right = 0, len(word) - 1
    out = []
    while left < right:
        out.extend((word[left], word[right], str(left + 1)))
        left += 1
        right -= 1
    return ''.join(out)


def char_concat_mutant(word):
    result = ''.join(word[i] + word[-i - 1] + str(i + 1)
                     for i in range(len(word) // 2))
    return result + (word[len(word) // 2] if len(word) % 2 else '')


def jumbled_string_oracle(s, n):
    if not s:
        return s
    seen = {s: 0}
    values = [s]
    while True:
        nxt = values[-1][::2] + values[-1][1::2]
        if nxt in seen:
            first = seen[nxt]
            return values[n] if n < len(values) else values[first + (n - first) % (len(values) - first)]
        seen[nxt] = len(values)
        values.append(nxt)
        if len(values) == n + 1:
            return values[-1]


def jumbled_string_reference(s, n):
    indices = tuple(range(len(s)))
    states = [indices]
    while True:
        old = states[-1]
        new = old[::2] + old[1::2]
        if new == indices:
            return ''.join(s[i] for i in states[n % len(states)])
        states.append(new)
        if len(states) == n + 1:
            return ''.join(s[i] for i in new)


def jumbled_string_mutant(s, n):
    chars = list(s)
    while n > 0:
        if n % 2:
            chars = chars[::2] + chars[1::2]
        n //= 2
        if n > 0:
            chars = chars[1::2] + chars[::2]
    return ''.join(chars)


def sequence_oracle(n):
    # Block k contains the concatenation 1..k; binary-search the block.
    def prefix_digits(k):
        total = 0
        power = 1
        digits = 1
        while power <= k:
            end = min(k, power * 10 - 1)
            total += (end - power + 1) * digits
            power *= 10
            digits += 1
        return total
    def cumulative(k):
        total = 0
        power = 1
        digits = 1
        while power <= k:
            end = min(k, power * 10 - 1)
            count = end - power + 1
            # Each integer j contributes its digits to blocks j..k.
            total += digits * count * (2 * (k + 1) - power - end) // 2
            power *= 10
            digits += 1
        return total
    hi = 1
    while cumulative(hi) < n:
        hi *= 2
    lo = 0
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if cumulative(mid) < n:
            lo = mid
        else:
            hi = mid
    offset = n - cumulative(hi - 1)
    power = 1
    while True:
        count = min(hi, power * 10 - 1) - power + 1
        span = count * len(str(power))
        if offset <= span:
            width = len(str(power))
            value = power + (offset - 1) // width
            return int(str(value)[(offset - 1) % width])
        offset -= span
        power *= 10


def sequence_reference(n):
    # Digit positions of j recur in every block k >= j.
    def upto(k):
        acc = 0
        start = 1
        width = 1
        while start <= k:
            last = min(k, start * 10 - 1)
            count = last - start + 1
            acc += width * count * (2 * (k + 1) - start - last) // 2
            start *= 10
            width += 1
        return acc
    left, right = 0, 1
    while upto(right) < n:
        right *= 2
    while left + 1 < right:
        middle = (left + right) // 2
        if upto(middle) >= n:
            right = middle
        else:
            left = middle
    position = n - upto(left) - 1
    start, width = 1, 1
    while start <= right:
        last = min(right, start * 10 - 1)
        span = (last - start + 1) * width
        if position < span:
            return int(str(start + position // width)[position % width])
        position -= span
        start *= 10
        width += 1
    raise AssertionError('block digit not found')


def sequence_mutant(n):
    # Recovered behavior models a different, ordinary Champernowne sequence.
    n -= 1
    width = 1
    while True:
        first = 10 ** (width - 1)
        span = 9 * first * width
        if n < span:
            return int(str(first + n // width)[n % width])
        n -= span
        width += 1


def _sieve(limit):
    flags = bytearray(b'\x01') * (limit + 1)
    flags[:2] = b'\x00\x00'
    for p in range(2, math.isqrt(limit) + 1):
        if flags[p]:
            flags[p * p:limit + 1:p] = b'\x00' * ((limit - p * p) // p + 1)
    return [i for i, flag in enumerate(flags) if flag]


def dominant_oracle(a, b):
    primes = _sieve(b)
    positions = set(_sieve(len(primes)))
    return sum(p for index, p in enumerate(primes, 1)
               if a <= p <= b and index in positions)


def dominant_reference(a, b):
    def prime(x):
        return x > 1 and all(x % divisor for divisor in range(2, math.isqrt(x) + 1))
    total = position = 0
    for number in range(2, b + 1):
        if prime(number):
            position += 1
            if number >= a and prime(position):
                total += number
    return total


def dominant_mutant(a, b):
    def prime(x):
        return x > 1 and all(x % divisor for divisor in range(2, math.isqrt(x) + 1))
    return sum(number for number in range(a, b + 1) if prime(number - 1))


def mod256_oracle(number):
    return number & 255


def mod256_reference(number):
    return number - 256 * (number // 256)


def mod256_mutant(number):
    return number % 256 if number >= 0 else -(abs(number) % 256)


def cost_of_carpet_oracle(room_width, room_length, roll_width, roll_cost):
    if min(room_width, room_length, roll_width) <= 0:
        return 'error'
    lengths = [along for across, along in ((room_width, room_length), (room_length, room_width))
               if across <= roll_width]
    return round(min(lengths) * roll_width * roll_cost, 2) if lengths else 'error'


def cost_of_carpet_reference(room_width, room_length, roll_width, roll_cost):
    if room_width <= 0 or room_length <= 0 or roll_width <= 0:
        return 'error'
    options = []
    if roll_width >= room_width:
        options.append(room_length * roll_width * roll_cost)
    if roll_width >= room_length:
        options.append(room_width * roll_width * roll_cost)
    if not options:
        return 'error'
    return round(sorted(options)[0], 2)


def cost_of_carpet_mutant(room_width, room_length, roll_width, roll_cost):
    if min(room_width, room_length, roll_width) <= 0:
        return 'error'
    required_length = max(room_width, room_length)
    rolls_needed = (required_length + roll_width - 1) // roll_width
    return round(rolls_needed * roll_cost, 2)


# Each pass is (status, cases, oracle, reference, mutant, contract reason).
# Rejections are ("rejected", reason); no executable oracle is constructed.
DEFINITIONS = {
    87: ('rejected', 'delimiter characters may be both opening and closing across pairs; precedence is unspecified'),
    90: ('passed', [('middle_vowels', ('assault',)), ('endpoints', ('aEi',)),
                    ('consonants', ('rhythm',)), ('upper', ('BEAUTIFUL',)),
                    ('single', ('A',))], short_form_oracle, short_form_reference,
         short_form_mutant, 'only interior a/e/i/o/u are removed, ignoring case'),
    99: ('passed', [('spaces', ('one   space',)), ('edges', ('   space WALK   ',)),
                    ('short_caps', ('I AM HERE',)), ('mixed', ('hELLo aT wORLD',))],
         drop_cap_oracle, drop_cap_reference, drop_cap_mutant,
         'words longer than two are capitalized; all original spaces remain'),
    101: ('passed', [('first_four', (4, 1)), ('second_four', (4, 2)),
                     ('last_four', (4, 90)), ('first_three', (3, 1)),
                     ('tenth_three', (3, 10)), ('first_two', (2, 1))],
          palin_oracle, palin_reference, palin_mutant,
          'b is one-indexed among increasing a-digit palindromes'),
    104: ('passed', [('mixed', ([0, 1, 2, 1, 5, 6, 2, 1, 1, 0],)),
                     ('all_binary', ([1, 0, 1],)), ('all_other', ([2, 3],)),
                     ('empty', ([],))], binary_cleaner_oracle,
          binary_cleaner_reference, binary_cleaner_mutant,
          'retain zero and one values and separately record indices above one'),
    105: ('passed', [('one', (1,)), ('two', (2,)), ('six', (6,)),
                     ('forty_seven', (47,))], sumin_oracle, sumin_reference,
          sumin_mutant, 'sum of min(x,y) over the specified integer square'),
    106: ('passed', [('tame', ('tame',)), ('fruity', ('fruity',)),
                     ('no_vowel', ('crwth',)), ('sentence', ('tame fruity',))],
          kontti_oracle, kontti_reference, kontti_mutant,
          'swap prefixes through first vowel with ko of -kontti for each word'),
    107: ('rejected', 'placeholder is blank in the rule and edge cases while examples disagree about # insertion'),
    110: ('passed', [('code_exact', ('hamster', 'hamster')),
                     ('help', ('hamster', 'helpme')),
                     ('missing_a', ('hmster', 'abcde')),
                     ('duplicates', ('aaab', 'abcz'))],
          hamster_me_oracle, hamster_me_reference, hamster_me_mutant,
          'sorted unique code letters head alphabetic runs, wrapping before first key'),
    111: ('passed', [('four', ('zaz', 'zazapulz')),
                     ('repeated', ('aa', 'aaaa')),
                     ('none', ('ab', 'bbb')),
                     ('exact', ('abc', 'abc'))],
          subsequence_oracle, subsequence_reference, subsequence_mutant,
          'example establishes position-distinct subsequences, not contiguous matches'),
    113: ('passed', [('single_spaces', ('This is an example!',)),
                     ('double', ('double  spaces',)),
                     ('edges', ('  abc  de ',)), ('empty', ('',))],
          reverse_words_oracle, reverse_words_reference, reverse_words_mutant,
          'reverse nonspace runs without changing any spaces'),
    116: ('passed', [('reorder', ('Carlos Ray Norris', 'Norris Carlos')),
                     ('missing', ('Carlos Ray Norris', 'Norris!')),
                     ('case', ('Carlos Ray', 'carlos')),
                     ('duplicate', ('Ana Bob', 'Ana Ana')),
                     ('empty', ('Ana', ' '))], could_be_oracle,
          could_be_reference, could_be_mutant,
          'case-sensitive constituent name combinations, not more copies than original'),
    117: ('passed', [('even', ('abcdef',)), ('odd', ('abc!def',)),
                     ('one', ('a',)), ('empty', ('',)),
                     ('ten', ('abcdefghij',))], char_concat_oracle,
          char_concat_reference, char_concat_mutant,
          'pair outside-in characters and number pairs; discard odd center'),
    127: ('passed', [('one', ('qwertyuio', 1)),
                     ('two', ('qwertyuio', 2)),
                     ('zero', ('abcd', 0)),
                     ('large', ('abcdefg', 1_000_000_007)),
                     ('single', ('x', 25))], jumbled_string_oracle,
          jumbled_string_reference, jumbled_string_mutant,
          'repeat the even-index then odd-index permutation n times'),
    128: ('passed', [('one', (1,)), ('two', (2,)), ('three', (3,)),
                     ('block_boundary', (10,)), ('two_digit', (60,)),
                     ('large', (10**18,))], sequence_oracle,
          sequence_reference, sequence_mutant,
          'one-indexed digit in concatenation of successive 1..k blocks'),
    132: ('passed', [('low', (0, 10)), ('middle', (6, 20)),
                     ('single', (3, 3)), ('none', (4, 4)),
                     ('wide', (50, 150))], dominant_oracle,
          dominant_reference, dominant_mutant,
          'sum primes in inclusive range whose one-based prime index is prime'),
    133: ('rejected', 'the 2/888 example repeats a single digit in multiple positions rather than specifying periodic notation'),
    137: ('rejected', 'word boundaries and capitalization of multiword patterns are not determined by the replacement list'),
    138: ('passed', [('positive', (258,)), ('negative', (-258,)),
                     ('negative_multiple', (-512,)), ('zero', (0,)),
                     ('edge', (-10000,))], mod256_oracle,
          mod256_reference, mod256_mutant,
          'explicit Python remainder behavior for negative operands'),
    142: ('passed', [('both_fit', (2.0, 3.0, 4.0, 5.0)),
                     ('one_fit', (5.0, 2.0, 3.0, 7.0)),
                     ('neither', (4.0, 5.0, 3.0, 2.0)),
                     ('zero', (0.0, 3.0, 4.0, 5.0)),
                     ('fractional', (2.5, 4.2, 3.0, 3.7))],
          cost_of_carpet_oracle, cost_of_carpet_reference,
          cost_of_carpet_mutant,
          'one piece cut from full-width roll; choose feasible room orientation at minimum area cost'),
}


def check_trusted_cases():
    """Validate each pass with batch8-compatible fail-closed evidence checks."""
    assert tuple(DEFINITIONS) == POSITIONS
    return {position: _evidence(cases, oracle, reference, mutant)
            for position, (status, *rest) in DEFINITIONS.items()
            if status == 'passed'
            for cases, oracle, reference, mutant, reason in (rest,)}


def build_group_b_batch(inventory, bank, previous, *, batch_number,
                        positions, functions, cumulative_examined,
                        cumulative_ready, cumulative_rejected):
    definitions = {}
    rejections = {}
    for position in positions:
        status, *rest = DEFINITIONS[position]
        if status == 'passed':
            definitions[position] = tuple(rest)
        else:
            rejections[position] = (rest[0], "contract_under_specified")
    return build_review_batch(
        inventory, bank, previous, batch_number=batch_number,
        positions=list(positions), functions=list(functions),
        definitions=definitions, rejections=rejections,
        cumulative_examined=cumulative_examined,
        cumulative_ready=cumulative_ready,
        cumulative_rejected=cumulative_rejected,
    )
