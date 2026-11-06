# C-upper-inclusive: worked fixture and scoring contract

This is a **synthetic verifier fixture**, not a Phase 16 model result or a
member of any teaching, probe, control, or confirmation pool. It makes the
first candidate error type reviewable before a source pool exists.

## Example task

> Write `count_in_closed_range(values: list[int], lower: int, upper: int) -> int`.
> Assume `lower <= upper`. Return the number of entries `x` in `values`
> satisfying `lower <= x <= upper`. Count duplicates separately. Do not
> modify the input list.

Correct predicate: `lower <= x <= upper`.
Illustrative wrong predicate: `lower <= x < upper`.

| Test kind | `values, lower, upper` | Oracle | Wrong predicate |
|---|---|---:|---:|
| ordinary interior | `[2], 1, 5` | 1 | 1 |
| lower boundary | `[1], 1, 5` | 1 | 1 |
| outside | `[0, 6], 1, 5` | 0 | 0 |
| empty | `[], 1, 5` | 0 | 0 |
| single point | `[5], 5, 5` | 1 | 0 |
| duplicate upper | `[5, 5], 1, 5` | 2 | 0 |
| pair 1 anchor | `[2], 1, 5` | 1 | 1 |
| pair 1 add `R` | `[2, 5], 1, 5` | 2 | 1 |
| pair 2 anchor | `[1, 3], 1, 7` | 2 | 2 |
| pair 2 add `R` | `[1, 3, 7], 1, 7` | 3 | 2 |

The wrong predicate has a clear pattern: it passes the interior, lower and
outside checks; the two anchors pass; adding a value exactly at the upper
bound changes the oracle result but leaves the candidate result unchanged in
both pairs. The two pairs use distinct upper bounds. The scorer labels this
`upper_boundary_signature` and `full_correct = false`.

The signature is an **observable test pattern**, not proof that the program
literally contains `< upper`. A program failing only one upper pair, or also
failing an ordinary interior case, receives `other_failure`. Syntax errors,
runtime errors and timeouts have separate labels. A program is fully correct
only if every frozen test passes.

## Transcript interface

`../scripts/score_boundary.py` accepts a JSON object with `task_id`, `variant`
(`count`, `sum`, or `select`), `execution_status`, and, for status `ok`, a list
of test cases. Each case has `case_id`, `kind`, trusted `expected`, and runner
reported `actual`. Upper-pair cases also have `pair_id` and integer `upper`.
For `sum`, the added case records nonzero `added_value`; for `select`, it
records a unique `added_item` ID and the selection order is input order.

The scorer validates the full case coverage and oracle pair relationships,
then reports `full_correct`, `label`, `signature_pairs`, passed cases and total
cases. Expected values must come from the locked oracle manifest. The scorer
never executes candidate code and must not be fed values supplied by the
model as if they were oracle outputs. Before screening a real model, build and
verify a separate isolated runner with network, filesystem, time and memory
limits; the existing Phase 1 research runner is insufficient as a security
boundary for arbitrary model code.

Mechanics were checked with ten synthetic fixtures under
`../tests/test_score_boundary.py`. They are verifier tests only and provide no
evidence about error prevalence or model learning.
