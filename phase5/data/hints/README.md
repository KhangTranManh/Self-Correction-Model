# Phase 5 hint feasibility

The primary Phase 5 behavioral comparison uses only `neutral` and `status`.
Both conditions cover all 480 frozen sources. Status truthfully discloses the
initial correctness label, so it measures assisted repair rather than autonomous
error detection.

`v1/` is a rejected parser audit. Train/development inspection found false
positives from percentages, chained equalities, approximation, LaTeX, and
algebraic expressions. No protected row was manually inspected while revising
the rule.

`v2/` is frozen. Its conservative rule accepts only an explicitly numbered
model step containing a literal, self-contained equality with one numeric
operation and one equals sign. It does not use the reference answer or verifier
trace to locate the step. Coverage for initially wrong rows is:

| Split | Eligible | Wrong rows |
|---|---:|---:|
| Train | 1 | 120 |
| Development | 0 | 40 |
| Protected test | 0 | 80 |

Location/type is therefore infeasible and disabled for this split. Do not
weaken the rule, refill rows, or run those conditions after seeing this result.
The controlling decision is locked in `../protocol/review_protocol_v1_lock.json`.
