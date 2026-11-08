# P16-R0 A0 CPU review log v1 (2026-11-08)

Status: **review begun in frozen order; zero corrections admitted**.

The first review batch examines A0 positions 0, 2 and 6, the first three
records marked `pending_independent_oracle_review`. It uses task contracts but
does not use published expected values or dataset reference solutions to
derive a verdict. The frozen admission inventory is not modified.

## Batch 1 decisions

| Position | Item | Review result |
|---:|---|---|
| 0 | `taco_train_14050` / `remove_parentheses` | Contract passes. Two independently implemented trusted CPU oracles agree on eight contract-derived cases. A boolean nesting mutant is caught on nested-parentheses cases. Actual model execution, full semantic history, cross-history near-duplicate review and human approval remain pending, so the item is not admitted. |
| 2 | `taco_train_18158` / `order` | Rejected fail-closed at contract review. Exact scheduling, resource overlap and completion semantics are delegated to unavailable example tests; using published tests to infer them would violate the independent-oracle boundary. |
| 6 | `taco_train_24349` / `transpose` | Contract passes, but a dedicated grid oracle is still required to preserve simultaneous-note alignment while fret widths expand or contract. Review stops here before later A0 positions. |

For position 0, the frozen candidate bank already records a singleton
near-duplicate component. A fresh bank-local token 5-gram comparison found a
maximum similarity below 0.011. This is useful automated evidence, not a
substitute for the required cross-history semantic and human review.

The reproducible artifact is
`../results/a0_cpu_review_batch1_v1.json`, built by
`../scripts/audit_a0_cpu_review_batch1.py`. It explicitly records that no
untrusted model code was executed locally and no correction was admitted.

## Next gate

Batch 2 resolves the position-6 blocker, then reviews positions 10 and 12:

| Position | Item | Review result |
|---:|---|---|
| 6 | `taco_train_24349` / `transpose` | Two independent grid implementations agree on seven contract-derived cases covering width expansion/contraction, simultaneous notes, a slide marker and both fret bounds. The trusted linewise model-equivalent mutant fails four cases. |
| 10 | `taco_train_16524` / `identify_weapon` | Two independent mapping implementations agree on all six valid characters and three invalid inputs. The trusted model-equivalent mutant incorrectly prefixes invalid results. |
| 12 | `taco_train_19256` / `closest_pair_tonum` | Brute-force and square-parameterization oracles agree on nine limits. The trusted model-equivalent mutant misses all valid pairs whose largest member is at most 10. |

The hash-chained artifact is `../results/a0_cpu_review_batch2_v1.json`.
Cumulatively, five positions have been examined: four have trusted oracles
ready for remaining gates, one is contract-rejected, and zero are admitted.
No exact recovered model output was executed locally.

Continue at position 16. Positions 0, 6, 10 and 12 still need sandbox
execution of the exact hash-frozen recovered output, full history and
cross-history semantic-near-duplicate review, and human approval. Training,
protected opening and effect claims remain unauthorized.

## Batch 3

Batch 3 reviews positions 16, 21 and 22:

| Position | Item | Review result |
|---:|---|---|
| 16 | `taco_train_04114` / `string_transformer` | Rejected fail-closed. Reversing words does not uniquely determine how unequal leading, trailing and inter-word space runs move, while the contract explicitly requires those cases. Published tests were not used to resolve the ambiguity. |
| 21 | `taco_train_14075` / `case_id` | Regex and parser oracles agree on 16 contract-derived valid and invalid case-style examples. The trusted model-equivalent mutant fails camel cases and malformed separators/casing. |
| 22 | `taco_train_00638` / `diff` | Iterative and vectorized first-maximum oracles agree on seven cases. The trusted model-equivalent mutant incorrectly returns the first pair when every difference is zero. |

Artifact: `../results/a0_cpu_review_batch3_v1.json`, hash-chained to batch 2.
Cumulatively, eight positions have been examined: six are oracle-ready for
remaining gates, two are contract-rejected and zero are admitted. Continue at
position 23. Exact recovered model execution, full semantic history and human
approval remain pending for every oracle-ready record.

## Batch 4

Batch 4 reviews positions 23, 28 and 29. All three contracts pass:

| Position | Item | Review result |
|---:|---|---|
| 23 | `taco_train_17641` / `calculate_scrap` | Closed-form integer ceiling and monotone binary-search oracles agree on seven cases. The trusted model-equivalent mutant floors fractional kilogram requirements. |
| 28 | `taco_train_05174` / `race` | Integer-second and rational-duration oracles agree on seven cases. The trusted model-equivalent mutant does not reduce minutes modulo 60 and computes minutes again in the seconds slot. |
| 29 | `taco_train_11943` / `day_plan` | Integer half-even and Decimal half-even schedule oracles agree on seven cases. The trusted model-equivalent mutant hard-codes ten-minute breaks instead of distributing available time. |

Artifact: `../results/a0_cpu_review_batch4_v1.json`, hash-chained to batch 3.
Cumulatively, eleven positions have been examined: nine are oracle-ready, two
are contract-rejected and zero are admitted. Continue at position 30. All
remaining sandbox/history/semantic/human gates stay pending.

## Batch 5

Batch 5 reviews positions 30, 31 and 36:

| Position | Item | Review result |
|---:|---|---|
| 30 | `taco_train_10757` / `ka_co_ka_de_ka_me` | Vowel-run and regular-expression boundary oracles agree on eight cases covering the required prefix, terminal vowels, vowel clusters and case preservation. The trusted model-equivalent mutant inserts `ka` before rather than after vowel runs and lowercases the input. |
| 31 | `taco_train_02417` / `fisHex` | Direct hexadecimal conversion and parity-reduction oracles agree on seven cases. The trusted model-equivalent mutant XORs character code points instead of hexadecimal values. |
| 36 | `taco_train_25279` / `pseudo_sort` | Rejected fail-closed. Sorting only by first letter does not specify the order of distinct words tied on that letter, and punctuation removal does not specify whether separated fragments join or remain separate tokens. |

Artifact: `../results/a0_cpu_review_batch5_v1.json`, hash-chained to batch 4.
Cumulatively, fourteen positions have been examined: eleven are oracle-ready,
three are contract-rejected and zero are admitted. Continue at position 37.
Exact recovered-output sandbox execution, full semantic history, manual cross-
history near-duplicate review and human approval remain pending for every
oracle-ready record.
