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

## Batch 6

Batch 6 increases the CPU review unit from three to five records and examines
positions 37, 39, 40, 41 and 42 in frozen order:

| Position | Item | Review result |
|---:|---|---|
| 37 | `taco_train_21380` / `parse_fen` | Rejected fail-closed. The contract depicts eight rows but does not uniquely specify the returned row delimiter or trailing-newline policy, so exact serialization cannot be independently certified. |
| 39 | `taco_train_18261` / `diamond` | Width-sequence and row-distance oracles agree on six odd/invalid-size cases. The trusted model-equivalent mutant reverses the diamond widths. |
| 40 | `taco_train_14218` / `gordon` | Character-loop and regex oracles agree on four contract-derived strings. The trusted mutant appends markers incorrectly and removes word separators. |
| 41 | `taco_train_02162` / `finding_k` | Difference-GCD and bounded exhaustive oracles agree on six arrays, including the all-equal sentinel. The trusted mutant confuses distinct-value counting with the modulus condition. |
| 42 | `taco_train_16843` / `change_case` | Manual-parser and regex-parser oracles agree on nine valid/invalid conversions, including acronym boundaries and mixed separators. The trusted mutant handles only fragments of the three grammars. |

Artifact: `../results/a0_cpu_review_batch6_v1.json`, hash-chained to batch 5.
Cumulatively, nineteen positions have been examined: fifteen are oracle-ready,
four are contract-rejected and zero are admitted. Continue at position 43.
All exact-output sandbox, full-history/semantic and human gates remain pending.

## Batch 7

Batch 7 examines positions 43, 44, 47, 48 and 51 in frozen order. All five
contracts pass:

| Position | Item | Review result |
|---:|---|---|
| 43 | `taco_train_09007` / `base_finder` | Unique-digit-set and highest-digit-plus-one oracles agree on five base-encoded sequences (binary through decimal). The trusted model-equivalent mutant computes range length instead of unique digit count. |
| 44 | `taco_train_07486` / `search_substr` | Scan and find-based oracles agree on seven cases covering overlapping, non-overlapping and empty-search behavior. The trusted model-equivalent mutant swaps the overlap/non-overlap step logic. |
| 47 | `taco_train_13728` / `archers_ready` | Short-circuit and iterative oracles agree on six cases including the empty-list edge. The trusted model-equivalent mutant omits the empty-group check (`all([])` returns `True`). |
| 48 | `taco_train_10648` / `combs` | Brute-force offset and exhaustive placement oracles agree on six tooth-collision cases. The trusted model-equivalent mutant uses a greedy scan that miscomputes optimal interleaving. |
| 51 | `taco_train_12930` / `wheres_wally` | Regex and manual-scan oracles agree on nine cases covering standalone, attached, punctuated and apostrophe-prefixed words. The trusted model-equivalent mutant hard-codes specific punctuation patterns and misses start-of-string and apostrophe-before cases. |

Artifact: `../results/a0_cpu_review_batch7_v1.json`, hash-chained to batch 6.
Cumulatively, twenty-four positions have been examined: twenty are oracle-ready,
four are contract-rejected and zero are admitted. Continue at position 53.
All exact-output sandbox, full-history/semantic and human gates remain pending.

## Batch 8

Batch 8 examines positions 53, 56, 58, 60 and 61 in frozen order. Four
contracts pass and one is rejected:

| Position | Item | Review result |
|---:|---|---|
| 53 | `taco_train_04217` / `morse_converter` | Rejected fail-closed. The contract lists digit-to-Morse mappings but does not specify how multi-digit inputs are separated or the exact return type/serialization. |
| 56 | `taco_train_22255` / `uncollapse` | Greedy-scan and regex oracles agree on six collapsed-digit-word strings. The trusted model-equivalent mutant replaces digit words with numeric characters instead of space-separating them. |
| 58 | `taco_train_06251` / `string_merge` | Index and iterative oracles agree on five merge cases. The trusted model-equivalent mutant duplicates the dividing letter using split/rejoin. |
| 60 | `taco_train_10013` / `checkchoose` | Comb-iteration and factorial oracles agree on eight cases including x=0 and no-solution sentinels. The trusted model-equivalent mutant skips x=0 and mishandles m=0. |
| 61 | `taco_train_19936` / `custom_fib` | Sliding-window and offset oracles agree on six sequences. The trusted model-equivalent mutant uses absolute indexes instead of window-relative indexes. |

Artifact: `../results/a0_cpu_review_batch8_v1.json`, hash-chained to batch 7.
Cumulatively, twenty-nine positions have been examined: twenty-four are
oracle-ready, five are contract-rejected and zero are admitted. Continue at
position 62. All exact-output sandbox, full-history/semantic and human gates
remain pending.

## Batch 9

Batch 9 examines positions 62, 64, 66, 68 and 70 in frozen order. Four
contracts pass and one is rejected:

| Position | Item | Review result |
|---:|---|---|
| 62 | `taco_train_03599` / `rank_of_element` | Direct side-count and strict-smaller-plus-equal-before oracles agree on six cases. The frozen mutant incorrectly includes equal elements after index zero. |
| 64 | `taco_train_17176` / `pick_peaks` | Rising-edge and plateau-scan oracles agree on eight cases covering strict peaks, plateau starts and excluded endpoints. The frozen mutant records the falling edge of a plateau as a duplicate peak. |
| 66 | `taco_train_15240` / `blocks` | Counter-layer and repeated-alphabet-pass oracles agree on six cases. The frozen mutant sorts all repeated characters into one block and omits dashes. |
| 68 | `taco_train_10087` / `calc_ms` | Exponentiation and iterative multiplication oracles agree on five levels. The frozen mutant cubes the previous count and divides by 27. |
| 70 | `taco_train_15788` / `find_pattern` | Rejected fail-closed. The statement delegates essential pattern semantics to unavailable test cases and does not define empty or singleton sequences. |

Artifact: `../results/a0_cpu_review_batch9_v1.json`, hash-chained to batch 8.
Cumulatively, thirty-four positions have been examined: twenty-eight are
oracle-ready, six are contract-rejected and zero are admitted. Continue at
position 71. The first 24 ready records have completed sandbox closure; the
four new ready records still need that gate. Full-history/semantic, manual
cross-history and human gates remain pending for all ready records.

## Batch 10

Batch 10 examines positions 71, 73, 74, 76 and 77. Positions 71 (`remove`),
73 (`bits_war`) and 77 (`numberOfSteps`) have agreeing contract-derived
oracle pairs that catch their frozen mutants. Position 74 (`solve`) is
rejected because base factors 2 and 5 and the no-rule result are unspecified.
Position 76 (`button_sequences`) is rejected because simultaneous release and
press precedence is unspecified. Artifact:
`../results/a0_cpu_review_batch10_v1.json`, hash-chained to batch 9.
Cumulatively, 39 positions are reviewed, 31 are oracle-ready, eight are
contract-rejected and zero are admitted. Continue at position 80.
