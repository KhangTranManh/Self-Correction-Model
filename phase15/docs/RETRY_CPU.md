# CPU investigation: when should the system generate another answer?

Recorded 2026-11-02. Exploratory analysis of previously opened data. This is
a separate controller investigation; it does not change P15-A1's episode
requirements or declare its go/no-go.

## Result

Agreement is a useful stopping baseline. We cannot yet assess whether a
judge can identify that both candidates are wrong: all 38 saved Phase 15
judge calls were selected to have exactly one correct candidate.

On the Phase 15 screen, the first answer is correct on 52/100 problems.
The first two answers agree on 34 problems, of which 33 are correct. Among
the 66 disagreements, both answers fail the project's verifier on 28
(42.4%). Among agreements, both fail on 1/34 (2.9%). Correctness here means
the project's answer parser and symbolic comparison, not human assessment
of the mathematical reasoning.

Only archived Phase 8 **v1** first-pass outputs are available locally. They
are complete and match their saved summary hashes. They are different from
the final **v2** experiment reported in Phase 8's final report: v1 first,
repeat and greedy correctness counts are 302/288/309; published v2 counts
are 283/309/307. The following replay describes v1 only and does not revise
the published v2 results.

| Policy on 400 Phase 8 v1 problems | Correct | Mean calls | Mean output tokens |
|---|---:|---:|---:|
| First answer | 302/400 (75.5%) | 1 | 202.05 |
| Second sample alone | 288/400 (72.0%) | 1 | 194.10 |
| Greedy alone | 309/400 (77.25%) | 1 | 193.59 |
| Vote over two sampled answers plus one greedy answer | 320/400 (80.0%) | 3 | 589.73 |
| Stop when the two samples agree; otherwise generate greedy and vote | 320/400 (80.0%) | 2.3475 | 478.72 |
| Stop on agreement; otherwise replace with greedy | 318/400 (79.5%) | 2.3475 | 478.72 |

The stopping rule saves 21.75% of calls and 18.82% of output tokens against
generating all three answers. It produces exactly the same decisions as
the three-answer vote: two agreeing answers already form a majority. This
is a structural property, not evidence of a learned controller. These
three answers have mixed decoding temperatures, so this is not standard
three-sample vote@3. Input tokens, judge calls and measured latency are not
included in these cost estimates.

The gated vote fixes 19 first-answer errors and harms one correct answer.
Its paired exploratory 95% bootstrap interval for improvement over the
first answer is +2.5 to +6.75 percentage points (2,000 source resamples,
seed 20261102). This is not a fresh confirmatory test or a claim against
five-answer voting; no five-answer replay is possible with these files.

## Where a controller might add value

In Phase 8 v1 there are 60 problems where the first two answers both fail.
Of those, 49 have disagreeing answers; the third answer is correct on
14/49. A plain vote cannot recover those 14: the three answers are distinct
and its frozen tie rule keeps the first. Even with agreement, 11 problems
have a shared wrong answer; the third is correct on 3/11, but the majority
keeps the repeated wrong answer. These counts show opportunities, not an
achievable policy: identifying and selecting those answers is the missing
capability.

Disagreement alone ranks both-wrong risk with AUC 0.776 in Phase 8 v1 and
0.715 in Phase 15. After restricting to disagreements, maximum answer length
has only AUC 0.609 and 0.551 respectively; hitting the cap is near chance.
Non-numeric parsing has disagreement-only AUC 0.511 in Phase 8 and 0.671 in
Phase 15. The latter may mainly identify format failures: a robust parser
is a necessary comparator before claiming semantic error detection. No
threshold was fitted and no learned controller was trained.

## Missing data and next GPU experiment

Phase 12-14 raw per-problem judge outputs and hidden states are absent from
their documented local mirror locations. Summary tables cannot substitute
for those inputs. Phase 15 has 38 one-right judge cases and zero judged
both-wrong cases, so judge sensitivity, specificity and AUC for both-wrong
detection are not identifiable from the current audit.

The next useful GPU task is a development feasibility run on the existing
100 Phase 15 screen problems:

1. Use the pinned Qwen2.5-7B-Instruct NF4 environment and saved candidate
   pair, and generate an additional three independent samples per problem
   to provide five candidates for replay. Keep each generation in its own
   context and record seeds and token costs.
2. Judge every initial pair in both orders, including agreeing, both-wrong,
   both-right and one-right cases. Do not select judge inputs using gold.
   Define an explicit A/B/NEITHER verdict; treat NEITHER as a request for
   another attempt, not a verified assertion that both are wrong.
3. Keep gold references out of generator, judge and controller prompts.
   Use references only for offline scoring. Record input/output token costs
   for every generation and judge call.
4. Compare first-answer, greedy, five-candidate plurality, agreement-gated
   three-candidate plurality, and judge-triggered retry at matched total
   token budgets. Report fixes and harms, both-wrong detection, rejection
   of correct pairs, and whether newly correct candidates are selected.
5. Freeze the resulting policy before testing on fresh, source-disjoint
   data. These 100 already opened problems are development only. Fitting
   an abstention threshold on them is not an evaluation result.

This requires 300 additional solution calls and 200 new judge calls for
development, before any fresh validation run. Runtime needs a short
throughput check; it is not guaranteed to fit the old seven-hour wall.
An abstention prompt must be specified before generation. This document
is a proposed next experiment, not an amendment silently changing P15-A1.

## Reproduction

```powershell
py -3.13 phase15/scripts/analyze_retry_cpu.py --out outputs/phase15_kaggle_pilot_v1/v2/retry_cpu.json
```

The JSON records input SHA-256 values, validates all three Phase 8 stage
hashes and the Phase 15 audit hash, checks complete Phase 8 source coverage,
and asserts that gated voting reproduces full three-answer voting exactly.
The original artifacts and historical transcripts are unchanged.
