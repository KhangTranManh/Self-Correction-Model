# Phase 13 preregistration — confirm, constrain, and combine (directions A, B, C)

Written on 2026-10-09 before any Phase 13 generation or training. One
execution lock (`data/execution_lock_v1.json`) covers this file, the holdout,
the direction-B data, and every script. All three directions share one
holdout, one GPU run, and **one** protected opening, so no direction is
designed after seeing another's results.

## Motivation

Phase 12 passed both primary endpoints on SVAMP, but the untrained judge was
already above chance there, invented answers (10.5%) missed their target, and
vote@5 stayed most accurate. Phase 13 asks three questions:

- **A — Is it the method or the dataset?** Re-test the Phase 12 judge, unchanged,
  on a harder set where Phases 9–11 failed.
- **B — Can invented answers be removed?** Train a judge that must end with
  "Verdict: Solution A" or "Verdict: Solution B".
- **C — Can the judge add accuracy on top of voting?** Call it only when a
  three-way vote is not unanimous.

## Data

- **Holdout:** GSM8K **test** problems 750–1318 (`openai/grade-school-math`,
  `test.jsonl`, SHA-256 `3730d312…`). Indices 0–749 are excluded outright
  because Phase 1 evaluated on test problems from offset 150; 8 further
  problems overlapped earlier project files. **561 problems** remain
  (`data/sources_v1/`), all references verified. GSM8K-test difficulty matches
  the GSM8K sets where Phases 9–11 failed.
- **Direction-B training data** (`data/dpo_b_v1_report.json`,
  `outputs/phase13_v1/dpo_b/`): from Phase 10 training-pool judgments, one-right
  pairs only, constrained prompt. Chosen = the first verifier-correct judgment
  plus "Verdict: Solution <right>". Rejected = a judgment that sided with the
  wrong solution plus "Verdict: Solution <wrong>", or a judgment that invented
  a third answer (no valid verdict). Every pair also order-swapped. 1,648 pairs:
  1,470 training (735 A-right, 735 B-right; 776 sided-wrong, 694 invented
  rejections) and 178 validation.

## Models

All judges use the original solver `Kxck/Self_Correction_v1`:

| Judge | Adapter | Prompt |
|---|---|---|
| `base` | none | Phase 9 judge prompt |
| `p12` | Phase 12 order-swapped DPO LoRA (SHA-256 `628982a0…`), unchanged | Phase 9 judge prompt |
| `base_c` | none | constrained prompt (Phase 9 prompt + verdict rule) |
| `p13b` | new direction-B DPO LoRA | constrained prompt |

`p13b` training: identical to Phase 12 (LoRA r 16, alpha 32; DPO beta 0.1;
learning rate 2e-5 cosine, 5% warmup; one epoch; effective batch 16; max
length 3,072; FP16 base, FP32 LoRA; validation every 50 steps), seed
20261009.

## Procedure

1. Five attempts per problem at temperature 0.7: s1 Phase 1 prompt, s2–s5
   blind prompt.
2. Judged pairs: (s1, s2) and (s1, s3) whose final answers differ (gold-free),
   each in both orders, by all four judges (greedy).
3. **Pick** of one judgment: for constrained judges, the solution named by the
   last "Verdict: Solution X" line; otherwise (or without a valid verdict
   line) the solution whose final answer it matches, else "invented".
   **Consistent verdict:** both orders pick the same solution.

## Strategies

- keep (s1); vote@3 (s1–s3); vote@5 (s1–s5); agree-gated (s1 if s1 = s2,
  else vote@3).
- **self_check_J:** s1 if s1 = s2; else judge J's consistent verdict on
  (s1, s2), or vote@3 if inconsistent. Calls: 2, 4, or 5.
- **split_judge_J (direction C):** s1 if s1 = s2 = s3; otherwise judge J's
  consistent verdict on (s1, sk) for the first k in (2, 3) with sk ≠ s1, or
  vote@3 if inconsistent. Calls: 3 or 5.

## Endpoints

| Dir. | Endpoint | Definition | Threshold |
|---|---|---|---|
| **A** | **A-P1** | `p12` consistent accuracy on one-right pairs (baseline: `base`) | > 50%, one-sided binomial p < 0.05 after Holm over {A-P1, A-Sec}, coverage ≥ 30% |
| **A** | **A-P2** | self_check_p12 − vote@k, k = 3 if its mean calls ≤ 3, else 5 | lower 95% bound > −2 points |
| **A** | A-Sec | `p12` single-order accuracy on one-right pairs | > 50%, p < 0.05 after Holm |
| A | diagnostics | `p12` position-2 rate; invented rate | report (Phase 12 targets 40–60%, < 5%) |
| **B** | **B1** | `p13b` invented rate over single-order judgments | < 5% |
| **B** | **B2** | `p13b` consistent accuracy on one-right pairs | > 50%, one-sided binomial p < 0.05, coverage ≥ 30% |
| B | report | `base_c` metrics; `p13b` − `p12` tie-break score (source-clustered CI); position-2 rate | report |
| **C** | **C1** | split_judge_p12 − vote@3 | 95% CI above 0 and exact paired p < 0.05 |
| **C** | **C2** | split_judge_p12 − vote@5 (mean calls reported) | lower 95% bound > −2 points |
| C | report | split_judge_p13b vs vote@5 | report |

Intervals are paired bootstrap 95% (10,000 resamples; source-clustered for
pair-level measures). Fixes, harms, and mean model calls are reported for
every strategy.

## Interpretation rules

- **A passes:** the Phase 12 result is not just SVAMP's ease; the judgment
  transfers to GSM8K-level problems. **A fails:** the Phase 12 gain was largely
  the dataset.
- **B1 and B2 pass:** a constrained verdict removes invented answers without
  losing real judgment.
- **C1 passes:** the judge adds accuracy beyond a three-way vote. **C2 passes:**
  judge-on-split is not worse than five-way voting while usually cheaper.

## Integrity

Holdout gold is read only by `scripts/analyze.py`, once, after all four
judges have run. The analysis was tested on synthetic data with hand-checked
values before freezing. Direction-B training uses only Phase 10 training-pool
sources. Deviations go to `RUN_STATUS.md`.
