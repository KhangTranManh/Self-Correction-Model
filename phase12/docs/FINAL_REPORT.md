# Phase 12 final report — both-orders judging and order-swapped DPO

**Status: closed on 2026-10-09. Both primary endpoints passed. On SVAMP,
both-orders judging is clearly above chance — already for the untrained judge
(68.7%) and more so after order-swapped DPO (74.9%, coverage 60%) — and the
DPO self-check is non-inferior to compute-matched voting. This is the
project's first preregistered evidence that the model's own choice between
its answers carries real, trainable signal. Two qualifications: the
untrained judge already passes P1 on this easier dataset, and invented
answers (10.5%) miss their 5% target. Voting over five attempts remains the
most accurate method.**

## 1. Question

Phase 11 found the judge's failure on GSM8K had three parts: order bias,
weak real skill, and invented third answers. Phase 12 asked whether judging
both orders and training on order-swapped preference pairs turn that weak
skill into a reliable judgment, on a new dataset.

## 2. How it was run

- **Holdout:** SVAMP (`arkilpatel/SVAMP`, 1,000 problems, integer answers).
  40 problems already used in Phase 3's frozen evaluation were excluded; 960
  remain. SVAMP was never used for training.
- **Training:** 1,822 DPO pairs built locally from Phase 10 training-pool
  judgments (Python 3.10 + SymPy 1.14): each contrast in the original order
  and with Solution A/B exchanged (labels in the judgments exchanged too),
  plus invented-answer judgments as extra rejected examples. 1,592 training
  (726 A-right, 726 B-right, 140 both-wrong) and 230 validation pairs.
- **Model:** original solver `Kxck/Self_Correction_v1` + a new LoRA (r 16,
  alpha 32). DPO, beta 0.1, learning rate 2e-5 cosine, one epoch (100 steps of
  16 pairs), FP16 base with FP32 LoRA. Validation preference accuracy rose
  0% → 76.1% (step 50) → 77.8% (step 100); validation loss 0.693 → 0.515.
- **Holdout procedure:** five attempts per problem (s1 Phase 1 prompt,
  s2–s5 blind, temperature 0.7). Pairs (s1, s2) and (s1, s3) with different
  final answers — 501 pairs — were each judged in both orders by the
  untrained and the DPO judge (2,004 judgments). A **consistent verdict**
  needs both orders to pick the same solution.
- **Endpoints** were revised once before any Phase 12 generation (see
  `RUN_STATUS.md`). Execution lock `4f42f926…`, verified on the GPU before
  generation and before the single opening. Tesla V100-SXM2-32GB, vLLM 0.7.0.
  All 15 result files match the GPU copies by SHA-256.

## 3. Results

The first answer was correct on 788/960 (82.1%). 367 judged pairs (from 287
problems) had exactly one right solution.

### Endpoints

| Type | Endpoint | Result | Threshold | |
|---|---|---|---|---|
| **Primary P1** | DPO judge, consistent both-orders accuracy | **74.9%** (164 right, 55 wrong), coverage **59.7%**, Holm p ≈ 9e−14 | > 50%, p < 0.05, coverage ≥ 30% | ✅ pass |
| (baseline) | Untrained judge, same measure | 68.7% (147 right, 67 wrong), coverage 58.3%, p ≈ 2e−8 | — | also above 50% |
| **Primary P2** | Self-check (DPO) − vote@3 (compute-matched; self-check uses 2.63 calls on average) | **−0.63 pts**, 95% CI [−1.56, +0.31] | lower bound > −2 | ✅ pass (non-inferior) |
| **Secondary** | DPO judge, single-order accuracy (734 judgments) | **61.9%**, clustered CI [57.5, 66.2], Holm p ≈ 7e−11 | > 50%, p < 0.05 | ✅ pass |
| Diagnostic | Position-2 rate (DPO) | **55.1%** (untrained 54.3%) | 40–60% | ✅ within |
| Diagnostic | Invented answers (DPO) | **10.5%** (untrained 13.2%) | < 5% | ❌ above |
| Report | Tie-break score (inconsistent = 0.5) | DPO 0.649 vs untrained 0.609; difference +3.95 pts, CI [+0.78, +7.12] | report only | |

### End-to-end accuracy (960 problems)

| Strategy | Accuracy | Fixes (of 172 wrong) | Harms (of 788 right) | Mean model calls |
|---|---:|---:|---:|---:|
| keep | 82.08% | — | — | 1 |
| agree_gated | 87.60% | 80 | 27 | 2.26 |
| vote@3 (maj3) | 87.60% | 80 | 27 | 3 |
| **vote@5 (maj5)** | **90.10%** | **95** | **18** | 5 |
| self_check, untrained judge | 86.46% | 79 | 37 | 2.63 |
| **self_check, DPO judge** | **86.98%** | 79 | 32 | 2.63 |

Self-check (DPO) − keep: +4.9 pts [+2.8, +7.1]. Self-check (DPO) − untrained
self-check: +0.5 pts [−0.2, +1.4], not significant. Self-check (DPO) −
vote@5: −3.1 pts [−4.5, −1.8]. vote@5 − keep: +8.0 pts [+5.9, +10.1].

### Single-order picks (DPO judge)

| Right solution shown as | Picks A | Picks B | Invented |
|---|---:|---:|---:|
| A (n = 367) | 208 ✅ | 116 | 43 |
| B (n = 367) | 87 | 246 ✅ | 34 |

## 4. Interpretation

1. **The judgment signal is real on SVAMP.** Even the untrained judge's
   consistent verdicts are right 68.7% of the time, and order bias is small
   here (position-2 rate 54%). On GSM8K in Phase 11 the same judge was at
   chance with a strong bias. Part of the Phase 12 success therefore reflects
   the **dataset** (SVAMP problems are 1–2 arithmetic steps, so errors are
   easier to spot), not only the method.
2. **Order-swapped DPO adds a measurable, significant improvement on top.**
   Consistent accuracy 68.7% → 74.9%, single-order accuracy 56.3% → 61.9%,
   tie-break score +3.95 points (CI excludes zero), fewer invented answers
   (13.2% → 10.5%) and fewer harms (37 → 32). Unlike Phase 11, the trained
   preference now transfers to decisions on fresh problems.
3. **Practically, self-check matches cheap voting but not five-way voting.**
   At about 2.6 model calls it is non-inferior to vote@3, but vote@5 is still
   3.1 points better. Voting remains the most accurate method overall.
4. **Coverage is the limiter.** About 40% of one-right pairs get no
   consistent verdict, and the invented-answer rate (10.5%) is twice the
   target.

## 5. What improved and what did not

| | Result |
|---|---|
| ✅ Real judgment (P1) | 74.9% consistent accuracy, p ≈ 1e−13 |
| ✅ Trainable | DPO beats the untrained judge on the same pairs (tie-break +3.95, CI > 0) |
| ✅ Order bias | position-2 rate 55% — within the 40–60% target |
| ✅ Practical value (P2) | non-inferior to compute-matched vote@3 |
| ❌ Invented answers | 10.5%, target < 5% |
| ❌ Beating strong voting | 3.1 points below vote@5 |
| ⚠️ Attribution | the untrained judge already passes P1 on SVAMP; the dataset change confounds direct comparison with Phase 11 |

## 6. Deviations and limits

- The endpoint table was revised on 2026-10-08, before any Phase 12
  generation; the analysis was retested on synthetic data and the lock
  refrozen. No other deviation; the pipeline completed in one attempt.
- The A/B label exchange in swapped training judgments was a token swap of
  standalone "A"/"B"; spot-checked, not exhaustively verified.
- SVAMP is easier than GSM8K (first-answer accuracy 82% vs 71–75%), which
  limits comparison with Phases 9–11.
- One configuration, one checkpoint, arithmetic word problems only.

## 7. Disposition and next step

- `phase12-dpo-judge` is the first judge adapter with a passing preregistered
  model-level result. It is **not promoted as a replacement for voting**: it is
  kept as the best judge so far and served by `serving/serve.sh`.
- The SVAMP holdout is opened; never train on it or tune against it.
- Next steps worth testing (detailed with effort estimates in the root
  `README.md`, "Next steps"): (a) confirm on a harder fresh dataset (for
  example MATH levels 1–3 or ASDiv) with the existing adapter, to separate the
  method from SVAMP's ease; (b) reduce invented answers by requiring the
  verdict to be Solution A or B and retraining; (c) call the judge only when a
  vote is split, and compare with vote@5 at matched compute.

## Artifacts

Mirrored under `outputs/phase12_remote_v100/outputs/phase12_v1/`: `samples/`,
`judge_base/`, `dpo_lora/` (adapter, log, summary), `judge_dpo/`, and
`analysis/report.json`. Training data: `outputs/phase12_v1/dpo/`.
