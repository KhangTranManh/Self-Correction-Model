# Phase 12 preregistration — order-bias removal and both-orders judging

Written on 2026-10-06 before any Phase 12 generation or training. The
execution lock (`data/execution_lock_v1.json`) hashes this file, the holdout,
the DPO data report and files, and every generation, training, and analysis
script.

## Motivation

The Phase 11 post-hoc analysis (exploratory) found three parts to the judge's
failure: a strong order bias toward the solution shown second, a weak real
skill (about 59% right when it picks one of the two solutions), and invented
third answers in 15–19% of cases. Phase 12 asks whether removing the order
bias and the invented answers reveals a judgment that is reliably above
chance, and whether that judgment beats voting.

## Data

- **Holdout (new dataset):** SVAMP (`github.com/arkilpatel/SVAMP`,
  `SVAMP.json`, SHA-256 `5be77703…`), 1,000 problems with integer answers.
  40 problems already appear in Phase 3's frozen evaluation and are excluded;
  960 remain (`data/sources_v1/`). Every reference passed the Phase 1
  verifier. No SVAMP problem was ever used for training.
- **Training:** `outputs/phase12_v1/dpo/` (`data/dpo_v1_report.json`), built
  locally from Phase 10 training-pool judgments with Python 3.10.11 + SymPy
  1.14.0. Each Phase 11-style contrast is emitted in the original order and
  with Solution A and B exchanged (standalone "A"/"B" tokens in the judgments
  exchanged accordingly); an incorrect judgment that invents a third answer is
  added as a second rejected example. 1,822 pairs: 1,592 training (726
  A-right, 726 B-right, 140 both-wrong; 796 original + 796 swapped; 770
  invented-answer and 762 sided-with-wrong rejections) and 230 validation,
  split by source.

## Model and training

Original solver `Kxck/Self_Correction_v1` (revision `6437f94…`) with a new
LoRA (r 16, alpha 32, dropout 0.05, all attention and MLP projections). DPO as
in Phase 11 (summed response log-probabilities, adapter-disabled reference,
beta 0.1, learning rate 2e-5 cosine with 5% warmup, effective batch 16,
maximum length 3,072, FP16 base with FP32 LoRA, gradient scaler from 1,024)
for **one epoch**, validation every 50 steps, seed 20261006. No search.

## Holdout procedure

1. Five attempts per source at temperature 0.7: s1 with the Phase 1 prompt,
   s2–s5 with the Phase 7/8 blind prompt.
2. **Judged pairs:** (s1, s2) and (s1, s3) whenever their final answers differ
   (gold-free). Each pair is judged greedily with the Phase 9 judge prompt in
   **both orders**: "ab" (s1 shown as Solution A) and "ba" (s1 shown as
   Solution B), by the untrained judge and by the DPO judge.
3. **Both-orders verdict:** the solution whose final answer **both** orderings'
   judgments match; otherwise the pair is **inconsistent**.

## Definitions

- **One-right pair:** a judged pair where exactly one of the two solutions is
  verifier-correct.
- **Consistent verdict:** both orderings' judgments match the same solution;
  otherwise the pair is **inconsistent** (no verdict).
- **Coverage:** the share of one-right pairs with a consistent verdict.
- **Self-check:** s1 if s1 = s2; otherwise the DPO judge's consistent verdict
  on (s1, s2); if inconsistent, fall back to the vote of s1, s2, s3. Model calls
  per source: 2 when s1 = s2, else 4 (two judge calls), or 5 with the fallback.
- **vote@k (compute-matched):** plurality vote over s1..sk, with k the smallest
  odd k in {3, 5} that is at least the self-check's mean model calls on the
  holdout.
- **Single-order judgment:** one judge call in one ordering; it "picks" the
  shown solution whose final answer it matches, or "invents" an answer that
  matches neither.
- **Position-2 rate:** among single-order judgments that pick A or B, the
  share that pick the solution shown second (B). The right solution is shown
  first and second equally often, so an unbiased judge sits near 50%.

## Endpoints (decided before any Phase 12 generation)

| Type | Aim | Definition | Threshold |
|---|---|---|---|
| **Primary P1** | Real judgment | Accuracy of the **DPO judge's consistent verdicts** on one-right pairs; the untrained judge on the same holdout is the reported baseline | **> 50%**, one-sided exact binomial **p < 0.05 after Holm** (family: P1 and Secondary), and **coverage ≥ 30%** |
| **Primary P2** | Practical value | End-to-end accuracy of the self-check (judge if consistent, otherwise vote) versus **vote@k at matched compute** over all 960 sources | Lower bound of the paired bootstrap **95% CI of the difference > −2 points** (non-inferiority) |
| **Secondary** | Model-level claim | DPO judge accuracy over **single-order** judgments on one-right pairs (both orderings counted) | **> 50%**, one-sided binomial **p < 0.05 after Holm**; a source-clustered CI is reported because the two orderings of a pair are dependent |
| Diagnostic | Order bias | Position-2 rate of single-order DPO judgments | **40–60%** |
| Diagnostic | Invented answers | Share of single-order DPO judgments matching neither solution | **< 5%** |
| Report | Tie-break score | Mean over one-right pairs of 1 (right), 0 (wrong), 0.5 (inconsistent) | report only |

Also reported (descriptive or raw p): all endpoints for the untrained judge;
DPO versus untrained tie-break score (source-clustered CI); self-check versus
agree-gated, keep, untrained self-check, and vote@5; vote@5 versus keep;
fixes, harms, and mean model calls per strategy.

## Interpretation rules

- P1 pass: first evidence that the model's own judgment between its answers is
  real once order bias is removed.
- P1 and P2 pass: model-level self-checking is at least as good as voting at
  matched compute (non-inferior within 2 points).
- P1 pass, P2 fail: real but weak judgment; next lever is step-level
  supervision or a larger model.
- Secondary pass without P1: skill exists but the both-orders rule loses too
  much coverage or consistency.
- Neither P1 nor Secondary: judgment between final answers is not learnable
  at 7B from this signal; prefer step-level verification or a larger model.

## Integrity

Holdout gold is read only by the analyzer, once. The DPO training data come
only from Phase 10 training-pool sources. Deviations go to `RUN_STATUS.md`.
The analysis logic was tested on synthetic data with hand-checked values before freezing. The endpoint table was revised on 2026-10-08 before any Phase 12 generation.
