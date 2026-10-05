# Phase 11 preregistration — preference training (DPO) of the judgment step

Written on 2026-10-02 before any Phase 11 generation or training. The
execution lock (`data/execution_lock_v1.json`) hashes this file, the holdout,
the DPO data report, and every generation, training, and analysis script.

## Motivation

Phase 10 found that the model can already produce a correct judgment of two
conflicting solutions (one of four samples is correct for 70–87% of one-right
pairs) but picks it only about half the time, and SFT on its own correct
judgments did not change that. SFT showed the model only what to imitate.
Phase 11 uses **DPO**: for the *same* problem and the *same* two solutions, it
raises the likelihood of a correct judgment relative to an incorrect one,
preferably one that sided with the wrong solution. This targets the model's
own selection between answers — a model-level change, not a harness.

## Hypothesis

DPO on contrasting judgments of the same pair increases how often the model's
greedy judgment sides with the right solution, and makes self-check at least
competitive with agreement-gated voting at equal compute.

## Data (built locally, no GPU)

- **Training data:** `outputs/phase11_v1/dpo/` from the Phase 10 training-pool
  pairs and judge candidates (hash-verified Phase 10 mirror), labeled with
  Python 3.10.11 and SymPy 1.14.0 — the same verifier versions as the GPU,
  confirmed by reproducing Phase 10's acceptance counts exactly
  (471 / 360 / 88). 542 balanced preference pairs (241 A-right, 241 B-right,
  60 both-wrong) split by source into 472 training and 70 validation; in 332
  the rejected judgment sided with the wrong solution
  (`data/dpo_v1_report.json`).
- **Holdout:** `data/sources_v1/holdout.jsonl`, all 234 remaining never-used
  eligible GSM8K train sources, disjoint from every Phase 1–10 source.

## Model and training

Original solver `Kxck/Self_Correction_v1` (revision `6437f94…`) with a new
LoRA (r = 16, alpha = 32, dropout 0.05, all attention and MLP projections).
DPO with summed response log-probabilities; reference = the same base with the
adapter disabled, precomputed once; beta 0.1; learning rate 2e-5 cosine with
5% warmup; two epochs; effective batch 16 pairs; maximum length 3,072; FP16
base, FP32 LoRA, gradient scaler starting at 1,024; seed 20261002. No search.

## Holdout procedure

1. Five attempts per holdout source at temperature 0.7: s1 with the Phase 1
   prompt, s2–s5 with the Phase 7/8 blind prompt.
2. **Judgment benchmark:** every pair (s1, sk), k = 2..5, whose final answers
   differ (gold-free). The untrained judge (J_base) and the DPO judge (J_dpo)
   judge each pair greedily with the Phase 9 judge prompt.
3. Pipeline per source: A0 = s1, B1 = s2, B2 = s3.
4. Single protected opening by `scripts/analyze.py`.

## Endpoints (Holm over the two primaries)

- **P1 — judgment:** on benchmark pairs where exactly one of s1/sk is right,
  the rate at which the judge's final answer matches the right one; DPO vs
  base. Source-clustered bootstrap 95% CI (10,000 resamples) and an exact
  sign test on discordant pairs.
- **P2 — end-to-end:** self_check_dpo − agree_gated accuracy over all 234
  sources (paired bootstrap CI, exact paired sign test).

Each passes only with a positive difference, a 95% CI excluding zero, and
Holm-adjusted p < 0.05. Secondary (raw p): self_check_dpo vs keep, vs
self_check_base, vs maj5; maj5 vs keep; P1 split by A-right and B-right
(position bias check); fixes and harms.

## Interpretation rules

- P1 passing is the first evidence in this project that training changed the
  model's own judgment.
- P1 without P2: better judgment, still not worth more than a third vote.
- Neither: preference training on self-generated judgments does not teach
  this selection at 7B and this scale.

## Integrity

Holdout gold is read only by the analyzer, once. No Phase 5–10 protected or
holdout source enters training. Deviations go to `RUN_STATUS.md`. This is the
last never-used pool under the current filters; any later phase needs a new
source (for example a different dataset).
