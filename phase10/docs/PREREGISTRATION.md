# Phase 10 preregistration — training the judgment step

Written on 2026-10-01 before any Phase 10 generation. The execution lock
(`data/execution_lock_v1.json`) hashes this file, the sources, and every
generation, training, and analysis script.

## Motivation

Phase 9 showed that disagreement between two independent attempts catches
82–85% of wrong first answers, but when exactly one of two conflicting
solutions is right, the model's self-check picks it only 44–52% of the time.
The model already writes the correction format ("Phát hiện lỗi / Nguyên nhân /
Sửa lại"); its **judgment** is at chance. Phase 10 trains that judgment only.

## Hypothesis

Supervised fine-tuning on the model's own verifier-correct judgments of its
own conflicting solutions improves (P1) how often it sides with the right
solution and (P2) end-to-end accuracy of the self-check pipeline relative to
agreement-gated voting at equal compute.

## Sources

`data/sources_v1/` (see `report.json`):

- **Training pool:** 1,900 GSM8K train sources from the Phase 4 expansion set,
  previously used only as Phase 4 training material, minus the 100 Phase 4
  development sources. The original solver was never trained on them.
- **Holdout:** 300 fresh GSM8K train sources never used by any phase (534
  eligible after excluding Phases 1–9 and the training pool).

## Model

Only the original solver (`Kxck/Self_Correction_v1`, revision `6437f94…`).
The judge is a new LoRA on this base. vLLM 0.7.0 FP16 for generation;
Transformers + PEFT for training; one Tesla V100-SXM2-32GB.

## Procedure

1. **Holdout generation (before training):** A0 with the Phase 1 prompt and
   B1–B4 with the Phase 7/8 blind prompt, all at temperature 0.7; the untrained
   greedy judge J_base on (A0, B1) with the Phase 9 judge prompt.
2. **Training samples:** on each training source, s1 (Phase 1 prompt) and
   s2–s4 (blind prompt) at temperature 0.7.
3. **Pairs:** (A = s1, B = sk) whenever final answers differ; at most two per
   source; labeled by the verifier (A_right, B_right, both_wrong).
4. **Judge candidates:** four temperature-0.7 samples of the Phase 9 judge
   prompt per pair from the untrained model.
5. **SFT data (rejection sampling):** per pair keep the first candidate whose
   final answer the verifier accepts. Select equal numbers of A_right and
   B_right pairs (cap 1,200 examples in total) plus both_wrong pairs up to a
   quarter of that count. Split 90/10 by source; validation is for loss only.
6. **Training:** LoRA r = 16, alpha = 32, dropout 0.05, all attention and MLP
   projections, learning rate 1e-4 cosine with 3% warmup, one epoch,
   effective batch 16, maximum length 3,072, loss on assistant tokens only,
   seed 20261001. No hyperparameter search; no holdout use.
7. **Trained judge:** greedy J_trained on the same holdout (A0, B1) prompts.
8. **Single protected opening** by `scripts/analyze.py`.

## Strategies on the holdout

keep (A0); maj3 (A0, B1, B2); agree_gated (A0 if A0 = B1 else maj3); maj5
(A0, B1–B4); self_check_base (A0 if agree else J_base); self_check_trained
(A0 if agree else J_trained). Agreement uses the Phase 9 gold-free
comparison.

## Primary endpoints (Holm over the two)

- **P1 — judgment accuracy:** on holdout disagreements where exactly one of
  A0/B1 is right, the rate at which the judge's final answer matches the
  right one; trained vs base, paired.
- **P2 — end-to-end:** self_check_trained accuracy − agree_gated accuracy on
  all 300 sources (equal compute of about 2.3 generations).

Each requires a positive difference with a paired bootstrap 95% CI
(10,000 resamples) excluding zero and a Holm-adjusted exact paired sign-test
p < 0.05.

Secondary (descriptive, raw p): self_check_trained vs keep, vs
self_check_base, vs maj5; maj5 vs keep; fixes and harms; the rate of the
correction format on disagreements.

## Interpretation rules

- P1 passing without P2 means the judgment improved but not enough to beat
  a cheap third vote.
- P2 passing means a trained self-check is a better use of the same compute
  than voting — the first model-internal correction result in this project.
- Neither passing means rejection-sampling SFT does not teach judgment at
  this scale.

## Integrity

Training uses only training-pool sources and their gold answers. Holdout gold
is read only by the analyzer, once. No Phase 5–9 protected source is used.
Deviations are recorded in `RUN_STATUS.md`.
