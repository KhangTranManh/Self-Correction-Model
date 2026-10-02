# Phase 10 final report — training the judgment step

**Status: closed on 2026-10-01. Negative result. LoRA fine-tuning on the
model's own verifier-correct judgments did not improve its choice between two
conflicting solutions (P1 failed), and the trained self-check was
significantly worse than agreement-gated voting at equal compute (P2 failed).
Voting over five attempts again gained about nine points. No adapter is
promoted.**

## 1. Question

Phase 9 showed that disagreement between two independent attempts catches
82–85% of wrong first answers, but when exactly one of two conflicting
solutions is right, the model's self-check picks it only 44–52% of the time.
Phase 10 asked whether supervised fine-tuning on the model's *own*
verifier-correct judgments can teach that judgment.

## 2. How it was run

### Sources (`data/sources_v1/`)

- **Training pool:** 1,900 GSM8K train sources from the Phase 4 expansion set
  (used before only as Phase 4 training material), minus the 100 Phase 4
  development sources. The original solver was never trained on them.
- **Holdout:** 300 fresh GSM8K train sources never used by any phase, ranked
  from 534 eligible after excluding Phases 1–9 and the training pool.

### Model

Only the original solver `Kxck/Self_Correction_v1` (Qwen2.5-7B-Instruct after
Phase 1 SFT). The judge is a new LoRA on this base; base weights are
unchanged.

### Procedure

1. **Holdout before training:** A0 (Phase 1 prompt) and B1–B4 (Phase 7/8 blind
   prompt) at temperature 0.7; the untrained greedy judge J_base on (A0, B1)
   with the Phase 9 judge prompt.
2. **Training samples:** 4 attempts per training source (7,600 in total):
   s1 with the Phase 1 prompt, s2–s4 blind, temperature 0.7.
3. **Pairs:** (A = s1, B = sk) whenever final answers differ; at most two per
   source; labeled by the verifier → **1,621 pairs** (672 A-right, 414
   B-right, 535 both-wrong).
4. **Judge candidates:** 4 temperature-0.7 judge samples per pair from the
   untrained model (6,484 generations).
5. **Rejection sampling:** per pair, keep the first candidate whose final
   answer the verifier accepts. Accepted: 471/672 A-right (70%), 360/414
   B-right (87%), 88/535 both-wrong (16%).
6. **Balanced SFT set:** 360 A-right + 360 B-right + 88 both-wrong = 808
   examples, split by source into 723 training and 85 validation.
7. **Training:** LoRA r = 16, alpha = 32, dropout 0.05, all attention and MLP
   projections; learning rate 1e-4 cosine, 3% warmup; one epoch (46 steps of
   effective batch 16); max length 3,072; loss on assistant tokens only; FP16
   base with FP32 LoRA and a gradient scaler starting at 1,024. No examples
   were dropped for length. Training took 13 minutes. Validation loss
   0.166 → 0.156. Adapter SHA-256
   `14a2933fc62708cfb00f0b5933ba8c37743f81c87b9c803a2c09f1ea73abc2d9`.
8. **Trained judge:** greedy J_trained with the LoRA on the same holdout
   (A0, B1) prompts, then a single protected analysis.

### Statistics and integrity

Paired bootstrap 95% intervals (10,000 resamples) and exact paired sign tests,
Holm over the two primary endpoints. The execution lock
(`data/execution_lock_v1.json`,
`77a97773dc851cb16d8530587004fe3e2fc00480eaf234f7ffacd483e91a5780`) was
verified before generation and again before the single opening. All 22
result files and the adapter mirrored locally match the GPU copies by
SHA-256.

## 3. Results (holdout, n = 300)

The first answer was correct on 221/300 (73.67%); A0 and B1 disagreed on 114.

### Accuracy, fixes, and harms

| Strategy | Accuracy | Fixes (of 79 wrong) | Harms (of 221 right) | Generations |
|---|---:|---:|---:|---:|
| keep | 73.67% | — | — | 1 |
| maj3 = agree_gated | 79.00% | 29 | 13 | 3 / ~2.4 |
| **maj5** | **82.67%** | **34** | **7** | 5 |
| self_check_base | 73.00% | 23 | 25 | ~2.4 |
| **self_check_trained** | **73.00%** | **26** | **28** | ~2.4 |

### Primary endpoints

| Endpoint | Result | 95% CI | Holm p | Gate |
|---|---|---|---|---|
| **P1** judgment: picks the right one when exactly one of A0/B1 is right (n = 73) | base 52.1% → trained 50.7% (−1.4 pp) | [−11.0, +9.6] | 1.00 | **failed** |
| **P2** self_check_trained − agree_gated | −6.00 pp | [−9.67, −2.33] | 0.006 | **failed (significantly worse)** |

### Secondary

| Contrast | Difference | 95% CI | Raw p |
|---|---:|---:|---:|
| self_check_trained − keep | −0.67 pp | [−5.33, +4.00] | 0.89 |
| self_check_trained − self_check_base | 0.00 pp | [−2.67, +2.67] | 1.00 |
| self_check_trained − maj5 | −9.67 pp | [−13.67, −6.00] | < 0.001 |
| maj5 − keep | +9.00 pp | [+5.00, +13.00] | < 0.001 |

The correction format ("Phát hiện lỗi …") appeared in 62.3% of untrained and
63.2% of trained judgments on disagreements.

## 4. Interpretation

1. **The training did not change the judgment.** Trained and untrained
   self-check reached identical accuracy, and the trained judge sided with the
   right solution at the same chance rate (51% vs 52%).
2. **Why rejection sampling was weak here.** The kept judgments were the
   model's own samples, already high-likelihood for it (validation loss moved
   only from 0.166 to 0.156). With a near-chance chooser, a verifier-correct
   final answer often reflects a lucky pick rather than a correct check, so
   the training signal did not separate good checking from guessing.
3. **Voting remains the reliable method.** For the third time on fresh data,
   five independent attempts beat everything else (+9.0 points, with the
   fewest harms).
4. **For the Phase 0 goal:** the model notices that its attempts disagree but
   still cannot reliably tell which attempt is right, even after targeted
   behavior training at this scale.

## 5. What improved and what did not

### Did not improve (the target of this phase)

| Measure | Untrained judge | Trained judge | Change |
|---|---:|---:|---|
| Picks the right solution when exactly one is right | 52.1% | 50.7% | none (chance) |
| Self-check accuracy | 73.00% | 73.00% | none |
| Wrong first answers fixed | 23 | 26 | +3 |
| Correct first answers broken | 25 | 28 | +3 (offsets the fixes) |
| Correction format used on disagreements | 62.3% | 63.2% | none |

The trained judge is not better at the one skill it was trained for. Its
three extra fixes are cancelled by three extra harms.

### Did improve or was newly established

| Finding | Evidence |
|---|---|
| **Voting replicates on a third fresh set.** | 73.67% → 82.67% (+9.0 points, CI [+5.0, +13.0]); Phase 8 and Phase 9 showed the same direction on their own sets. |
| **Voting is also the safest method.** | It breaks the fewest correct answers (7 of 221) while fixing the most wrong ones (34 of 79). |
| **When one solution is right, the right answer is usually reachable.** | Among four sampled judgments, at least one was correct for 87% of B-right and 70% of A-right pairs, but the greedy judge picks it only half the time. The capability exists; the selection is the weak point. |
| **When both solutions are wrong, the model rarely recovers.** | Only 16% of both-wrong pairs had any correct judgment in four tries. |
| **A reusable data asset now exists.** | 1,621 verified disagreement pairs and 6,484 labeled judgments, containing both correct and incorrect judgments of the same pairs — the input a preference (DPO) objective needs. |
| **A working training and serving stack on the V100.** | Verified LoRA training (FP16 scaler fix, 13 minutes for 723 examples) and vLLM serving of all checkpoints with the validated methods (`serving/`). |

### Bottom line

Phase 10 produced **no improvement in self-judgment** and no promotable
model. Its gains are knowledge, not accuracy: it ruled out self-imitation as a
way to teach judgment, confirmed voting a third time, and showed that the
missing piece is choosing among answers the model can already produce.

## 6. Deviations and limits

- Before any Phase 10 generation, a V100 training smoke test found that the
  default FP16 gradient-scaler start (65536) skipped every update; training
  was fixed to start at 1,024.
- The training-sample stage first failed on a U+2028 character inside some
  GSM8K questions (a `str.splitlines()` reader). The fix was a newline-only
  reader inside Phase 10's code; the lock was refrozen after the holdout
  generation had already completed. It changed only file reading; no output or
  label had been inspected.
- The continuous local backup kept being stopped by a Windows console
  interrupt; results were copied with one-shot syncs at each stage.
- Scale limits: 808 examples, one epoch, one fixed configuration, one
  checkpoint, arithmetic word problems only. P1 rests on 73 one-right pairs,
  so its interval is wide (±10 points); it rules out a large improvement, not
  a small one.

## 7. Disposition and next step

- The `phase10-judge` adapter is not promoted; it is kept for reproducibility
  and is exposed by `serving/serve.sh` for inspection.
- The Phase 10 holdout is opened; never train on it or tune against it.
- Voting over independent attempts remains the best-supported method.
- If the judgment goal continues, the next phase needs a stronger signal
  than self-imitation, for example a preference objective that contrasts a
  correct and an incorrect judgment of the *same* pair (Phase 10 already
  generated both), or a check that re-derives each solution independently
  before comparing. Either needs a new fresh holdout.

## Artifacts

Mirrored under `outputs/phase10_remote_v100/outputs/phase10_v1/`:
`holdout_samples/`, `holdout_judge_base/`, `train_samples/`,
`train_pairs.jsonl`, `judge_candidates/`, `sft/`, `judge_lora/` (with the
adapter), `holdout_judge_trained/`, and `analysis/report.json`.
