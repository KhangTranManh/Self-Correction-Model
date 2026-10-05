# Phase 11 final report — preference training (DPO) of the judgment step

**Status: closed on 2026-10-02. Negative result. DPO taught the model to
prefer correct judgments in likelihood (67% on unseen validation pairs) but
did not significantly change which solution it chooses on fresh problems
(P1: 46.6% → 49.4%, not significant), and the DPO self-check did not beat
agreement-gated voting (P2 failed). Voting over five attempts gained 12.8
points, the largest gain measured so far. No adapter is promoted.**

## 1. Question

Phase 10 showed that SFT on the model's own correct judgments does not teach
it to pick the right solution. Phase 11 used a contrastive signal instead:
for the *same* problem and the *same* two conflicting solutions, DPO raises a
correct judgment over an incorrect one — preferably one that sided with the
wrong solution. The aim was a model-level change in judgment, not a harness.

## 2. How it was run

### Data (built locally, no GPU)

- **Training pairs:** from the Phase 10 training-pool judgments (hash-verified
  mirror). A local Python 3.10.11 + SymPy 1.14.0 environment reproduced Phase
  10's acceptance counts exactly (471 / 360 / 88) before labeling. 542
  balanced preference pairs (241 A-right, 241 B-right, 60 both-wrong), split
  by source into 472 training and 70 validation. In 332 pairs the rejected
  judgment sided with the wrong solution.
- **Holdout:** all 234 remaining never-used eligible GSM8K train sources,
  disjoint from Phases 1–10.

### Model and training

Original solver `Kxck/Self_Correction_v1` with a new LoRA (r 16, alpha 32,
dropout 0.05, all attention and MLP projections). DPO with summed response
log-probabilities, reference = base with the adapter disabled (precomputed);
beta 0.1; learning rate 2e-5 cosine, 5% warmup; two epochs (59 steps of 16
pairs); maximum length 3,072 (no pair dropped); FP16 base, FP32 LoRA, gradient
scaler starting at 1,024; logits computed on response tokens only. Training
took 37 minutes on a Tesla V100-SXM2-32GB (peak 18.4 GB in the smoke test).
Adapter SHA-256
`f8251341d5399489e193b4de6a543bf398bcb74dedf1cfaf3fe037d595a1917d`.

| Step | 0 | 10 | 20 | 40 | 50 | 59 |
|---|---:|---:|---:|---:|---:|---:|
| Validation loss | 0.693 | 0.690 | 0.684 | 0.674 | 0.674 | 0.674 |
| Validation preference accuracy | 0% | 55.7% | 60.0% | 64.3% | 68.6% | 67.1% |

Training-set preference accuracy reached 100% in the second epoch
(memorization of seen pairs); the validation figures are the meaningful ones.

### Holdout procedure

Five attempts per source at temperature 0.7 (s1 with the Phase 1 prompt,
s2–s5 blind). Every pair (s1, sk) with different final answers — 362 pairs —
was judged greedily by the untrained judge and by the DPO judge with the
Phase 9 judge prompt. Pipeline per source: A0 = s1, B1 = s2, B2 = s3. Single
protected opening; Holm over the two primaries. The lock
(`data/execution_lock_v1.json`, `3e37081046d1…`) was verified on the host
before generation and before the opening. All 16 result files mirrored
locally match the GPU copies by SHA-256.

## 3. Results (holdout, n = 234)

The first answer was correct on 167/234 (71.37%).

### End-to-end accuracy

| Strategy | Accuracy | Fixes (of 67 wrong) | Harms (of 167 right) |
|---|---:|---:|---:|
| keep | 71.37% | — | — |
| maj3 = agree_gated | 77.35% | 25 | 11 |
| **maj5** | **84.19%** | **35** | **5** |
| self_check_base | 70.94% | 18 | 19 |
| self_check_dpo | 72.22% | 23 | 21 |

### Primary endpoints

| Endpoint | Result | 95% CI | Holm p | Gate |
|---|---|---|---|---|
| **P1** picks the right solution (251 one-right pairs from 133 sources) | base 46.6% → DPO 49.4% (+2.8 pp; 28 better, 21 worse) | [−2.4, +8.0] (source-clustered) | 0.39 | **failed** |
| **P2** self_check_dpo − agree_gated | −5.13 pp | [−9.83, −0.43] | 0.10 | **failed** |

### Position bias (P1 split)

| Pair type | n | Untrained judge | DPO judge |
|---|---:|---:|---:|
| First answer (A) right | 141 | 31.9% | 34.0% |
| Blind attempt (B) right | 110 | 65.5% | 69.1% |

Both judges favor Solution B, the second solution shown. DPO raised both rows
slightly without removing the bias.

### Secondary

| Contrast | Difference | 95% CI | Raw p |
|---|---:|---:|---:|
| self_check_dpo − keep | +0.85 pp | [−4.70, +6.41] | 0.88 |
| self_check_dpo − self_check_base | +1.28 pp | [−1.71, +4.27] | 0.58 |
| self_check_dpo − maj5 | −11.97 pp | [−16.67, −7.26] | < 0.001 |
| maj5 − keep | +12.82 pp | [+7.69, +17.95] | < 0.001 |

## 4. What improved and what did not

| | Result |
|---|---|
| ❌ Picking the right solution on fresh problems | +2.8 points, not significant; still below chance overall (49.4%) |
| ❌ Self-check vs equal-compute voting | 5 points worse |
| ❌ Position bias | unchanged: the judge mostly sides with Solution B |
| ✅ Preference in likelihood | 67% of unseen validation pairs prefer the correct judgment (from 0%) — DPO moved the model's scores, but not enough to change greedy choices |
| ✅ Voting | 71.4% → 84.2% (+12.8 points), the fourth consistent replication and the largest gain so far, with only 5 harms |

## 5. Interpretation

1. **A likelihood shift is not a decision change.** DPO separated correct from
   incorrect judgments on held-out training-distribution pairs, but on fresh
   problems the greedy judge's choice barely moved. 472 pairs and one
   checkpoint may be too little signal, or the judgment may not be learnable
   from self-generated judgments at 7B.
2. **The judge has a strong order bias.** It sides with Solution B about twice
   as often as with Solution A. Much of its apparent "judgment" is position,
   which also explains why it trails voting.
3. **Across Phases 9–11, the model-level result is consistent:** the model
   notices when its own attempts disagree, but neither prompting, SFT, nor
   DPO has made it reliably choose the correct one.

## 5b. Post-hoc exploratory analysis: bias versus skill

*Not preregistered.* Computed after the single protected opening from the same
251 one-right pairs, with the same verifier (Python 3.10, SymPy 1.14). It
explains the primary results; it does not change any gate.

| Judge, case | Picks A | Picks B | Writes a third answer |
|---|---:|---:|---:|
| Untrained, A right (n = 141) | 32% ✅ | 49% | 19% |
| Untrained, B right (n = 110) | 18% | 65% ✅ | 16% |
| DPO, A right | 34% ✅ | 52% | 14% |
| DPO, B right | 20% | 69% ✅ | 11% |

Three effects combine:

1. **Order bias.** The judge sides with Solution B (shown second) about half
   the time even when A is right.
2. **A weak real skill underneath.** It picks A more often when A is right
   (32% vs 18%) and B more often when B is right (65% vs 49%): about 14–16
   points of genuine sensitivity. Restricted to cases where it picks A or B,
   it is right about 59% of the time.
3. **Invented answers.** In 15–19% of cases it writes a final answer matching
   neither solution, which is almost always wrong. DPO reduced this from 19%
   to 14% when A is right and from 16% to 11% when B is right.

Bias-free balanced accuracy (average of the two "picks the right one" rates)
is 48.7% untrained and 51.6% after DPO. Removing order bias alone would
likely recover roughly 55–60% decisive accuracy, not the 75–80% the Phase 0
goal requires.

## 6. Deviations and limits

- None to the preregistered protocol. The pipeline completed in one attempt.
- The local backup script initially pointed at the previous host's project
  root and copied nothing; it was corrected (it is not part of the lock) and
  all outputs were then mirrored and verified.
- 234 holdout sources is the last never-used pool under the current filters;
  P1's interval excludes large improvements but not small ones.
- One configuration only (beta, learning rate, epochs fixed in advance).

## 7. Disposition and next step

- The `phase11-dpo-judge` adapter is not promoted; it is kept for
  reproducibility and served for inspection by `serving/serve.sh`.
- The Phase 11 holdout is opened. No never-used GSM8K train source remains
  under the current filters; any further phase needs a new evaluation source.
- Voting over independent attempts remains the best-supported method.
- If the judgment goal continues: remove order bias first (judge both
  orderings, or train on order-swapped pairs), discourage invented third
  answers, and use a richer signal than final-answer correctness — for
  example step-level verification of each solution — on a new dataset. The
  proposed Phase 12 plan is in the root `README.md` ("Next steps").

## Artifacts

Mirrored under `outputs/phase11_remote_v100/outputs/phase11_v1/`: `samples/`,
`judge_base/`, `dpo_lora/` (adapter, log, summary), `judge_dpo/`, and
`analysis/report.json`. Training data: `outputs/phase11_v1/dpo/`.
