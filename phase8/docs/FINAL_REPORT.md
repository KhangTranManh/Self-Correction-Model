# Phase 8 final report — resampling, distractor, and probe-routed blind re-solve

**Status: completed on 2026-09-30 under execution-lock amendment v2. The
preregistered end-to-end gate passed for all three checkpoints, but the gain
is explained mainly by a second, answer-hidden attempt rather than by model
self-correction. No checkpoint is promoted.**

The protected analysis was opened once, after all generation and probe scoring
finished. Its report is `outputs/phase8_analysis_v2/report.json`, with verdict
rows SHA-256
`0c598805b60b2203e222a8c26ccfe9a7b1e43ad5c33516575b505c1eb9ed277b`.

## Questions

Phase 7 found that blind re-solving fixed far more initially wrong answers
than re-solving with the earlier answer visible. Phase 8 asked:

1. How much does a plain second sample or greedy decoding already help?
2. Does the model's *own* earlier answer hurt more than an irrelevant wrong
   answer of similar length?
3. Can a frozen hidden-state probe route blind re-solves profitably, without
   an oracle?

## Design as run

- **Sources:** 400 fresh, verifier-checked GSM8K questions, disjoint from every
  Phase 1–7 source inventory; manifest SHA-256
  `a4d2829140292cce1b86ccf3c758e7b30641ee829452698a9ca139462eae7e24`.
  Natural first-answer prevalence was not rebalanced.
- **Checkpoints:** original solver (`Kxck/Self_Correction_v1`), Phase 4
  warm-start V2 (merged FP16), and correction SFT V3 (LoRA on merged V2),
  with lineage and adapter hashes from `data/model_lineage_v1.json`.
- **First answers:** the original solver with the Phase 7 initial prompt at
  temperature 0.7. Two controls on every source: a second temperature-0.7
  sample with a new seed, and a greedy answer with the same prompt.
- **Three greedy arms per checkpoint:** blind re-solve; own first answer
  visible; distractor visible. The distractor is a verifier-wrong answer from
  a different Phase 7 source, matched to within 10% token length by a frozen
  hash rule. 395/400 sources matched; the 5 unmatched sources are excluded
  from distractor contrasts only.
- **Probe:** the exact frozen Phase 5 probes (layer 14; C = 0.01 / 0.01 / 1.0;
  threshold 0.5), recovered locally with matching SHA-256. No probe was
  refit.
- **Pipeline:** keep an unflagged first answer; replace a flagged answer with
  the same checkpoint's blind re-solve. Gold verification is offline only.
- **Statistics:** source-level paired bootstrap 95% intervals with 10,000
  resamples; exact paired sign tests with Holm correction across three
  checkpoints for each named contrast family.

## Results

The original solver answered 283/400 correctly on the first attempt
(70.75%); 117 first answers were wrong.

### Step 1 — a second attempt alone

| Original-solver pass | Accuracy | Paired change vs first (95% CI) |
|---|---:|---:|
| First answer (temperature 0.7) | 70.75% | — |
| Second temperature-0.7 sample | 77.25% | +6.50 pp [+2.25, +10.75] |
| Greedy, same prompt | 76.75% | +6.00 pp [+1.75, +10.25] |

An independent second attempt, without any feedback, already gains about six
points on this pool.

### Step 2 — what visible context does to re-solving

Wrong-to-correct fixes among the 117 initially wrong sources (114 matched for
distractor rows), and correct-to-wrong harms among the 283 initially correct:

| Checkpoint | Blind fixes | Own-visible fixes | Distractor fixes (of 114) | Blind harms | Own-visible harms |
|---|---:|---:|---:|---:|---:|
| Original solver | 50 | 21 | 35 | 33 | 72 |
| Warm-start V2 | 53 | 17 | 28 | 34 | 69 |
| Correction SFT V3 | 58 | 19 | 29 | 29 | 79 |

Paired wrong-row contrasts (difference in fix rate, 95% CI, Holm p):

| Checkpoint | Own − blind | Distractor − blind | Own − distractor |
|---|---:|---:|---:|
| Original solver | −24.8 pp [−34.2, −15.4], p < 0.001 | −12.3 pp [−21.1, −3.5], p = 0.013 | −12.3 pp [−21.9, −2.6], p = 0.060 |
| Warm-start V2 | −30.8 pp [−40.2, −21.4], p < 0.001 | −20.2 pp [−29.8, −10.5], p < 0.001 | −9.6 pp [−19.3, 0.0], p = 0.142 |
| Correction SFT V3 | −33.3 pp [−43.6, −23.1], p < 0.001 | −24.6 pp [−34.2, −14.0], p < 0.001 | −8.8 pp [−18.4, +0.9], p = 0.142 |

Showing any candidate answer sharply reduces repair, and an irrelevant wrong
answer accounts for most of the drop. The additional penalty for the model's
*own* answer is 9–12 points in every checkpoint, but it does not survive Holm
correction. Specific self-anchoring is therefore **inconclusive**, not
absent. Own-answer visibility also changes about a quarter of initially
correct answers to wrong (69–79 of 283), more than double the blind arm.

### Step 3 — probe-routed blind re-solve

| | Original | V2 | V3 |
|---|---:|---:|---:|
| Flagged (of 400) | 152 | 152 | 163 |
| Probe wrong recall | 65.0% | 64.1% | 65.0% |
| Probe precision | 50.0% | 49.3% | 46.6% |
| Correct preservation | 73.1% | 72.8% | 69.3% |
| KEEP-all accuracy | 70.75% | 70.75% | 70.75% |
| **Probe-routed accuracy** | **74.50%** | **75.00%** | **77.50%** |
| Routed − KEEP (95% CI) | +3.75 [+1.00, +6.50] | +4.25 [+1.50, +7.25] | +6.75 [+3.75, +9.75] |
| Holm p | 0.015 | 0.015 | < 0.001 |
| Routed fixes / harms | 23 / 8 | 27 / 10 | 33 / 6 |
| BLIND-all accuracy | 75.00% | 75.50% | 78.00% |
| Random equal-count routing | 71.25% | 73.25% | 74.75% |

The preregistered primary gate (routed accuracy above KEEP-all, with the 95%
CI excluding zero after Holm correction, for at least one checkpoint) **passes
for all three checkpoints**.

## Interpretation

1. **The gain is real but not self-correction.** Re-solving every answer blind
   is as accurate as probe routing or slightly better, and the original
   solver's greedy same-prompt control (76.75%) matches the routed pipeline
   without any probe or re-solve instruction. The improvement comes mainly
   from a second, answer-hidden attempt, partly from switching temperature
   0.7 to greedy decoding.
2. **The probe adds efficiency, not accuracy.** It re-solves 38–41% of rows
   while retaining nearly all of the BLIND-all gain. It beats equal-count
   random routing by 1.75–3.25 points, but that comparison was descriptive,
   not a preregistered test.
3. **Visible answers are harmful context.** The large Phase 7 blind-versus-
   visible gap is reproduced on fresh sources, but most of it is also caused
   by an irrelevant wrong answer. The model does not reliably use a visible
   candidate as evidence; it tends to follow it.
4. **The standalone model still does not detect and repair its own errors.**
   The working system is an external harness: sample, probe-flag, re-solve
   blind, and optionally verify.

## Deviations and limits

- **Amendment v2 (2026-09-30).** The RTX 3090 host and every v1 raw output
  were lost, so all generation restarted on one Tesla V100-SXM2-32GB (vLLM
  0.7.0, FP16, xFormers). Requests were batched 64 per call with unchanged
  prompts, per-request seeds, decoding, and 768-token cap. Batching can change
  greedy outputs relative to one-at-a-time decoding; every checkpoint and arm
  used the same method. See `../RUN_STATUS.md` and
  `../data/execution_lock_v2.json`
  (`c0858a6d5758606082db300726a12e24e32338bf070d943d1b8a3673a0e65543`).
- **Regenerated donors.** The Phase 7 donor pool was regenerated with the
  frozen Phase 7 initial protocol (339 rows; 239 correct, 100 wrong). These
  rows are donor text only, not Phase 7 results.
- **Dropped replay.** The exploratory same-seed Phase 7 replay could not run
  without the lost historical text.
- **Probe provenance improved.** The exact Phase 5 probes were used instead of
  the planned rebuilt probe.
- **Retries.** V3 failed twice before generating any answer because the host
  lacked a C compiler and Python headers for vLLM's LoRA Triton kernels. They
  were installed, and the resumable pipeline continued; no output was
  regenerated or discarded.
- **Truncation.** 1.5% of first-pass answers (9/1200) and 3.2% of three-arm
  answers (116/3585) hit the 768-token cap; they are scored as generated.
- **Scope.** All sources are arithmetic word problems. The first answer is
  always the original solver's, so V2 and V3 act only as re-solvers. The
  v1 timing deviation and inadvertent row print recorded in `RUN_STATUS.md`
  concerned lost v1 outputs and did not affect v2 decisions.

## Disposition

- No checkpoint, probe, or threshold is promoted as an autonomous
  self-corrector.
- A harness of *probe → blind re-solve* is a documented, protected-set-
  supported accuracy gain over KEEP-all on this pool, with the caveat that
  blind re-solving everything performs at least as well.
- The Phase 8 protected pool is now opened and must not be used to tune
  prompts, thresholds, seeds, or models.
- Any continuation should open a new phase with a fresh holdout. A natural
  next question is whether multiple independent attempts plus a selector
  (for example majority vote or a verifier-free agreement signal) beat the
  single-probe route, since the gain appears to come from resampling.

## Artifacts

All v2 outputs were generated at `/root/AGI_phase8/` on the V100 host and
mirrored to `outputs/phase8_remote_v100/`; the `outputs/...` paths below are
relative to that mirror. All 35 mirrored files match the remote SHA-256, and
every summary hash matches its sibling file. The frozen distractor map is also
committed under `phase8/data/distractors_v2/`.

- Donor pool: `outputs/phase7_initials_regen_v2/`
- First passes: `outputs/phase8_first_pass_v2/{initial,sample_repeat,greedy_same_prompt}/`
- Distractor map: `phase8/data/distractors_v2/`
- Three arms: `outputs/phase8_three_arms_v2/<checkpoint>/`
- Probe scores: `outputs/phase8_probe_scores_v2/<checkpoint>/`
- Protected analysis: `outputs/phase8_analysis_v2/report.json` and `verdicts.jsonl`
