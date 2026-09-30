# Phase 9 final report — independent attempts, voting, and self-check

**Status: closed on 2026-10-01. Option A (plurality vote over five independent
attempts) passed its preregistered gate for all three checkpoints. Option B
(model self-check of two conflicting solutions) failed and was significantly
worse than mechanical voting. No checkpoint is promoted.**

## 1. Question

Phase 8 showed that a second attempt *without seeing the first answer* drives
most of the gain, and that a visible candidate answer gets copied rather than
checked. Phase 9 asked whether the model can check itself using only its own
independent attempts, with no external probe:

- **Option A — voting:** several independent attempts; the plurality final
  answer wins.
- **Option B — self-check:** compare the first answer with one independent
  blind attempt. If they agree, keep it. If they disagree, the model sees both
  solutions, checks them, and writes a final solution.

## 2. How it was run

### Sources

`data/source_pool_v1/`: 40 development and 400 protected GSM8K train
questions, disjoint from every Phase 1–8 source.

Several Phase 1–4 inventory files hashed by the Phase 5 audit are missing (34)
or changed (142) on this workstation, so strict re-hashing was impossible.
The builder therefore:

1. rebuilt the Phase 8 selection from the 252 inventory files still present and
   required an exact match — 1374 eligible sources and the identical 400
   frozen IDs (passed);
2. over-excluded every JSON/JSONL under the Phase 1–4 directories, all Phase
   5–7 data, and the Phase 8 pool (974 eligible — exactly 1374 − 400, so
   nothing beyond the Phase 8 pool was additionally removed);
3. ranked by SHA-256(`phase9_source_pool_v1|20260930|index`) and took 40
   development then 400 protected rows.

### Generation

| Output | Model | Prompt | Decoding |
|---|---|---|---|
| A0 — shared first answer | original solver | frozen Phase 1 math prompt | temperature 0.7, seed 20260930 + index |
| B1–B4 — blind attempts (per checkpoint) | original / merged V2 / V3 on merged V2 | exact Phase 7/8 blind prompt | temperature 0.7, top-p 1, hashed seeds |
| J — self-check (per checkpoint) | same | problem + check instruction + "Solution A" = A0 + "Solution B" = B1 | greedy |

The frozen Phase 5 probes (layer 14; C = 0.01 / 0.01 / 1.0; threshold 0.5)
scored A0. Hardware and runtime: one Tesla V100-SXM2-32GB, vLLM 0.7.0, FP16,
batches of 64, 768-token output cap, 4096 context. About 7,000 generations in
75 minutes (15:39–16:54 UTC).

### Strategies scored

| Strategy | Rule | Generations |
|---|---|---:|
| keep | A0 | 1 |
| blind1 | B1 | 2 |
| maj3 | plurality of A0, B1, B2 | 3 |
| maj5 | plurality of A0, B1–B4 | 5 |
| agree_gated | A0 if A0 = B1, else maj3 | ~2.3 |
| self_check | A0 if A0 = B1, else J | ~2.3 |
| probe_blind | B1 if the probe flags A0, else A0 | ~1.4 |

Answers are compared without gold using the Phase 1 extraction and SymPy
equality; unparseable answers never agree. Plurality ties go to B1, B2, …,
then A0.

### Statistics

Paired differences on the same 400 sources; source-level bootstrap 95%
intervals (10,000 resamples); exact paired sign tests on discordant sources;
Holm correction across the three checkpoints within each named contrast.
Primary gates: **H-A** maj5 > keep and **H-B** self_check > keep, each
requiring a CI above zero and Holm p < 0.05 for at least one checkpoint.

### Integrity

- Preregistration and execution lock written before any generation
  (`data/execution_lock_v1.json`, 23 files, LF-normalized hashing,
  `ead98d7c7dd4b526e8a9440b2fdca0766fa0147109069e038fb7b604d9a2e482`).
- Lock verified on the GPU host before generation and again before the single
  protected opening. The analyzer refuses to run twice.
- All 29 output files mirrored locally match the GPU copies by SHA-256.
  Verdict rows SHA-256
  `50c17468bd0fa931fe71b5b15c7ff64df2f3ba8484650627a21fe9f8c6c6a757`.

## 3. Results (protected, n = 400)

The shared first answer was correct on 300/400 sources (75.0%); 100 were wrong.

### Accuracy by strategy

| Strategy | Original | V2 | V3 |
|---|---:|---:|---:|
| keep | 75.00% | 75.00% | 75.00% |
| blind1 | 68.50% | 71.00% | 72.75% |
| maj3 | 79.25% | 78.75% | 79.75% |
| **maj5** | **84.50%** | **85.50%** | **83.25%** |
| agree_gated | 79.25% | 78.75% | 79.75% |
| **self_check** | **72.25%** | **72.75%** | **74.00%** |
| probe_blind | 75.25% | 75.00% | 75.25% |

`agree_gated` equals `maj3` by construction: when A0 and B1 agree, the vote of
three returns that shared answer.

### Fixes (of 100 wrong) and harms (of 300 correct)

| Strategy | Original fixes / harms | V2 | V3 |
|---|---:|---:|---:|
| blind1 | 36 / 62 | 36 / 52 | 35 / 44 |
| maj3 = agree_gated | 33 / 16 | 35 / 20 | 31 / 12 |
| **maj5** | **46 / 8** | **50 / 8** | **42 / 9** |
| self_check | 23 / 34 | 20 / 29 | 22 / 26 |
| probe_blind | 23 / 22 | 24 / 24 | 18 / 17 |

### Primary contrasts (difference, 95% CI, Holm p)

| Contrast | Original | V2 | V3 |
|---|---|---|---|
| maj5 − keep | +9.50 pp [+6.00, +13.00], p < 0.001 | +10.50 [+7.00, +14.25], p < 0.001 | +8.25 [+5.00, +11.75], p < 0.001 |
| self_check − keep | −2.75 [−6.50, +1.00], p = 0.55 | −2.25 [−5.75, +1.25], p = 0.55 | −1.00 [−4.50, +2.25], p = 0.67 |

**H-A passes for every checkpoint. H-B fails for every checkpoint.**

### Secondary contrasts (difference, Holm p)

| Contrast | Original | V2 | V3 |
|---|---|---|---|
| maj5 − blind1 | +16.00, p < 0.001 | +14.50, p < 0.001 | +10.50, p < 0.001 |
| maj5 − probe_blind | +9.25, p < 0.001 | +10.50, p < 0.001 | +8.00, p < 0.001 |
| maj3 − keep | +4.25, p = 0.043 | +3.75, p = 0.058 | +4.75, p = 0.016 |
| agree_gated − keep | +4.25, p = 0.043 | +3.75, p = 0.058 | +4.75, p = 0.016 |
| **self_check − agree_gated** | **−7.00, p < 0.001** | **−6.00, p < 0.001** | **−5.75, p = 0.001** |
| self_check − probe_blind | −3.00, p = 0.33 | −2.25, p = 0.47 | −1.25, p = 0.59 |

### Error detection

| Detector | Original | V2 | V3 |
|---|---:|---:|---:|
| A0 ≠ B1: flagged | 147 | 137 | 126 |
| A0 ≠ B1: wrong recall / precision | 85% / 58% | 85% / 62% | 82% / 65% |
| A0 ≠ B1: correct preservation | 79% | 83% | 85% |
| Probe: flagged | 157 | 153 | 166 |
| Probe: wrong recall / precision | 69% / 44% | 68% / 44% | 70% / 42% |
| Probe: correct preservation | 71% | 72% | 68% |

### Self-check behavior on disagreements

| | Original | V2 | V3 |
|---|---:|---:|---:|
| Disagreements | 147 | 137 | 126 |
| Final answer = A0's | 50 (28 correct) | 46 (23) | 45 (18) |
| Final answer = B1's | 74 (23 correct) | 64 (19) | 52 (17) |
| Another answer | 23 (0 correct) | 27 (1) | 29 (5) |
| Real fixes (A0 wrong → right) | 23 | 19 | 17 |
| False alarms (A0 right → wrong) | 34 | 29 | 26 |
| **Sided with the right one, when exactly one is right** | **51/98 (52%)** | **42/88 (48%)** | **35/79 (44%)** |

The model writes in the Phase 1 correction format ("Phát hiện lỗi /
Nguyên nhân / Sửa lại"), but its choice between two conflicting solutions is
at chance. Full verified examples are in [CASE_REPORT.md](CASE_REPORT.md).

### Development split (n = 40, technical smoke, not gated)

First answers 33/40 correct. maj5 reached 35–37/40; self_check 32–33/40.

## 4. Interpretation

1. **The model can tell *that* something is wrong, not *which* answer is
   right.** Disagreement between two independent attempts catches 82–85% of
   first-answer errors, better than the hidden-state probe, but the model's
   written judgment between the two candidates is near chance.
2. **Voting is the working method.** Five independent attempts with plurality
   selection add about ten points without training while breaking fewer
   correct answers than any other strategy. It is a sampling harness, not
   self-correction.
3. **Visible candidates still hurt.** Consistent with Phase 8, showing two
   solutions to compare is worse than sampling a third independent attempt.
4. **A single sampled retry is not enough.** At temperature 0.7 it is below
   keep (44–62 harms); the Phase 8 single-retry gain came from greedy
   decoding.

## 5. Deviations and limits

- No deviation from the preregistered protocol; the pipeline completed in
  one attempt.
- The local backup process stopped when the Claude Code session restarted; it
  was restarted outside the session, and all 29 files were then verified.
- The case report was built on the GPU host with the same Python 3.10 and
  SymPy 1.14.0 as the analysis.
- Arithmetic word problems only; the first answer always comes from the
  original solver; the three checkpoints share A0, so their results are
  correlated.

## 6. Disposition

- No checkpoint, prompt, or probe is promoted. Plurality voting over
  independent attempts is the best supported accuracy method so far.
- The Phase 9 protected pool is opened; never train on it or tune against it.
- **Next target (Phase 10):** the judgment step. Train the model to choose
  correctly between two conflicting solutions and explain why, then compare
  against the maj3/agree_gated and maj5 baselines on a fresh holdout.

## Artifacts

Mirrored under `outputs/phase9_remote_v100/`: `outputs/phase9_first_v1/`,
`outputs/phase9_checkpoints_v1/<checkpoint>/`,
`outputs/phase9_probe_scores_v1/<checkpoint>/`,
`outputs/phase9_analysis_v1/`, and `logs/phase9_pipeline.log`.
