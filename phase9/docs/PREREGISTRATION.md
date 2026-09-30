# Phase 9 preregistration — independent attempts, voting, and self-check

Written on 2026-09-30 before any Phase 9 generation. The execution lock
(`phase9/data/execution_lock_v1.json`) hashes this file, the source pool, and
every generation and analysis script.

## Motivation

Phase 8 found that a second attempt *without seeing the first answer* explains
most of the probe-routed gain, and that any visible candidate answer pulls the
model toward copying. Phase 9 asks whether the model can check itself using
only its own independent attempts, with no external probe:

- **Option A (voting):** several independent attempts, plurality answer wins.
- **Option B (self-check):** compare the first answer with one independent
  blind attempt. If they agree, keep. If they disagree, the model sees both
  solutions and writes a final one.

## Sources

`data/source_pool_v1/`: 40 development and 400 protected GSM8K train
questions, disjoint from every Phase 1–8 source. The builder first reproduced
the frozen Phase 8 selection exactly (1374 eligible; identical 400 IDs), then
over-excluded all JSON/JSONL under the Phase 1–4 directories, all Phase 5–7
data, and the Phase 8 pool (974 eligible). The development rows are a
technical smoke only; no prompt, rule, or threshold is tuned on them.

## Generation

- **A0 (shared first answer):** original solver, frozen Phase 1 math prompt,
  temperature 0.7, seed 20260930 + index.
- **Per checkpoint** (original solver, merged warm-start V2, V3 LoRA on merged
  V2):
  - **B1–B4:** four blind attempts with the exact Phase 7/8 blind prompt,
    temperature 0.7, top-p 1, hashed per-request seeds.
  - **J (self-check judge):** greedy; the problem, the header in
    `scripts/collect_checkpoint.py`, then "Solution A" = A0 and
    "Solution B" = B1, both unlabeled as to correctness. Generated for every
    source; used only where A0 and B1 disagree.
- vLLM 0.7.0, FP16, 768-token cap, 4096 context, batches of 64, one Tesla
  V100-SXM2-32GB.
- **Probe:** the exact frozen Phase 5 probes score A0 (layer 14, threshold 0.5).

## Answer agreement

Final answers are compared without gold using the Phase 1 extraction and SymPy
equality (`scripts/answers.py`). An unparseable answer never agrees. Plurality
ties are broken by attempt order B1, B2, …, then A0.

## Strategies (per checkpoint)

| Strategy | Rule | Generations |
|---|---|---:|
| keep | A0 | 1 |
| blind1 | B1 | 2 |
| maj3 | plurality of A0, B1, B2 | 3 |
| maj5 | plurality of A0, B1–B4 | 5 |
| agree_gated | A0 if A0 = B1, else maj3 | 2–3 |
| self_check | A0 if A0 = B1, else J | 2–3 |
| probe_blind | B1 if the probe flags A0, else A0 | 1 + flag rate |

## Hypotheses and gates

Primary (each family Holm-corrected across the three checkpoints):

- **H-A:** maj5 accuracy > keep.
- **H-B:** self_check accuracy > keep.

A primary claim requires a positive paired difference whose 95% source-level
bootstrap interval (10,000 resamples) excludes zero **and** Holm-adjusted
exact paired sign-test p < 0.05, for at least one checkpoint.

Secondary (Holm within each named contrast; descriptive otherwise): maj5 vs
blind1; maj3 vs keep; agree_gated vs keep; self_check vs agree_gated;
self_check vs probe_blind; maj5 vs probe_blind. Also reported: disagreement
(A0 ≠ B1) versus the probe as error detectors (recall, precision, correct
preservation); on disagreement rows, whether J copies A0, copies B1, or gives
another answer, and its correctness; fixes, harms, and mean generations.

## Interpretation rules

- self_check beating agree_gated would indicate that the model's own judgment
  of two visible solutions adds value beyond mechanical voting. If
  self_check is worse, Phase 8's copying problem extends to comparing two
  candidates.
- Accuracy gains from voting are a sampling harness, not autonomous
  self-correction.
- A null or inconclusive interval is not evidence of no effect.

## Integrity

Protected labels are opened once, after all generation and probe scoring,
by `scripts/analyze.py`, which refuses to run twice. No Phase 9 output enters
training. The Phase 5–8 protected sets are not reused. Deviations are recorded
in `RUN_STATUS.md` with timing relative to protected opening.
