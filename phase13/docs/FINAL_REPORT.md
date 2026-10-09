# Phase 13 final report — confirm, constrain, and combine

**Status: closed on 2026-10-09.**

- **A (confirm on harder data): judgment passed, practical value failed.** The
  unchanged Phase 12 judge, on 561 GSM8K test problems (the difficulty where
  Phases 9–11 failed), is right 79.3% of the time when its both-orders verdict
  is consistent (untrained 71.0%). Its self-check trails compute-matched
  vote@5.
- **B (remove invented answers): passed.** A constrained-verdict DPO judge cuts
  invented answers from 9.9% to 2.1%, raises coverage from 52% to 63%, and
  keeps a 74.0% consistent accuracy.
- **C (judge only on split votes): failed.** It does not beat vote@3 and
  trails vote@5 by 6.6 points.
- **Exploratory layer study:** the information about which solution is right
  is already present in the untrained model's hidden states (about 70%
  cross-dataset, layers 16–23). DPO changes how the model uses it, not whether
  it has it.

Vote@5 remains the most accurate method (77.5%). No adapter replaces voting.

## 1. Questions

Phase 12 passed both primary endpoints on SVAMP, but the untrained judge was
already above chance there, invented answers (10.5%) missed their target, and
vote@5 stayed more accurate. Phase 13 asked:

- **A:** Is the Phase 12 result the method or the dataset?
- **B:** Can a forced "Verdict: Solution A/B" line remove invented answers
  without losing judgment?
- **C:** Can the judge add accuracy on top of voting if it is called only when
  a three-way vote is split?

All three were preregistered together, ran in one GPU pipeline, and were
scored in a single protected opening.

## 2. How it was run

- **Holdout:** GSM8K test problems 750–1318 (`openai/grade-school-math`,
  SHA-256 `3730d312…`). Indices 0–749 were excluded outright (Phase 1's
  evaluation range) and 8 more overlapped earlier files: **561 problems**,
  none ever used for training.
- **Attempts:** five per problem at temperature 0.7 (s1 Phase 1 prompt, s2–s5
  blind prompt), 2,805 in total. Pairs (s1, s2) and (s1, s3) with different
  final answers — **486 pairs** — were each judged greedily **in both orders**
  by four judges (972 judgments each):

| Judge | Adapter | Prompt |
|---|---|---|
| `base` | none | Phase 9 judge prompt |
| `p12` (A, C) | Phase 12 order-swapped DPO LoRA, unchanged (`628982a0…`) | Phase 9 judge prompt |
| `base_c` | none | constrained (must end "Verdict: Solution A/B") |
| `p13b` (B) | new constrained-verdict DPO LoRA | constrained |

- **Direction-B training:** 1,648 order-swapped pairs built locally from Phase
  10 training-pool judgments (one-right pairs only; chosen = correct judgment +
  right verdict; rejected = sided-with-wrong judgment + wrong verdict, or an
  invented-answer judgment). 1,470 training (735 A-right / 735 B-right) and 178
  validation pairs. DPO as in Phase 12 (beta 0.1, learning rate 2e-5, one epoch,
  92 steps). Validation preference accuracy rose 0% → 82.6% (step 50) → 84.3%
  (step 92); validation loss 0.693 → 0.400.
- **Integrity:** preregistration and lock (`data/execution_lock_v1.json`,
  26 files, `15f33493…`) frozen before generation; verified on the GPU before
  generation and before the opening. Tesla V100-SXM2-32GB, vLLM 0.7.0. All 21
  result files match the GPU copies by SHA-256.

## 3. Results

The first answer was correct on 371/561 (66.1%). **288 judged pairs (from 217
problems) had exactly one right solution.**

### Judges (one-right pairs)

| Measure | `base` | **`p12` (A)** | `base_c` | **`p13b` (B)** |
|---|---:|---:|---:|---:|
| Consistent both-orders accuracy | 71.0% (98/40) | **79.3%** (119/31) | 63.8% (90/51) | **74.0%** (134/47) |
| Coverage (consistent share) | 47.9% | 52.1% | 49.0% | **62.8%** |
| Single-order accuracy | 52.3% | 61.8% | 54.9% | **64.1%** |
| Single-order clustered 95% CI | [47.3, 57.1] | [57.3, 66.3] | [50.4, 59.1] | [59.4, 68.6] |
| Invented answers | 16.7% | 9.9% | 4.9% | **2.1%** |
| Position-2 rate (target 40–60%) | 53.8% | 55.7% | 37.0% | 45.9% |
| Tie-break score (inconsistent = 0.5) | 0.601 | **0.653** | 0.568 | **0.651** |

### Preregistered endpoints

| Dir. | Endpoint | Result | Threshold | |
|---|---|---|---|---|
| A | **A-P1** | `p12` 79.3% consistent, coverage 52.1%, Holm p ≈ 2e−13 | > 50%, p < 0.05, coverage ≥ 30% | ✅ pass |
| A | **A-P2** | self-check (3.09 calls) vs **vote@5**: −5.35 pts [−7.66, −3.03] | lower bound > −2 | ❌ fail |
| A | A-Sec | `p12` single-order 61.8%, Holm p ≈ 8e−9 | > 50%, p < 0.05 | ✅ pass |
| B | **B1** | `p13b` invented 2.1% | < 5% | ✅ pass |
| B | **B2** | `p13b` 74.0% consistent, coverage 62.8%, p ≈ 4e−11 | > 50%, p < 0.05, coverage ≥ 30% | ✅ pass |
| C | **C1** | split-judge vs vote@3: −1.07 pts [−2.85, +0.53], p = 0.31 | CI > 0, p < 0.05 | ❌ fail |
| C | **C2** | split-judge vs vote@5: −6.60 pts [−9.27, −4.10] | lower bound > −2 | ❌ fail |

**A-P2 note:** the self-check averaged 3.09 model calls, just above 3, so the
preregistered rule matched it against vote@5. Against vote@3 (descriptive
only) it is 72.19% vs 72.01%.

**B report:** `p13b` vs `p12` tie-break difference −0.17 points, clustered CI
[−4.39, +3.95] — the same overall quality, reached with more coverage and far
fewer invented answers.

### End-to-end accuracy (561 problems)

| Strategy | Accuracy | Fixes (of 190 wrong) | Harms (of 371 right) | Mean calls |
|---|---:|---:|---:|---:|
| keep | 66.13% | — | — | 1 |
| agree-gated = vote@3 | 72.01% | 59 | 26 | 2.43 / 3 |
| self-check `base` | 70.59% | 55 | 30 | 3.09 |
| **self-check `p12`** | **72.19%** | 60 | 26 | 3.09 |
| self-check `base_c` | 71.30% | 54 | 25 | 3.08 |
| self-check `p13b` | 71.30% | 56 | 27 | 3.03 |
| split-judge `p12` (C) | 70.94% | 61 | 34 | 4.09 |
| split-judge `p13b` | 69.34% | 58 | 40 | 4.09 |
| **vote@5** | **77.54%** | **82** | **18** | 5 |

## 4. Exploratory layer study (not preregistered)

Run after the opening on the same GPU with Transformers
(`layer/extract_hidden.py`, `layer/probe_analysis.py`; report
`outputs/phase13_v1/layer/probe_report.json`). For every judged prompt, the
hidden state of the **last prompt token** — after reading both solutions,
before writing anything — was saved at every layer (0–28). A probe
(StandardScaler + L2 logistic regression, C = 0.01, as in Phase 5) predicts
which shown position holds the right solution on one-right pairs. For `base`
and `p12`, the probe was **trained on Phase 12 SVAMP and tested on Phase 13
GSM8K**; its layer was chosen by source-grouped CV on SVAMP only.

| Layers | SVAMP CV (p12) | GSM8K cross-dataset (p12) |
|---|---:|---:|
| 0 (embeddings) | 50% | 50% |
| 1–15 | 63–70% | 58–66% |
| **16–23** | **75–77%** | **66–70%** |
| 24–28 | 73–75% | 64–68% |

| Judge | Chosen layer | GSM8K accuracy | GSM8K AUC |
|---|---:|---:|---:|
| `base` | 21 | 69.3% | 0.765 |
| `p12` | 21 | 70.0% | 0.776 |
| `base_c` (within-GSM8K CV) | 19 | 71.5% | — |
| `p13b` (within-GSM8K CV) | 23 | 71.5% | — |

Pair level (average of both orders, GSM8K one-right pairs, chosen layer):

| Pairs | `p12` probe accuracy |
|---|---:|
| All (n = 288) | 75.0% |
| Judge-consistent (n = 150) | 76.0% (the judge's written verdict: 79.3%) |
| **Judge-inconsistent (n = 138)** | **73.9%** |

Exploratory strategy — `p12` self-check that uses the probe instead of vote@3
when the judge is inconsistent: 71.84%, versus vote@3 72.01% (−0.18 [−2.50,
+2.14]) and vote@5 77.54% (−5.70 [−8.56, −2.85]).

Findings:

1. **The model holds the answer internally before writing.** About 70%
   cross-dataset (AUC ≈ 0.77), concentrated in layers 16–23 and slightly
   weaker at the final layers, consistent with Phase 6's attenuation finding.
2. **DPO barely changes the internal signal** (69.3% → 70.0%). The untrained
   model already has it; training changes how consistently the judgment is
   expressed, not what is known.
3. **The probe resolves pairs the written judge leaves open** (73.9% on
   judge-inconsistent pairs), but this does not raise end-to-end accuracy:
   when both solutions are wrong, choosing either loses, whereas a further vote
   can still find a right answer.

## 5. Interpretation

1. **The method transfers.** On GSM8K-level problems the both-orders judge is
   clearly above chance, untrained (71.0%) and more so after Phase 12 DPO
   (79.3%), with no retraining on GSM8K. Part of Phase 12's SVAMP result was
   the method, not only the easier data.
2. **Forcing a verdict fixes invented answers** without losing quality, and
   yields the widest coverage. The untrained constrained judge, however,
   overcorrects toward Solution A (position-2 rate 37%); training restores
   balance (46%).
3. **The ceiling is selection between two answers.** Judge, constrained
   judge, and probe all choose between existing solutions. When both are wrong
   none can recover, so five independent attempts with a vote remain clearly
   better (fewer harms and more fixes).

## 6. Deviations and limits

- No deviation from the preregistered protocol; the pipeline completed in one
  attempt. Two code defects (a text hash applied to a binary adapter, and a
  duplicated `raise`) were caught and fixed before the lock.
- A-P2's compute-matching rule selected vote@5 because the self-check used
  3.09 calls on average; the rule was applied as written.
- The layer study is exploratory: it ran after the opening, and only its
  cross-dataset design (train on SVAMP, test on GSM8K) protects it from
  tuning on the holdout. `base_c` and `p13b` probes use within-GSM8K CV.
- Arithmetic word problems only; one 7B checkpoint; one training configuration.

## 7. Disposition and next steps

- `phase13-verdict-judge` (`p13b`) is the best-behaved judge (fewest invented
  answers, highest coverage); `phase12-dpo-judge` (`p12`) has the highest
  consistent accuracy. Neither replaces vote@5. Both are served by
  `serving/serve.sh`.
- The GSM8K test holdout (750–1318) is opened; never train on it or tune
  against it.
- Next lever: detect when **both** candidates are likely wrong and spend more
  attempts only then — for example a probe or judge signal that triggers extra
  sampling — and test it against vote@5 at matched compute on a new dataset.

## Artifacts

Mirrored under `outputs/phase13_remote_v100/outputs/phase13_v1/`:
`samples/`, `judge_base/`, `judge_p12/`, `judge_base_c/`, `judge_p13b/`,
`dpo_b_lora/` (adapter, log, summary), `analysis/report.json`, and
`layer/probe_report.json`. The hidden-state arrays (about 1.2 GB) remain on the
GPU host at `/root/AGI_phase12/outputs/phase13_v1/layer/`. Direction-B training
data: `outputs/phase13_v1/dpo_b/`.
