# Paper checklist — definitions, status, and results

Paper: *Position bias hides self-arbitration skill* (working title). Source
checklist: `Checklist trước khi viết paper Position bias che giấu kỹ năng tự
phân xử.docx` (2026-10-09).

**Central claim.** When a 7B model arbitrates between two of its own
conflicting solutions, position bias makes the skill look like chance, while
the hidden state still carries a right/wrong signal of about 70–75%.

**Model.** Qwen2.5-7B-Instruct → `Kxck/Self_Correction_v1` (Phase 1 SFT) →
LoRA judge adapters `p12` (Phase 12 order-swapped DPO) and `p13b` (Phase 13
constrained-verdict DPO). Untrained judges: `base` (Phase 9 prompt) and
`base_c` (constrained prompt).

**Data.** GSM8K test 750–1318 (Phase 13: 561 problems, 486 judged pairs, 288
one-right pairs from 217 problems) and SVAMP (Phase 12: 960 problems, 501
judged pairs, 367 one-right pairs from 287 problems). Each pair is judged in
both orders: "ab" shows s1 as Solution A, "ba" shows it as B.

**Status legend.** ✅ done · ⏳ needs GPU · ⚠️ done with a caveat · ⬜ not started.
All CPU numbers come from `paper/scripts/cpu_checks.py` →
`paper/results/cpu_checks.json` and `paper/scripts/a1_sensitivity.py` →
`paper/results/a1_sensitivity.json` (Python 3.10 + SymPy 1.14, the GPU
verifier versions). Everything in this file is post-hoc unless marked as a
preregistered gate (see A4).

---

## A. Data integrity (required)

### A1 ✅⚠️ Contamination audit of the GSM8K holdout

*Definition.* Compare the 561 holdout questions with every JSON/JSONL file in
`phase1/`–`phase13/` and `outputs/` (705 files): exact normalized-question
match plus 8-word-shingle Jaccard ≥ 0.5; separately against all 7,473 GSM8K
train questions.

*Result.*
- **42 holdout problems** match exactly (Jaccard 1.0), all in
  `outputs/p0_base_100_log.jsonl` — the 2026-09-02 **P0 evaluation** of Qwen
  base versus `Self_Correction_v1` on a "600 held-out GSM8K" pool. That run was
  evaluation only; the Phase 13 exclusion scan had covered the `phaseN/`
  folders but not `outputs/`.
- **No match in any training data**, and no GSM8K-train near-duplicate
  (Jaccard ≥ 0.5) for any holdout question.

*Amendment (sensitivity, 519 problems / 271 one-right pairs after removing the
42):* every conclusion is unchanged.

| Judge | Consistent accuracy (561 → 519) | Coverage | Tie-break | Invented |
|---|---|---:|---:|---:|
| base | 71.0% → 70.3% | 47.2% | 0.596 | 16.8% |
| p12 | 79.3% → 78.3% | 52.8% | 0.649 | 9.8% |
| base_c | 63.8% → 63.4% | 49.5% | 0.566 | 5.0% |
| p13b | 74.0% → 73.8% | 63.5% | 0.651 | 2.0% |

In the paper, report the 519-problem numbers as primary (or both), and state
that 42 problems had appeared in an earlier evaluation-only run.

### A2 ✅ Splits by source problem

*Definition.* No source problem may straddle train/validation or CV folds; both
orders and both pairs (s1, s2), (s1, s3) of a problem stay together.

*Result.* DPO train/validation problem overlap is **0** for Phase 11 (401/56
problems), Phase 12 (520/73), and Phase 13B (471/58). Probe CV uses
`GroupKFold(groups = problem id)`; cross-dataset probes train on SVAMP and test
on GSM8K (disjoint by construction).

### A3 ✅ Verdict parse rates

| Judge (GSM8K) | No extractable answer | Unparseable answer | Hit 768-token cap | Missing "Verdict" line |
|---|---:|---:|---:|---:|
| base | 0.1% | 0.5% | 3.4% | — |
| p12 | 0.0% | 0.3% | 4.0% | — |
| base_c | 0.2% | 0.4% | 3.7% | **19.3%** |
| p13b | 0.0% | 0.1% | 4.8% | 4.6% |

SVAMP: base 1.0% unparseable, p12 0.7%; caps 3.5–3.8%. Parsing is not a
problem. For `base_c`, 19% of judgments omit the verdict line and fall back to
answer matching; this is part of why the untrained constrained judge looks
different (A-biased, see B1).

### A4 ✅ Preregistered versus exploratory

| Preregistered gates (cite as confirmatory) | Exploratory / post-hoc (label as such) |
|---|---|
| Phase 12: P1 74.9% (p12, SVAMP), P2 non-inferior vs vote@3, Secondary 61.9% | Everything in this file (B1–B5, C2, F3, A1 sensitivity) |
| Phase 13: A-P1 79.3%, A-Sec 61.8%, A-P2 fail, B1 2.1%, B2 74.0%, C1/C2 fail | Phase 11 section 5b bias-vs-skill table |
| Phases 9–11 primary gates (judge at chance on GSM8K) | Phase 13 layer study (probe by layer, probe pair level) |

---

## B. Output level: bias hides skill (required)

### B1 ✅ Position decomposition — **the paper's key table**

*Definition.* Single-order judgments on one-right pairs, split by where the
right solution is shown; skill index = P(pick A | A right) − P(pick A | B right),
with a problem-clustered bootstrap CI.

**Untrained judge (`base`), GSM8K, 288 one-right pairs × 2 orders:**

| Right solution shown as | Picks A | Picks B | Invented |
|---|---:|---:|---:|
| A | **49.3%** | 34.4% | 16.3% |
| B | 27.8% | **55.2%** | 17.0% |

**Skill index +0.215** [+0.128, +0.304]. Single-order accuracy is only 52.3% —
near chance — yet the judge clearly responds to which solution is right; the
apparent chance level comes from position bias plus invented answers.

| Judge | Dataset | Skill index [95% CI] | Position-2 rate | Invented |
|---|---|---:|---:|---:|
| base | GSM8K | 0.215 [0.128, 0.304] | 53.7% | 16.7% |
| p12 | GSM8K | 0.319 [0.236, 0.402] | 55.7% | 9.9% |
| base_c | GSM8K | 0.135 [0.045, 0.223] | 37.0% | 4.9% |
| p13b | GSM8K | 0.316 [0.222, 0.408] | 45.9% | 2.1% |
| base | SVAMP | 0.248 [0.166, 0.332] | 54.3% | 13.2% |
| p12 | SVAMP | 0.330 [0.251, 0.411] | 55.1% | 10.5% |

Note: on this Phase 13 set the untrained judge's position-2 rate is 53.7%; the
stronger bias reported in Phase 11 (picks B 65% when B right vs 32% A when A
right) was on a different GSM8K holdout. Both show a positive skill index
masked at the single-order level.

### B2 ✅ Three numbers side by side

| Judge | Dataset | Tie-break score [CI] | Consistent accuracy | Coverage | Right / all one-right pairs |
|---|---|---:|---:|---:|---:|
| base | GSM8K | 0.601 [0.560, 0.642] | 71.0% | 47.9% | 34.0% |
| p12 | GSM8K | 0.653 [0.613, 0.692] | 79.3% | 52.1% | 41.3% |
| base_c | GSM8K | 0.568 [0.524, 0.610] | 63.8% | 49.0% | 31.2% |
| p13b | GSM8K | 0.651 [0.605, 0.696] | 74.0% | 62.8% | 46.5% |
| base | SVAMP | 0.609 [0.568, 0.649] | 68.7% | 58.3% | 40.1% |
| p12 | SVAMP | 0.649 [0.610, 0.688] | 74.9% | 59.7% | 44.7% |

Always report all three. "79%" is consistent-only; over all one-right pairs
the p12 judge is right 41% of the time and abstains (inconsistent) on 48%.

### B3 ✅ Paired judge comparisons (GSM8K, tie-break per pair, problem-clustered bootstrap, Holm over 4)

| Comparison | Difference | 95% CI | Holm p |
|---|---:|---:|---:|
| p12 − base | +5.2 pts | [+1.6, +8.8] | **0.016** |
| p13b − base | +5.0 pts | [+0.7, +9.4] | 0.054 (0.043 on 519 problems) |
| p13b − base_c | +8.3 pts | [+4.1, +12.5] | **0.0008** |
| p13b − p12 | −0.2 pts | [−4.3, +3.9] | 0.97 |

### B4 ✅ Single-order accuracy (model-level, no harness)

| Judge | GSM8K [clustered CI] | SVAMP |
|---|---:|---:|
| base | 52.3% [47.4, 57.2] | 56.3% [51.8, 60.8] |
| p12 | 61.8% [57.3, 66.2] | 61.9% [57.5, 66.2] |
| base_c | 54.9% [50.4, 59.3] | — |
| p13b | 64.1% [59.4, 68.7] | — |

### B5 ✅ Random-choice baseline

*Definition.* When s1 and s2 differ, pick one at random (expected accuracy);
otherwise keep s1. "Judge or random" uses the judge's consistent verdict and
random choice otherwise — no extra attempt, so it isolates the judge.

| Strategy | GSM8K | SVAMP |
|---|---:|---:|
| keep s1 | 66.1% | 82.1% |
| random s1/s2 | 65.7% | 81.6% |
| judge or random — base | 67.8% | 84.3% |
| judge or random — p12 | **69.9%** | **84.7%** |
| judge or random — p13b | 69.1% | — |
| self-check p12 (fallback vote@3, extra call) | 72.2% | 87.0% |
| vote@3 | 72.0% | 87.6% |
| vote@5 | 77.5% | 90.1% |

The random baseline is no better than keeping s1, so re-solving alone adds
nothing without a chooser. **The judge's own contribution is +4.2 points (p12)
over random choice on GSM8K and +3.1 on SVAMP**; the rest of the earlier "+6"
came from the vote@3 fallback's extra attempt.

---

## C. Representation level: the hidden state knows (required)

### C1 ⏳ Probe consistency across orders

*Definition.* Share of pairs where the probe picks the same solution in both
orders, next to the judge's own consistency (p12: 52% of one-right pairs).
*Status.* Script ready (`paper/scripts/c1_c4_probe.py`). The hidden-state
arrays lived only on the released GPU host; re-extraction (about 30 minutes)
is planned for the next GPU session.

### C2 ✅⚠️ Surface-feature baseline

*Definition.* Logistic regression on surface features of both solutions
(length, word and line counts, number count, log answer magnitude, integer and
negative answer, parse success; for A, B, and B − A), same rows and labels as
the probe.

| Model | SVAMP → GSM8K | Within-GSM8K CV |
|---|---:|---:|
| Surface, C = 0.01 | 62.5% | 65.6% |
| Surface, C = 1.0 | 64.2% | 65.6% |
| Hidden-state probe (p12, layer 21) | 70.0% (AUC 0.776) | — |
| Hidden-state probe (base_c / p13b) | — | 71.5% |

**The surface baseline is ~63–66%, near the checklist's "claim weakens"
threshold.** The probe beats it by about 5–7 points, so the representation
claim must be stated as "beyond surface cues by ~5–7 points", and a probe +
surface combined model (does the hidden state add information beyond surface?)
should be run when activations are re-extracted (C2b).

### C3 ⏳ Likelihood baselines

*Definition.* (a) Pick the solution with higher solver log-probability; (b)
pick the solution matching a fresh greedy solve. If the probe only matches
these, write "the model implicitly knows its own answer" rather than "the model
verifies solutions". *Status.* Needs GPU (~20 min).

### C4 ⏳ One probe protocol for all four judges

*Definition.* Train on SVAMP, test on GSM8K, layer chosen by SVAMP CV, for
base, p12, base_c, and p13b (constrained judges currently have only
within-GSM8K CV). *Status.* Script ready; needs SVAMP hidden states for base_c
and p13b plus re-extraction of the lost GSM8K arrays (~30 min GPU, same
session as C1).

### C5 ⏳ Logit lens by layer

*Definition.* At the end of the prompt and just before the verdict, track a
bias curve (overall lean to position 2) and a skill curve (A-right vs B-right
gap) across 28 layers. *Status.* Needs light GPU; candidate main figure.

---

## D. Origin of the bias (should have)

### D1b ✅ Phase 1 SFT data prior

*Result.* In all **631/631** samples of `phase1/backup_server/processed/phase1_sft.jsonl`,
the first answer is verifier-wrong ("SAI") and the later answer (the
correction) is the right one. The training data contain a perfect
"later = right" prior — supporting H1 (bias trained in). Caveat: confirm this
file is the exact `Self_Correction_v1` training set.

### D1 ⏳ Qwen2.5-7B-Instruct vs Self_Correction_v1 position bias (H1) — ~1 h GPU
### D2 ⏳ Label vs position 2×2 (H3) — ~1 h GPU
### D3 ⏳ Restate the problem after both solutions (H2, recency) — ~1 h GPU

Interpretation rules (written before running): H1 true → "self-correction
training created the bias"; H1 false and H2 true → innate recency; all false →
descriptive only. Holm over three hypotheses.

---

## E. Comparison with Ren et al. 2023 (should have)

- **E1 ⏳** Four-candidate selection (s1–s4) with permutation averaging — ~1–2 h GPU.
- **E2 ⬜** Put our tie-break score (the analogue of their order-averaged
  selection) next to their reported numbers; needs their exact figures.
- **E3 ✅ (draft)** Condition differences:

| | Ren et al. 2023 | This work |
|---|---|---|
| Candidates | 4 | 2 (E1: 4) |
| Model | PaLM-2 Large | Qwen2.5-7B (Self_Correction_v1) |
| Labels | model-graded | deterministic verifier (SymPy) |
| Selection | letter probability | generated judgment with reasoning |
| Reasoning step | no | yes (step-by-step check) |
| Data | TruthfulQA, TL;DR | GSM8K test, SVAMP |
| Trained for self-correction | no | yes (Phase 1 SFT; Phase 12–13 DPO judges) |

---

## F. Practical (optional)

- **F1 ⏳** Abstain-to-vote when the probe is unsure; check whether both-wrong
  pairs coincide with low-confidence pairs (needs re-extracted activations).
- **F2 ⏳** Probe-weighted voting over five solutions (needs pointwise probe; GPU).
- **F3 ✅ Ceiling.**

| | GSM8K | SVAMP |
|---|---:|---:|
| s1 ≠ s2 | 240 / 561 | 246 / 960 |
| of which both wrong | **103 (42.9%)** | 65 (26.4%) |
| Perfect choice between s1 and s2 | 77.9% | 91.0% |
| vote@5 | 77.5% | 90.1% |

A perfect two-way judge would only tie vote@5: with 43% of GSM8K disagreements
having no right candidate, two-way arbitration has a hard ceiling.

- **F4 ✅** Agreement-gated voting (vote@3 on disagreement) is included in every
  comparison: 72.0% GSM8K, 87.6% SVAMP.
- **F5 ✅ (definition).** Compute = number of model generation calls (solver
  attempts plus judge calls; a both-orders judgment costs 2). Token counts
  should be added as a secondary measure.

## G. Robustness (optional)

- **G1 ⏳** Repeat B1–B2 with an untrained Llama or Mistral 7–8B judge (several GPU hours).

---

## Claim wording — updated with evidence

| Current sentence | Rewrite | Evidence |
|---|---|---|
| "Self-evaluation 79%, Phase 0 goal met" | "79% when consistent (52% coverage); 41% of all one-right pairs right, tie-break 0.65" | B2 |
| "The model truly recognizes the right answer" | "A judgment signal well above chance (skill index +0.21 untrained, +0.32 trained)" | B1; C2 shows surface cues reach ~63–66% |
| "Preregistered, tested on new problems" | Keep, adding that 42 of 561 had appeared in an evaluation-only run; results unchanged without them | A1, A4 |
| "Self-correction +6 points vs no correction" | "+4.2 points over random choice (judge's own contribution); the rest is the vote fallback" | B5 |
| "Knowledge exists before training" | Keep provisionally (probe base 69.3% vs p12 70.0%), confirm with C4 | C4 ⏳ |
| "Could beat vote@5 at lower cost" | "A perfect two-way judge only ties vote@5 (77.9% vs 77.5%)" | F3 |
| "The recipe is anti-bias + penalizing invention" | "This combination is effective"; no ablation | — |

## Next GPU session (one rental, ~3 h)

1. Re-extract hidden states (6 sets + SVAMP base_c/p13b) → **C1, C4, C2b, F1** (~35 min).
2. **C3** likelihood baselines (~20 min).
3. **C5** logit lens (~20 min).
4. **D1, D2, D3** bias origin (~3 × 1 h; can be trimmed to D1 first).
5. **E1** four-candidate selection if time remains.
