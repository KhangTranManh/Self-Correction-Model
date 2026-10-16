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

### C1 ✅⚠️ Probe consistency across orders

*Result (3090 re-extraction, `outputs/paper_3090/c1_c4_report.json`).* At the
SVAMP-chosen layer, the probe picks the same solution in both orders on 55–57%
of one-right GSM8K pairs (all four judges), against judge consistency of
48% (base), 52% (p12), 49% (base_c), 63% (p13b). On the pairs where the probe is
consistent (160–165 pairs) it is right 78–83%. **The probe is only modestly
more order-invariant than the judge; it is not order-free** (~44% of pairs flip).

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
threshold.**

**C2b ✅ (3090, `c2b_f1.json`, layer 17 by SVAMP CV, SVAMP → GSM8K, 576 rows):**

| Model | base acc / AUC | p12 acc / AUC |
|---|---:|---:|
| Surface only | 64.2% / 0.742 | 64.2% / 0.742 |
| Hidden only | 67.9% / 0.734 | 68.6% / 0.737 |
| Hidden + surface | 68.9% / 0.754 | 69.1% / 0.759 |
| Hidden, on rows where surface is wrong (n = 206) | 63.1% | 64.6% |

The hidden state adds a little beyond surface cues (+1–1.5 points accuracy,
+0.015 AUC over surface alone) and is right on ~64% of the rows surface cues get
wrong, so the two signals are partly distinct. **Write: "the hidden state carries
a signal beyond surface cues, but surface cues alone reach AUC 0.74".**

### C3 ✅ Likelihood baselines (`c3_likelihood.json`)

One-right (s1, sk) pairs; greedy-match scores 0.5 when the fresh greedy solve
matches neither or both (79% of pairs are decisive).

| Selector | GSM8K (288 pairs) | SVAMP (367 pairs) |
|---|---:|---:|
| Higher mean token log-prob | 60.8% [54.5, 66.9] | 55.6% [50.1, 60.9] |
| Higher summed log-prob | 61.1% [55.1, 67.2] | 61.0% [55.6, 66.4] |
| Matches a fresh greedy solve | **81.1%** [76.6, 85.6] | **81.1%** [77.2, 84.7] |
| Judge p12, single order (B4) | 61.8% | — |
| Hidden-state probe (C4, p12) | 68.6% | — |

**Likelihood alone is ~61% (about the judge's single-order accuracy). Agreement
with one more greedy solve reaches 81%**, but that is a third solve acting as a
tie-breaker (one extra solver call, like vote@3): a compute baseline, not a free
signal. Wording: the probe (69%) beats likelihood (61%), but one extra greedy
solve beats the probe.

### C4 ✅ One probe protocol for all four judges

| Judge | Layer (SVAMP CV) | SVAMP → GSM8K acc | AUC | Within-GSM8K CV best |
|---|---:|---:|---:|---:|
| base | 17 | 67.9% | 0.734 | 71.5% (L20) |
| p12 | 17 | 68.6% | 0.737 | 71.5% (L20) |
| base_c | 18 | 65.3% | 0.726 | 71.5% (L21) |
| p13b | 20 | 67.2% | 0.741 | 71.3% (L22) |

Same protocol for all four judges: 65–69% cross-dataset, ~71.5% within GSM8K,
**no training effect on the probe**. The 3090 re-extraction gives 1–2 points
less than the V100 run (69–70%, layer 21), from fp16 numerical differences;
report the 3090 numbers as the uniform protocol and the V100 numbers as the original.

### C5 ✅⚠️ Logit lens by layer (`c5_logit_lens.json`)

d_l = logit(" A") − logit(" B") at the verdict letter, per layer; constrained
judges, all 972 GSM8K tasks.

| Readout | Judge | Final layer: share preferring A | Final layer: accuracy (one-right) | Best layer accuracy |
|---|---|---:|---:|---:|
| Forced ("Verdict: Solution" right after the prompt) | base_c | 65% | 64% | 64% (L24–28) |
| | p13b | 60% | 65% | 65% |
| Written (just before the judge's own verdict letter) | base_c | 66% | 57% | 60% (L26) |
| | p13b | 53% | 66% | 67% (L26) |

Figure: `paper/figures/c5_logit_lens.png` (script `paper/scripts/c5_figure.py`).
The skill curve (signed logit gap, A-right vs B-right) is flat at 0 up to
layer 20 and rises from layer 21–22 (to about +1.5–2.4 logits at layers 25–27).
Middle layers (14–20) read almost 100% "A" at chance accuracy; the lens becomes
informative only from about layer 22 (logit lens is unreliable in middle layers).
**Two findings:** (1) forced to answer before any reasoning, the model already
picks the right solution 64–65% of the time, about the probe's level; (2) for the
untrained base_c, writing out a check *lowers* final-layer accuracy to 57%,
while p13b keeps 66%. Usable as a figure, not as the headline. Note
the contrast with C4: a linear probe already reads the right/wrong signal at
layers 16–21, several layers before it becomes decodable as the verdict token
(≥ 22) — the information is present before it is mapped onto "A"/"B".

---

## D. Origin of the bias (should have)

### D1b ✅⚠️ Phase 1 SFT data prior

*Result.* In all **631/631** samples of `phase1/backup_server/processed/phase1_sft.jsonl`,
the first answer is verifier-wrong ("SAI") and the later answer (the
correction) is the right one. The data carry a perfect "later = right" prior,
which is **consistent with** H1; it does not show that the prior caused the bias.
*Exact training file (checked 2026-10-16).* The 513-row training file is in the repo:
`data/processed/phase1_sft_train.jsonl` (config `phase1.yaml`: the training copy of `phase1_sft.jsonl`, with the long
`<thinking>` trace removed and the critique + verified corrected solution kept; `phase1_sft.jsonl` has the same 513
problem ids). Counts on the 513 rows (147 GSM8K, 366 MBPP): every row has the message order
system, user, assistant (first attempt), tool, assistant (correction) = 513/513; the tool message reports
"Kết quả kiểm tra: SAI" (verifier says the first attempt is wrong) = 513/513. For the 78 GSM8K rows whose tool message
states the gold answer, the correction contains that answer in its last 400 characters = 78/78 (MBPP and the other
GSM8K rows were not re-verified). So the "later answer is right, first is wrong" prior holds for the training
set itself, by construction and by this count. The earlier 631/631 count was on a superset-era backup and is superseded.
Wording stays "consistent with": a prior in the data does not show that it caused the bias.

### D0 ✅ 3090 control

The Self_Correction base judge re-run on the 3090 reproduces 86.3% of the V100
outputs exactly and makes the same pick on 95.1% of one-right single-order
judgments; tie-break 0.599 vs 0.601 on the V100. All D comparisons use this
3090 run (d0) as the reference.

### D1–D3 ✅ (`paper/results/d_e_analysis.json`, GSM8K, 288 one-right pairs)

| Judge (greedy; Phase 9 prompt unless noted) | Skill index | Net position-2 pref. | Consistent acc. | Coverage | Tie-break | Single-order | Token cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| d0 Self_Correction base (3090) | +0.21 | +0.090 | 70.5% | 48.3% | 0.599 | 52.8% | 3.8% |
| **d1 Qwen2.5-7B-Instruct (untrained parent)** | **+0.65** [0.57, 0.72] | −0.024 | **94.9%** | **61.1%** | **0.774** | **72.4%** | 20.6% |
| d1_long: same, 1536-token cap | +0.69 [0.62, 0.76] | −0.016 | 95.1% | 70.8% | 0.819 | 79.3% | 0.8% |
| d2 labels reversed (first shown = "Solution B") | +0.20 | +0.108 | 68.3% | 49.3% | 0.590 | 51.9% | 3.7% |
| d3 problem restated after the solutions | +0.25 | +0.009 | 67.5% | 58.7% | 0.602 | 54.7% | 3.4% |

Net position-2 preference per single-order judgment: +1 picks the second shown
solution, −1 the first, 0 invented. The three hypotheses as defined in the
checklist (problem-clustered bootstrap, two-sided, Holm over 3; thresholds
written before running):

| Hypothesis | Estimate | 95% CI | p | p (Holm) | Checklist rule | Verdict |
|---|---:|---|---:|---:|---|---|
| D1 (H1) Instruct − Self_Correction | **−0.115** | [−0.191, −0.036] | 0.005 | **0.016** | ≥ 10 points and CI excludes 0 | **confirmed** |
| D2 (H3) relabel − d0 | +0.017 | [−0.048, 0.085] | 0.63 | 0.63 | — | preference follows position |
| D3 (H2) restate − d0 | −0.082 | [−0.159, −0.005] | 0.040 | 0.08 | drop ≥ 10 points | not confirmed |

Instruct on its own: net position-2 preference −0.024 [−0.073, +0.026] (no
bias). Tie-break Instruct − d0: +0.175 [+0.130, +0.222], p < 0.001 (exploratory).
D2 with d0 forms the 2×2 (label order × position order: each run has both orders).

**Reading (by the rules written before running):**
- **H1 supported by the checklist rule: the untrained parent has no position
  preference and judges far better** (skill +0.65 vs +0.21; 95% accurate when
  consistent). This is **consistent with** the bias, and much of the lost skill,
  coming from the Phase 1 self-correction SFT, whose data carry a "later answer =
  right" prior (D1b). Two confounds remain: Instruct judges another model's
  solutions (T1 tests this) and the D1b data provenance.
- **H3: the preference follows position, not the label** (renaming changes nothing).
- **H2: restating the problem removes the net position-2 preference** (+0.090 →
  +0.009) and raises coverage (48% → 59%) but not accuracy (tie-break 0.602);
  nominal p = 0.04, not significant after Holm.
- Token cap: Instruct hits 768 tokens on 20.6% of judgments. With 1536 tokens
  (d1_long, `paper/results/d1_g1_robust.json`) the cap rate is 0.8% and the
  result is stronger: tie-break 0.819, coverage 70.8%, single-order 79.3%; net
  position-2 difference vs d0 −0.106 [−0.186, −0.028], tie-break +0.220
  [+0.174, +0.268]. The 768-token numbers are the comparable ones (same cap as d0).
- Instruct here judges solutions written by Self_Correction_v1, its fine-tuned
  child, not its own: see T1.

**Framing after T1–T5:** the D1 gap (0.77 vs 0.60) is not an SFT effect: on its own
solutions the parent reaches only 0.61–0.66 (T1, T3, both scorings). It is also
not a general "others are easier" effect (T5). Keep the original central claim
(self-arbitration looks like guessing because of position bias); add "the parent
also has a position bias on its own solutions (towards the first one)" and "a
stronger judge does well on the weaker model's disagreements".

### T1 ✅ The parent model judges its OWN conflicting solutions — most important

*Problem.* In D1, Qwen2.5-7B-Instruct judges solutions written by
Self_Correction_v1, so it is not self-arbitration, and the solutions' style
(written by its child) may help or hurt it.
*Design (fixed before running, `paper/scripts/gpu/t1_parent_self.py`).*
Qwen2.5-7B-Instruct solves each of the 561 Phase 13 GSM8K problems three times
with the Phase 13 recipe (s1 Phase 1 prompt, s2–s3 blind_resolve prompt,
temperature 0.7, 768 tokens, Phase 13 seeds with a "t1" tag); pairs (s1, s2) and
(s1, s3) with different final answers are judged by the same model in both
orders (Phase 9 prompt, greedy, 768 tokens as in table D). Same verifier, same
metrics as table D, problem-clustered bootstrap.
*Primary metric:* tie-break on one-right pairs. *Compared with:* d0 (Self_Correction
judging its own solutions, 0.599) and d1 (Instruct judging its child's, 0.774).
*Interpretation rule (written before the result):* tie-break ≥ ~0.70 → the claim
"SFT reduced the arbitration skill" stands; ≈ 0.60 → most of the D1 gap comes
from the parent judging another model's work. Different pairs than d0/d1 (its own
solutions), so the comparison is unpaired (problem-clustered CIs per arm).
*Result (`paper/results/t1_parent_self.json`).* Instruct's s1 is right on 81.3%
of problems (Self_Correction: 66.1%); 264 one-right pairs on 211 problems.

| Judge → solutions | Pairs | Skill index | Position-2 rate (A/B picks) | Consistent acc. | Coverage | **Tie-break** | Single-order | Token cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| d0 Self_Correction → own | 288 | +0.22 | 55.4% | 70.5% | 48.3% | **0.599** | 52.8% | 3.8% |
| **T1 Instruct → own** | 264 | +0.44 [0.36, 0.52] | **36.7%** | 91.5% | **31.1%** | **0.629** | 53.2% | 20.0% |
| d1 Instruct → Self_Correction's | 288 | +0.65 | 48.5% | 94.9% | 61.1% | **0.774** | 72.4% | 20.6% |

Tie-break differences (independent problem-clustered bootstraps, different pair
sets): T1 − d0 **+0.030 [−0.019, +0.080]**; T1 − d1 **−0.146 [−0.189, −0.100]**.

**Reading by the rule written before the result: the tie-break falls to ~0.6
(0.63, not ≥ 0.70), so most of the D1 gap comes from the parent judging another
model's work, not from SFT destroying the skill.** Judging its own conflicting
solutions, the parent is no better than Self_Correction on the primary metric.
It also has a position bias, in the *opposite* direction (63% of its A/B picks go
to the first shown solution, vs Self_Correction's lean to the second), which
drives coverage down to 31%; when it is consistent it is right 91.5%.
Caveats: 20% of T1 judgments hit the 768-token cap (kept for comparability with
d0/d1); the pairs differ in difficulty (the parent's disagreements arise on
harder problems for it, since its s1 is right 81% vs 66%).

**Consequences:**
- D1 still shows that the *direction and size* of the bias differ between the
  two models (position 2 vs position 1), but the claim "SFT reduced the
  arbitration skill" is **not** supported. Keep D1/D1b as "the SFT data prior is
  consistent with Self_Correction's lean to the later solution".
- ~~Both models arbitrate others' solutions better than their own~~ — **not
  supported after T5**: it holds for Instruct (0.77–0.84 on Self_Correction's
  pairs vs 0.61–0.66 on its own) but not for Self_Correction (0.56–0.64 on
  Instruct's pairs vs 0.60 on its own). See T3–T5.

T2 on Instruct's own solutions (same rules as T2): keep 81.3%, random 76.5%,
vote@3 87.0%; judge-or-random 80.3% (−6.7 [−8.6, −4.7] vs vote@3, below keep);
self-check 87.2% at 2.88 calls (+0.2 [−0.7, +1.1] vs vote@3) — the parent's
self-check only ties vote@3 on its own solutions.

### T2 ✅ B5 with the parent as judge (`paper/results/t2_parent_b5.json`)

*Definition.* Rows added to the B5 table for the Instruct judge on the GSM8K
Phase 13 solutions (written by Self_Correction_v1): "judge or random" (consistent
verdict, else expected random pick) and "self-check" (consistent verdict, else
vote@3). Compute = model calls (F5): self-check 2 calls when s1 = s2, else 4
(+1 for the vote@3 fallback); vote@3 3 calls. Paired problem-level bootstrap vs vote@3.

| GSM8K, 561 problems | Accuracy | Mean calls | vs vote@3 |
|---|---:|---:|---|
| keep s1 | 66.1% | 1 | |
| random s1/s2 | 65.7% | 2 | |
| vote@3 | 72.0% | 3 | |
| vote@5 | 77.5% | 5 | |
| judge or random — Self_Correction base (3090) | 68.2% | | −3.8 [−6.0, −1.7] |
| judge or random — p12 | 69.9% | | −2.1 [−4.1, −0.2] |
| judge or random — **Instruct** | 71.7% | | −0.4 [−2.3, +1.7] |
| judge or random — Instruct, 1536 tokens | 72.9% | | +0.9 [−1.1, +2.9] |
| self-check — Self_Correction base (3090) | 70.6% | 3.08 | −1.4 [−3.0, 0.0] |
| self-check — p12 | 72.2% | 3.09 | +0.2 [−1.1, +1.6] |
| self-check — **Instruct** | **74.0%** | 3.12 | **+2.0 [+0.5, +3.4]** |
| self-check — Instruct, 1536 tokens | **74.7%** | 3.08 | **+2.7 [+1.1, +4.5]** |

**Practical answer: yes, at the same compute (~3.1 calls vs 3) the parent as
judge beats vote@3 by +2.0 points (CI excludes 0)**, while the Self_Correction
judges only tie it; nothing reaches vote@5 (77.5%). Calls count both judge
orders; token counts are not matched (Instruct judgments are longer). Post-hoc.
T1 has finished; the rows for Instruct on its own solutions are given under T1.

### T3–T5 ✅ Robustness of T1 (`paper/results/t3_t5_analysis.json`, `t4_parse_audit.json`)

*Definitions.*
- **T3** — T1 re-judged with a 1536-token cap (T1 hit 768 on 20%), same pairs.
- **T4** — check that Self_Correction's s1 accuracy (66.1%) is not lowered by
  parsing/format: count verifier-wrong s1 answers whose last "Đáp số" line starts
  with the gold number. **T4b** — rescore everything with a lenient reading (a
  solution or judge answer is also read from the first number on its last
  "Đáp số" line when the parser does not yield a number).
- **T5** — Self_Correction_v1 judges Instruct's T1 pairs (another model's work),
  768 tokens as in d0. Gives the reverse direction of "judging others is easier".

*T4 result.* Self_Correction s1: 4/561 format misses ("1 USD/bánh",
"55 m/s", "2.00", "227 phút hoặc…") → 66.1% → 66.8% lenient. **The 66.1% is
sound.** Instruct s1: **29/561 format misses** (answers written as sentences,
"Đáp số: Mỗi trẻ nhận được 9 trứng.") → 81.3% → 86.5%. So T1's strict labels are
noisy: under lenient scoring Instruct's one-right pairs drop from 264 to **133**
(many "disagreements" differ only in how the answer is written).

*2 × 2: judge × whose solutions (tie-break on one-right pairs; strict / lenient):*

| Judge ↓ / Solutions → | Self_Correction's (288 / 279 pairs) | Instruct's (264 / 133 pairs) |
|---|---:|---:|
| Self_Correction (768) | d0 **0.599 / 0.599** (own) | T5 **0.642 / 0.564** (other's) |
| Instruct (768) | d1 **0.774 / 0.789** (other's) | T1 **0.629 / 0.605** (own) |
| Instruct (1536) | d1_long **0.819 / 0.839** (other's) | T3 **0.655 / 0.658** (own) |

| Instruct on own (T3, 1536) | strict | lenient |
|---|---:|---:|
| Single-order accuracy | 58.3% | 62.8% |
| Skill index | +0.50 | +0.32 |
| Position-2 rate (A/B picks) | 34.8% | 37.6% |
| Coverage / consistent acc. | 40.9% / 88.0% | 57.1% / 77.6% |
| Invented | 25.4% | 6.0% |

Differences (independent problem-clustered bootstraps): T3 − d1_long −0.164
[−0.211, −0.116] strict, −0.181 [−0.251, −0.111] lenient; T3 − d0 +0.056
[+0.005, +0.108] strict, +0.059 [−0.016, +0.136] lenient; T5 − d0 +0.043
[−0.010, +0.097] strict, −0.035 [−0.105, +0.032] lenient.

**Reading.**
1. **T3: the token cap did not cause T1's low score.** With 1536 tokens Instruct
   on its own pairs reaches 0.655–0.658, still ~0.16–0.18 below its score on
   Self_Correction's pairs and only ~0.06 above d0. Its single-order accuracy on
   its own pairs is 58–63%, so "both models ~53% on their own solutions" must be
   written as "Self_Correction 53%, Instruct 53–63% depending on token cap and
   scoring", not as a shared 53%.
2. **T5: the reverse direction does not hold.** Self_Correction judging Instruct's
   solutions (0.564–0.642) is not better than judging its own (0.599). So the
   claim "judging another model's solutions is easier" holds **only for one
   model (Instruct)** and must be written for that model only.
3. The 2 × 2 suggests a different explanation than self vs other: **Instruct's own
   disagreements are hard for both judges (~0.56–0.66), while Self_Correction's
   disagreements are easy for the stronger judge (0.77–0.84).** Pair difficulty
   (whose errors) and judge strength both matter; "self-arbitration is uniquely
   hard" is not established. Note the pair sets differ by column, and Instruct's
   pairs come from problems hard for a model that is right 81–87% of the time.
4. Lenient scoring changes the levels a little but none of the conclusions.

### T6 ✅ CI for the first-position pick rate (`paper/results/t6_position_ci.json`)

*Definition.* Share of A/B picks (invented excluded) that go to the solution shown
first, on one-right pairs; problem-clustered bootstrap (10,000), two-sided
bootstrap p against 0.5.

| Judge → solutions | Strict: rate [95% CI] (n picks) | p vs 0.5 | Lenient: rate [95% CI] (n) | p vs 0.5 |
|---|---|---:|---|---:|
| d0 Self_Correction → own | 44.6% [40.8, 48.4] (480) | 0.005 | 44.5% [40.7, 48.4] (467) | 0.005 |
| d1 Instruct → Self_Correction's | 51.5% [48.6, 54.4] (478) | 0.34 | 50.8% [47.9, 53.7] (480) | 0.60 |
| d1_long Instruct (1536) → Self_Correction's | 50.9% [48.1, 53.5] (523) | 0.55 | 50.6% [47.9, 53.3] (522) | 0.70 |
| **T1 Instruct → own** | **63.3% [58.7, 67.8]** (354) | < 0.001 | 56.6% [49.1, 63.8] (205) | 0.087 |
| **T3 Instruct (1536) → own** | **65.2% [61.1, 69.4]** (394) | < 0.001 | **62.4% [57.0, 67.9]** (250) | < 0.001 |
| T5 Self_Correction → Instruct's | 48.5% [44.7, 52.5] (375) | 0.49 | 46.7% [40.4, 52.9] (199) | 0.32 |

**Reading.** The parent's lean to the first solution on its own pairs is clear
with 1536 tokens under both scorings (62–65%, CI above 0.5); at 768 tokens it is
clear under strict scoring and borderline under lenient (56.6%, CI includes
0.5, fewer picks). Write "the parent picks the first-shown solution 62–65% of
the time on its own pairs (T3)". Also new: **neither judge shows a position
lean on the *other* model's pairs** (d1, d1_long, T5 all ≈ 50%) — the bias shows
up when judging one's own conflicting solutions, which fits the central claim.

---

## E. Comparison with Ren et al. 2023 (should have)

- **E1 ✅** Four candidates (s1–s4, Self_Correction base judge, 4 cyclic
  rotations) on the 347 GSM8K problems whose four answers are not all equal
  (1,388 judgments). Position picks: **1st 36.4%**, 2nd 20.0%, 3rd 11.0%,
  4th 16.1%, invented 16.5%. With four candidates the lean is to the *first*
  position (primacy), unlike the two-way position-2 lean. Single-judgment
  accuracy 44.9% vs 45.5% for a uniformly random candidate (no skill);
  plurality over the 4 rotations 53.9% vs vote@4 65.1% on the same problems
  (keep 46.7%). Permutation averaging helps (+8.5 points over one rotation) but
  the judge stays far below voting.
  *Effect size vs two candidates — corrected 2026-10-12.* An earlier version
  compared "+7.1 points with 2 candidates" (single-order 52.8% → tie-break
  0.599) with "+8.5 with 4" (one rotation 45.5% → plurality 53.9%) and called
  both debiasing gains. Neither is: the 2-candidate number is the
  invented-answer term (see E2), and the 4-candidate plurality drops invented
  picks while the one-rotation score counts them as wrong, so part of the +8.5
  is the same artefact. The 4-candidate result that stands: picks lean to the
  first position (36.4%), and single judgments are at chance (44.9% vs 45.5%).
  The size of a true debiasing gain with 2 vs 4 candidates is not measured.
- **E2 ⚠️ corrected 2026-10-12 — comparable number pending (Phase 14).**
  Ren et al. 2023 (arXiv 2312.09300), Table 4, PaLM-2 Large, Sample and Select
  over n = 4 answers, accuracy vanilla → de-biased (letter probabilities averaged
  over shuffled orders): TruthfulQA 58.26 → 58.87 (+0.61), TL;DR 70.20 → 70.70
  (+0.50); calibration-AUC slightly worse (53.17 → 52.13; 46.65 → 43.94).

  **What was wrong here.** The earlier table put our "tie-break − single-order"
  (+7.1, +8.5, +3.5, +5.0) next to their gain. Per pair, tie-break minus the mean
  of the two single-order scores is exactly 0.5 × [neither order picks the right
  solution and at least one order invents an answer]:

| Two judgments of a pair | Single-order (mean) | Tie-break | Difference |
|---|---:|---:|---:|
| right, right | 1 | 1 | 0 |
| wrong, wrong | 0 | 0 | 0 |
| right, wrong (pure position bias) | 0.5 | 0.5 | **0** |
| right, invented | 0.5 | 0.5 | 0 |
| wrong, invented | 0 | 0.5 | **+0.5** |
| invented, invented | 0 | 0.5 | **+0.5** |

  A judge with pure position bias and no invented answers gets 0; the difference
  tracks invented answers (Phase 13: +7.8 / +3.5 / +1.9 / +1.0 points for judges
  with 16.7% / 9.9% / 4.9% / 2.1% invented). It is not a debiasing gain and not on
  Ren et al.'s scale (multiple choice, no invented option). The judge with almost
  no invented answers (p13b) shows +1.0, close to their +0.5–0.6.

  **Comparable measure:** Ren et al.'s own method, log-probability order
  averaging — m = log P(A) − log P(B) at the verdict token, pick by
  sign(m_ab − m_ba), compared with the single-order accuracy of the same
  log-probabilities on the same pairs (Phase 14 endpoint E2, all four models,
  `base_c` judge; also the forced verdict with no written check). The Phases
  12–13 vLLM judge runs did not store log-probabilities and the C5 file stores
  only per-layer aggregates, so this number does not exist yet for any model; it
  comes from Phase 14. Until then the report must not claim our gain is larger
  than Ren et al.'s.

  **Phase 14 result (2026-10-13, endpoint E2, opened once 03:55 UTC):** log-prob
  order averaging on the two new models gives +0.6 [−3.7, +4.8] (Llama ASDiv),
  +5.0 [+0.9, +9.3] (Llama MATH), +0.7 [−2.3, +3.6] (Mistral ASDiv), +3.3
  [−0.7, +7.3] (Mistral MATH) points; the bias-removal part is only −0.8 to +1.4
  although the log-odds bias toward A is 1.3–2.4. E2 passes in 1/4 cells →
  "conditional". **Measured on Ren et al.'s scale, debiasing helps about as little
  as they found.** Self_Correction and Qwen-Instruct values come from the Phase 14
  completion run (descriptive).
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

- **F1 ✅** p12 probe (layer 17). Confidence |p − 0.5| is the same on one-right
  (0.200) and both-wrong (0.196) disagreements: AUC 0.51 for flagging both-wrong
  pairs, so **the probe cannot tell when neither solution is right**. Abstention
  rule (judge if consistent, else probe if confident, else vote@3; τ = 0.05
  chosen on SVAMP): 72.5% on GSM8K vs self_check_p12 72.2% (+0.4 [−1.4, +2.1])
  and vote@5 77.5% (−5.0 [−7.8, −2.3]). Probe-only rule 73.1% vs vote@3 72.0%
  (+1.1 [−1.1, +3.2]).
- **F2 ✅ (exploratory only, `f2_pointwise.json`)** Pointwise probe: one solution per
  prompt ("Is the final answer of this solution correct?"), last-token hidden
  state of the untrained solver, label = verifier-correct, trained on all 4,800
  SVAMP samples, layer 23 chosen by SVAMP CV, tested on all 2,805 GSM8K samples
  (base rate 65.0% correct): **accuracy 79.3%, AUC 0.838** (layers 19–28 all
  78–80%). On the 288 one-right pairs, picking the higher-scored solution is
  right 74.3%, above the pairwise judge-prompt probe (68.6%).

| GSM8K (561 problems) | Accuracy |
|---|---:|
| keep (s1) | 66.1% |
| probe best-of-5 | 74.5% (−3.0 [−5.7, −0.4] vs vote@5) |
| vote@5 | 77.5% |
| **probe-weighted vote@5** (sum of P(correct) per answer cluster) | **79.3% (+1.8 [+0.2, +3.4] vs vote@5)** |

  The first strategy in the project to beat vote@5, at the same five solver
  calls plus five forward passes (no generation). Post-hoc and single-split:
  must be confirmed on a fresh holdout before it is claimed; stays in the paper
  as an exploratory result. It also suggests
  that the pairwise framing itself hurts: the model's pointwise signal (AUC
  0.84) is stronger than its pairwise one (AUC 0.74).
- **F2b ✅ (exploratory) Pointwise surface baseline** (`outputs/paper_3090/f2b_surface.json`,
  script `paper/scripts/gpu/f2b_surface.py`). Same rows, labels and split as F2;
  surface features of the single solution (the C2 set).

| Model (SVAMP → GSM8K, 2,805 samples) | Acc. | AUC | Best-of-5 | Weighted vote@5 (vs vote@5 77.5%) | One-right pair pick |
|---|---:|---:|---:|---:|---:|
| Surface only (C = 1) | 72.2% | 0.722 | 66.8% | 78.1% (+0.5 [−0.7, +1.8]) | 62.5% |
| Surface only (C = 0.01) | 72.3% | 0.720 | 65.6% | 77.9% (+0.4 [−0.9, +1.6]) | 60.8% |
| Hidden only (layer 23) | 79.3% | 0.838 | 74.5% | 79.3% (+1.8 [+0.2, +3.4]) | 74.3% |
| Hidden + surface | 80.0% | 0.849 | 76.1% | 79.7% (+2.1 [+0.5, +3.7]) | 75.7% |

  (Accuracy base rate: 65.0% of GSM8K samples are correct.) **Unlike the
  pairwise case (C2b: surface AUC 0.742 ≈ hidden 0.737), the pointwise hidden
  state is well beyond surface cues: AUC 0.838 vs 0.722 (+0.12), pair pick 74%
  vs 61–63%.** Surface-weighted voting does not beat vote@5 (CI includes 0);
  only the hidden-state weighting does. F2 stays exploratory, but its signal is
  not a surface artefact.
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

- **G1 ✅ (Phase 14, preregistered endpoint E1)** Untrained Llama-3.1-8B-Instruct and
  Mistral-7B-Instruct-v0.3 judging their own conflicting solutions on ASDiv (822) and
  MATH 1–3 (785), Phase 13 protocol unchanged: single-order accuracy 50.5–57.3% (looks
  like chance), skill index +0.321 / +0.362 (Llama ASDiv / MATH), +0.335 / +0.344
  (Mistral), all CIs above 0 → **E1 general (4/4)**; tie-break 0.61–0.65; consistent
  accuracy 77–80% at 39–50% coverage; both lean to position 1 (position-2 rate 42–45%).
  Details and amendments: `phase14/RUN_STATUS.md`, `paper/REPORT_SPEC.md` §4.6. GSM8K and
  SVAMP for these models: completion run (descriptive).

---

## Central claim — status of the four pieces after all checks

| Piece (checklist) | Status | Evidence |
|---|---|---|
| 1. One order looks like guessing | ✅ for Self_Correction on its own solutions (single-order 52.8%) **and for untrained Llama and Mistral on theirs (50.5–57.3%, Phase 14)**; ⚠️ for the parent on its own it is 53% at 768 tokens but 58–63% at 1536 (T3); the parent does well (72–82%) on Self_Correction's solutions (D1) | B1, B4, D1, T1, T3, G1 |
| 2. Removing output bias reveals skill (~70% untrained) | ✅ skill index +0.21 [0.13, 0.30] (B1), **+0.32 to +0.36 for Llama and Mistral (E1 4/4, Phase 14)**, consistent accuracy 70.5% at 48% coverage (B2; 77–80% for the new models). But **log-prob order averaging adds only 0.6–5 points (E2 1/4), close to Ren et al.**: the skill shows in the decomposition and in consistent pairs, not as a large debiasing gain | B1, B2, E2, G1 |
| 3. Hidden state knows (~70–75%), transfers, not flipped by order | ⚠️ knows and transfers (65–69% cross-dataset, 71.5% within; AUC 0.73–0.74), **but is flipped by order on ~44% of pairs**, adds only a little over surface cues (AUC 0.759 vs 0.742), and cannot flag both-wrong pairs | C1, C2b, C4, F1 |
| 4. Order-swapped DPO improves expression, not internal knowledge | ✅ probe unchanged across the 4 judges (65–69%) while tie-break rises 0.60 → 0.65 | C4, B3 |

**Revised central claim after T1–T5:** "When a 7B self-correction model arbitrates
between its own conflicting solutions, position bias makes it look like guessing
(single-order 53%); order-averaging, the hidden state and a pointwise readout
still carry a right/wrong signal. Its untrained parent also shows a position bias
on its own solutions (towards the first one) and stays well below its accuracy on
the fine-tuned model's solutions (tie-break 0.61–0.66 vs 0.77–0.84)." Drop "not
flipped by order" (C1), any "SFT caused the bias" wording, and any general "judging
others is easier" wording (T5 does not support it for Self_Correction).

## Claim wording — updated with evidence

| Current sentence | Rewrite | Evidence |
|---|---|---|
| "Self-evaluation 79%, Phase 0 goal met" | "79% when consistent (52% coverage); 41% of all one-right pairs right, tie-break 0.65" | B2 |
| "The model truly recognizes the right answer" | "A judgment signal well above chance (skill index +0.21 untrained, +0.32 trained)" | B1; C2 shows surface cues reach ~63–66% |
| "Preregistered, tested on new problems" | Keep, adding that 42 of 561 had appeared in an evaluation-only run; results unchanged without them | A1, A4 |
| "Self-correction +6 points vs no correction" | "+4.2 points over random choice (judge's own contribution); the rest is the vote fallback" | B5 |
| "The hidden state is not flipped by order" | "The probe is somewhat more order-stable than the judge (56% vs 48–52% of pairs consistent), not order-free" | C1 |
| "Knowledge exists before training" | Keep: same probe protocol gives 65–69% for all four judges; DPO does not change it | C4 |
| "Self-correction SFT created the bias" | Do not write this. "Both models show a position bias on their own solutions, in opposite directions (Self_Correction → later, consistent with its SFT data format; parent → first)" | D1, D1b, T1, T3 |
| "Judging another model's solutions is easier" | Write for Instruct only: "the parent judges the fine-tuned model's disagreements far better (0.77–0.84) than its own (0.61–0.66)"; the reverse (Self_Correction on Instruct's: 0.56–0.64 vs 0.60 on its own) does not hold | T1, T3, T5 |
| "Both models are ~53% on their own solutions" | "Self_Correction 53%; Instruct 53% (768 tokens) to 58–63% (1536 tokens, lenient scoring)" | T1, T3, T4b |
| "The hidden state verifies solutions" | "The hidden state carries a signal beyond surface cues (AUC 0.76 combined vs 0.74 surface) and likelihood (61%), but cannot flag pairs where both are wrong" | C2b, C3, F1 |
| "Debiasing helps much more than in Ren et al." | Do not write. On their scale (log-prob order averaging) Phase 14 finds +0.6 to +5.0 points, bias-removal part ≈ 0 (E2 conditional, 1/4 cells): "consistent with Ren et al." | E2, Phase 14 |
| "The hidden state beats surface cues" | "For Self_Correction and Llama, slightly (pair probe AUC 0.72–0.74 vs surface 0.64–0.74); not for Mistral (surface AUC 0.80–0.88 vs probe 0.76–0.82)" | C2b, Phase 14 probes |
| "Could beat vote@5 at lower cost" | "A perfect two-way judge only ties vote@5 (77.9% vs 77.5%); a pointwise probe-weighted vote@5 reaches 79.3% (+1.8, exploratory), which surface features alone do not (78.1%, n.s.)" | F3, F2, F2b |
| "The parent prefers the first solution" | "On its own pairs the parent picks the first-shown solution 62–65% of the time (T3, CI above 50%); neither model shows a position lean on the other model's pairs" | T6 |
| "The recipe is anti-bias + penalizing invention" | "This combination is effective"; no ablation | — |

## Remaining

- **Phase 14 completion run** (started 2026-10-13 05:58 UTC): Self_Correction and Qwen-Instruct
  on all four datasets, Llama and Mistral on GSM8K and SVAMP, bridge set, base_c regression —
  fills Table 3 and the §4.6 grid for all four models (descriptive cells).
- **Report spec**: `paper/REPORT_SPEC.md` maps every report table to one definition set
  (updated with the Phase 14 endpoint results).
- **D1b**: the exact 513-row training file is not in the repo (format confirmed by design).
