# Phase 3 Canonical Results

Updated 2026-09-07.

## Frozen behavioral comparison

All routers were evaluated on the same protected 200-row manifest: 100 verified
correct previous answers and 100 verified wrong previous answers, balanced
between math and code.

| Metric | Decision-Only V1 | DPO Pilot V1 | DPO Semantic V2 |
|---|---:|---:|---:|
| Balanced decision accuracy | 65.0% | 66.5% | 65.5% |
| KEEP recall | 81.0% | 85.0% | 85.0% |
| REVISE recall | 49.0% | 48.0% | 46.0% |
| REVISE precision | 72.1% | 76.2% | 75.4% |
| Code decision accuracy | 62.0% | 62.0% | 59.0% |
| Math decision accuracy | 68.0% | 71.0% | 72.0% |
| Unseen-template accuracy | 58.0% | 55.5% | 53.5% |
| Main exact-contract rate | 100% | 100% | 100% |
| Unseen exact-contract rate | 100% | 95.5% | 95.5% |
| Final selective-repair accuracy | 51.5% | 54.5% | 55.5% |
| Harmful correct-to-wrong revisions | 8 | 2 | 1 |
| Missed wrong answers | 51 | 52 | 54 |

The DPO models improve preservation and final selective-repair accuracy, but
they do so by becoming more conservative. They do not improve the target skill:
detecting plausible wrong answers.

## Semantic V2 data intervention

Semantic V2 directly addressed the reference-versus-model shortcut in the first
DPO pilot:

- 100 same-problem pairs / 200 preference rows;
- 100 KEEP / 100 REVISE;
- 73 GSM8K, 20 APPS, 7 MBPP;
- 48 Base-correct versus V1-wrong;
- 52 V1-correct versus Base-wrong;
- both members are natural model outputs;
- correct members pass 100/100 fresh checks;
- wrong members fail 100/100 fresh semantic checks;
- no frozen overlap and no synthetic wrong answers.

The historical pool has one output per model/problem, so the two members cannot
share exact model origin without new multi-sampling. This limitation is explicit
rather than hidden.

Training preference accuracy rose from 54% to 59%, but frozen REVISE recall fell
to 46%. The intervention removed one shortcut without solving semantic detection.

## Representation result

The cleanest preserved comparison is Decision-Only V1 versus Semantic V2 on the
same 256-source probe split and activation positions:

| Metric | Decision-Only V1 | Semantic V2 | Delta |
|---|---:|---:|---:|
| Test balanced accuracy | 67.3% | 57.7% | -9.6 pp |
| KEEP recall | 80.8% | 76.9% | -3.9 pp |
| REVISE recall | 53.8% | 38.5% | -15.3 pp |
| ROC-AUC | 0.734 | 0.599 | -0.135 |
| Code balanced accuracy | 57.7% | 53.8% | -3.9 pp |
| Math balanced accuracy | 76.9% | 61.5% | -15.4 pp |

The answer-length-only control reaches 57.7%, equal to Semantic V2's selected
probe result. Semantic V2 therefore provides no evidence of a stronger internal
correctness representation.

## Five-stage follow-up

The ordered follow-up ran entirely on the remote RTX 3090. Only the frozen
classifier passed its gate:

| Stage | Behavioral BA | KEEP recall | REVISE recall | Probe BA | Result |
|---|---:|---:|---:|---:|---|
| Decision-Only V1 | 65.0% | 81.0% | 49.0% | 67.3% | canonical baseline |
| 01 frozen classifier | 69.2%* | 76.9%* | 61.5%* | n/a | pass |
| 02 decision-token QLoRA | 64.0% | 94.0% | 34.0% | 65.4% | rejected |
| 04 same-origin mixed SFT | 61.0% | 98.0% | 24.0% | 57.7% | rejected |
| 05 verifier-reward pilot | 62.0% | 98.0% | 26.0% | 59.6% | rejected |

`*` Stage 01 uses the protected 52-row representation-probe test split, not the
200-row generative benchmark, so its values are not directly interchangeable
with the LLM behavior rows.

Stage 03 found near-chance grouped shortcut predictability: 50.0% from answer
character n-grams, 50.5% from simple surface features, and 52.0% from model
origin. Stage 04 generated 762 candidates but only 23 problems yielded a strict
same-model-origin plausible wrong answer. The 200-row training mix therefore
contained 23 same-origin pairs and 77 verified cross-origin fallbacks.

All three weight-updating follow-ups moved the router toward KEEP. Their probe
REVISE recall also fell, so the negative result is not explained solely by a
changed output threshold.

## Frozen-200 representation follow-up

The frozen classifier was subsequently tested without refitting on all 200
protected behavioral rows. These sources have zero overlap with the 256-row
probe-fitting corpus.

| Readout | Balanced accuracy | KEEP recall | REVISE recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Precommitted layer/threshold | 50.5% | 1.0% | 100.0% | 71.7% |
| Rank-balanced diagnostic | 67.0% | 67.0% | 67.0% | 71.7% |

The threshold fails under distribution shift, although ordering remains useful:
ROC-AUC is 81.0% for math and 68.3% for code. Across 100 repeated source-disjoint
folds on the original probe distribution, mean ROC-AUC is 77.6% and mean
balanced accuracy is 68.1%, but KEEP recall remains unstable. Four-layer
concatenation and a small MLP generalize worse than the single-layer linear
score. Layer-28 and layer-21 residual interventions do not produce a meaningful
causal change in forced-choice decision log-odds.

The supported claim is therefore narrower: correctness is partially and
linearly rank-decodable, but the absolute calibration is not portable and the
decoded direction has not been shown to cause the native decision.

## Source-disjoint calibration follow-up

A preregistered calibration pass mined 600 stochastic Self_Correction_v1
outputs and retained 88 unique natural GSM8K failures after fresh verification.
The final corpus contains 100 balanced fitting rows and a disjoint 40-row
natural audit, with zero source overlap against probe-256, frozen-200, or the
Decision-Only SFT corpus. No LLM or probe weights were changed.

| Split | Balanced accuracy | KEEP recall | REVISE recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Balanced fit, 5-fold OOF | 66.0% | 86.0% | 46.0% | 70.8% |
| Natural audit | 63.5% | 95.8% | 31.3% | 84.6% |

The selected threshold was 0.61. The BA and REVISE-recall gates failed, so this
calibrator is not deployable. The result again supports rank information in the
hidden state, while rejecting the claim that an absolute KEEP/REVISE operating
point has been made portable.

## Promotion decision

| Experiment | Decision | Reason |
|---|---|---|
| Decision-Only V1 | canonical router | strongest preserved representation result and better REVISE recall |
| Router V3 mini | diagnostic only | policy shift did not establish robust representation gain |
| DPO Pilot V1 | rejected for promotion | balanced accuracy up, REVISE and unseen behavior down |
| DPO Semantic V2 | rejected for promotion | REVISE, code, unseen template, and probe all regress |
| Frozen classifier V1 | diagnostic only | ranking signal remains, but both external threshold tests are non-portable |
| Decision-Token V2 | rejected for promotion | REVISE recall falls to 34% |
| Same-Origin Router V1 | rejected for promotion | only 23 strict new pairs; REVISE recall falls to 24% |
| Verifier-Reward GRPO V1 | rejected for promotion | constrained pilot still collapses toward KEEP |

## Evidence locations

- Decision-Only frozen run: `phase3/runs/two_stage_selective_repair/`
- DPO Pilot report: `phase3/runs/dpo_pilot_v1/dpo_pilot_report.md`
- Semantic V2 report: `phase3/runs/dpo_semantic_v2/semantic_v2_report.md`
- Semantic V2 probe: `outputs/phase3_dpo_semantic_v2/representation_probe/probe_report.md`
- Five-stage report: `outputs/phase3_five_stage_pipeline/pipeline_report.md`
- Frozen representation follow-up:
  `outputs/phase3_five_stage_pipeline/frozen_representation_followup_report.md`
- Source-disjoint calibration:
  `outputs/phase3_five_stage_pipeline/router_calibration_v1/report.md`
- New per-sample decisions and visible rationales:
  `outputs/phase3_five_stage_pipeline/stages/{02_decision_token,04_same_origin,05_reward_grpo}/frozen_eval/`

## Activation Steering V2 diagnostic (2026-09-13)

A frozen causal intervention tested standardized logistic-probe and CAA
directions at layers 14, 21, and 28. The evaluation used 20 source-disjoint
Router Recovery V2 rows balanced across math/code and KEEP/REVISE. No weights
were updated.

Final-token steering produced no decision or accuracy improvement. Layer 14's
probe direction had a small monotonic positive log-odds slope (`+0.002423`), but
all alphas retained 40% balanced accuracy, 30% KEEP recall, and 50% REVISE
recall. Extending that direction over the last 32 or all prompt tokens also
failed. The best apparent result, all-prompt `alpha=+2`, reached 50% balanced
accuracy by raising KEEP recall to 60% while reducing REVISE recall to 40%.

This is a bias shift, not improved semantic error detection. Activation
steering is therefore stopped at the diagnostic stage. Full report:
`outputs/phase3_activation_steering_v2/report.md`.

## Phase closure and future handoff

Phase 3 is closed. Do not repeat another LLM weight update, retune a threshold,
or select new rows against its frozen benchmark. A future phase may import the
immutable 108-pair recovery pool, but must define new holdouts and a new
preregistered hypothesis before collecting more hard code errors.
