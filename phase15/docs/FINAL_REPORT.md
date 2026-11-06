# Phase 15 final report — development study closed

Closed 2026-11-03. Phase 15 asked whether one or two verified corrections
could yield a useful temporary weight update on new tasks without the
correction in context. The work reached development diagnostics through
P15-A5. It did not reach the preregistered main evaluation or meta-trained
self-edit arm. Therefore the phase has an operational stop decision, not a
scientific yes/no result about fast learning in 7B models.

## What ran

The learner in the completed pilot was Qwen/Qwen2.5-7B-Instruct, loaded with
NF4 on a T4. A0 was the unchanged model, A1 supplied the verified correction
in context, and A2 applied a temporary three-step LoRA update. The matched
P15-A3 comparison used the same seed for every item across these three arms,
40 S-format episodes, two probes and one unrelated control per episode. The
adapter was reset after each episode. P15-A4 and P15-A5 used only saved P15-A3
outputs on CPU. No protected evaluation split was opened.

The engineering smoke validated loading, updates, reset and audit. P15-A1
stopped at its seven-hour wall after 31/40 S-format episodes; J-format could
form only one leakage-safe disjoint episode in that screened pool and was
closed as infeasible there. P15-A2 finished 40 S-format episodes, but its arm
seeds differed, so those arm differences cannot isolate an intervention.
P15-A3 reran all 40 with paired seeds. P15-A4 tested deterministic suffix
normalization. P15-A5 decomposed strict parser success from correctness under
the frozen rescued-answer extractor.

## Results on the matched P15-A3 data

| Measure | A0 base | A1 context | A2 LoRA |
|---|---:|---:|---:|
| Probe strict correctness | 31/80 | 37/80 | 40/80 |
| Probe rescued-answer correctness | 45/80 | 44/80 | 49/80 |
| Probe S-format clean | 38/80 | 46/80 | 53/80 |
| Control strict correctness | 30/40 | 34/40 | 29/40 |
| Control rescued-answer correctness | 36/40 | 38/40 | 34/40 |

Against paired A0 items, A2 changed rescued-answer probe correctness with
13 fixes and 9 harms, and control correctness with 3 fixes and 5 harms.
A1 changed rescued-answer probe correctness with 6 fixes and 7 harms.
These are descriptive development counts; the rescued extractor is not a
proof of correct intermediate reasoning.

P15-A4 preserved the original A0 generations and appended a canonical final
line to 49 outputs where the existing numeric answer could be safely rescued.
It raised strict probe correctness from 31/80 to 45/80 and strict control
correctness from 30/40 to 36/40, with no observed correctness harms. All 120
outputs became S-format clean. Of the 49 normalized outputs, 29 were still
incorrect. Output recovery therefore solves the measured format problem but
does not solve wrong answer content.

## Decision and limits

The frozen P15-A5 gate required A2's rescued-answer control loss to be no
worse than 1/40 while retaining at least +4/80 probe gain and more probe fixes
than harms. The probe conditions passed; controls fell by 2/40. The gate
failed, so the planned IA3 follow-up on this S-format branch is closed. Hard
token blocking was not pursued because P15-A4 already achieved complete
S-format cleanliness without changing generation. These are development
resource decisions, not formal hypothesis tests.

S-format is a poor remaining test of learning from corrections because a
deterministic output processor recovers its main measurable gain. The only
residual A2 answer signal is small and accompanied by control regression.
No claim is made about IA3 in other settings, meta-trained self-editing,
continual updates, other model families, or the original preregistered P1–P3
endpoints. Those endpoints were not evaluated. Phase 16 starts a separate
protocol to define a substantive, verifiable error type before considering
another model intervention.

## Audit trail

- P15-A3 audit: `../results/p15_a3_audit.jsonl`, SHA-256
  `ef2a05b5e4c534451028a2b8cf97fa0787bb381f83171ce0e9b2d9542562bf79`.
- Frozen episode manifest: `../kaggle/pilot_s_format_continuation_v1/episodes.json`,
  SHA-256 `1839aba51f447c4f0da8b1b95a8fea9775072a81736f50f1faf96e418ac1c2e9`.
- Independent P15-A3 labels: `../results/P15_A3_RESCORE.json`.
- P15-A4 and P15-A5: `PILOT_A4.md`, `P15_A4_RESCORE.json`, `PILOT_A5.md`,
  `P15_A5_RESCORE.json`. P15-A5 verified all 40 episodes, 120 matched-seed
  items, and zero disagreements with saved correctness and all-error joint
  labels.
- Chronological record: `../RUN_STATUS.md`.
