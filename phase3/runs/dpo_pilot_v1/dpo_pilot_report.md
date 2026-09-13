# Phase 3 DPO Pilot V1 Report

## Status

- Training: complete
- Frozen 200-row benchmark: complete
- Representation probe: complete
- Local artifact backup: complete
- vLLM serving: completed as an operational test; no persistent server is assumed

## Training setup

- Starting checkpoint: Decision-Only V1 adapter over `Kxck/Self_Correction_v1`
- Method: DPO-style preference optimization on the existing LoRA parameters
- Data: 120 clean contrastive pairs / 240 behavioral rows
- Composition: 180 code rows and 60 math rows; 140 APPS, 40 MBPP, and 60 GSM8K
- Epochs: 1
- Learning rate: `5e-6`
- Beta: `0.1`
- Optimizer steps: 30
- Trainable parameters: 80,740,352
- Runtime: 638.34 seconds on one RTX 3090 (24 GB)
- Truncation: none; maximum observed sequence length was 2,315 tokens

The preference audit on the training set changed as follows:

| Metric | Decision-Only V1 before | DPO Pilot V1 after | Delta |
|---|---:|---:|---:|
| Preference accuracy | 49.17% | 52.92% | +3.75 pp |
| Mean chosen-minus-rejected margin | 0.0154 | 0.1777 | +0.1623 |
| Median chosen-minus-rejected margin | -0.0277 | 0.0512 | +0.0788 |

This is an in-training audit, not an estimate of generalization.

## Frozen behavioral benchmark

The same fixed 200-row benchmark was used for both models. Initial-answer labels were freshly reverified on the new server and passed 200/200. The DPO evaluation saved 200 scored decisions and a separate, non-scored rationale for every row.

| Metric | Decision-Only V1 | DPO Pilot V1 | Delta |
|---|---:|---:|---:|
| Balanced decision accuracy | 65.0% | 66.5% | +1.5 pp |
| KEEP recall | 81.0% | 85.0% | +4.0 pp |
| REVISE recall | 49.0% | 48.0% | -1.0 pp |
| REVISE precision | 72.06% | 76.19% | +4.13 pp |
| Code decision accuracy | 62.0% | 62.0% | 0.0 pp |
| Math decision accuracy | 68.0% | 71.0% | +3.0 pp |
| Unseen neutral-template accuracy | 58.0% | 55.5% | -2.5 pp |
| Exact contract rate (main prompt) | 100% | 100% | 0.0 pp |
| Exact contract rate (unseen prompt) | 100% | 95.5% | -4.5 pp |
| Final selective-repair accuracy | 51.5% | 54.5% | +3.0 pp |
| Correct preservation rate | 92.0% | 98.0% | +6.0 pp |
| Harmful correct-to-wrong revisions | 8 | 2 | -6 |
| Missed wrong answers | 51 | 52 | +1 |

The DPO policy became more conservative. It preserved correct answers better and improved overall balanced accuracy slightly, but did not improve error detection. It issued 63 REVISE decisions versus 68 for V1, and missed one additional wrong answer.

## Representation probe

The language-model weights were frozen during extraction. Regularized logistic-regression probes used the same fixed 256-source train/dev/test split and final-prompt-token activations from layers 7, 14, 21, and 28.

| Metric | Decision-Only V1 | DPO Pilot V1 | Delta |
|---|---:|---:|---:|
| Dev-selected layer | 7 | 7 | — |
| Test balanced accuracy | 57.7% | 59.6% | +1.9 pp |
| KEEP recall | 76.9% | 80.8% | +3.9 pp |
| REVISE recall | 38.5% | 38.5% | 0.0 pp |
| ROC-AUC | 0.629 | 0.635 | +0.006 |
| Code balanced accuracy | 57.7% | 57.7% | 0.0 pp |
| Math balanced accuracy | 57.7% | 61.5% | +3.8 pp |

The answer-length-only control also reached 57.7% balanced accuracy. Therefore the small aggregate probe gain is not strong evidence that DPO learned a better internal representation of correctness. The probe conclusion remains `weak_internal_error_signal`.

## Success-criteria decision

| Criterion | Result |
|---|---|
| Balanced accuracy improves | PASS: +1.5 pp |
| REVISE recall improves without KEEP collapse | FAIL: REVISE -1.0 pp; KEEP +4.0 pp |
| Code accuracy improves | FAIL: unchanged |
| Hidden-state probe improves, especially code | FAIL overall: small aggregate gain, no code or REVISE-recall gain |

Overall, this pilot does not meet the stated success criteria. DPO changed policy preference toward KEEP and reduced harmful false revisions, but it did not strengthen detection of wrong answers or code-error representation. It should not replace Decision-Only V1 as the main REVISE router without another training/data iteration.

## Saved artifacts

- Trained adapter: `outputs/phase3_dpo_pilot_v1/final_adapter/`
- Training report: `outputs/phase3_dpo_pilot_v1/train_report.json`
- Preference audits: `outputs/phase3_dpo_pilot_v1/reference_logps.jsonl` and `outputs/phase3_dpo_pilot_v1/final_preference_audit.jsonl`
- Frozen benchmark report: `phase3/runs/dpo_pilot_v1/frozen_eval/two_stage_report.md`
- Per-row decisions and rationales: `phase3/runs/dpo_pilot_v1/frozen_eval/decision_generations.jsonl` and `phase3/runs/dpo_pilot_v1/frozen_eval/decision_rationales.jsonl`
- Representation report: `outputs/phase3_dpo_pilot_v1/representation_probe/probe_report.md`
- Probe activations and per-row predictions: `outputs/phase3_dpo_pilot_v1/representation_probe/activations/` and `outputs/phase3_dpo_pilot_v1/representation_probe/probe_test_predictions.jsonl`

The locally backed-up adapter SHA-256 is `27a784b7515ec818a79862a15194d2e57402d867e3b8fae49deae05655ec89e3`.

## Serving

The rented server exposed an OpenAI-compatible vLLM endpoint during the run with these model IDs:

- `self-correction-v1` — base checkpoint
- `phase3-dpo-pilot-v1` — the DPO LoRA adapter

Both model IDs passed a live chat-completions smoke test. Server addresses and
credentials are intentionally not treated as durable project configuration.
