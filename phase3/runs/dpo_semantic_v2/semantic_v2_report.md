# Phase 3 DPO Semantic V2 Report

## Dataset

- 100 same-problem contrastive pairs / 200 preference rows
- Labels: 100 KEEP / 100 REVISE
- Domains: 73 math / 27 code
- Datasets: 73 GSM8K / 20 APPS / 7 MBPP
- Origin direction: 48 Base-correct versus V1-wrong; 52 V1-correct versus Base-wrong
- Both answers in every pair are natural model generations; no reference answer is included in a training prompt
- Fresh verification: correct 100/100 pass; wrong 100/100 fail
- All 100 wrong answers are semantic and plausible under the construction filters
- Frozen benchmark overlap: zero
- Same-origin pairs: zero, because the verified historical pool contains only one generation per model/problem

## Training

- Initial checkpoint: Decision-Only V1 LoRA over `Kxck/Self_Correction_v1`
- Method: 4-bit LoRA DPO
- Epochs: 1
- Learning rate: `5e-6`
- Beta: `0.1`
- Runtime: 447.44 seconds on RTX 3090 24 GB
- Training preference accuracy: 54.0% before to 59.0% after
- Mean chosen-minus-rejected margin: 0.1594 before to 0.3651 after
- Adapter SHA-256: `cef8cc95ca960670bb7e878c613aa8154873714587023342fcab4ed1c7edf0c1`

## Frozen behavioral benchmark

| Metric | Decision-Only V1 | DPO Pilot V1 | DPO Semantic V2 |
|---|---:|---:|---:|
| Balanced decision accuracy | 65.0% | 66.5% | 65.5% |
| KEEP recall | 81.0% | 85.0% | 85.0% |
| REVISE recall | 49.0% | 48.0% | 46.0% |
| Code decision accuracy | 62.0% | 62.0% | 59.0% |
| Math decision accuracy | 68.0% | 71.0% | 72.0% |
| Unseen-template accuracy | 58.0% | 55.5% | 53.5% |
| Final selective-repair accuracy | 51.5% | 54.5% | 55.5% |

Semantic V2 became more conservative: it preserved correct answers well but missed 54/100 wrong answers. The increase in final repair accuracy is driven by fewer harmful revisions and the fixed repair model, not stronger error detection.

## Representation probe

The same frozen 256-source probe set and the same layers were used. Decision-Only V1 activations were reused from the previously saved baseline; Semantic V2 activations were freshly extracted on an RTX 3090.

| Metric | Decision-Only V1 | DPO Semantic V2 | Delta |
|---|---:|---:|---:|
| Dev-selected layer | 28 | 7 | — |
| Test balanced accuracy | 67.3% | 57.7% | -9.6 pp |
| KEEP recall | 80.8% | 76.9% | -3.9 pp |
| REVISE recall | 53.8% | 38.5% | -15.3 pp |
| ROC-AUC | 0.734 | 0.599 | -0.135 |
| Code balanced accuracy | 57.7% | 53.8% | -3.9 pp |
| Math balanced accuracy | 76.9% | 61.5% | -15.4 pp |

Conclusion: `partial_internal_error_signal`, but Semantic V2 is materially weaker than Decision-Only V1. The new DPO data removed the reference-versus-model shortcut, yet the decision-token objective still pushed the policy toward KEEP instead of learning a stronger representation of semantic correctness.

## Live smoke test

The OpenAI-compatible vLLM service exposed both the base checkpoint and Semantic V2. Semantic V2 followed the exact decision decision contract on both requests:

- `17 + 25`, previous answer `42`: KEEP (correct)
- `17 + 25`, previous answer `43`: KEEP (incorrect)

This failure is consistent with the frozen behavioral and representation results.

## Artifacts

- Dataset summary: `phase3/data/semantic_model_dpo/semantic_dpo_summary.json`
- Preferences: `phase3/data/semantic_model_dpo/dpo_preferences.jsonl`
- Adapter: `outputs/phase3_dpo_semantic_v2/final_adapter/`
- Training report: `outputs/phase3_dpo_semantic_v2/train_report.json`
- Frozen evaluation: `phase3/runs/dpo_semantic_v2/frozen_eval/`
- Representation report: `outputs/phase3_dpo_semantic_v2/representation_probe/probe_report.md`
- Probe predictions: `outputs/phase3_dpo_semantic_v2/representation_probe/probe_test_predictions.jsonl`
