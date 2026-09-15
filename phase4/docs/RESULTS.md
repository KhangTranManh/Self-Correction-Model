# Final results and evidence

Phase 4 closed on 2026-09-15. All values are development results, not sealed
confirmation or model promotion.

## Original 100-source math comparison

All runs reuse the same 73-correct initial answers. Historical and replacement
GPU observations are separate measurements.

| Policy/run | Accuracy | KEEP | REVISE | Invalid | Fixes | Harms |
|---|---:|---:|---:|---:|---:|---:|
| Rebuilt base | 73% | 95 | 3 | 2 | 0 | 0 |
| V2 historical | 74% | 91 | 8 | 1 | 1 | 0 |
| GRPO V3 | 73% | 92 | 7 | 1 | 0 | 0 |
| Correction SFT V3 | 74% | 86 | 14 | 0 | 3 | 2 |
| DPO V1 replacement GPU | 73% | 94 | 5 | 1 | 0 | 0 |
| V2 replacement rerun | 73% | 93 | 5 | 2 | 0 | 0 |

Evidence under `phase4/runs/`:

- `warmstart/dev_base_summary.json`, `dev_v2_summary.json`, `dev_grpo_v3_summary.json`.
- `expansion_v1/dev_correction_sft_v3_summary.json`, `paired_v2_vs_sft_v3.json`.
- `preference_v1/dev_dpo_v1_summary.json`, `dev_v2_replacement_summary.json`.
- `selection.json`: pilot reference and reproducibility note.
- Training reports and weights: `outputs/phase4_*`.

DPO V1 pair accuracy was 97%, loss 0.5863 and mean margin 0.2353. These are
pair-ranking metrics, not final-answer accuracy.

## Blind preference collection

16,064 reviews of 1,004 sources: 14,104 KEEP, 1,568 REVISE and 392 invalid.
108 verified fixes and 42 harmful revisions are attempt counts, not distinct
sources. Available pairs were train KEEP/REVISE 14/44 and dev 2/4. Balanced
outputs have 28 train and four dev rows. The frozen training gate failed.

Evidence: `phase4/data/blind_preferences_v2/collection_manifest.json`,
`blind_attempts.jsonl`, `summary.json`, `train.jsonl` and `dev.jsonl`.
The prepared DPO V2 config is marked not run due to insufficient coverage.

## Three-round autonomous review

The 80-source diagnostic has 20 examples per math/code × initial correct/wrong
bucket. Three seeds produce 240 complete trajectories and 720 reviews with no
verifier feedback.

| Seed | Round 1 | Round 2 | Round 3 |
|---|---:|---:|---:|
| 20260914 | 52.50% | 50.00% | 48.75% |
| 20260915 | 50.00% | 48.75% | 50.00% |
| 20260916 | 51.25% | 50.00% | 48.75% |
| Mean | 51.25% | 49.58% | 49.17% |

All seeds start at 50%. Round 1 retains three fixes and zero initial-answer
harms across trajectories; round 3 retains one fix and three harms. The
paired source bootstrap is described in [the protocol](THREE_ROUND_DIAGNOSTIC.md).

Evidence: `phase4/runs/three_round_v1/protocol.json`, `trajectories.jsonl`,
`summary.json`, `run.log`. Local/GPU hashes matched for all four at completion.
No weights were updated or promoted.
