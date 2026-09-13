# Phase 3 DPO Pilot V1 — Interrupted Remote Run (Historical)

> Superseded status: a clean replacement-server rerun later completed training,
> downloaded the adapter, finished all 200 decisions and rationales, ran the
> frozen benchmark, and ran a representation probe. Use
> `dpo_pilot_report.md` and `frozen_eval/two_stage_summary.json` for final
> results. The hashes below belong only to the interrupted remote attempt.

## Confirmed completed

- Starting policy/reference: Decision-Only V1 LoRA on `Kxck/Self_Correction_v1`.
- Input: 240 decision preferences from 120 verified same-problem pairs.
- Training: one epoch, 30 optimizer steps, peak learning rate `5e-6`, beta `0.1`.
- GPU: NVIDIA GeForce RTX 3090 24 GB; peak observed training VRAM about 19.9 GB.
- Runtime: 643.10 seconds.
- Trainable parameters: 80,740,352.
- Final adapter was saved remotely before evaluation.
- Final adapter SHA-256: `4547044993ae8c3eff3783facef85337211af8af6fb65a59c2068a2f891fb175`.

Preference diagnostics on the 240 training rows:

| Metric | Decision-Only V1 | DPO after |
|---|---:|---:|
| Preference accuracy | 49.17% | 53.33% |
| Mean chosen margin | 0.0154 | 0.1794 |
| Median chosen margin | -0.0277 | 0.0631 |

## Frozen benchmark status

- Fresh initial-answer verification: 200/200 passed after making the APPS
  resource-limit launcher independent of the Torch-heavy parent process.
- Scored DPO decisions generated: 200/200.
- Observable rationale audit generated: at least 40/200 at the last successful poll.
- Existing 200/200 neutral repair generations were reused because the repair
  model and frozen inputs did not change.
- Aggregate metrics, unseen-template decisions, and the representation probe
  were not completed/downloaded before the remote endpoint became unreachable.

## Interruption

Both SSH `159.48.242.34:21602` and Jupyter `159.48.242.34:21603` became
unreachable during rationale generation. Multiple retries failed. Resume the
same instance if possible: the evaluator uses JSONL caches and should continue
without regenerating completed rows. Do not claim benchmark or probe success
from this partial state.
