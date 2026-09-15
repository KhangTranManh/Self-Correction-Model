# Phase 4 — On-Policy Selective Revision

**Closed: 2026-09-15. No reliable autonomous correction gain established.**

Phase 4 tested warm-starts, GRPO, expanded correction SFT, preference DPO,
hint-free competing reviews and three-round autonomous review. None established
a reliable solution to Phase 3's error discrimination problem. The historical
V2 gain (73% to 74%) did not reproduce on the replacement GPU (73%). V2 remains
the archived pilot reference, not a promoted replacement for the Phase 3 router.

| Experiment | Result | Disposition |
|---|---|---|
| Warm-start V2 | 74% historical development; 73% rerun | Gain not reproduced |
| GRPO V3 | 73% development | Negative result |
| Correction SFT V3 | 74%; three fixes, two harms | No gain over historical V2 |
| Preference DPO V1 | 97% pair accuracy; 73% behavioral accuracy | No behavioral gain |
| Blind preferences V2 | 16,064 reviews; balanced 28 train / four dev pairs | Coverage gate failed; training not run |
| Three-round review | Mean 51.25% round 1 → 49.17% round 3 | Extra rounds reduced accuracy |

The 80-row three-round diagnostic is deliberately balanced and starts at 50%.
Its percentages cannot be compared directly with the 100-row math development
set. Confirmation remains unopened; no compliant sealed confirmation was run.

Read [the final report](docs/FINAL_REPORT.md), [results](docs/RESULTS.md),
[code ownership](docs/CODEBASE.md) and [pilot log](docs/PILOT_LOG.md).
Final dispositions are recorded in [configs/experiments.yaml](configs/experiments.yaml).

Phase 4 authorizes no further generation, tuning or model selection. Historical
CLIs and artifacts are retained for reproducibility. New research must open a
separate phase with its own hypothesis, settings and protected holdouts.

## Local closure audit

```powershell
.venv-phase4-verify/Scripts/python.exe phase4/scripts/validate_local_state.py --hash-adapters
```

## Historical design and execution

Phase 4 tests a new hypothesis after the closed Phase 3 router experiments:

> Multi-turn, approximately on-policy reinforcement learning with objective
> transition rewards can improve correction of initially wrong answers without
> teaching the policy to revise initially correct answers.

Phase 3 remains immutable. None of its frozen behavior or probe rows may be
used for Phase 4 training, sample selection, threshold selection, or early
stopping.

## Initial model

- Actor and frozen reference: `Kxck/Self_Correction_v1`.
- Phase 3 router adapters are not loaded.
- Correctness and reward come only from deterministic math/code verifiers.
- A newer-model control was not run in Phase 4.

## Training unit

Each episode records two independently verified states:

```text
problem → initial answer → neutral review → KEEP or REVISE+answer
        → verify initial → verify final → transition reward
```

The model never sees verifier results before its decision. A cycle uses the
latest adapter to sample initial answers, then GRPO samples review actions from
that same policy family. After training, all rollouts are refreshed.

## RTX 3090 execution model

The current GPU has 24 GB VRAM. vLLM generation and QLoRA/GRPO training run in
separate stages; they are not kept in memory at the same time.

```text
1. start vLLM
2. collect initial-answer rollouts
3. stop vLLM
4. train one bounded GRPO cycle
5. save adapter and reports
6. restart vLLM with the new adapter
7. collect a fresh cycle
```

## Repository contract

```text
phase4/
├── configs/transition_rl_v1.yaml
├── data/
│   ├── README.md
│   └── rollouts/                 generated, gitignored, copied to both machines
├── docs/PREREGISTRATION.md
├── lib/transition.py
├── runs/                         generated evidence, gitignored, copied to both
└── scripts/
    ├── collect_rollouts.py
    ├── collect_warmstart.py
    ├── score_rollouts.py
    ├── data/prepare_math_pilot.py
    ├── training/train_transition_grpo.py
    ├── serving/serve_actor.sh
    └── ops/sync_phase4.ps1
```

## Input contract

The collector reads a JSONL problem manifest. Every row must have `id`,
`domain`, and `question`. Math rows also require `reference_answer`; code rows
require `entry_point` and `tests`. A new source-disjoint training pool and a new
sealed evaluation manifest must be built before the first scientific run.

## Local/GPU storage

The canonical code lives in this repository on the workstation and is copied to
`/root/agi/` on the GPU. Generated evidence lives under `phase4/data/rollouts/`
and `phase4/runs/` on both machines. Use the sync script after every generation
or training stage. Adapter weights remain under `outputs/phase4_*` and must also
be pulled back immediately.

The sync script uses SSH authentication interactively or through an SSH key. It
contains no password.

For historical reproduction, `scripts/ops/restart_preference_gpu.sh` installs the
environments, reconstructs selected V2, and starts preference training. Make
and check code changes locally before uploading. DPO saves resumable
checkpoints every 25 steps; `scripts/ops/watch_gpu_evidence.py` runs locally
to copy stable checkpoints and final evidence back during training. Completed
DPO V1 did not improve behavioral accuracy; see `docs/PILOT_LOG.md`.

On Windows, if the local PowerShell execution policy blocks scripts, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  phase4/scripts/ops/sync_phase4.ps1 -Direction PushCode
```
