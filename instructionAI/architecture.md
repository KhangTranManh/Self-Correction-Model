# Project Architecture

This document defines ownership and dependency boundaries for the current
repository. The root README routes current work. Phase 3 documentation owns
its closed results; Phase 4 documentation owns its closed correction pilots.

## Phase boundaries

```text
Phase 1 verified solver
        │
        ├── Kxck/Self_Correction_v1 (base solver/repair model)
        │
        ▼
Phase 3 immutable Base/V1 attempts
        │
        ├── verifier-labelled source inventory
        ├── router training datasets
        └── protected frozen evaluations
                │
                ▼
        Decision-Only V1 router
                │
                ├── DPO Pilot V1 (not promoted)
                └── DPO Semantic V2 (not promoted)
```

Phase 2 is a separate KTO exploration. Its data and scripts are not imported by
the canonical Phase 3 pipeline.

## Top-level ownership

| Path | Owner | Purpose |
|---|---|---|
| `phase1/` | Phase 1 | Verified solve/correct pipeline and historical results |
| `phase2/` | Phase 2 | KTO preference experiment |
| `phase3/` | Phase 3 | Closed discrimination/selective-repair research package |
| `phase4/` | Phase 4 | Closed selective-correction pilots and immutable evidence |
| `phase5/` | Phase 5 | Active guided-repair and pre-hint probe design; CPU source inventory prepared |
| `outputs/` | Runtime | Local adapters, logs, activations; excluded from Git |
| `instructionAI/` | Project | Cross-phase architecture and data invariants |

## Canonical Phase 3 layers

### `phase3/lib/`

Reusable implementation primitives only:

- provenance resolution;
- prompt construction;
- math/code verification;
- APPS sandbox execution.

Libraries must not select experiment samples or contain hard-coded run paths.

### `phase3/scripts/data/`

CPU-oriented acquisition and deterministic dataset builders. Builders may read
immutable attempts and verifier metadata, but may not read frozen model outputs
to choose training samples.

### `phase3/scripts/training/`

GPU training entry points. All hyperparameters and artifact locations come from
YAML configs. Training scripts do not silently rebuild datasets.

### `phase3/scripts/evaluation/`

Frozen behavioral evaluation, model comparison, activation extraction, and
linear probes. Evaluation outputs belong under `phase3/runs/` or a clearly named
runtime output directory.

### `phase3/scripts/serving/`

vLLM launchers only. Serving is operational and never changes model weights.

### `phase3/configs/`

- one YAML file per training run;
- `experiments.yaml` as the canonical model/status registry;
- no secrets or provider passwords.

### `phase3/data/`

```text
source/ and apps_pilot/candidates/    original problem records
attempts/ and apps_pilot/attempts/    immutable natural model generations
behavior/ and buckets/                compact provenance/index metadata
decision_only/                        Decision-Only V1 SFT data
revised_router/                       Router V3 experimental data
contrastive_pairs/                    first 120-pair DPO source
dpo_pilot/                            DPO Pilot V1 preferences
semantic_model_dpo/                   100 model-vs-model semantic pairs
two_stage_selective_repair/           protected frozen evaluation manifest
```

### `phase3/runs/`

Research evidence: raw decisions, model-visible rationales, verifier results,
metrics, and generated reports. These files are not training inputs.

## Model roles

| Model | Parent | Role | Status |
|---|---|---|---|
| `Kxck/Self_Correction_v1` | Phase 1 model | Solver and repair model | canonical solver |
| Decision-Only V1 | Self_Correction_v1 | KEEP/REVISE router | canonical router |
| Router V3 mini | Decision-Only V1 | REVISE-heavy router experiment | diagnostic only |
| DPO Pilot V1 | Decision-Only V1 | First decision-token DPO | not promoted |
| DPO Semantic V2 | Decision-Only V1 | Model-vs-model semantic DPO | not promoted |

LoRA adapters are not standalone models. Serving or extraction loads
`Kxck/Self_Correction_v1` first, then attaches the selected adapter.

## Artifact lifecycle

```text
source problem
  → natural Base/V1 attempt
  → deterministic verification
  → immutable inventory/bucket
  → deterministic dataset builder
  → reviewed YAML config
  → adapter training
  → immediate local backup + SHA-256
  → frozen behavioral evaluation
  → representation probe
  → promote or reject in experiments.yaml
```

No later result may retroactively change an earlier frozen manifest. A new
manifest or experiment version is required.

Phase 3 is closed as of 2026-09-13. Its canonical and negative-result artifacts
remain immutable. New discrimination research must create a new phase rather
than adding rows, adapters, or tuned thresholds to Phase 3.

## Security and operational rules

Phase 4 is closed as of 2026-09-15. No reliable correction gain was established
and no compliant confirmation was run. Warm-start V2 remains an archived pilot
reference, not a promoted canonical model. Final dispositions live in
`phase4/configs/experiments.yaml`; ownership and closure rules live in
`phase4/docs/CODEBASE.md` and `phase4/docs/FINAL_REPORT.md`. Preserve historical
configs and generated evidence. Any new research must open a separate phase.

- Make and check implementation changes locally before uploading to the GPU.
- Use Luna or Terra for routine log monitoring and straightforward checks when
  delegation is available; retain experiment design and result decisions with
  the main agent, as requested by the user.
- Use CPython 3.10/SymPy 1.14.0 for Phase 4 verification on both machines;
  Python-version differences can change parsing of trailing currency symbols.

- Never inspect, copy, document, or commit `.env` values.
- Never store SSH passwords in scripts, reports, or configs.
- vLLM binds to `127.0.0.1` by default; public binding requires an explicit
  `PHASE3_HOST=0.0.0.0` choice and should be protected externally.
- Stop GPU processes or the rented instance when work is complete.
- Download adapters and small reports immediately after each stage; do not wait
  for all downstream evaluation to finish.
