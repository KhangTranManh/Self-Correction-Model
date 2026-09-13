# Closed Phase 3 codebase

Phase 3 closed on 2026-09-13. This tree is an immutable research package for
audit and reproduction, not an active experiment workspace.

## Directory contract

| Path | Contents | Lifecycle |
|---|---|---|
| `configs/` | run configs and `experiments.yaml` registry | frozen |
| `data/` | source, training, calibration, and protected datasets | catalogued; canonical files immutable |
| `lib/` | verifier, provenance, prompt primitives | maintenance fixes only |
| `runs/` | raw decisions, rationales, activations, reports | immutable evidence |
| `scripts/data/` | deterministic acquisition/builders | reproduction only |
| `scripts/training/` | QLoRA, DPO, classifier, reward pilots | historical reproduction only |
| `scripts/evaluation/` | frozen evaluation, probes, steering | reproduction only |
| `scripts/serving/` | vLLM adapter serving | comparison only |
| `scripts/maintenance/` | catalog and integrity tools | active maintenance |
| `../outputs/` | local adapters and runtime artifacts | external to committed data |

Stable paths are intentionally retained. Moving canonical data into a prettier
tree would invalidate configs, recorded hashes, and provenance references.
Cleanliness is provided by lifecycle catalogs rather than path churn.

## Canonical entry points

- Data sources: `prepare_source_pool.py`, `prepare_apps_pilot.py`.
- Natural attempts: `collect_initial_attempts.py`, `collect_apps_pilot.py`.
- Final router corpus: `build_decision_only.py`.
- Protected benchmark: `build_two_stage_eval.py`.
- Probe: `build_representation_probe.py`, `extract_representation_probe.py`,
  `run_representation_probe.py`.
- Router training: `train_behavior_pilot.py`.
- Frozen behavior: `evaluate_two_stage_selective_repair.py`.
- Serving: `serving/serve_phase3.sh`.
- Integrity: `validate_local_state.py`,
  `maintenance/build_data_catalog.py`.

Other builders/trainers remain solely to reproduce rejected or incomplete
experiments. Their lifecycle is recorded in `configs/experiments.yaml` and
`data/catalog.json`.

## Final lineage

```text
Kxck/Self_Correction_v1
└── Decision-Only V1                         final Phase 3 baseline
    ├── Router V3 Mini                       diagnostic
    ├── DPO Pilot V1                         rejected
    ├── DPO Semantic V2                      rejected
    ├── Decision-Token V2                    rejected
    ├── Same-Origin Router V1                rejected
    ├── Verifier-Reward Pilot                rejected
    ├── Frozen classifier / calibration      diagnostic, rejected for deployment
    ├── Activation Steering V2               rejected
    └── Router Recovery V2                   incomplete, abandoned
```

LoRA adapters are not standalone models. Historical reproduction loads
`Kxck/Self_Correction_v1` and then attaches the selected adapter.

## Data lifecycle

- `canonical`: necessary to reproduce the final Decision-Only baseline.
- `canonical_frozen`: protected evaluation/probe; never train on it.
- `historical_negative`: necessary to explain a rejected experiment.
- `completed_gate_failed`: completed calibration or diagnostic that failed.
- `incomplete_closed`: verified partial work that never crossed its run gate.

Large unexecuted generation queues are not evidence. The Router Recovery full
scan was compacted to its summary and reproducible builder; verified pairs and
raw generations actually used in analysis were preserved.

## Change policy

- Do not add new training data or experiments to Phase 3.
- Maintenance may fix security, compatibility, or deterministic-reproduction
  bugs without changing recorded scientific outcomes.
- Never rewrite frozen manifests, raw attempts, scored outputs, or hashes.
- Never store `.env`, access tokens, SSH passwords, or provider details here.
- A new hypothesis must start a new phase with new holdouts.
