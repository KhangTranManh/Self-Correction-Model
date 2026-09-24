# Phase 6 codebase and artifact ownership

Phase 6 is closed. This directory is intentionally compact and flat: the
scripts are small, single-purpose CLIs and their names identify the diagnostic
stage. Do not delete them merely because the phase is not active; they are the
reproduction path for the negative result.

## Ownership map

| Path | Purpose | Mutability |
|---|---|---|
| `configs/` | Frozen protocol contracts and parameters. | Historical; do not tune in place. |
| `data/` | Selected source manifests and hash locks. | Immutable evidence; never training data. |
| `scripts/prepare_*` | Deterministic source/manifests preparation. | Reproduction only. |
| `scripts/run_*_vllm.py` | GPU generation under a frozen contract. | Reproduction only. |
| `scripts/extract_harness_activations.py` | Frozen forward-pass activation extraction. | Reproduction only. |
| `scripts/calibrate_*` and `scripts/analyze_*` | CPU analyses and reports. | Reproduction only. |
| `docs/` | Human-readable results, closure, and ownership. | Canonical Phase 6 documentation. |
| `outputs/phase6_*` | Raw generations, verifier records, activations, and reports. | Audit evidence; gitignored; never training data. |

## Script groups

### Data and locks

- `prepare_pilot.py`
- `prepare_fuzzy_hint_pilot.py`
- `prepare_harness_confirmation_sources.py`
- `finalize_harness_confirmation_holdout.py`
- `build_harness_confirmation_routes.py`

These scripts create inputs in the only permitted order: source pool, initial
rollout, balanced holdout, then frozen routes. Each lock records the hashes of
its upstream inputs.

### GPU runtime and activations

- `run_vllm_pilot.py`
- `run_fuzzy_hint_vllm.py`
- `run_harness_vllm.py`
- `extract_harness_activations.py`

Runtime scripts retain full prompts and raw model outputs. The harness runtime
has three distinct conditions: self-confidence routing, frozen-probe routing,
and oracle-known-wrong repair. Do not merge their metrics or prompts.

### Analysis

- `analyze_pilot.py`
- `analyze_fuzzy_hint.py`
- `analyze_layer_path.py`
- `calibrate_verbal_confidence.py`
- `analyze_router_repair_decomposition.py`
- `analyze_harness_confirmation.py`

Each analyzer reads saved verifier records and writes a machine-readable report
plus a Markdown summary. Analysis must never mutate an upstream manifest.

## Reproduction order

For the fresh harness confirmation only:

1. `prepare_harness_confirmation_sources.py`
2. Initial generation with `run_harness_vllm.py --stage initial`
3. `finalize_harness_confirmation_holdout.py`
4. Confidence generation and `extract_harness_activations.py` for every
   checkpoint
5. `build_harness_confirmation_routes.py`
6. Recheck generation with `run_harness_vllm.py --stage recheck`
7. `analyze_harness_confirmation.py`

Do not rerun this sequence against the locked holdout to choose a new setting.
It is a reproducibility path, not an optimization loop.
