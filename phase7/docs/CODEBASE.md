# Phase 7 ownership and artifact rules

| Path | Planned role |
|---|---|
| `README.md` | Phase question, status, and result boundary |
| `configs/experiments.yaml` | Mutable Phase 7 status registry until a protocol lock is created |
| `configs/blind_resolve_v1.yaml` | Draft experimental settings; freeze a versioned lock before GPU re-solving |
| `docs/PREREGISTRATION.md` | Hypotheses, metrics, decisions, and data safeguards |
| `docs/EXECUTION_PLAN.md` | CPU-first and GPU-later execution order |
| `data/` | Future source inventory, selected manifests, split locks, and receipts |
| `scripts/` | CPU candidate builder, FP16 vLLM smoke and resumable initial collector; 4-bit fallback scripts; paired runner and analysis remain future work |
| `outputs/phase7_*` | Future raw outputs, activations, audit logs and reports; gitignored |

Current Phase 7 files contain a plan and CPU candidate manifest only. Keep old Phase 5/6 manifests and
results immutable. No Phase 7 output may silently enter training data. A
source audit, exact protocol lock and checkpoint hash verification must precede
protected model generation. Store raw GPU evidence both on the GPU and on this
workstation, with matching hashes.
