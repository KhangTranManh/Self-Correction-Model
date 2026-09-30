# Phase 7 ownership and artifact rules

| Path | Role |
|---|---|
| `README.md` | Phase question, status, and result boundary |
| `configs/experiments.yaml` | Status registry; hash-bound pre-smoke snapshot left unchanged to preserve the protocol locks |
| `configs/blind_resolve_v1.yaml` | Frozen experimental settings (hashed in `data/protocol/`) |
| `docs/PREREGISTRATION.md` | Hypotheses, metrics, decisions, and data safeguards |
| `docs/EXECUTION_PLAN.md` | Executed CPU-first and GPU-later order |
| `docs/FINAL_REPORT.md` | Protected result and limitations |
| `data/` | Source inventory, balanced split, and protocol locks |
| `scripts/` | CPU candidate builder, FP16 vLLM smoke, resumable initial and paired collectors, paired analysis, and 4-bit fallback scripts |
| `outputs/phase7_*` | Raw outputs and reports; gitignored, and no longer present on this workstation |

Phase 7 is a completed diagnostic. Keep old Phase 5/6 manifests and results
immutable. No Phase 7 output may enter training data. The raw
`outputs/phase7_*` evidence was later lost from this workstation; the final
report, protocol locks, and split files remain the record. Phase 8
regenerated a separate donor pool with the frozen initial protocol for
distractor text only.
