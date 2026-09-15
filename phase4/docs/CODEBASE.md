# Phase 4 code ownership

Phase 4 is closed. Historical CLI and artifact paths remain compatible.
Maintenance must not change Phase 3 or rewrite generated evidence.

| Path | Responsibility |
|---|---|
| `lib/rollout.py` | JSONL/hashing helpers, schema, verification and HTTP completion |
| `lib/transition.py` | Strict deployment parsing and transition/reward rules |
| `scripts/collect_*.py` | Generation orchestration and raw/resumable collection |
| `scripts/score_rollouts.py` | Fresh objective rollout scoring |
| `scripts/evaluate_multiseed.py` | Paired policy evaluation on fixed datasets |
| `scripts/evaluate_three_rounds.py` | Evolving-answer review and paired metrics |
| `scripts/data/` | Source preparation, pair building, sealing and validation |
| `scripts/training/` | SFT/DPO/GRPO, adapter merge and inspection |
| `scripts/serving/` | Localhost vLLM launcher |
| `scripts/ops/` | Backup, filtered sync and historical GPU recovery |
| `scripts/validate_local_state.py` | CPU-only closure and optional adapter audit |
| `configs/experiments.yaml` | Final dispositions and evidence pointers |

Shared rollout helpers live in the library. `collect_rollouts.py` re-exports
its earlier helpers for compatibility. The refactor preserves prompt text,
strict parsing, generation settings, verifier behavior and stored evidence.
Training code retains its heavy GPU dependencies; the closure audit does not
need them. Use CPython 3.10 and SymPy 1.14.0 for verification.

```powershell
.venv-phase4-verify/Scripts/python.exe phase4/scripts/validate_local_state.py --hash-adapters
.venv-phase4-verify/Scripts/python.exe -m compileall -q phase4
```

The audit checks dispositions, hashes, diagnostic balance, blind collection
coverage and three-round metrics without opening confirmation candidates or
secrets. Adapters remain in `outputs/phase4_*`, data in `phase4/data/` and
results in `phase4/runs/`. Historical bootstrap and training CLIs are preserved
for reproduction; they are not authorization for new Phase 4 research.
