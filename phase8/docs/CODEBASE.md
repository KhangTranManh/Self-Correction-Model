# Phase 8 ownership and reproduction

Amendment v2 scripts are the canonical run. Files named in
`data/execution_lock_v2.json` must not change; the lock hashes LF bytes, so
verify on an LF checkout (a Windows `core.autocrlf=true` checkout reports a
mismatch for line endings alone).

| Path | Role |
|---|---|
| `docs/PREREGISTRATION.md` | Hypotheses, controls, metrics, and gates |
| `docs/FINAL_REPORT.md` | Protected results and disposition |
| `RUN_STATUS.md` | Host history, amendment v2, and every deviation |
| `data/fresh_source_pool_v1/` | 400 protected sources (opened) |
| `data/model_lineage_v1.json` | Checkpoint revisions and adapter hashes |
| `data/execution_lock_v1.json` | Superseded v1 lock (historical) |
| `data/execution_lock_v2.json` | Canonical lock for the completed run |
| `data/distractors_v2/` | Frozen length-matched distractor map |
| `scripts/batched.py` | Ordered, resumable, batched vLLM generation |
| `scripts/fetch_adapters.py` | Download pinned private adapters and verify hashes |
| `scripts/regen_phase7_donors.py` | Regenerate the Phase 7 donor pool with the frozen protocol |
| `scripts/collect_first_pass_vllm.py` | First answer, resample, and greedy controls |
| `scripts/freeze_distractors.py` | Deterministic donor matching |
| `scripts/collect_three_arms_vllm.py` | Blind, own-visible, and distractor arms |
| `scripts/score_probe.py` | Score first answers with the frozen Phase 5 probe |
| `scripts/analyze_protected.py` | Single protected opening and report |
| `scripts/ops/run_remote.sh` | Sequential, skip-if-done GPU pipeline |
| `scripts/ops/supervise.sh` | Retry wrapper (up to five attempts) |
| `scripts/sync_remote.py` | Mirror remote outputs locally (credentials from a local JSON file) |

Scripts for the lost v1 run (`reproduce_phase7_initials.py`,
`fit_locked_probe.py`, `mark_phase5_lineage.py`, protocol audits, and
`build_transfer.py`) are kept for history and are not part of the v2 run.

## Reproduction order

On one GPU with at least 32 GB, Python 3.10, and vLLM 0.7.0, run
`scripts/ops/supervise.sh`. The host needs `gcc` and `python3.10-dev` for
vLLM's LoRA Triton kernels, plus `scikit-learn==1.6.1` for the frozen probes.
Place `HF_TOKEN` in the project-root `.env` for the private adapters, and
remove it afterwards.

Outputs are written to `outputs/phase7_initials_regen_v2/`,
`outputs/phase8_first_pass_v2/`, `outputs/phase8_three_arms_v2/`,
`outputs/phase8_probe_scores_v2/`, and `outputs/phase8_analysis_v2/`. The
completed run is mirrored under `outputs/phase8_remote_v100/`. The analyzer
refuses to run twice.
