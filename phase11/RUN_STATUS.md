# Phase 11 run record

2026-10-02, local CPU (no GPU):

- A local Python 3.10.11 + SymPy 1.14.0 environment (`.venv-verify310`,
  gitignored) was created to match the GPU verifier. It reproduced Phase 10's
  rejection-sampling acceptance counts exactly (471 / 360 / 88) from the
  hash-verified Phase 10 mirror.
- Holdout frozen at `data/sources_v1/` (234 rows — every remaining
  never-used eligible GSM8K train source; 534 − 300 = 234).
- DPO data built from Phase 10 training-pool judgments: 542 balanced
  preference pairs, 472 training and 70 validation
  (`data/dpo_v1_report.json`; files under `outputs/phase11_v1/dpo/`).
- Preregistration written and execution lock frozen
  (`data/execution_lock_v1.json`) before any Phase 11 generation.
2026-10-02, remote GPU (new host: Tesla V100-SXM2-32GB, 94 GB RAM, Ubuntu
22.04, Python 3.10.12; project root `/root/AGI_phase11/`):

- `scripts/ops/bootstrap.sh` installed gcc, Python 3.10 headers, vLLM 0.7.0,
  SymPy 1.14.0, and the pinned base model; the lock verified on the host.
- DPO smoke test (temporary script and output path; the four longest
  training pairs, 3,668–4,697 tokens): peak 18.4 GB, 3.3 s per pair; the
  preference margin started at 0 (policy equals reference) and rose on every
  step with no skipped updates.
- 11:21 UTC: pipeline attempt 1 started under `scripts/ops/supervise.sh`.
- Stage times (UTC): holdout samples 11:21–11:29 (1,170); base judge
  11:29–11:33 (362 pairs); DPO training 11:33–12:10 (59 steps; final
  validation preference accuracy 67.1%); DPO judge 12:10–12:16; protected
  analysis 12:16. `PHASE11_PIPELINE_COMPLETE` at 12:16:50 in one attempt.
- The one-shot local sync first copied nothing: `scripts/sync_remote.py`
  still pointed at the Phase 8–10 host root (`/root/AGI_phase8`). It was
  corrected to `/root/AGI_phase11` (the script is not in the lock); all 16
  result files then matched the GPU copies by SHA-256.
- 12:18 UTC: `serving/serve.sh original` started on the host (now loading only
  adapters present on the host), serving `original-solver` and
  `phase11-dpo-judge`; a demo ran the `single`, `vote`, and `self_check`
  client methods through the server.
- Closed on 2026-10-02; see `docs/FINAL_REPORT.md`.
