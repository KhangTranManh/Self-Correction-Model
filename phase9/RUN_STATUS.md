# Phase 9 run record

2026-09-30, local CPU:

- Source pool frozen at `data/source_pool_v1/` (40 development, 400
  protected). The builder reproduced the Phase 8 selection exactly (1374
  eligible, identical 400 IDs) from the 252 of 286 Phase 5-listed inventory
  files still present, then over-excluded all Phase 1-7 JSON/JSONL plus the
  Phase 8 pool (974 eligible). The raw GSM8K file matches its recorded
  SHA-256 after CRLF -> LF normalization.
- Preregistration written and execution lock frozen
  (`data/execution_lock_v1.json`, 23 files, LF-normalized hashing, SHA-256
  `ead98d7c7dd4b526e8a9440b2fdca0766fa0147109069e038fb7b604d9a2e482`)
  before any Phase 9 generation.

2026-09-30, remote GPU (the Phase 8 Tesla V100-SXM2-32GB host, reused):

- Code uploaded to `/root/AGI_phase8/` alongside the existing Phase 8
  environment, base model cache, merged V2, and verified adapters. The lock
  verified on the host.
- Pipeline attempt 1 started 15:39 UTC under `scripts/ops/supervise.sh`;
  outputs are mirrored locally to `outputs/phase9_remote_v100/` by
  `scripts/sync_remote.py`.
- Stage times: first answers 15:39–15:41; original solver 15:42–15:58;
  V2 15:58–16:15; V3 16:15–16:53 (LoRA and long self-check prompts).
  `PHASE9_PIPELINE_COMPLETE` at 16:54 UTC in a single attempt.
- The Claude Code session restarted during V3. The GPU job was unaffected
  (`setsid nohup`); the local backup process had been tied to the session and
  stopped, so it was restarted through Windows WMI outside the session.
- A final one-shot sync copied the analysis. All 29 remote output files
  match the local mirror by SHA-256.
- `scripts/build_case_report.py` ran on the GPU host (Python 3.10, SymPy
  1.14.0) to produce `docs/CASE_REPORT.md`.
- Closed on 2026-10-01; see `docs/FINAL_REPORT.md`.
