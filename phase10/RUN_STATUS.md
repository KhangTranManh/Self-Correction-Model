# Phase 10 run record

2026-10-01 (UTC times; the V100 host clock read 2026-09-30 for the first part
of the run):

- Sources frozen at `data/sources_v1/`: 1,900 training sources (Phase 4
  expansion minus 100 Phase 4 development sources) and a 300-source holdout
  from 534 never-used eligible sources.
- Preregistration written before any generation. Before launch, a V100
  training smoke test on dummy text found two problems: the smoke script
  itself lacked `model.train()` (so gradient checkpointing was off and memory
  ran out), and the default FP16 gradient-scaler start of 65536 overflowed and
  skipped every update. `train_judge.py` was fixed to start the scaler at 1024
  (loss then fell on each step, 2.2 s per 2.5k-token example, 20.9 GB), and
  the training step runs with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  vLLM loaded a PEFT-saved LoRA correctly. The lock was frozen after these
  fixes and before any Phase 10 generation.
- 17:35 UTC: pipeline attempt 1 started. The holdout stages completed
  (1,500 attempts; 300 untrained judgments).
- 17:49 UTC: the training-sample stage failed while reading the training
  pool, because `str.splitlines()` in the shared Phase 8 JSONL reader also
  splits on U+2028, which occurs inside some GSM8K questions. No output was
  produced by that stage. The supervisor was stopped, and Phase 10 now uses
  a newline-only reader patched into the shared helper at import time (the
  locked Phase 8/9 file is unchanged). The execution lock was refrozen
  (`77a97773dc851cb16d8530587004fe3e2fc00480eaf234f7ffacd483e91a5780`).
  This came after the holdout generation but changes only file reading, not
  prompts, decoding, training, or analysis, and no holdout output or label
  was inspected.
- 17:51 UTC: relaunched; the completed holdout stages resumed as complete and
  training-sample generation began.
- The continuous local backup kept terminating with a console Ctrl+C status
  (0xC000013A) when launched from the session, through WMI, and through Task
  Scheduler. Results are instead copied with one-shot syncs
  (`scripts/sync_remote.py --once`) at each stage boundary; the GPU disk
  keeps every append-only audit in the meantime.
- Stage times (UTC): training samples 17:51–18:37 (7,600); pairs 18:37
  (1,621); judge candidates 18:38–19:39 (6,484); SFT data 19:39 (808
  examples); LoRA training 19:39–19:53 (46 steps); trained judge 19:53–19:58;
  protected analysis 19:58. `PHASE10_PIPELINE_COMPLETE` at 19:58:20.
- All 22 result files match the GPU copies by SHA-256; the local adapter
  matches its recorded hash `14a2933f…`.
- 20:05 UTC: `serving/serve.sh original` started on the GPU host, serving
  `original-solver`, `warmstart-v2`, and `phase10-judge`; a demo question ran
  through the `single`, `vote`, and `self_check` client methods.
- Closed on 2026-10-01; see `docs/FINAL_REPORT.md`.
