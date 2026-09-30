# Phase 9 — independent attempts, voting, and self-check

**Status: closed on 2026-10-01. No checkpoint is promoted.** Read the
[final report](docs/FINAL_REPORT.md) and the
[case report](docs/CASE_REPORT.md).

On 400 fresh protected GSM8K sources, Phase 9 tested two ways for the model
to check itself using only its own independent attempts.

## Result in brief

- **Option A — vote over five attempts:** 75.0% → 83.25–85.5% for all three
  checkpoints (Holm p < 0.001). Passed.
- **Option B — self-check of two conflicting solutions:** 72.25–74.0%, no
  better than the first answer and significantly worse than voting. Failed.
- Disagreement between two independent attempts catches 82–85% of wrong
  first answers, better than the Phase 5 probe (68–70%).
- When one of two solutions is right, the self-check picks it only 44–52% of
  the time: the model knows *that* it may be wrong, not *which* answer is
  right.

## Documents and code

- [Preregistration](docs/PREREGISTRATION.md) · [Run record](RUN_STATUS.md)
- `data/source_pool_v1/` — frozen sources; `data/execution_lock_v1.json` — lock
- `scripts/prepare_sources.py` — source builder with Phase 8 reproduction check
- `scripts/collect_first.py`, `scripts/collect_checkpoint.py`,
  `scripts/score_probe.py` — GPU generation and probe scoring
- `scripts/analyze.py` — single protected opening
- `scripts/build_case_report.py` — wrong → correction → proof report
- `scripts/ops/` — pipeline and retry wrapper; `scripts/sync_remote.py` — backup

`scripts/verify_lock.py` hashes LF-normalized text, so it verifies on both
Windows and Linux checkouts. The protected pool is opened: never train on it
or tune against it.
