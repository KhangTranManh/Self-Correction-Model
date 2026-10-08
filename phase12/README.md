# Phase 12 — both-orders judging and order-swapped DPO

**Status: closed on 2026-10-09. Both primary endpoints passed.** Read the
[final report](docs/FINAL_REPORT.md).

Phase 12 judged each pair of the model's own conflicting solutions in both
orders (A-B and B-A) and trained a DPO LoRA on order-swapped preference pairs
(1,822 pairs), then tested it on SVAMP (960 problems never used for training).

## Result in brief

- **P1 passed:** the DPO judge's consistent both-orders verdicts are right
  74.9% of the time (coverage 60%; untrained judge 68.7%).
- **P2 passed:** the DPO self-check (86.98%, ~2.6 calls) is non-inferior to
  compute-matched vote@3 (87.60%; difference −0.63, CI [−1.56, +0.31]).
- **Secondary passed:** single-order accuracy 61.9% (untrained 56.3%).
- Order bias is within target (position-2 rate 55%); invented answers 10.5%
  miss the < 5% target.
- Vote@5 remains the most accurate method (90.10%).
- Caveat: on SVAMP the untrained judge is already above chance, so part of
  the gain reflects the easier dataset; DPO's added improvement is
  significant (tie-break +3.95 points, CI [+0.78, +7.12]).

## Documents and code

- [Preregistration](docs/PREREGISTRATION.md) · [Run record](RUN_STATUS.md)
- `data/raw/SVAMP.json`; `data/sources_v1/` — holdout; `data/dpo_v1_report.json`;
  `data/execution_lock_v1.json` — lock
- `scripts/prepare_sources.py`, `scripts/build_dpo.py` — run locally
  (Python 3.10 + SymPy 1.14)
- `scripts/generate.py` — vLLM stages (both orders); `scripts/train_dpo.py`
- `scripts/analyze.py` — single opening; `scripts/ops/` — bootstrap,
  pipeline, retries; `scripts/sync_remote.py` — backup

The adapter is served as `phase12-dpo-judge` by `../serving/serve.sh`. The
holdout is opened: never train on it or tune against it.
