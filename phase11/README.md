# Phase 11 — preference training (DPO) of the judgment step

**Status: closed on 2026-10-02. Negative result; no adapter promoted.** Read
the [final report](docs/FINAL_REPORT.md).

Phase 11 trained a DPO LoRA on the original solver to prefer a correct
judgment over an incorrect judgment of the *same* two conflicting solutions
(542 balanced pairs built from Phase 10 outputs), then tested it on the last
234 never-used GSM8K problems.

## Result in brief

- DPO learned the preference in likelihood: 67% of unseen validation pairs.
- On fresh problems it barely changed the choice: picks the right solution
  46.6% → 49.4% (+2.8 points, not significant; P1 failed).
- DPO self-check 72.2% vs agreement-gated voting 77.4% (P2 failed).
- The judge sides with the second solution shown about twice as often as the
  first (order bias), before and after DPO.
- Voting over five attempts: 71.4% → 84.2% (+12.8 points).
- Post-hoc (exploratory): beneath the order bias the judge has a weak real
  skill (about 59% correct when it picks one of the two solutions) and invents
  a third answer in 15–19% of cases; see section 5b of the final report.

## Documents and code

- [Preregistration](docs/PREREGISTRATION.md) · [Run record](RUN_STATUS.md)
- `data/sources_v1/` — holdout; `data/dpo_v1_report.json` — DPO data report;
  `data/execution_lock_v1.json` — lock
- `scripts/build_dpo.py` — preference pairs (run locally with Python 3.10 + SymPy 1.14)
- `scripts/generate.py` — vLLM stages; `scripts/train_dpo.py` — DPO LoRA
- `scripts/analyze.py` — single opening; `scripts/ops/` — bootstrap, pipeline, retries
- `scripts/sync_remote.py` — backup

The adapter is served as `phase11-dpo-judge` by `../serving/serve.sh`. The
holdout is opened: never train on it or tune against it.
