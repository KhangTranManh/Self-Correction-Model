# Phase 13 — confirm, constrain, and combine (directions A, B, C)

**Status: closed on 2026-10-09.** Read the [final report](docs/FINAL_REPORT.md).

Phase 13 tested three follow-ups to Phase 12 on 561 GSM8K test problems
(indices 750–1318, never used for training), with one preregistration, one
lock, and one protected opening.

## Result in brief

| Direction | Question | Result |
|---|---|---|
| **A** | Does the Phase 12 judge hold up on harder (GSM8K) problems? | ✅ judgment: 79.3% consistent accuracy (untrained 71.0%), coverage 52% · ❌ practical: self-check trails compute-matched vote@5 by 5.3 points |
| **B** | Does a forced "Verdict: Solution A/B" line remove invented answers? | ✅ invented 9.9% → 2.1%, coverage 63%, consistent accuracy 74.0% |
| **C** | Does calling the judge only on split votes beat voting? | ❌ no gain over vote@3; 6.6 points below vote@5 |

Vote@5 remains the most accurate method (77.5%; first answer alone 66.1%).

**Exploratory layer study:** a probe on the judge prompt's hidden states
(trained on SVAMP, tested on GSM8K) finds which solution is right about 70% of
the time, strongest at layers 16–23, and already in the untrained model. DPO
changes how that knowledge is expressed, not whether the model has it.

## Documents and code

- [Preregistration](docs/PREREGISTRATION.md) · [Run record](RUN_STATUS.md)
- `data/raw/gsm8k_test.jsonl`; `data/sources_v1/` — holdout;
  `data/dpo_b_v1_report.json`; `data/execution_lock_v1.json` — lock
- `scripts/prepare_sources.py`, `scripts/build_dpo_b.py` — run locally
  (Python 3.10 + SymPy 1.14)
- `scripts/constrained.py` — constrained prompt and verdict parsing
- `scripts/generate.py` — vLLM stages for the four judges;
  `scripts/train_dpo_b.py` — direction-B DPO
- `scripts/analyze.py` — single opening for A, B, C;
  `scripts/ops/` — pipeline and retries; `scripts/sync_remote.py` — backup
- `layer/extract_hidden.py`, `layer/probe_analysis.py`, `layer/run_layer.sh` —
  exploratory layer study (Transformers)

The direction-B adapter is served as `phase13-verdict-judge` by
`../serving/serve.sh`. The holdout is opened: never train on it or tune
against it.
