# Phase 13 run record

2026-10-09, local CPU (no GPU):

- Phase 12 committed (`bb29ca3`) before Phase 13 work began.
- Holdout: GSM8K test problems 750–1318 downloaded from
  `openai/grade-school-math` (SHA-256 `3730d312…`); indices 0–749 excluded
  outright (Phase 1 evaluation range), 8 overlaps with earlier files excluded;
  561 problems frozen (`data/sources_v1/`).
- Direction-B DPO data built locally (Python 3.10.11 + SymPy 1.14.0): 1,648
  constrained-verdict, order-swapped pairs (1,470 training, 178 validation).
  A swapped B-right pair was spot-checked: chosen ends with
  "Verdict: Solution B", rejected with "Verdict: Solution A".
- While writing `generate.py`, two defects were caught before freezing: the
  Phase 12 adapter check first used the LF-normalized text hash (wrong for a
  binary file; changed to raw bytes, which match `628982a0…`), and an editing
  slip left a duplicate `raise` that would have always failed the p12 judge
  (removed).
- The analysis (A, B, C) was tested on synthetic data with hand-checked
  values. Preregistration written and lock frozen (`data/execution_lock_v1.json`,
  26 files, `15f33493…`).

2026-10-09, remote GPU (the Phase 12 host, Tesla V100-SXM2-32GB; project root
`/root/AGI_phase12/`, reusing its environment, base-model cache, and the Phase
12 adapter):

- The Phase 12 vLLM server was stopped to free the GPU (the first stop command
  matched its own SSH session; a self-excluding pattern was used).
- The lock verified on the host; the pipeline started under
  `scripts/ops/supervise.sh` (order: samples → A judges → B baseline → B
  training → B judge → single analysis).
- Stage times (UTC, 2026-10-08 host clock): samples 18:45–19:03 (2,805);
  A untrained judge 19:03–19:13 and Phase 12 judge 19:13–19:30 (486 pairs ×
  2 orders each); B untrained constrained judge 19:30–19:41; B DPO training
  19:41–20:45 (reference pass ~15 min, 92 steps; validation preference
  accuracy 82.6% at step 50, 84.3% at step 92); B judge 20:45–21:01; single
  protected analysis 21:01. `PHASE13_PIPELINE_COMPLETE` at 21:02:24 in one
  attempt.
- All 21 result files match the GPU copies by SHA-256.
- 21:03 UTC: `serving/serve.sh original` started (`original-solver`,
  `phase12-dpo-judge`, `phase13-verdict-judge`).
- Exploratory layer study (after the opening): the vLLM server was stopped;
  `layer/run_layer.sh` extracted last-prompt-token hidden states at all 29
  layers for SVAMP (`base`, `p12`) and GSM8K (`base`, `p12`, `base_c`,
  `p13b`) with Transformers (21:11–21:32, about 3 minutes per set), then fit
  the probes (21:32–21:36). `probe_report.json` was copied locally; the
  hidden-state arrays (about 1.2 GB) were left on the GPU host.
- Closed on 2026-10-09; see `docs/FINAL_REPORT.md`.
