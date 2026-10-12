# Phase 14 run record

2026-10-12, local, setup only (no GPU, no model output):

- Raw data downloaded and hashed (`data/raw/manifest.json`): ASDiv XML
  (2,305 problems) and the seven MATH test parquet files (5,000 problems),
  converted once to `math_test.jsonl`.
- Model revisions pinned from the Hugging Face API (`scripts/config.py`).
  Llama-3.1-8B-Instruct and gemma-2-9b-it are gated; Mistral-7B-Instruct-v0.3
  and Phi-3.5-mini-instruct are not.
- Sources frozen with Python 3.10.21 + SymPy 1.14.0 (`data/sources_v1/`):
  ASDiv 2,019 eligible (132 near-SVAMP, 146 non-numeric, 7 formula mismatches
  excluded), MATH levels 1–3 2,057 eligible (405 non-numeric or unsafe
  references excluded). 353 earlier-phase files scanned; no question overlap.
- While building the MATH pool, four silent misreadings by the Phase 1 parser
  were found (leading-number fallback, mixed numbers read as products,
  `15 \pi` read as 15, units with exponents). The pool was rebuilt three times
  before any use to exclude or normalize these references; the final pool was
  checked by an independent full parse (0 of 4,076 differ). The shared
  verifier is unchanged, because earlier phases' locks depend on it.
- Pipeline dry-run on fake generations (pilot → main size → main analysis)
  completed; the metric identity check (tie-break minus single-order equals
  the invented-answer term) passed.
- Added for comparability with earlier results: the original solver as an
  anchor model, and a bridge set that re-judges the Phase 13 pairs.
- Added hidden-state extraction with Transformers after vLLM generation
  (answer, judge, and verdict sites; all layers) and an exploratory probe
  analysis with likelihood and surface baselines. Dry-run with fake outputs
  and fake hidden states recovered the planted layers cross-dataset.
- Aligned with the report draft (2026-10-12): Qwen2.5-7B-Instruct added as a
  second anchor; GSM8K (Phase 13 holdout) and SVAMP (Phase 12 holdout) added
  as comparison datasets solved by each model; tie-break CI, all-pairs
  accuracy, the section 5 baselines, the forced-verdict logit lens, pointwise
  and probe + surface selectors, and `report_tables.py` added. Noted that the
  report's Table 3 gain matches the invented-answer term (see the
  preregistration).
- Models decided by the user: Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3
  (Gemma-2 dropped: BF16-trained, FP16 risk on a V100).
- Order decided by the user: run the two new models first (main), the
  regression test and the cells that need Phase 12/13 outputs later
  (comparison). Added `regression.py` and `analyze.py --add-comparison`; the
  two-step analysis was dry-run on fake data (endpoints unchanged by the
  addition).
- Decided by the user: one identical protocol for all four models; Self_Correction
  is re-run fresh like the new models. The report's own outputs are used only
  for the regression check (`gsm8k13_report`, `svamp12_report`), never in the
  main tables. A unified method table was written into the preregistration.
- E2 decided by the user: log-prob order averaging (the tie-break minus
  single-order difference is reported, not an endpoint). Preregistration
  settled.
- Next: rent the GPU; freeze the pilot lock locally, upload, run the smoke test.
