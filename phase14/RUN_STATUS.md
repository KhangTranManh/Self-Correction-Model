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

2026-10-12, local, still before any lock (aligning Phase 14 with the paper checks):

- Answer reading: added a third mode `rescued` (sensitivity check) to
  `scripts/answers.py`: the lenient reading, except that a non-numeric lenient
  value is replaced by the first number on the last "Đáp số"/"Answer" line
  (project comma rule kept: "1,5" → 1.5, "1,200" → 1200). Paper check T4 found
  29/561 Qwen2.5-7B-Instruct first attempts written as sentences ("Đáp số: Mỗi
  trẻ nhận được 9 trứng.") and scored wrong by the lenient reading (4/561 for
  Self_Correction). `rescued` changes correctness only; judged pairs and picks
  keep the lenient reading. `analyze.py` and `regression.py` now score
  lenient, strict and rescued. Lenient stays primary.
- Name mapping, to be used everywhere: Phase 14 `lenient` = the project parser
  = "strict" in the paper checks (T3–T5); Phase 14 `rescued` = "lenient" in the
  paper checks; Phase 14 `strict` (boxed or labelled only) has no paper-check
  counterpart.
- `config.INSTRUCT_REFERENCE` corrected: it held 0.655 / 0.658 (the 1536-token
  T3 view) under swapped mode names. Phase 14 judges a 768-token view, so the
  reference is now T1 (768 tokens): lenient 0.629, rescued 0.605.
- Added `data/gsm8k13_audit_excluded.json` (42 ids from the paper's A1 audit),
  so every GSM8K cell is also scored on the audited 519 problems.
- Scoring speed: `answers.same` now memoizes the phase9 comparison and is installed
  into `phase9.scripts.answers` and `phase13.scripts.constrained`, so `vote` and
  `pick` reuse it. Profiling one fake cell showed 60% of scoring time in SymPy
  `simplify`; with the memo the same cell scores in 60 s instead of 98 s with a
  byte-identical result (checked on the dry-run copy).
- Added `scripts/dry_run.py`: CPU dry run of the whole pipeline in a separate
  work folder (copies code, frozen data and the Phase 12/13 mirrors; fakes only
  the vLLM generation and the Transformers extraction with planted signals; runs
  every other script for real, in `run_remote.sh` order, including the regression
  check on the real Phase 13/12 outputs).
- Added `scripts/ops/bootstrap.sh` (host setup: environment, GPU and disk check,
  HF_TOKEN from `.env` without printing it, access check, pinned downloads of the
  four models, skipping Mistral's and Llama's duplicate consolidated weights).
  Local access check with the project token: all four models accessible,
  Llama-3.1-8B-Instruct license accepted.
- **Pilot lock frozen locally** (`data/execution_lock_pilot.json`, 42 files,
  lock sha256 74b050d0…). Upload bundle built and checked by extracting it into an
  empty folder: lock verified, all four datasets load (pilot 100 + 100, GSM8K 561,
  SVAMP 960), Phase 13 judgments readable. From here on, a change to a locked file
  is an amendment recorded here before the generation it affects.
- GPU host: not yet rented. Its type goes here before the smoke test; a GPU other
  than the V100 is a recorded deviation (FP16 kept for every model).
- Amendment before any generation (pilot lock re-frozen): the smoke test now also
  checks the Transformers hidden-state path on every model. `smoke_model.py`
  judges both orders on the primary (768-token) views and saves the toy attempts
  and judgments; new `smoke_hidden.py` re-reads them with the exact
  `extract_hidden.py` site functions and checks layer count and width against the
  model config, finiteness (FP16 overflow), tokenization prefixes, vLLM vs
  Transformers verdict log-probabilities (same argmax ≥ 90%), logit-lens last
  layer = final margin, timing per forward and peak memory. `run_remote.sh smoke`
  runs both per model; `verify_lock.py` covers the new script. The old pilot lock
  (74b050d0…) was deleted and re-frozen: 43 files, lock sha256 d62023f8….

2026-10-13, GPU host rented: **Tesla V100-SXM2-32GB** (driver 580.82.07), 94 GB RAM,
10 CPU cores, **100 GB disk**, Python 3.10.12. Same GPU type as the
preregistration (V100, FP16): no hardware deviation.

- Disk: four models (~61 GB) + environment (~10 GB) + hidden states (~25–30 GB)
  would fill the 100 GB disk. Bootstrap threshold lowered from 120 to 90 GB free.
- Amendment before any generation (decided by the user): the main run goes **one
  model at a time** with the new `scripts/ops/run_sequential.sh`: each model does
  all its generations (endpoint datasets, GSM8K, SVAMP, and its comparison sets:
  the bridge set, or for the original solver the report's regression sets), then
  all its hidden states, and its weights are deleted from the Hugging Face cache
  before the next model. Smoke and pilot keep all four models on disk (no hidden
  states yet). Same scripts, arguments and per-cell work as `run_remote.sh`; only
  the order changes. Consequence: comparison cells already exist when the
  endpoints are opened, so `analyze.py` scores them in the same single opening
  instead of a later `--add-comparison`; E1, E2 and the generality rule use only
  the uniform cells, as before. `run_sequential.sh` added to the lock list.
- Amendment before any generation (decided by the user): `meta-llama/Llama-3.1-8B-Instruct`
  could not be downloaded on the host (403: the access request is still awaiting
  Meta's review; the earlier local check used model metadata, which is public for
  gated repos, and wrongly reported access). `llama31_8b` now loads
  `NousResearch/Meta-Llama-3.1-8B-Instruct` @ d10aef7999a2b5ba950ab3974312feeedbfe0b77
  (ungated). Hub file hashes: its four weight shards have the same LFS sha256 as
  Meta's repo @ 0e9e39f2…, and config.json, generation_config.json, tokenizer.json
  and special_tokens_map.json are identical; only tokenizer_config.json (the chat
  template) differs and is NousResearch's copy. Bootstrap re-hashes the downloaded
  shards against Meta's hashes; the smoke report stores the rendered prompt so the
  template used is on record. Bootstrap's access check now downloads config.json
  instead of reading metadata. Ollama (GGUF, 4-bit by default) was considered and
  rejected: not the same weights or format as the other three FP16 models.
- Smoke 1 (first pass, 2026-10-13): Llama (NousResearch weights) passed the
  Transformers check (33 × 4096, finite, verdict argmax agreement 12/12, lens =
  final margin) but not the vLLM check: (a) decode equality 9/12 — the copy's
  tokenizer_config sets clean_up_tokenization_spaces=True, so tokenizer.decode
  removes spaces that vLLM keeps (12/12 with the flag off); (b) 5/12 toy answers
  scored right (7/12 actually right: two answers had no "Đáp số" line and the
  lenient fallback read the last number, e.g. "120 (cm2)" → 2). Its rendered
  prompt had no "Cutting Knowledge Date" system block, i.e. NousResearch's chat
  template is not Meta's current one.
- Amendment before any generation: (1) `llama31_8b` takes its tokenizer from
  `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic` @ 442e7f52…, whose
  tokenizer_config.json, tokenizer.json and special_tokens_map.json have the same
  git blob ids as Meta's @ 0e9e39f2 (weights stay NousResearch, sha256-identical
  to Meta's): together exactly Meta's model and chat template. `config.tokenizer_source`
  is used by generate, smoke and extract_hidden; bootstrap downloads the tokenizer
  files. (2) The 768-token primary view is decoded with
  clean_up_tokenization_spaces=False, matching vLLM's own detokenization for every
  model; the smoke decode check uses the same. Smoke is re-run for all four models
  on the final code.
- Smoke 1 also showed: Mistral decode equality 0/12 — vLLM's text starts with a
  space that SentencePiece decoding drops (checked on generated ids on the host;
  that is the only difference). Amendment: `generate.primary_view` builds the
  768-token view with clean_up_tokenization_spaces=False and restores that leading
  whitespace, so a truncated attempt is written exactly as a 768-cap vLLM run;
  smoke now checks `decode_matches` and that the stored view equals the 768-cap
  run's text. Transformers vs vLLM verdict log-probabilities: same argmax on
  100% of toy judgments for all four models; mean absolute difference 0.01 (Qwen
  models), 0.58 (Llama), 0.86 (Mistral) — the teacher-forced pass re-tokenizes the
  generated text; E2 uses vLLM's own values.
- Smoke 2 (Meta tokenizer and chat template for Llama; the rendered prompt now has
  Meta's "Cutting Knowledge Date / Today Date: 26 Jul 2024" system block): Llama
  vLLM check passed (prefix 12/12, decode 12/12, primary view = 768-cap text 12/12,
  9/12 toy answers right). Its Transformers check failed only on verdict agreement
  8/9: the one disagreement is a near-tie (Transformers A−B +0.39, vLLM −0.86);
  the other eight agree with margins 1.0–8.0 nats. Amendment: the agreement check
  now requires the same choice on every judgment with a vLLM margin ≥ 1 nat (at
  least 3 such judgments), with the overall rate still reported. Smoke re-run.
- **Smoke 3 (final code, all four models; `outputs/phase14_remote/smoke/`):** every
  technical check passes for every model — 768-token prefix 12/12, decode 12/12,
  stored primary view = 768-cap run text 12/12; Transformers sites with the right
  shapes (Qwen 29 × 3584; Llama, Mistral 33 × 4096), all finite, tokenization
  prefixes 100%, logit-lens last layer = final margin, verdict choice identical on
  every clear (≥ 1 nat) judgment. Toy accuracy: Self_Correction 12/12, Qwen-Instruct
  10/12, Llama 9/12, **Mistral 5/12**.
- Decision by the user: **Mistral is kept.** Its 5/12 is below the smoke's ≥ 8/12
  criterion, which exists to catch FP16 garbage; Mistral's failures are reasoning
  errors in coherent text (2/9 for 1/6, 46 for 120, 24 for 18, one refusal, one
  English final line), and all its technical checks pass. Not replaced by
  phi35_mini. Its lower solve accuracy is a limitation to report.
- **Pilot complete** (2026-10-12 19:50–21:26 UTC, all four models, no errors; outputs
  mirrored to `outputs/phase14_remote/`). Main size written by `pilot_report.py --write`:
  **ASDiv 822, MATH 1–3 785** (min one-right pairs per problem 0.42 and 0.44 over
  the two new models).
- **Amendment after the pilot, before any main generation (decided by the user):**
  the GPU host has ~7.5 h left of its 10 h, and the full main (four models × four
  datasets + hidden states) was measured at ~18 h. The main run is cut to the
  endpoint cells: Llama-3.1-8B and Mistral-7B on ASDiv and MATH 1–3 at the full
  pilot-fixed N, hidden-state sites answer, judge and direct (the verdict site was
  only an agreement check, done in smoke). `config.RUN_MODELS` = the two new
  models (`PILOT_MODELS` keeps the four for the completed pilot); the anchors are
  not regenerated in main. `analyze.py` requires only the endpoint cells; comparison
  cells (GSM8K for the new models, `run_sequential.sh extra`) are optional and, if
  run after the opening, are added with `--add-comparison` without touching E1/E2.
  Not run: SVAMP for the new models, the anchors' fresh cells, the bridge set, and
  the base_c regression re-run (the base-judgment reproduction of the report was
  verified on CPU in the dry run). E1, E2, thresholds and the generality rule are
  unchanged.
- Dry run of the cut main (`dry_run.py --order sequential --endpoints-only`, CPU, fake
  generations): pilot report with the four pilot models, opening on the four endpoint
  cells only, probe analysis and report tables all complete; planted skill, bias
  direction and layers recovered. (dry_run.py now fakes the pilot for PILOT_MODELS.)
- Main started 2026-10-12 21:45 UTC (main lock 5b782972…). Measured on Llama: ASDiv
  samples 37 min, base judge 11 min, base_c ~15 min — slower than the pilot-based
  estimate, putting the end of the planned order near the GPU cutoff. Amendment
  (order only, same scripts and arguments): when Llama's block finishes, a watcher
  stops `run_sequential.sh` and Mistral continues with the new `scripts/ops/run_tail.sh`:
  all of Mistral's generations first; then the protected opening (`analyze.py`, CPU)
  runs in the background while Mistral's hidden states are extracted; an early probe
  analysis (Llama's complete hidden states) runs on CPU at the start. Generation is
  resumable, so a stage interrupted by the switch continues from its audit log.
- Llama timings on MATH: samples 57 min, base judge 30 min, base_c 37 min (2,042 judged
  tasks), hidden answer 6 min, judge 19 min. Projected end of Mistral's generations
  ~04:30 UTC with the host stopping ~04:50. Amendment (scheduling only): Mistral's
  generations run through `scripts/ops/gen_batch.py`, which calls generate.py's main with
  vLLM chunks of 256 requests instead of 64 (`phase8 batched.BATCH_SIZE`, used by both
  models so far). Per-request seeds (attempts) and greedy decoding (judges) are
  unchanged; small FP16 differences from batch composition are possible (cf. paper check
  D0). The chunk size used is recorded in each Mistral stage's settings
  (`batch_size: 256`). Llama's cells used 64.
- Mistral with 256-request chunks: ASDiv samples 17 min (Llama 37), all generations done
  03:44 UTC. **Protected opening on the host at 2026-10-13 03:55 UTC** (main lock
  5b782972…), mirrored to `outputs/phase14_remote/outputs/phase14_v1/main/analysis/`:
  - **E1 (skill index, base judge, lenient): 4/4 cells pass → general.** Llama ASDiv
    +0.321 [0.230, 0.413], Mistral ASDiv +0.335 [0.269, 0.400], Llama MATH +0.362
    [0.291, 0.428], Mistral MATH +0.344 [0.261, 0.431].
  - **E2 (log-prob order-averaging gain, base_c): 1/4 cells pass → conditional.** Llama
    ASDiv +0.6 pts [−3.7, +4.8], Mistral ASDiv +0.7 [−2.3, +3.6], Llama MATH +5.0
    [+0.9, +9.3] (pass), Mistral MATH +3.3 [−0.7, +7.3]. Bias-removal part −0.8 to
    +1.4 pts although the log-odds bias toward A is large (1.35–2.40); log-prob
    coverage 50–87% of pairs.
  - Preregistered prediction (bias-removal gain grows with |bias|): not supported
    (Spearman ρ = −0.20, four cells, descriptive).
- Probe analysis and report tables started on the host CPU right after the opening,
  in parallel with Mistral's hidden-state extraction (GPU time).
- Report tables (endpoint cells) written 04:30 UTC: `outputs/phase14_remote/logs/phase14_report_tables_endpoints.md`.
- Probe analysis (exploratory) finished 04:37 UTC on the host CPU, started at 03:55, so it
  covers the hidden states complete at that time: Llama ASDiv + MATH (all sites), Mistral
  ASDiv answer site (+ forced verdict). `probe_report.json` mirrored. Llama, cross-dataset:
  pair probe 66–67% (AUC 0.72), surface 62–64% (AUC 0.64–0.69), likelihood 51–63%;
  pointwise probe per-solution AUC 0.80–0.81; pair-probe order-flip rate 39–40%; forced
  verdict order-averaging gain −0.4 (ASDiv) and +4.8 pts (MATH). Mistral: forced-verdict
  gain +2.3 pts on ASDiv (log-odds bias 3.4); first-answer probe ASDiv → MATH AUC 0.78
  (log-prob baseline balanced accuracy 0.59).
- Not done before the GPU limit: Mistral MATH judge/direct hidden states and its probes;
  the `extra` GSM8K cells; SVAMP; anchors' fresh cells; bridge set; base_c regression re-run.
  Hidden-state arrays stay on the host only (several GB each; not mirrored).
- The host outlived the 10 h estimate. Mistral's remaining hidden states finished at 04:53
  UTC (all sites, both datasets); the final probe analysis (both models, all sites) and
  report tables completed at 05:46 UTC (`PHASE14_MAIN_COMPLETE`), mirrored to
  `outputs/phase14_remote/` (`probe_report.json`, `report_tables.md`; hidden-state
  summaries only — arrays stay on the host).
- Mistral probes (exploratory): pair probe 69–72% (AUC 0.76–0.82) but the surface
  baseline is higher, 73–81% (AUC 0.80–0.88); pointwise probe per-solution AUC 0.77–0.81;
  likelihood 63–67%; order-flip rate 25–40%; forced-verdict order-averaging gain +2.3
  (ASDiv) and +4.2 pts (MATH) with log-odds bias 3.4 toward A. For Mistral the hidden
  state does not beat surface cues (its wrong answers are often visibly long, rambling or
  in English), unlike Llama (probe AUC 0.72 vs surface 0.64–0.69).
- Phase 14 main closed: endpoints opened once (E1 general 4/4, E2 conditional 1/4).
  Not run: GSM8K/SVAMP cells, anchors' fresh cells, bridge set, base_c regression re-run.

2026-10-13 ~06:00 UTC, **completion run (amendment after the protected opening, decided
by the user; same host still running):** fill the whole grid {Self_Correction,
Qwen2.5-7B-Instruct, Llama-3.1-8B, Mistral-7B} × {GSM8K 561, SVAMP 960, ASDiv 822, MATH 1–3
785} with the identical protocol and every measure, plus the bridge set (Llama, Mistral,
Qwen re-judging the Phase 13 pairs) and the base_c regression re-run on the report's
outputs. One model at a time (`scripts/ops/run_complete.sh`; weights fetched by
`ops/fetch_model.py`, deleted after use); 256-request chunks (`ops/gen_batch.py`);
hidden-state sites answer, judge, direct. The new cells are scored with
`analyze.py --add-comparison`, now extended to every not-yet-scored cell; cells already in
the report (the four endpoint cells) are never recomputed, and E1, E2 and the generality
verdicts stay as opened at 03:55 UTC. The anchors' ASDiv/MATH cells are descriptive (the
endpoints are the two new models only). `config.RUN_MODELS` = all four models again. New
lock stage `completion` (main-lock files + the completion scripts).

2026-10-13, **locked claim and interpretation rules for the descriptive cells** (written
before any completion cell is scored; the last mirror is from 06:22 UTC, when only Llama's
GSM8K samples and base judgment existed and none had been read):

The claim is locked to what was opened at 03:55 UTC plus the Phase 9–13 deep dive:
(L1) single-order judging looks like chance while the skill index is clearly positive —
E1 general (4/4 endpoint cells); (L2) log-prob order averaging adds little, as in Ren et
al. — E2 conditional (1/4); the bias-removal part is ≈ 0 even under a large bias; the
preregistered dose-response prediction is not supported; (L3) the hidden state carries a
right/wrong signal, beyond surface cues for Self_Correction and Llama but not for Mistral.
Descriptive cells (anchors on all four datasets, new models on GSM8K/SVAMP, bridge,
regression) may only **support** or **limit** these statements. They never change the E1/E2
verdicts, are never added to the endpoint counts, and do not create new headline claims.
All cells are reported; none is dropped.

| Observation in a descriptive cell | How it is written |
|---|---|
| Skill index CI above 0 | "consistent with E1" (counted separately from the 4/4) |
| Skill index > 0, CI includes 0 | listed as "not distinguishable from 0 in this cell"; limitation, claim unchanged |
| Skill index CI below 0 | named as a counter-example in the abstract and the limitations; L1 stays "general on the two new models, ASDiv/MATH" and is written as not universal |
| Log-prob gain ≥ 3 pts with CI above 0 | "heterogeneous across models/datasets"; E2 stays conditional |
| Bias-removal part ≥ 3 pts with CI above 0 | reported as an exception to "bias removal ≈ 0"; dose-response re-described over all cells (Spearman, descriptive), endpoint verdict unchanged |
| Self_Correction fresh GSM8K/SVAMP within the CI of the Phase 13/12 numbers | "reproduced with new samples" |
| outside that CI | "run-to-run variation"; sections 4.1–4.5 keep the Phase 13/12 numbers |
| base_c re-run on the report's pairs within ±3 pts of Phase 13 | pipeline check passed; otherwise a limitation (no regeneration) |
| Bias direction per cell | reported as observed; no mechanism claimed |
| Probe vs surface | "beyond surface" only for models whose pair-probe AUC exceeds the surface AUC in both transfer directions |
| Bridge set (other models judging Self_Correction's Phase 13 pairs) | judge-strength comparison, descriptive; no "self vs other" claim beyond T1–T5 |


2026-10-13 ~15:00 local (08:00 UTC), **token-cap sensitivity queued (exploratory, after the opening):**
`scripts/ops/run_long_judge.sh` waits for the completion run, re-fetches Llama and re-judges the
endpoint pairs (ASDiv, MATH 1–3) with `base` and `base_c` at a 1536-token judge cap
(`scripts/ops/judge_long.py`: same tasks, prompts, seeds and greedy decoding as generate.py's judge
stage; only max_tokens and the stage name `judge_<judge>_1536` differ; tasks over the 4096-token
context are skipped and listed). No locked file changes; E1/E2 are not recomputed.
Reading rule, written before the run: for each cell, compare the three-part Shapley split
(`paper/scripts/phase14_decomposition.py`) at 768 and 1536. If Llama's `base` single-order gap
shrinks by >= 5 points with the CI of the difference above 0, the report states that a sizeable
part of Llama's loss at 768 is a token-cap artefact and gives both numbers; otherwise it states
that raising the cap does not recover it (consistent with most capped judgments being loops,
`paper/results/phase14_truncation.md`). The locked claim (L1–L3) does not change either way.

2026-10-14 04:36 local: **token-cap sensitivity done** (Llama, ASDiv + MATH, `base` and `base_c` re-judged at
1536 tokens; 0 tasks skipped for context). Three-part Shapley split (`paper/results/phase14_decomposition.md`):
the single-order gap does not shrink — `base` ASDiv 14.3 → 14.5, MATH 19.4 → 19.6; `base_c` ASDiv 5.0 → 5.1,
MATH 13.6 → 13.6 points; single-order accuracy 57.3 → 56.8% and 53.0 → 52.9% (`base`), 47.9 → 48.2% (`base_c`
MATH). Raising the cap halves the share of cut judgments (`base` MATH 14.3% → 7.1%; `base_c` MATH 19.7% →
6.2%), but the extra tokens end in a finished output that still names neither solution (no-valid-choice share
`base` MATH 12.4% → 19.8%, `base_c` MATH 1.4% → 14.7%). By the reading rule written before the run (gap shrink
< 5 points): **raising the cap does not recover Llama's loss** — consistent with most capped judgments being
loops (`paper/results/phase14_truncation.md`). The locked claim is unchanged.

2026-10-15, **amendment before the paper analyses (all exploratory, post-hoc; written before running them;
`paper/PAPER_TODO.md` holds the full definitions).** None of these changes E1/E2 or the locked claim L1–L3.
(i) *Seven-way classification of every one-right single-order judgment* (16 grid cells + bridge, `base` and
`base_c`), first match wins: 1 valid choice (project `pick()` gives A/B); 2 format error (rescued reading matches
exactly one shown solution); 3 cut while looping (`hit_token_cap`, truncation label b); 4 cut otherwise (capped,
label a/c/d); 5 re-solved (not capped, truncation label d); 6 other third answer (not capped, a number that
matches neither); 7 no answer (not capped, no number by either reading). The two report-regression cells have no
cap flag and get only classes 1, 2, 5–7.
(ii) *Rescued audit:* 50 class-2 judgments per model (seed 20261015, spread over datasets) are read and marked
"the rescued number is the solution the judge meant" or not. Thresholds: misreading < 5% → rescued is the main
sensitivity reading; 5–15% → reported with a warning; > 15% → appendix only. The audit is done by the analysis
assistant (Claude), not by an independent human rater; this is stated in the paper.
(iii) *Decomposition sensitivity:* lower bound of the loss due to invalid choices = π·0.5, upper bound = π·acc_AB;
the Shapley value must lie between them. Probit link instead of logit; criterion: the bias share stays ≤ 3 points
in every cell. Model-free identity single-order = (1 − π)·acc_AB, and π_A − π_B with a bootstrap CI.
(iv) *Out-of-sample F2:* answer-site pointwise probe trained on X, layer by 5-fold grouped CV on X, tested on Y;
probability-weighted vote over s1–s3 vs vote@3; "confirmed out of sample for a model" if the paired CI is above 0
in ≥ 2 of 3 train/test pairs (ASDiv→MATH, MATH→ASDiv, SVAMP→GSM8K).
(v) Practical table: calls = 2 + 2·P(s1 ≠ s2) for judge-or-random, plus P(disagree and inconsistent) for
self-check; paired bootstrap against vote@3 and vote@5.
