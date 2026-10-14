# Phase 14 — does the judgment generalize to other model families?

**Status: main run closed (2026-10-13).** Run on a V100 32 GB in one 10 h rental, cut by
recorded amendments to the endpoint cells: Llama-3.1-8B-Instruct (Meta weights via the
NousResearch copy, Meta tokenizer) and Mistral-7B-Instruct-v0.3 on ASDiv (822) and MATH
1–3 (785). **E1 (skill index) 4/4 cells → general; E2 (log-prob debiasing gain) 1/4 →
conditional.** Exploratory probes for both models. Not run: GSM8K/SVAMP cells, the anchors'
fresh cells, the bridge set. Full record and every amendment: [RUN_STATUS.md](RUN_STATUS.md);
mirrored outputs: `outputs/phase14_remote/`.

## Question

Phases 11–13 showed, on one Qwen-based checkpoint, that a judge's choice
between two of its own conflicting solutions mixes position bias, a weak real
skill, and invented answers, and that judging both orders exposes the skill.
Phase 14 repeats the Phase 13 protocol unchanged on two untrained models from
other families and two datasets (ASDiv, MATH levels 1–3). It is built to fill
the report draft's generalization grid: Self_Correction and Qwen2.5-7B-Instruct
run alongside as anchors, every model also solves and judges the Phase 13
GSM8K and Phase 12 SVAMP holdouts, every model re-judges the Phase 13 pairs,
and hidden states are read with Transformers for layer-wise probes and the
logit lens.

## Layout

| Path | Contents |
|---|---|
| `docs/PREREGISTRATION.md` | Draft design, endpoints, limits |
| `DATA_CATALOG.md`, `DATA_MANIFEST.json` | Where every output and hidden-state file is, its schema and sha256, and how to reload each result without re-running (`paper/scripts/phase14_data.py`) |
| `RUN_STATUS.md` | Setup log and, later, the run record |
| `data/raw/` | ASDiv XML, MATH test (parquet and normalized JSONL), `manifest.json` hashes |
| `data/sources_v1/` | Frozen pilot (100) and ranked main pools per dataset, `report.json` |
| `scripts/config.py` | Models (pinned revisions), datasets, caps, sizes |
| `scripts/prepare_sources.py` | Builds `data/sources_v1/` |
| `scripts/smoke_model.py` | GPU check per model (FP16, 768-prefix equality, verdict log-probs) |
| `scripts/smoke_hidden.py` | Transformers check per model on the smoke outputs: hidden-state shapes, finiteness, vLLM/Transformers log-prob agreement, logit lens, timing |
| `scripts/generate.py` | Attempts and judges, Phase 13 prompts imported unchanged |
| `scripts/metrics.py` | Per-cell measures, shared by pilot and main |
| `scripts/pilot_report.py` | Pilot scoring and main size (`--write` fixes it) |
| `scripts/analyze.py` | Single main opening, endpoints E1/E2, anchor and bridge comparisons |
| `scripts/extract_hidden.py` | Transformers pass over the generated text: hidden states at every layer |
| `scripts/probe_analysis.py` | Exploratory probes (surface, log-prob, probe, probe + surface, pointwise) and forced-verdict logit lens |
| `scripts/report_tables.py` | All results in the layout of the report template's tables (Markdown) |
| `scripts/regression.py` | Reproduces the report with the two original models (run in `comparison`) |
| `scripts/verify_lock.py` | Pilot and main execution locks |
| `scripts/dry_run.py` | CPU dry run of the whole pipeline on fake generations, in a separate work folder |
| `scripts/ops/` | `bootstrap.sh` (host setup), `run_sequential.sh smoke|pilot|main` (used on the 100 GB host: one model at a time, weights deleted after each), `run_remote.sh` (all models together), retry wrapper |
| `scripts/sync_remote.py` | Mirror remote outputs to `outputs/phase14_remote/` |

## Run order on the GPU host

1. Environment: `bash phase14/scripts/ops/bootstrap.sh` — Python 3.10, vLLM
   0.7.0 and Transformers in `.venv-vllm` (as in Phases 8–13), plus
   scikit-learn; reads `HF_TOKEN` from `.env` (licenses accepted for gated
   models), checks access, downloads the four models at their pinned revisions,
   and writes `logs/phase14_bootstrap.json` (GPU name, BF16 support). For the bridge set, upload `outputs/phase13_v1/`
   (samples, judges, analysis) from the machine that holds the Phase 13 mirror.
2. `bash phase14/scripts/ops/supervise.sh smoke` — replace a failing model with
   `phi35_mini` in `config.py`, record it in `RUN_STATUS.md`.
3. Locally:
   `python phase14/scripts/verify_lock.py --stage pilot --freeze`; upload.
4. `bash phase14/scripts/ops/supervise.sh pilot`.
5. `python phase14/scripts/pilot_report.py --write`, then
   `python phase14/scripts/verify_lock.py --stage main --freeze`.
6. `bash phase14/scripts/ops/supervise.sh main` — all four models (Llama-3.1-8B,
   Mistral-7B, Self_Correction, Qwen2.5-7B-Instruct) with one identical protocol
   on ASDiv, MATH, GSM8K and SVAMP; opens once and writes `report_tables.md`.
7. Later, after uploading `outputs/phase13_v1/` and `outputs/phase12_v1/`:
   `bash phase14/scripts/ops/supervise.sh comparison` — regression check on
   the report's own Self_Correction outputs and the bridge set. Main cells and
   endpoints are not recomputed.

Local work (source building, lock freezing) uses Python 3.10 with SymPy 1.14,
as in earlier phases.
