# Phase 14 — does the judgment generalize to other model families?

**Status: closed (2026-10-14); paper analyses finished 2026-10-16.** The endpoints were opened once on 2026-10-13:
Llama-3.1-8B-Instruct (Meta weights via the NousResearch copy, Meta tokenizer) and Mistral-7B-Instruct-v0.3 on ASDiv
(822) and MATH 1–3 (785). **E1 (skill index) 4/4 cells → general; E2 (log-prob debiasing gain) 1/4 → conditional.**
The completion run then filled the rest of the 4 × 4 grid (Self_Correction and Qwen2.5-7B-Instruct on all four
datasets, Llama and Mistral on GSM8K and SVAMP), the bridge set (each model re-judges the 288 Phase 13 pairs), the
regression cells, hidden states for probes on all models, and a 1536-token sensitivity run for Llama. Full record
and every amendment, including the 2026-10-15 amendment written before the paper analyses:
[RUN_STATUS.md](RUN_STATUS.md); mirrored outputs: `outputs/phase14_remote/`; hidden states: `F:\AGI_data\phase14_hidden`
(junction `outputs/phase14_hidden`). Consolidated results and claims: [../paper/summary/SUMMARY_ALL.md](../paper/summary/SUMMARY_ALL.md); the
final pipeline status written on 2026-10-14 is archived in [../paper/summary/sources/FINAL_STATUS.md](../paper/summary/sources/FINAL_STATUS.md).

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

## Results

Endpoints (opened once on 2026-10-13; problem-clustered bootstrap, 10,000 resamples):

| Cell | E1 skill index [95% CI] | E2 log-prob order-averaging gain [95% CI] |
|---|---|---|
| Llama · ASDiv | **+0.321 [0.230, 0.413]** | +0.6 [−3.7, +4.8] |
| Llama · MATH 1–3 | **+0.362 [0.291, 0.428]** | **+5.0 [+0.9, +9.3]** |
| Mistral · ASDiv | **+0.335 [0.269, 0.400]** | +0.7 [−2.3, +3.6] |
| Mistral · MATH 1–3 | **+0.344 [0.261, 0.431]** | +3.3 [−0.7, +7.3] |

**E1 4/4 → general. E2 1/4 → conditional.** Descriptive grid (16 cells): skill index above 0 in every cell;
single-order accuracy 45–64%; Self_Correction prefers the second position (52–56%), the other models the first.

Paper analyses (exploratory, rules written in `RUN_STATUS.md` on 2026-10-15 before running):

| Analysis | Result | File |
|---|---|---|
| Seven-way outcome of each judgment | valid choice in 51–90% (`base`), 79–99% (`base_c`) | `paper/results/phase14_invalid_types.md` |
| Four-factor split of the single-order gap | gap 6.4–43.3 points: bias 0–2.5, format 0.2–22.8, cut 0.5–13.9, no choice 2.6–16.4 | `paper/results/phase14_decomp4.md` |
| Rescued reading | Qwen 45–64% → 72–88% single-order; assistant audit: 2.2% clear misreadings | `paper/results/phase14_rescued_picks.md`, `phase14_rescued_audit.md` |
| Practical value | judge loses to vote@3 in 15/16 cells; vote@5 best everywhere | `paper/results/phase14_practical.md` |
| Probe-weighted vote@3 out of sample | confirmed for Llama and Mistral, not Self_Correction or Qwen | `paper/results/phase14_f2_oos.md` |
| Token cap 1536 (Llama) | gap unchanged; capped judgments are mostly loops | `paper/results/phase14_truncation.md` |
| Counts per cell | 21 cells | `paper/results/phase14_counts.md` |

Consolidated numbers and claim wording: [../paper/summary/SUMMARY_ALL.md](../paper/summary/SUMMARY_ALL.md); paper
draft: [../paper/manuscript/main.tex](../paper/manuscript/main.tex).

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
