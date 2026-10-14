# Phase 14 preregistration

**Status: design settled on 2026-10-12; nothing has been generated. All
decisions are made (models: Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3;
one protocol for all four models; E2 = log-prob order averaging). The file is
frozen by `scripts/verify_lock.py --stage pilot --freeze` before the first
GPU run;** after that, any change is an amendment recorded in `RUN_STATUS.md`
before the generation it affects.

## Motivation

Phases 11–13 found, on one checkpoint (`Kxck/Self_Correction_v1`, a Qwen2.5-7B
fine-tune), that the judge's choice between two of its own conflicting
solutions mixes three things: a position bias, a weak real skill, and invented
third answers. Judging both orders removed most of the bias and exposed the
skill (Phase 12–13). Phase 14 asks whether this holds beyond that checkpoint:
two untrained instruction models from other families, on two datasets of
different difficulty, under the unchanged Phase 13 protocol.

## Comparability with Phases 12–13

The protocol is Phase 13's, changed only where a new model forces it:

| Item | Phase 13 | Phase 14 |
|---|---|---|
| Solve prompt (s1) | Phase 1 Vietnamese prompt | same, imported from `phase1/src/core/prompts.py` |
| Blind prompt (s2–s5) | Phase 7/8 blind suffix | same, imported from `phase7/scripts/paired_prompts.py` |
| Judge prompts | Phase 9 (`base`), Phase 13 constrained (`base_c`) | same, imported |
| Attempts | 5, temperature 0.7, per-request seeds | same |
| Judged pairs | (s1, s2), (s1, s3) with different final answers, both orders, greedy | same |
| Token cap / context | 768 / 4096 | **primary view 768 / 4096**; attempts also generated to 1536 for a cap-sensitivity view |
| Answer reading | Phase 1 extraction + SymPy | same (lenient, primary) + strict (boxed or labelled only) + rescued (first number on the last "Đáp số" line when lenient is non-numeric; added 2026-10-12) as checks |
| Chat template | Qwen | each model's own template, user role only |
| Hardware | V100, vLLM 0.7.0, FP16 | same if the models pass the smoke test |

The 768-token primary view is the first 768 tokens of each 1536-token attempt.
With a fixed per-request seed this equals a 768-cap run; `scripts/smoke_model.py`
checks the equality on the GPU before any data run.

## Models (decided 2026-10-12)

**Selected: `llama31_8b` (meta-llama/Llama-3.1-8B-Instruct) and `mistral_7b`
(mistralai/Mistral-7B-Instruct-v0.3).** Untrained instruction models from two
families other than Qwen, of similar size (8B, 7B), both low risk in FP16 on a
V100. Llama is gated (accepted license and `HF_TOKEN` needed); Mistral is not.
If either fails the smoke test it is replaced by `phi35_mini` and the swap is
recorded before any pilot run. In the report's grid these fill the rows
"Llama hoặc Mistral" and the third-family row (Mistral instead of Gemma/Phi).

Candidates considered, revisions pinned on 2026-10-12 (`scripts/config.py`):

| Key | Model | Gated | V100 FP16 risk |
|---|---|---|---|
| `llama31_8b` | meta-llama/Llama-3.1-8B-Instruct | yes (license) | low |
| `mistral_7b` | mistralai/Mistral-7B-Instruct-v0.3 | no | low |
| `gemma2_9b` | google/gemma-2-9b-it | yes (license) | **high**: trained in BF16, uses logit soft-capping; FP16 on a V100 may overflow or be unsupported by the attention backend |
| `phi35_mini` | microsoft/Phi-3.5-mini-instruct | no | low (but 3.8B, a size difference) |

Gemma-2 was the first default but is not run: it is BF16-trained and FP16 on
a V100 risks overflow. Using it would need a BF16 GPU, which changes precision
from earlier phases.

## Comparison with earlier results and the report template

Phase 14 fills the report draft "Position bias che khuất khả năng tự phân
xử" (2026-10-12): its generalization grid (section 4.6) is models
{Self_Correction, Qwen2.5-7B-Instruct, Llama-3.1-8B, Mistral-7B} × datasets
{GSM8K, SVAMP, ASDiv, MATH 1–3}, each model judging its own solutions. The
additions below exist for that comparison and do not count toward the
endpoints.

- **Anchor models.** `Kxck/Self_Correction_v1` (the checkpoint of Phases 5–13)
  and its untrained base `Qwen/Qwen2.5-7B-Instruct` run the same pipeline on
  every dataset (report rows 1–2).
- **GSM8K and SVAMP columns: the report's own datasets.** Every model solves
  and judges exactly the report's problems: the frozen Phase 13 GSM8K holdout
  (561; also scored on the audited 519 if `data/gsm8k13_audit_excluded.json`
  is supplied) and the Phase 12 SVAMP holdout (960), loaded through their
  original hash-checked loaders. Their labels were opened earlier; untrained
  models are only evaluated on them.
- **One protocol for all four models (decided 2026-10-12).** Self_Correction
  is re-run exactly like the new models: fresh attempts with Phase 14 seeds,
  both judges with verdict log-probabilities, the same hidden-state sites, on
  all four datasets. The main tables use only these uniform cells, so position
  bias, skill index, and the debiasing measures are computed the same way for
  every model. Self_Correction's GSM8K numbers will therefore differ somewhat
  from the report's (new samples give new pairs).
- **The report's own outputs are used only for the regression test.** The
  regression-only sets `gsm8k13_report` and `svamp12_report` read
  Self_Correction's attempts and base judgments from `outputs/phase13_v1/` and
  `outputs/phase12_v1/`; `regression.py` must reproduce the report's printed
  numbers from them. These cells never enter the main tables or endpoints.

| Report item | Phase 14 source |
|---|---|
| Table 1 (picks by right position), skill table | `metrics.judge_block`: picks, skill index + clustered CI, position-2 rate, invented rate |
| Table 2 | tie-break + CI, consistent accuracy, coverage, all-pairs accuracy |
| Table 3 (vs Ren et al.) | log-prob order averaging (verdict token after the written check, and forced verdict without it); tie-break minus single-order shown with its invented-answer share |
| Section 4.3 | `probe_analysis.py`: surface, log-prob, probe, probe + surface, pointwise probe; train SVAMP → test GSM8K and the other directions; order-flip rate |
| Logit lens (Figure 1) | forced verdict, logit lens at every layer |
| Section 4.5 | bridge set (each model judging Self_Correction's Phase 13 solutions) |
| Section 4.6 grid | per cell: skill index [CI], bias-removal gain, bias log-odds, coverage |
| Section 5 | keep, random s1/s2, judge-or-random, self-check, vote@3, vote@5 |

`scripts/report_tables.py` prints all of these in the report's layout.

Not reproduced here (definitions live outside this repository): the 4-candidate
comparison of Table 3 and DPO judges p12/p13b on new models. The reverse
direction of section 4.5 (Self_Correction judging Qwen2.5-7B-Instruct's
solutions) was run outside Phase 14 as paper check T5.
The surface features follow the report's wording (length, step count,
answer); their exact earlier definition should be checked.
- **Bridge set (`bridge13`).** Every model re-judges the 486 judged pairs of
  Phase 13 (GSM8K test 750–1318) with the Phase 9 and Phase 13 prompts. The
  solutions are Phase 13's own original-solver outputs, so only the judge
  differs, and the Phase 13 `base` (71.0% consistent) and `base_c` numbers on
  the same pairs are the reference. The original solver on this set must
  reproduce Phase 13's `base` and `base_c` judgments (a pipeline check; small
  differences from batching are reported, not corrected). Its labels were
  opened in Phase 13; untrained judges are only evaluated, never tuned, on
  it. It needs `outputs/phase13_v1/` on the GPU host; otherwise it is skipped
  and recorded.

## Datasets

Frozen in `data/sources_v1/` (built by `scripts/prepare_sources.py`; raw
files hashed in `data/raw/manifest.json`):

| Dataset | Source | Eligible | Pilot | Main pool | Notes |
|---|---|---:|---:|---:|---|
| `asdiv` | ASDiv (chaochun/nlu-asdiv-dataset) | 2,019 | 100 | 1,919 | grades 1–6, easier than GSM8K; 132 problems excluded as near-copies of SVAMP (opened in Phase 12), token Jaccard ≥ 0.5 |
| `math13` | MATH test, levels 1–3 (EleutherAI/hendrycks_math) | 2,057 | 100 | 1,957 | numeric answers only; references normalized and re-parsed (see below) |

Both exclude every question found in Phase 1–13 data files (353 files
scanned); none overlapped. Selection order is
SHA256(`phase14_sources_v1|dataset|20261012|raw_id`).

MATH references were filtered because the Phase 1 parser silently misreads
some forms: it falls back to a leading number, so `500a^7` reads as 500,
`1,3` (two answers) as 1.3, `443_5` (base 5) as 4435, `10{,}000` as 10, and
`45 \text{ units}^2` as 47; it reads `12\frac{3}{5}` as 36/5 (a product, not a
mixed number) and `15 \pi` as 15 ("pi" taken as a unit). Such references are
excluded or normalized (units, spacing, thousands separators removed), and all
4,076 stored references were re-parsed independently without the fallback;
none differed.

## Regression test (after the main run; order decided 2026-10-12)

The main run (all four models, uniform protocol, all four datasets) comes
first; the regression test runs later, when the report's outputs are uploaded
(`supervise.sh comparison`). This order is fixed here, before any generation.
The comparison run adds only the regression-only sets and the bridge set
(`analyze.py --add-comparison`), never recomputing uniform cells, E1, E2, or
their summary.

Checks, all on data the report already opened:

1. Self_Correction, untrained base judge, the report's own GSM8K 561 and SVAMP
   960 outputs: every printed report number reproduced to its printed precision.
2. Self_Correction, constrained judge re-run on the report's GSM8K pairs: same
   task set as Phase 13, share of identical texts and picks reported, Phase 13 numbers within
   ±3 points.
3. Qwen2.5-7B-Instruct, its uniform GSM8K cell (main run):
   tie-break (strict, lenient) with an interval, set against section 4.5 of the
   report (descriptive; the report used a 1536-token judge view and other
   seeds).

Risk of this order: the endpoints are opened before checks 1–2. If check 1
fails because of a metric, pair-selection, or verifier difference, every cell
is rescored from the saved generations (CPU only) and the difference is
recorded; the endpoint thresholds and rules do not change. If check 2 fails
(generation settings differ from Phase 13), the discrepancy is reported as a
limit and the affected cells are flagged; no cell is regenerated to match.

## Pilot (before the main holdout)

For each selected model and dataset, run the full pipeline on the 100 pilot
problems. `scripts/pilot_report.py` reads pilot labels (pilot problems are
burned and never reused) and proposes, per dataset, the main size

N = ceil(1.15 × 300 / min over models of one-right pairs per problem),

capped at the main pool. N is written to `data/main_size.json` and the main
lock is frozen before any main generation. The pilot may not change prompts,
models, thresholds, or endpoints, except to replace a model for a technical
failure (recorded).

## Unified method (identical for all four models and all four datasets)

**Generation.** For each problem: s1 with the Phase 1 prompt and s2–s5 with
the blind prompt, temperature 0.7, per-request seeds, 1536 tokens generated,
the first 768 used (primary view). Judged pairs: (s1, s2) and (s1, s3) whose
final answers differ, each judged in both orders (A/B swapped), greedy, by two
untrained judges of the same model:

| Judge | Prompt | Output used |
|---|---|---|
| `base` | Phase 9 judge prompt (written check, free final answer) | which solution its final answer matches: A, B, or invented |
| `base_c` | Phase 13 constrained prompt (must end "Verdict: Solution A/B") | the verdict line, plus log P(A) and log P(B) at the verdict token |

**Labels.** Deterministic verifier (Phase 1 SymPy, lenient primary, strict
as a check). One-right pair: exactly one of the two solutions is correct. All
judge measures below are computed on one-right pairs, both orders.

**Measures**, each with a 95% interval by bootstrap over problems (10,000):

| Measure | Definition | Judge |
|---|---|---|
| Single-order accuracy | share of single judgments that pick the right solution (invented = wrong) | both |
| **Skill index** | P(pick A \| right shown as A) − P(pick A \| right shown as B); a pure position bias cancels | `base` (primary), `base_c` (reported) |
| **Position bias, choice rate** | share of A/B picks that are B (0.5 = none) | both |
| **Position bias, asymmetry** | P(right \| right at B) − P(right \| right at A) | both |
| **Position bias, log-odds** | mean over pairs of (m_ab + m_ba)/2, m = log P(A) − log P(B); > 0 favors A | `base_c` |
| Invented-answer rate | share of judgments whose answer matches neither solution | both |
| Consistent accuracy, coverage | both orders pick the same solution; accuracy among those; their share | both |
| All-pairs accuracy | consistent and right / all pairs | both |
| Tie-break | consistent right 1, consistent wrong 0, otherwise 0.5 | both |
| **Debias gain (log-prob)** | accuracy of sign(m_ab − m_ba) minus single-order accuracy of the same log-probabilities; split into *bias removal* (single order with a leave-one-source-out bias estimate subtracted) and *two readings* (the rest) | `base_c` |
| Tie-break − single order | reported; equals 0.5 × share of pairs with no right pick and an invented answer (checked in code), so it is not a debias measure | both |
| Forced verdict | `base_c` prompt + "Verdict: Solution" with no written check: single-order and order-averaged accuracy, and per layer through the logit lens | `base_c` prompt |

Also reported per cell: accuracy of keep / random s1–s2 / judge-or-random /
self-check / vote@3 / vote@5, fixes and harms, the 1536-token view, cap-hit
rates.

## Endpoints (decided 2026-10-12)

| Endpoint | Definition | Threshold per cell |
|---|---|---|
| **E1 skill** | skill index, `base` judge, lenient scoring | source-clustered 95% CI above 0 |
| **E2 debias** | log-prob debias gain, `base_c` judge: accuracy of sign(m_ab − m_ba) minus single-order accuracy of the same log-probabilities | ≥ +3 points and CI above 0 |

**Generality rule (written before running):** a claim is general if it
passes in at least 3 of the 4 cells; if it passes in 1–2 cells it is reported
as conditional on those cells; never written as a general phenomenon otherwise.

**Testable prediction:** across cells, the *bias-removal* part of the gain
grows with |bias|. With four cells this is descriptive (Spearman reported, no
test). If a cell with almost no bias still shows a large *bias-removal* gain,
the bias explanation is wrong. A large *two-readings* gain with no bias is
expected and does not support the bias explanation.

### Why E2 is not "tie-break minus single-order accuracy"

The first proposal defined the debiasing gain as tie-break score minus
single-order accuracy. Per pair that difference is exactly
0.5 × [no ordering picks the right solution and at least one ordering
invents an answer], so it measures invented answers, not position bias. A
judge with a pure position bias and no invented answers gets 0; a judge that
always invents gets +50 points. Phase 13 confirms it: the difference was
+7.8 / +3.5 / +1.9 / +1.0 points for judges with 16.7% / 9.9% / 4.9% / 2.1%
invented answers. It is still reported, with the identity checked in code
(`scripts/metrics.py`).

This matters for the report's Table 3. Its "+7.1" for Self_Correction on
GSM8K (52.8 → 59.9) has the size of the tie-break minus single-order
difference of the untrained judge (52.3 → 60.1, invented 16.7%). If that is how
it was computed, it is the invented-answer term, not a debiasing gain, and it
is not on the same scale as Ren et al., whose multiple-choice setting has no
invented answers. The judge with 2.1% invented answers (p13b) shows +1.0,
close to Ren et al.'s +0.5–0.6. The log-prob order averaging above is Ren et
al.'s method and is the comparable number. The log-prob version measures debiasing at full
coverage and can be split into bias removal and two-readings averaging.

## Limits to report in advance

- **Contamination:** public instruction models have very likely seen GSM8K,
  MATH, and possibly ASDiv during training, including test sets. Absolute
  accuracy is not comparable to Phases 9–13; the judge measures within a cell
  are less affected but not immune.
- **Prompt language:** the solve prompt is Vietnamese ("Đáp số"), kept for
  comparability. The new models' Vietnamese ability may change accuracy and
  answer formatting. An English rerun is out of scope for the primary
  analysis.
- **Verifier blind spots** that remain on the model side: a model answer
  written as `15 \pi` reads as 15, and a mixed number as a product. These
  affect lenient and strict scoring equally; the analysis counts answers
  containing such forms.
- Chat templates differ between models; four cells cannot establish a trend;
  one run per cell.

## Hidden states (exploratory, in the main run)

Text is generated with vLLM as in Phases 8–13; hidden states come from a
separate Transformers forward pass over exactly that text
(`scripts/extract_hidden.py`, FP16, every layer), as in Phases 5 and 13. This
keeps generation comparable while exposing every layer. Extraction is
label-free and runs before the main opening; the probes run after it
(`scripts/probe_analysis.py`) and are not endpoints.

| Site | Position | Probe question |
|---|---|---|
| answer | last token of s1, s2, s3 after the solve prompt (+ mean token log-probability) | is the first answer right? (Phase 5 analogue) |
| judge | last prompt token of every judged task, before the judge writes | which shown solution is right? (Phase 13 analogue) |
| verdict | teacher-forced log P(A), log P(B) after "Verdict: Solution" | agreement of Transformers and vLLM log-probabilities |

Probe: StandardScaler + L2 logistic regression, C = 0.01 (Phases 5 and 13).
Train on one dataset, test on the other and on the bridge set; the layer is
chosen only by 5-fold source-grouped CV on the training dataset. Baselines on
the same test rows: likelihood (higher mean token log-probability) and surface
(length, digit count, log answer magnitude). Pair-level probe accuracy is
reported separately for pairs the judge got consistently right, consistently
wrong, and inconsistent, to test whether the internal signal knows more than
the written verdict. A probe that does not beat both baselines is not
evidence of an internal correctness signal.

## Optional (not part of the primary analysis)

- Order-swapped DPO judges as in Phase 12, if time allows (new phase or a
  recorded amendment before any main opening).

## Integrity

Pilot and main holdout labels are read only by `scripts/pilot_report.py`
(pilot) and once by `scripts/analyze.py` (main). Both locks are verified on
the GPU before generation and before scoring. Deviations go to
`RUN_STATUS.md`.
