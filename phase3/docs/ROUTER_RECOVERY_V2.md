# Router Recovery V2

**Final status: abandoned when Phase 3 closed.** Data collection stopped at 108
verified pairs, below the 200-pair minimum. Hidden extraction, classifier
selection, contrastive LoRA, and confirmation were not run.

## Purpose

Router Recovery V2 tests whether Decision-Only V1's partially decodable
correctness signal can become a reliable error detector without repeating the
KEEP collapse seen after DPO, decision-token LoRA, same-origin LoRA, GRPO, and
simple threshold calibration.

This document is retained as an archived experiment definition. It does not
authorize resuming the experiment inside Phase 3.

## Stage 00 — score and failure audit

**Question:** where does the current frozen score fail?

- Input: the completed 140-row calibration run and its per-row scores.
- Analysis: score distributions and metrics by domain, dataset, error type,
  answer length, and model origin.
- Output: error taxonomy, score histograms, and a prioritized collection list.
- Compute: CPU only, about 15–30 minutes after the reporting script exists.
- Gate: every proposed new data bucket must correspond to a measured failure;
  do not collect generic easy wrong answers.

## Stage 01 — source-disjoint hard pairs

**Question:** can surface-matched examples isolate semantic correctness?

Build at least 200, preferably 300, verified pairs. Every pair contains the same
problem, one correct model answer and one plausible wrong model answer. The two
members must share model origin and closely match length, format, interface, and
neutral-review wording.

Target 60% code and 40% math. Code REVISE rows should execute or compile but
fail substantive tests; math REVISE rows should be near-correct with a genuine
reasoning or final-answer error. The complete dataset remains 50/50 KEEP and
REVISE. Split by source, never by response.

- Existing verified pools: CPU only.
- New V1 generation, if required: GPU inference.
- Validation: fresh verifier, one pair per source, duplicate and protected-set
  overlap checks, label-blind template/length/origin audits.
- Output: `phase3/data/router_recovery_v2/`.

## Stage 02 — frozen representation extraction

**Question:** which frozen state best separates pair members?

Run Decision-Only V1 forward-only and extract layers 14, 21, and 28 at the final
prompt token and answer-end token. Save float16 arrays with a row manifest and
artifact hashes. No model or probe weight may change.

- Compute: one 24 GB GPU; estimated 20–60 minutes for 400–600 rows depending on
  code length.
- Output: `outputs/phase3_router_recovery_v2/02_hidden_states/`.

## Stage 03 — frozen decision heads

**Question:** can a small head expose the signal without modifying the LLM?

Compare only two families initially:

1. class-weighted logistic regression;
2. a regularized 1-hidden-layer MLP using focal loss.

Hyperparameters are selected on source-grouped development folds. A candidate
is ineligible if KEEP recall is below 70%, regardless of balanced accuracy.
Report paired-source accuracy in addition to ordinary classification metrics.

- Compute: CPU only; usually under one hour.
- Weight scope: decision head only; Decision-Only V1 stays frozen.

## Stage 04 — operating policy and uncertainty

**Question:** is the remaining problem primarily the operating point?

Fit policy parameters only on the dedicated threshold-calibration split. Test:

- one global binary threshold;
- separate predeclared math/code thresholds;
- a three-way KEEP / UNCERTAIN / REVISE policy.

The three-way policy sends borderline cases to a verifier instead of forcing a
possibly unsafe KEEP decision. Report coverage and selective accuracy at 70%,
80%, and 90% coverage. Domain thresholds are accepted only if both domains have
enough calibration examples and the internal test improves; arbitrary
per-dataset thresholds are prohibited.

- Compute: CPU only.
- Interpretation: this is policy calibration, not representation learning.

## Stage 05 — conditional contrastive LoRA

**Question:** must the representation itself change?

Run this stage only if Stages 03–04 fail the promotion gate. Start from
Decision-Only V1, train one epoch at learning rate `5e-6`, and combine:

- cross-entropy on the KEEP/REVISE decision token only;
- supervised contrastive loss that separates the correct and plausible-wrong
  member of the same problem.

Mask explanation tokens from language-model loss. Do not use whole-response
DPO. Early-stop based on source-disjoint internal REVISE recall while enforcing
the KEEP floor.

- Compute: one 24 GB GPU, approximately 1–3 hours for a QLoRA/LoRA pilot.
- Output is experimental and cannot replace Decision-Only V1 until Stage 06.

## Stage 06 — sealed confirmation

**Question:** does the selected method generalize to genuinely unseen data?

Construct and seal 200 new rows before looking at candidate-model results. They
must be source-disjoint from all training, calibration, old probe, and frozen
sets. The old frozen-200 has already been inspected and is therefore only an
exploratory comparison, not the final confirmation set.

Promotion requires all of:

- balanced accuracy at least 70%;
- KEEP recall at least 70%;
- REVISE recall at least 65%;
- no code-accuracy regression;
- no unseen-template regression;
- predictions contain both labels.

If metrics improve only through abstention, report coverage explicitly and do
not present selective accuracy as full-coverage router accuracy.

## Execution order and stop rules

```text
Stage 00 audit
    -> Stage 01 hard pairs
    -> Stage 02 hidden extraction
    -> Stage 03 frozen heads
    -> Stage 04 calibration / UNCERTAIN policy
         -> PASS: build sealed Stage 06 confirmation
         -> FAIL: Stage 05 contrastive LoRA
                    -> Stage 06 confirmation
```

Do not spend GPU time before Stage 01 passes all data validation. Do not run
Stage 05 merely because one binary threshold fails: first test whether a frozen
head and an explicit uncertainty band are sufficient.
