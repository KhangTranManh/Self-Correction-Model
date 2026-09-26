# Phase 8 preregistration — reproducibility, distractor, and non-oracle routing

Original version written on 2026-09-25 before Phase 8 model generation. The
fixed historical layer/C clarification was made after first-pass collection
started but before any answer content, score, or correctness result was
inspected; see `phase8/RUN_STATUS.md` for this timing deviation. The execution
lock fixes the final protocol before three-arm or probe generation. Phase 7 protected
results were already opened and are historical evidence, not a fresh
confirmatory test. Any new analysis on those 160 sources is labeled exploratory.
The primary Phase 8 test is the 400-source protected pool in
`phase8/data/fresh_source_pool_v1/`, frozen before new initial answers.

## Question and constraints

Phase 7's blind-versus-answer-visible paired effect is real under its fixed
prompts, but does not isolate a psychological anchoring mechanism. Phase 8
tests (1) how much an independent second sample or prompt change contributes,
(2) whether answer-visible harm exceeds length-matched irrelevant context,
and (3) whether an external probe can route blind re-solves profitably without
an oracle label. No solver weights are trained.

Do not modify this protocol after inspecting intermediate Phase 8 outcomes.
Record deviations separately with timestamp, reason, and whether they precede
or follow protected opening. No protected outcome may choose prompts, donor
rules, probe layer/C/threshold, model, or stop condition.

## Step 0: as-run audit (completed, no GPU)

`phase8/data/phase7_protocol_audit_v1.json` checks locked initial and paired
summaries and reconstructs the user-message text for all 334 initial answers.
Initial answers used vLLM 0.7.0 FP16, temperature 0.7, top-p 1, top-k -1,
seed 20260924 plus source index, and 768 output tokens. Paired outputs used
vLLM 0.7.0 FP16, greedy temperature 0, 768 output tokens, and one shared
request seed per source/checkpoint across arms. The initial prompt was
`build_prompt(problem)`; blind was that exact text plus
`\n\nGive a complete solution from scratch.`. User text is different on all 334
sources. Both paths invoked the tokenizer chat template with a single user
message and `add_generation_prompt=True`; rendered token IDs could not be
compared locally because the pinned tokenizer is absent. A subsequent
read-only tokenizer audit on the GPU host (`phase8/data/phase7_rendered_prompt_audit_v1.json`)
confirmed zero exact rendered-prompt or token-ID matches across all 334 rows.
The protected split
contains 80 initial wrong and 80 initial correct, not 100 wrong.

These differences affect comparisons of first versus second answer. They
cannot alone account for the blind-minus-visible difference because both
Phase 7 arms shared decoding, checkpoint, source, and base prompt. The visible
arm additionally included the earlier answer and its header.

## Step 1: reproducibility controls

Run on the 80 historically protected wrong sources with the original solver
using the exact original prompt and decoding, including the original per-row
request seed. Report exact full-text match, parsed final-answer match, and
verified correctness against the historical first answer. This is an
exploratory reproducibility check because Phase 7 protected is already open.
Reproducibility may be affected by GPU/backend changes and vLLM version. A
high same-seed match does **not** imply deterministic decoding; a low match
does **not** quantify anchoring.

On the new 400-source pool, generate the first answer once using the pinned
original-solver Phase 7 initial prompt and temperature 0.7. Generate two
independent controls with the original solver on every source: (A) a second
temperature-0.7 sample with a distinct predetermined seed and unchanged
prompt; (B) a greedy answer with unchanged prompt. These isolate resampling
and decoding before changing the prompt. Apply the same 768-token cap and
checkpoint revision. Score only after all outputs are complete.

## Step 2: paired distractor comparison

For each source and checkpoint, generate three greedy arms with identical
model lineage, output cap, chat template, and base prompt:

- Blind: `build_prompt(problem)` plus the Phase 7 fixed suffix.
- Own-answer visible: blind prompt plus the Phase 7 visible header and the
  complete initial answer from the original solver.
- Distractor: blind prompt plus the same visible header and a complete answer
  from a different Phase 7 source, selected by a deterministic hash rank.
  Donors must be from the same math domain, have a verifier-wrong final answer
  for their own source, and be within 10% of the target answer's tokenizer
  length. If none exists, mark that source unmatched and exclude it from
  three-arm paired contrasts, while retaining it for the blind/visible and
  full-pipeline analyses. Do not search or adjust donors after generation.

The distractor is an imperfect placebo: its semantic irrelevance can make it
easier to ignore than a plausible answer to the same question. Report this
limitation. Token-count matching uses the pinned model tokenizer, and the
matching table/hash is frozen before any Step 2 generation. Phase 7 old wrong
answers are donor text only; their source questions and gold answers do not
enter Phase 8 model prompts.

Primary mechanism contrasts are paired wrong-to-correct rates on naturally
initially-wrong Phase 8 rows: own-visible minus blind and distractor minus
blind; own-visible minus distractor is the direct specificity contrast.
Also report correct-to-wrong harms and total accuracy. Use paired source
bootstrap 95% intervals with 10,000 resamples and exact paired tests with
Holm correction across three checkpoints per named contrast family. A CI
crossing zero is inconclusive, not equivalence. No conclusion of "specific
anchoring" follows merely from a nonsignificant distractor/blind contrast.

## Step 3: probe and full non-oracle pipeline

First search for the exact Phase 5 `.joblib` models and verify checksums.
The local recovery audit currently reports all three missing. If unavailable,
extract activations from frozen Phase 5 train (240 rows) and development (80
rows) with the original checkpoint lineage and Phase 5 pre-hint template.
Fit the existing StandardScaler + L2 logistic probe using **train only**.
Development is an audit only; it does not select hyperparameters. Freeze the
historical selected settings: layer 14, C=0.01 for original/V2 and layer 14,
C=1.0 for V3, threshold 0.5. Do not fit on development alone. If newly fit,
call it a rebuilt probe, never the identical frozen Phase 5 artifact.

Before evaluating the 400 protected sources, lock model files, SHA-256 hashes,
checkpoint lineage, layer, scaler, classifier, threshold 0.5, score template,
route rule, prompt, decoding, and output parser. Save at least two checked
copies, one local and one GPU-host copy, and a third independent backup when
available. Protect the model input from gold answer, verifier result, probe
score, and any previous response in the blind solving session.

The primary pipeline uses the natural first answer on each of 400 sources,
scores its pre-hint hidden state, keeps unflagged answers, and replaces flagged
answers with the corresponding blind re-solve from the same checkpoint. Gold
verification is **offline only**. Report per-checkpoint wrong recall,
precision, correct preservation, flag rate, routed wrong-to-correct and
correct-to-wrong counts, initial accuracy, final accuracy, and paired accuracy
difference with 95% bootstrap CI. Benchmark against KEEP-all, BLIND-all, and
an equal-compute random-routing control fixed before protected opening.
The primary end-to-end claim requires final accuracy above KEEP-all with a
95% paired CI excluding zero for at least one checkpoint after Holm correction
over the three checkpoints; otherwise report no demonstrated gain.

## Execution boundary

All 400 source rows are hash-locked at
`a4d2829140292cce1b86ccf3c758e7b30641ee829452698a9ca139462eae7e24`.
Do not use any Phase 5 protected, Phase 6 holdout, or Phase 7 outcome to fit or
calibrate the probe. Do not use the new 400-source outcomes to change any
policy. Save append-only output audits; verify source, model, and output hashes
before analysis. Open protected labels and aggregate metrics once after all
required generation is complete. If GPU hosts change, keep all arms for a
checkpoint/source on the same host and record the hardware and backend.

Current RTX 3090 host is newly provisioned and has no project code or Python
environment in `/root`; successful SSH connectivity alone is not a completed
GPU preflight. Upload only after local protocol and source hashes are checked.
