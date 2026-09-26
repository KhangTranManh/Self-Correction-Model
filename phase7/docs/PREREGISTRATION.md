# Phase 7 preregistration draft - blind re-solving

**Status: preregistered protocol executed.** The source pool and
initial-collection inputs were frozen at
`data/protocol/initial_collection_v1_lock.json`; paired inputs were frozen at
`data/protocol/paired_generation_v1_lock.json`. See the
[final report](FINAL_REPORT.md) for the protected result and limitations.

## Question and hypotheses

The question is whether the weak repair in Phase 6 is primarily a limitation
of independent solving or of asking a model to work around an answer already
visible in context.

- **H1 (answer anchoring):** On initially wrong answers, blind re-solving has
  a higher verified wrong-to-correct rate than answer-visible re-solving on
  the same sources and checkpoints.
- **H2 (solving limit):** If both arms have similarly low verified success,
  the tested checkpoints have a low second-pass solving ceiling under this
  protocol. A null difference does not prove that anchoring never occurs.
- **Safety question:** Does a blind second pass change initially correct
  answers into wrong ones? A gain on wrong rows is not sufficient if the
  non-oracle router loses correct answers.

These are observational claims about frozen checkpoints and prompts. A paired
prompt intervention can show an answer-visibility effect, but cannot by itself
identify a unique psychological mechanism inside the model.

## Fresh sources and freeze order

Start from raw arithmetic word problems and build a new CPU source inventory.
Exclude all known Phase 1-6 train, development, probe, confirmation, and
protected sources by source ID and normalized question. Exclude the full Phase
5 selected 1,200-candidate manifest and the entire Phase 6 confirmation
160-source pool, including sources not used in its 40-row holdout. Record every
input file and its hash. Do not recycle any Phase 5/6 response as an initial
answer for this phase.

Hash-order up to 600 eligible problems before seeing Phase 7 model behavior.
Generate one natural initial answer per problem from the original solver, with
the frozen Phase 1 math solve prompt, then verify it. Stop at the first ordered
prefix containing at least 100 initially correct and 100 initially wrong
answers, or at the 600-problem ceiling. If the quota fails, report the
feasibility failure; do not extend or curate the pool after inspecting repair
outcomes. Select a balanced 40-source development set (20/20) and 160-source
protected set (80/80) by a predeclared hash rule. Exact IDs and hashes must be
locked before either re-solve arm runs. The initial answer is identical across
all checkpoint/arm comparisons on a source.

## Paired interventions

For each checkpoint and selected source, run both arms with the same weight
dtype, decoding budget, sampling policy, seed schedule, output format, and
problem text. Use separate single-turn conversations; do not append either
arm to the initial-answer conversation. Run order is fixed independently of
the verifier label, and a failed generation remains a failure in the audit.

1. **Blind re-solve:** supply the ordinary original problem/solve prompt in a
   fresh conversation. Do not include the old answer, its length, a probe
   score, a hint, or a statement that the model is being rechecked. The
   previous answer remains in harness metadata only.
2. **Answer-visible re-solve:** supply the same solve instruction and problem,
   plus the full prior answer identified as an unverified candidate. Ask for
   an independent complete solution, without claiming that the candidate is
   wrong. This measures the effect of prior-answer visibility under a
   re-solving instruction.

The output is a complete answer in the same format in both arms. The old
KEEP/REVISE correction contract is not used as the primary comparison.
Existing Phase 6 oracle/status results are historical context only and cannot
serve as a matched Phase 7 arm because they used different sources and prompts.

## Detector and end-to-end accounting

Use the existing frozen Phase 5 probe for each checkpoint with its recorded
layer, scaler, classifier and 0.5 wrong-probability threshold. Extract its
signal from the original problem-plus-initial-answer context before any
re-solve prompt. No probe refit, calibration, or threshold search is allowed.
The documented probe artifact is not currently present at its workstation
path. Retrieval and hash verification are a hard prerequisite for the
non-oracle analysis. If retrieval fails, report that analysis as unavailable;
do not silently fit a replacement probe. The paired solving comparison and
oracle-known-wrong diagnostic remain independently interpretable.

Generate the paired potential outcomes on all selected rows, then apply the
frozen route offline: unselected rows retain the old answer; selected rows
take the new answer from the chosen arm. This makes true positives, false
positives, misses, and correct-answer preservation measurable with identical
source coverage. Report the **oracle-known-wrong** diagnostic separately:
route only initially wrong rows, without telling the re-solving model that
they are wrong. Oracle selection is an analysis condition, not autonomous
detection.

## Metrics and interpretation

The primary contrast is the paired difference in verified wrong-to-correct
rate, blind minus answer-visible, on initially wrong protected rows. Report
the 2x2 paired outcome table, absolute rates, difference, and a source-level
paired 95% confidence interval for each checkpoint. Use exact paired tests
with Holm correction across the three checkpoints as secondary evidence.
Predeclare one pooled descriptive estimate but do not treat three checkpoint
results as independent source samples.

Also report initially correct-to-wrong rate, answer agreement with the old
answer, agreement between the two new arms, verifier validity, malformed or
truncated generations, and stratification by frozen probe route. The old
answer is compared by normalized final answer and by verifier result; textual
similarity alone is not evidence of a correct repair or of anchoring.

For the non-oracle route, report wrong recall, route precision, preservation
of initially correct answers, wrong-to-correct, correct-to-wrong, and final
accuracy for both arms. The initially balanced protected set starts at 50%
accuracy by construction. A result that improves wrong-row solving but causes
equal or larger correct-row harm is not a safe system gain.

Interpretation gates:

- Evidence favoring answer visibility as a bottleneck requires a positive
  blind-minus-visible paired difference on wrong rows, with a 95% interval
  excluding zero and Holm-adjusted paired `p < 0.05`. For an end-to-end safety
  claim, the frozen routed analysis must also show no more than a 2.5-point
  absolute increase in correct-to-wrong rate. This does not automatically
  promote a system.
- If both arms have no more than 15% wrong-to-correct success and the upper
  95% bound on the blind-minus-visible gain is below 10 percentage points,
  report support for a low second-pass solving ceiling **under the tested
  protocol**. If the interval is wider, call the result inconclusive rather
  than declaring anchoring absent.
- If both re-solve arms beat historical repair, the difference may involve
  correction framing or contract burden as well as answer visibility;
  historical comparisons are descriptive, not causal.

## Integrity and disposition

Freeze exact prompts, model revisions, adapter parents and hashes, candidate
order, split rule, decoding, seed schedule, probe artifacts, metrics, bootstrap
method, and failure handling before protected generation. A development smoke
checks technical feasibility and parser behavior only; it must not select the
best prompt, model, seed, or threshold. Open protected results once.

No training or promotion follows from this diagnostic alone. Preserve all
prompts, raw outputs, verifier verdicts, route decisions, hashes, and failures
locally and on the GPU. Phase 5 protected and Phase 6 holdouts remain closed;
their outputs are not training data. A new improvement attempt after Phase 7
needs its own fresh evaluation set.
