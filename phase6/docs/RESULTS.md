# Phase 6 detailed results

**Phase status: CLOSED (2026-09-24).** This file retains the detailed
diagnostic evidence. The canonical conclusion and closure rules are in
[FINAL_REPORT.md](FINAL_REPORT.md).

## Scope

The pilot used 16 deterministically hash-selected Phase 5 development sources:
8 initially correct and 8 initially wrong. It evaluated the original solver,
Warm-start V2, and Correction SFT V3 with vLLM 0.7.0 FP16 on a V100. The Phase 5
protected set was not read, no model weights changed, and no probe was refit.

Each checkpoint produced two independent continuations from the same problem
and initial answer:

1. an integer confidence from 0–100 that the initial answer was fully correct;
2. a neutral, model-visible step-by-step review followed by a final answer.

Generated review text is retained for audit and is not hidden chain-of-thought
or training data. Full raw prompts and outputs are under
`outputs/phase6_verbalization_pilot_v1/raw/`; the machine-readable analysis is
`outputs/phase6_verbalization_pilot_v1/report.json`.

## Confidence versus frozen probe

| Checkpoint | Valid | Spearman correlation | Verbal AUC | Probe AUC | Verbal Brier | Probe Brier |
|---|---:|---:|---:|---:|---:|---:|
| Original solver | 16/16 | 0.813 (`p=0.00013`) | 0.719 | 0.859 | 0.381 | 0.174 |
| Warm-start V2 | 16/16 | 0.581 (`p=0.018`) | 0.617 | 0.844 | 0.438 | 0.172 |
| Correction SFT V3 | 16/16 | 0.607 (`p=0.013`) | 0.727 | 0.859 | 0.414 | 0.183 |

The proposed null-correlation result did not occur. Verbalized uncertainty and
probe uncertainty share substantial rank information in this small pilot.
However, their scales differ sharply: mean verbalized wrong-probability was
only 5.3–10.6%, while mean probe wrong-probability was 66.2–69.0%. Most verbal
confidence values were 85, 95, or 100. The current evidence therefore favors
**shared ranking signal with a compressed, overconfident verbal channel**, not
two clearly independent pathways.

This remains a development-set diagnostic. Correlation does not establish that
the same internal direction causally produces the confidence report, and 16
sources are too few for a promotion claim.

## Step-by-step neutral review

| Checkpoint | Condition | Contract valid | Wrong→correct | Correct→wrong | Final accuracy |
|---|---|---:|---:|---:|---:|
| Original solver | Neutral direct | 14/16 | 0/8 | 0/8 | 43.8% |
| Original solver | Neutral + rationale | 0/16 | 1/8 | 4/8 | 31.2% |
| Original solver | Status direct | 13/16 | 0/8 | 0/8 | 50.0% |
| Warm-start V2 | Neutral direct | 15/16 | 0/8 | 1/8 | 43.8% |
| Warm-start V2 | Neutral + rationale | 4/16 | 0/8 | 2/8 | 37.5% |
| Warm-start V2 | Status direct | 15/16 | 0/8 | 0/8 | 50.0% |
| Correction SFT V3 | Neutral direct | 16/16 | 1/8 | 4/8 | 31.2% |
| Correction SFT V3 | Neutral + rationale | 14/16 | 0/8 | 0/8 | 50.0% |
| Correction SFT V3 | Status direct | 16/16 | 2/8 | 0/8 | 62.5% |

Because the original solver and V2 often ignored the requested XML wrapper,
rationale final correctness is conservatively verified from each complete raw
output. Contract validity is reported separately.

Step-by-step review did not improve wrong-to-correct repair for any checkpoint:
the original solver gained one fix but caused four harms, V2 caused two harms
without a fix, and V3 removed the four neutral-review harms but also removed its
single fix. V3's accuracy rose to 50% through correct-answer preservation, not
through error repair, and remained below the 62.5% status result.

## Disposition

- The pilot does not support the claim that explicit confidence and hidden
  correctness representations are uncorrelated.
- It does reveal a calibration/decision-use gap: confidence preserves some
  ranking but is strongly overconfident and less predictive than the probe.
- Additional generated reasoning does not currently unlock autonomous repair.
  Its main positive effect is reducing V3 over-revision on this subset.
- No checkpoint, prompt, threshold, or composite system is promoted.
- A confirmatory continuation needs a new, larger holdout and a prompt contract
  frozen before generation; these 16 development sources cannot serve as that
  confirmation.

## Fuzzy external-hint curve

A second, disjoint development pilot used eight sources (four initially
correct, four initially wrong). Each checkpoint received the identical
`P(error)=50/60/70/80/90%` grid for both correctness classes. This design tests
repair benefit and false-hint/sycophancy harm at the same time. The prompt,
manifest, and hashes were frozen before generation; the protected set remained
closed.

| Checkpoint | 50--80% behavior | 90% wrong→correct | 90% correct→wrong | Safe threshold |
|---|---|---:|---:|---|
| Original solver | KEEP on all 32 rows | 0/4 | 0/4 | none |
| Warm-start V2 | KEEP on all 32 rows | 0/4 | 0/4 | none |
| Correction SFT V3 | KEEP on all 32 rows | 0/4 | 2/4 | none |

At 90%, V3 issued REVISE for one wrong and two correct answers. The attempted
wrong-answer repair remained wrong, while both correct answers became wrong.
Thus the strongest fuzzy hint activated a revision bias before it produced any
verified repair benefit. The exact-status comparator also did not fix any of
these four wrong sources, so this tiny subset is a difficult repair sample as
well as a harness-sensitivity test.

No minimum useful external-signal threshold was found. This does not establish
that no such threshold exists: each class has only four sources. It does show
that, under the frozen prompt, increasing the hint to 90% is not a safe bridge
from the decoded representation to correction behavior, especially for V3.
Full rows are in `outputs/phase6_fuzzy_hint_pilot_v1/raw/`; the aggregate report
is `outputs/phase6_fuzzy_hint_pilot_v1/report.json`.

## Layer-path audit

The layer audit reused the existing 80-row Phase 5 development activations and
recorded logistic-probe candidates. It did not refit a probe. Holding
regularization fixed at `C=0.01` isolates the layer comparison:

| Checkpoint | Layer 14 BA / AUC | Layer 28 BA / AUC | Δ BA | Δ AUC |
|---|---:|---:|---:|---:|
| Original solver | 0.788 / 0.871 | 0.750 / 0.826 | -0.038 | -0.046 |
| Warm-start V2 | 0.775 / 0.870 | 0.762 / 0.825 | -0.013 | -0.045 |
| Correction SFT V3 | 0.762 / 0.866 | 0.738 / 0.836 | -0.025 | -0.029 |

All three checkpoints peak at layer 14 and lose ROC-AUC toward layer 28. The
mean loss is 2.5 balanced-accuracy points and 4.0 ROC-AUC points. However,
layer 28 remains strongly above chance. The evidence therefore supports a
mid-layer peak followed by **partial attenuation**, not a complete loss of the
correctness representation.

Layer 28 here is the final hidden-state probe position, not the vocabulary
logit readout. A teacher-forced KEEP/REVISE logit-lens experiment would require
a separate frozen GPU protocol. The current result is correlational layer-wise
evidence and must not be described as direct causal proof of a lost
know-to-say pathway.

## Verbal-confidence calibration

The stored 16-source confidence outputs were calibrated on CPU against the
deterministic verifier labels. The input score was the log-odds of the model's
reported probability that its initial answer was correct. Prior vLLM runs did
not store vocabulary-token logits, so this is a calibration of verbalized
confidence, not a direct KEEP/REVISE logit calibration.

| Checkpoint | Fitted T | Raw Brier | Temperature LOO Brier | Isotonic LOO Brier | Constant 50% Brier |
|---|---:|---:|---:|---:|---:|
| Original solver | 8.490 | 0.379 | 0.265 | 0.316 | 0.250 |
| Warm-start V2 | 26.298 | 0.436 | 0.282 | 0.316 | 0.250 |
| Correction SFT V3 | 11.130 | 0.414 | 0.277 | 0.307 | 0.250 |

The large temperatures compress the reported 85--100% confidence values toward
50%, confirming that overconfidence is the dominant calibration defect.
In-sample temperature scaling lowered NLL and Brier, and in-sample isotonic
regression reached zero ECE. Those apparent gains did not survive
leave-one-out evaluation: neither method beat the constant 50% Brier baseline
for any checkpoint. Isotonic was especially unstable because each model has
only 16 rows and three or four distinct confidence values.

This pilot therefore does not support the claim that a readout-only calibration
is sufficient. Temperature scaling is monotone and cannot create missing rank
information or move an answer across the 0.5 side of its original logit. Direct
decision-token logit calibration remains a separate experiment requiring a new
forward pass with logprobs enabled. See
`outputs/phase6_confidence_calibration_v1/report.json` for all metrics and
per-source leave-one-out predictions.

## Detection-router-repair decomposition

The final CPU-only analysis separates three questions rather than treating
correction as one score. It uses only stored development artifacts plus the
already-open Phase 5 protected artifacts; no model, prompt, probe, or protected
set was newly run.

### 1. Can calibrated verbal confidence decide when to recheck?

Applying thresholds to leave-one-out calibrated verbal confidence produces a
smooth routing curve only by external construction. It does not demonstrate a
smooth generative response. For example, at `P(wrong) >= 0.50`, the
temperature-calibrated verbal router routes 1/16 rows for the original solver
and 0/16 for V2/V3. Lowering the threshold routes many correct answers:
at 0.40, wrong recall is 50--88% but correct preservation is 0--75% across the
three checkpoints. Given the preceding leave-one-out calibration failure, no
verbal-confidence threshold is selected or promoted.

### 2. Can the model repair if it is told exactly that an answer is wrong?

This is an oracle detector condition: status explicitly states that the answer
is wrong, so it bypasses detection completely.

| Checkpoint | Protected valid REVISE / known-wrong | Protected verified repairs / valid REVISE |
|---|---:|---:|
| Original solver | 12/80 | 0/12 (0.0%) |
| Warm-start V2 | 77/80 | 5/77 (6.5%) |
| Correction SFT V3 | 79/80 | 7/79 (8.9%) |

V2 and V3 comply with REVISE nearly every time once status reveals an error,
but their repaired answer is still wrong roughly 91--94% of the time. This is
direct evidence of a separate solver/repair bottleneck, not only a calibration
or detection problem.

### 3. Does a frozen probe make a usable detector-to-repair harness?

At its locked 0.5 threshold on the 80-row development split, the raw frozen
probe routes 32 true-positive wrong rows for the original solver, 31 for V2,
and 32 for V3; it also creates 9, 9, and 10 false-positive routes respectively.
For the true positives, the matching cached status outputs provide an exact
repair measurement:

| Checkpoint | True-positive routes | Verified repairs among those routes |
|---|---:|---:|
| Original solver | 32 | 0 (0.0%) |
| Warm-start V2 | 31 | 2 (6.5%) |
| Correction SFT V3 | 32 | 5 (15.6%) |

The raw logistic probe probabilities outperform train-fit temperature and
isotonic post-hoc calibration on development; neither post-hoc calibrator is
promoted. Cached status outputs for initially correct rows say “no error was
found,” so they cannot simulate how the solver would react to a probe false
positive that says REVISE. Consequently this is a **partial harness analysis**,
not a deployable end-to-end result.

The pragmatic future system remains transparent rather than “self”:

```text
frozen probe detector -> external router -> repair model -> deterministic verifier
```

To evaluate it properly, freeze a new non-protected split and run the same
router-triggered REVISE prompt for both true and false positives. That one
generation run is necessary to measure end-to-end harms and final accuracy.
The full CPU report is
`outputs/phase6_router_repair_decomposition_v1/report.json`.

## Fresh end-to-end harness confirmation

This confirmation closes the prior false-positive gap without reopening the
Phase 5 protected set. A 160-source fresh candidate pool was filtered to
exclude every Phase 5 split, prior rollout source, and earlier Phase 6 pilot.
An original-solver initial rollout then deterministically selected a balanced
40-source holdout (20 verifier-correct and 20 verifier-wrong initial answers).
The source pool, holdout, fixed thresholds, and resulting routes were all
locked before recheck generation.

For every selected row, the recheck prompt stated only that an external
uncertainty monitor had selected it and that it might still be correct. It did
not reveal the verifier label. Oracle-known-wrong is separate and is applied
only to initially wrong rows to measure the repair ceiling. All results are
deterministically verifier-scored and the raw prompt/output records are saved.

| Checkpoint | Router | Routed | Wrong recall | Precision | Correct preservation | Wrong-to-correct | Correct-to-wrong | Final accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Original solver | Self confidence | 1/40 | 5.0% | 100.0% | 100.0% | 0/20 | 0/20 | 50.0% |
| Original solver | Frozen probe | 22/40 | 75.0% | 68.2% | 65.0% | 1/20 | 1/20 | 50.0% |
| Warm-start V2 | Self confidence | 40/40 | 100.0% | 50.0% | 0.0% | 1/20 | 0/20 | 52.5% |
| Warm-start V2 | Frozen probe | 17/40 | 65.0% | 76.5% | 80.0% | 0/20 | 0/20 | 50.0% |
| Correction SFT V3 | Self confidence | 0/40 | 0.0% | n/a | 100.0% | 0/20 | 0/20 | 50.0% |
| Correction SFT V3 | Frozen probe | 23/40 | 75.0% | 65.2% | 60.0% | 0/20 | 0/20 | 50.0% |

| Checkpoint | Oracle known-wrong rows | Verified repairs | Repair rate |
|---|---:|---:|---:|
| Original solver | 20 | 1 | 5.0% |
| Warm-start V2 | 20 | 2 | 10.0% |
| Correction SFT V3 | 20 | 2 | 10.0% |

The probe remains materially better than verbal confidence at locating wrong
answers, but it does not turn into final accuracy because re-solving rarely
repairs a routed error and sometimes harms a routed correct answer. The oracle
condition independently confirms the repair ceiling: knowing exactly which
answer is wrong only produces 5--10% verified repair. Therefore this result
does not support a model-only self-correction claim or promotion of a router,
threshold, or checkpoint. It supports the narrower claim that a frozen probe
can be an external detector in a future detector -> router -> repair ->
verifier system, subject to further repair-model work.

Artifacts: `phase6/data/harness_confirmation_source_pool_v1*.json`,
`phase6/data/harness_confirmation_holdout_v1*.json`,
`phase6/data/harness_confirmation_routes_v1*.json`, and
`outputs/phase6_harness_confirmation_v1/`. The machine-readable aggregate is
`outputs/phase6_harness_confirmation_v1/report.json` and row-level routed
outcomes are in `analysis_audit.jsonl`.
