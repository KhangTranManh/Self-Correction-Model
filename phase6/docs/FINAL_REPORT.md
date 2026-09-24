# Phase 6 final report

**Status: closed on 2026-09-24.**

## Research question

Phase 5 established that a frozen hidden-state probe can decode a correctness
signal before review, while the model's neutral review generation does not use
that signal safely. Phase 6 asked whether the gap could be closed without
training by improving readout, elicitation, calibration, or external routing.

## Final answer

No tested mechanism turns the existing signal into reliable autonomous
self-correction. A frozen probe is a useful **external detector**, but the
model's repair policy remains too weak for a promoted correction system.

## Evidence

| Question | Result | Decision |
|---|---|---|
| Does verbal confidence expose the probe signal? | It is positively rank-correlated on a 16-source pilot, but concentrated near 85--100% correct and less predictive than the probe. | Do not use verbal confidence as a router. |
| Does model-visible reasoning unlock repair? | Step-by-step neutral review did not improve wrong-to-correct repair. | Rejected as a remedy. |
| Can a fuzzy external hint activate safe revision? | No useful point from 50% to 90%; V3 at 90% created harms with no fix. | No hint threshold promoted. |
| Is the signal absent near output? | Layer 14 exceeds final hidden layer 28, but the final layer stays above chance. | Partial attenuation only; no causal loss claim. |
| Can post-hoc confidence calibration solve routing? | Temperature scaling and isotonic regression fail leave-one-out generalization. | No calibration promoted. |
| Can detection be separated from repair? | Prior status/probe artifacts show low verified repair even for known errors. | Repair is an independent bottleneck. |
| Does a fresh end-to-end harness work? | On a locked new 40-source balanced holdout, probe recall is 65--75%, but final accuracy remains 50%. | Harness is not deployable. |

## Fresh end-to-end confirmation

The final diagnostic closes a specific gap in the earlier evidence: cached
status responses could not measure the effect of routing an initially correct
answer. A new 160-source pool excluded all Phase 5 splits, previous initial
rollouts, and prior Phase 6 pilots. The original solver's initial outputs then
selected a deterministic 40-source holdout: 20 initially correct and 20
initially wrong. Source pool, holdout, prior thresholds, and routes were locked
before recheck generation.

Non-oracle routes were told only that an external uncertainty monitor selected
the answer and that it might still be correct. Oracle-known-wrong is reported
separately and applies only to initially wrong rows.

| Checkpoint | Router | Wrong recall | Precision | Correct preservation | Wrong-to-correct | Correct-to-wrong | Final accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| Original solver | Frozen probe | 75.0% | 68.2% | 65.0% | 1/20 | 1/20 | 50.0% |
| Warm-start V2 | Frozen probe | 65.0% | 76.5% | 80.0% | 0/20 | 0/20 | 50.0% |
| Correction SFT V3 | Frozen probe | 75.0% | 65.2% | 60.0% | 0/20 | 0/20 | 50.0% |

Fixed calibrated verbal confidence is qualitatively unstable on the same
holdout: it routes 1/40 rows for the original solver, all 40 rows for V2, and
0/40 rows for V3. It is not a viable detector.

| Checkpoint | Oracle-known-wrong rows | Verified repairs |
|---|---:|---:|
| Original solver | 20 | 1 (5.0%) |
| Warm-start V2 | 20 | 2 (10.0%) |
| Correction SFT V3 | 20 | 2 (10.0%) |

This oracle result is the key boundary: even perfect knowledge of which answer
is wrong does not produce enough correct repairs. Better detection alone cannot
solve the current system.

## Model dispositions

- `Kxck/Self_Correction_v1` remains the historical solver/repair checkpoint;
  it is not promoted as an autonomous self-corrector.
- Phase 4 Warm-start V2 and Correction SFT V3 remain archived pilots only.
- Frozen Phase 5 probes remain diagnostic external readouts. Their thresholds
  are not promoted for deployment.
- No adapter, prompt, calibration, or harness is promoted from Phase 6.

## Closure and future work

Phase 6 data, locks, scripts, raw outputs, activations, and reports must be
preserved as negative-result evidence. They are not training data. No further
threshold tuning, prompt iteration, or evaluation may use the Phase 6 holdouts
or the Phase 5 protected set.

Any continuation must open a new phase with a new hypothesis and a new frozen
evaluation split. The appropriate research target is a demonstrably stronger
repair policy, evaluated separately from detection, before composing a
detector -> router -> repair -> verifier harness.
