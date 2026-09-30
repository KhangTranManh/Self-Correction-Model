# Phase 8 — reproducibility, distractor, and probe-routed blind re-solve

**Status: completed on 2026-09-30 under execution-lock amendment v2. The
preregistered end-to-end gate passed for all three checkpoints; no checkpoint
is promoted.** Read the [final report](docs/FINAL_REPORT.md) for results and
limits.

Phase 7 showed that blind re-solving fixed far more initially wrong answers
than re-solving with the earlier answer visible. Phase 8 tested, on 400 fresh
protected GSM8K sources:

1. how much a plain second sample or greedy decoding already helps;
2. whether the model's own earlier answer hurts more than a length-matched
   irrelevant wrong answer; and
3. whether the frozen Phase 5 probe can route blind re-solves profitably
   without an oracle.

## Result in brief

- A second attempt alone raises the original solver from 70.75% to about 77%.
- Showing any candidate answer sharply reduces repair. An irrelevant wrong
  answer explains most of the drop; the extra penalty for the model's *own*
  answer (9–12 points) is not significant after Holm correction.
- Probe → blind re-solve beats KEEP-all for every checkpoint (+3.75, +4.25,
  and +6.75 points; all Holm p ≤ 0.015). Blind re-solving every answer is at
  least as accurate, so the probe mainly saves compute.

The working system is an external harness, not evidence that the standalone
model detects and repairs its own errors.

## Documents

- [Preregistration](docs/PREREGISTRATION.md) — hypotheses, controls, and gates
- [Final report](docs/FINAL_REPORT.md) — protected results and disposition
- [Run record](RUN_STATUS.md) — host history, amendment v2, and deviations
- [Codebase](docs/CODEBASE.md) — scripts, data, and reproduction order

## Data boundaries

`data/fresh_source_pool_v1/` holds the 400 protected sources (manifest
SHA-256 `a4d2829140292cce1b86ccf3c758e7b30641ee829452698a9ca139462eae7e24`),
selected before any answer existed and disjoint from Phases 1–7. The pool
is now opened. Do not use it, its outputs, or the frozen distractor map to
tune prompts, thresholds, seeds, or models, and never train on it. A
continuation needs a new phase with a fresh holdout.
