# Phase 6 - Signal Readout and Harness Confirmation

**Status: CLOSED (2026-09-24). No model, router, calibration, or threshold is
promoted.**

Phase 6 followed the completed Phase 5 diagnosis: frozen hidden-state probes
can decode a pre-hint correctness signal, but neutral model generation does not
reliably use it. The phase tested whether that gap could be narrowed by
verbalized confidence, additional model-visible review, soft hints, calibration,
or an external probe-to-repair harness. No model weights were trained.

## Final answer

The project has evidence of **error information**, but not reliable
model-only self-correction.

- Verbal confidence shares some ranking information with the frozen probe but
  is severely overconfident and is not a stable routing signal.
- More explicit reasoning, fuzzy error hints, and post-hoc confidence
  calibration did not create a safe correction policy.
- Frozen probes remain materially better detectors than verbal confidence.
- Repair is the binding limitation: on the fresh 40-source holdout, even an
  oracle that identifies every initially wrong answer yields only 1/20 verified
  repairs for the original solver and 2/20 for V2 and V3.

The frozen probe therefore remains an external diagnostic component, not proof
that any checkpoint autonomously detects, decides, and repairs its own errors.

## Evidence map

| Diagnostic | Locked scope | Main result |
|---|---|---|
| Verbal confidence and rationale | 16 Phase 5 development sources | Confidence correlates with probe ranks but is compressed; rationale did not improve repair. |
| Fuzzy external hints | 8 development sources | No safe operating point from 50% through 90%; V3 at 90% harmed correct answers without fixing a wrong answer. |
| Layer-path audit | Frozen Phase 5 development activations | Layer 14 is stronger than final hidden layer 28, indicating partial attenuation rather than disappearance. |
| Confidence calibration | Stored 16-source confidence outputs | Temperature scaling and isotonic regression fail leave-one-out generalization. |
| Detection/repair decomposition | Existing development and already-open protected artifacts | Status/probe detection does not remove the low repair ceiling. |
| Fresh harness confirmation | New 40-source balanced holdout | Probe recall is 65--75%, but final accuracy remains 50% because repair is weak and false-positive routes can harm correct answers. |

The detailed evidence and all result boundaries are in
[docs/RESULTS.md](docs/RESULTS.md). The definitive closure statement is in
[docs/FINAL_REPORT.md](docs/FINAL_REPORT.md).

## Repository layout

```text
phase6/
  configs/  frozen contracts for each diagnostic
  data/     selected manifests and immutable locks
  scripts/  preparation, GPU runtime, activation, and analysis CLIs
  docs/     closure, results, and code ownership
```

`outputs/phase6_*` contains raw generations, verifier records, activations, and
machine-readable reports. These outputs are audit evidence only and must never
enter training data. See [docs/CODEBASE.md](docs/CODEBASE.md) for file ownership
and reproduction order.

## Closure rules

1. Do not tune a prompt, threshold, layer, or calibration against any Phase 6
   holdout.
2. Do not train on Phase 6 raw outputs, selected sources, or locks.
3. Do not reopen the Phase 5 protected set through this phase.
4. Future work requires a new phase, a new hypothesis, and a newly frozen
   evaluation set.
