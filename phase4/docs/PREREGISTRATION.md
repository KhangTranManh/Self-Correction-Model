# Phase 4 preregistration

## Hypothesis

On a new protected, balanced benchmark, transition-reward GRPO starting from
`Kxck/Self_Correction_v1` will improve REVISE recall over Decision-Only V1
without reducing KEEP recall below the precommitted floor.

## Primary comparison

- Historical reference: Phase 3 Decision-Only V1.
- New candidate: Phase4 Transition-RL V1.
- The Phase 3 frozen 200 is historical context only and cannot select Phase 4
  data, rewards, checkpoints, thresholds, or hyperparameters.

## Required new data

- Training problems must be source-disjoint from every Phase 3 protected row.
- The final evaluation must be newly sampled, sealed before training, balanced
  by initial correctness and domain, and untouched until confirmation.
- Incorrect answers must be natural outputs from the current actor.
- Cached verifier labels must be freshly reconfirmed before use.

## Primary metrics and gate

- balanced decision accuracy >= 0.70;
- KEEP recall >= 0.75;
- REVISE recall >= 0.65;
- code accuracy does not regress against the frozen Phase 4 baseline;
- final verified accuracy exceeds initial verified accuracy;
- both KEEP and REVISE are emitted;
- the gate is evaluated once on the sealed confirmation set.

## Abort conditions

- correct-to-wrong transition rate exceeds 5% on development data;
- one decision exceeds 95% of development predictions;
- verifier mismatch or train/evaluation source overlap is detected;
- reward can be increased through formatting without changing verified outcome.

## Pilot

The first run is limited to 100–300 training problems and one bounded GRPO
cycle. It is a pipeline and direction check, not a promotion run.
