# Phase 5 — guided repair and error-signal diagnosis

**Status: Initial GPU answer collection complete (815 answers: 575 correct, 240 wrong). Guided reviews, probe fits, and confirmation have not run.**

Phase 4 closed without a reliable autonomous correction gain. Phase 5 asks a
narrower question: for short, verifiable arithmetic solutions, does modest
guidance help an existing checkpoint repair a natural mistake, and is an
error signal present in the model before any hint is given?

The experiment has two linked parts. First, compare neutral review with
guidance about error presence, location, and type. Second, probe frozen hidden
states from the same three checkpoints to test whether correct answers and
natural wrong answers can be distinguished before the model responds.
A local source audit excluded 4,000 prior-phase GSM8K questions and found
3,174 short arithmetic problems passing the reference-step rules. It selected
1,200 by a fixed hash order. All 1,200 reference answers passed fresh
verification under CPython 3.10; no prior-phase source overlap was found.
See [the candidate report](data/CANDIDATE_REPORT.md). The bounded GPU pass
stopped at the first ordered prefix with 240 natural wrong answers; the
source-disjoint correct/wrong split has not been frozen yet. The full initial
audit and summary are backed up under `../outputs/phase5_remote_v100/`.

Read [the preregistration](docs/PREREGISTRATION.md) and
[protocol](configs/pilot_v1.yaml) before collecting model data.

The three **project checkpoints**, chosen for diagnostic contrast, are:

| Name | Role | Historical Phase 4 observation |
|---|---|---|
| `Kxck/Self_Correction_v1` | Original solver and repair baseline | Autonomous review remained weak |
| Warm-start V2 | Retained pilot reference | 74% development result did not reproduce (73% rerun) |
| Correction SFT V3 | More revision behavior | 3 fixes and 2 harms on the 100-row development set |

They share a lineage. Results cannot establish how unrelated current models
behave. Model paths and hashes must be verified from the local Phase 4 registry
before execution. A newer model would require a separately frozen comparison.

## Research sequence

1. Inventory sources and checkpoint provenance; exclude every Phase 1–4 train,
   development, probe, and confirmation source from Phase 5 evaluation.
2. Freeze an arithmetic task definition, source splits, hint rules, verifier,
   sample budget, prompts, metrics, and analysis before generating outcomes.
3. Collect one natural initial answer per problem using the common original
   solver. All three checkpoints review the same answer.
4. Run the hint comparison on fixed development data. Truthful hints are
   constructed independently of each reviewing model's output.
5. Extract frozen, pre-hint representations. Fit simple probes on probe-train,
   choose layers on probe-dev, and evaluate once on the protected probe-test.
6. Report paired answer transitions, harms, per-hint outcomes, and probe
   controls. Open a separately sealed confirmation split once after choices
   are frozen, if all feasibility gates pass.

No GPU has been assigned. GPU-dependent work starts only after the user gives a
new machine. Code, configs and result definitions are created and checked
locally first, then uploaded. Phase 3 and Phase 4 remain closed and unchanged.

## Repository layout

```text
phase5/
  README.md
  configs/experiments.yaml       # status and model identities
  configs/pilot_v1.yaml          # frozen proposal; no results
  data/README.md                # source ownership and exclusions
  docs/PREREGISTRATION.md       # hypotheses, safeguards and decisions
  docs/CODEBASE.md               # planned code and evidence boundaries
```

See the [Phase 4 final report](../phase4/docs/FINAL_REPORT.md) for the evidence
that motivates this new phase.
