# Phase 5 — guided repair and error-signal diagnosis

**Status: Initial GPU answer collection complete (815 answers: 575 correct, 240 wrong). Source split, guided reviews, probes, and protected evaluation have not run.**

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

Read the [next-stage execution plan](docs/EXECUTION_PLAN.md),
[preregistration](docs/PREREGISTRATION.md), and
[protocol](configs/pilot_v1.yaml) before generating review or probe outcomes.

The three **project checkpoints**, chosen for diagnostic contrast, are:

| Name | Role | Historical Phase 4 observation |
|---|---|---|
| `Kxck/Self_Correction_v1` | Original solver and repair baseline | Autonomous review remained weak |
| Warm-start V2 | Retained pilot reference | 74% development result did not reproduce (73% rerun) |
| Correction SFT V3 | More revision behavior | 3 fixes and 2 harms on the 100-row development set |

They share a lineage. Results cannot establish how unrelated current models
behave. Model paths and hashes must be verified from the local Phase 4 registry
before execution. A newer model would require a separately frozen comparison.

Checkpoint sources, checked against the `Kxck` Hugging Face account on
2026-09-23:

| Phase 5 role | Exact source | Parent |
|---|---|---|
| Original solver | Hugging Face `Kxck/Self_Correction_v1` (full merged model) | — |
| Warm-start V2 | Local `outputs/phase4_exploration_warmstart_v2/final_adapter` and private Hugging Face `Kxck/phase4_exploration_warmstart_v2` (LoRA) | `Kxck/Self_Correction_v1` |
| Correction SFT V3 | Local `outputs/phase4_correction_sft_v3/final_adapter` and private Hugging Face `Kxck/phase4_correction_sft_v3` (LoRA) | V2 merged into the original solver |

The two Phase 4 adapters were uploaded as private Hugging Face repos on
2026-09-23. Their remote weight SHA-256 hashes match the local files and Phase 4
registry. The older Hugging Face repos `Kxck/AGI_v2` and `Kxck/AGI_V3` are
different checkpoints; do not substitute them. The merged V2 parent must be
recreated on the GPU before loading V3. The upload procedure is recorded in
`scripts/ops/upload_phase4_adapters.py` and the uploaded model cards are in
`docs/model_cards/`.

The Phase 5 collector reads the exact environment variable `HF_TOKEN` from
the project-root `.env` on the machine where it runs. `.env` is gitignored and
excluded from transfer archives; place it separately on any new GPU if
authenticated Hugging Face access is needed. Never put token values in configs,
logs, or archives.

## Execution sequence

The source audit and natural-answer collection are complete. Next, freeze a
balanced 480-source split from the 815 answers: 120 correct and 120 wrong in
train, 40 and 40 in development, and 80 and 80 in protected test. Select by a
deterministic rule before seeing review or probe outcomes. The 240 wrong cases
are all natural model errors; do not replace cases after seeing hint coverage.

Audit whether a model error can be located and described truthfully before
using location/type hints. The earlier selection rules required the alignment
rule before the first GPU pass, but it was not frozen then. Document this
deviation and treat current location/type analyses as exploratory unless a new
preregistered holdout is created. Neutral/status reviews and the pre-hint probe
can proceed on the frozen split. Verify the V2/V3 adapter parents and hashes,
then lock prompts, decoding, dtype, seeds, metrics, and stop rules before any
further GPU call. The initial run used emulated BF16 on the V100, as recorded;
do not describe those answers as FP16.

Run paired reviews across the three checkpoints with the same initial answer,
then fit probes on train, select on development, and evaluate protected data
once. Report fixes, harms, contract validity, hint coverage, probe controls,
and uncertainty. The [execution plan](docs/EXECUTION_PLAN.md) gives the
artifact and decision gates. Steps through protocol freeze are local CPU work;
the GPU is needed again for review and activation extraction. Phase 3 and
Phase 4 remain closed.

## Repository layout

```text
phase5/
  README.md
  configs/experiments.yaml       # status and model identities
  configs/pilot_v1.yaml          # pilot protocol and pending decisions
  data/README.md                # source ownership and exclusions
  docs/PREREGISTRATION.md       # hypotheses, safeguards and decisions
  docs/EXECUTION_PLAN.md        # next steps and feasibility gates
  docs/CODEBASE.md               # code and evidence boundaries
```

See the [Phase 4 final report](../phase4/docs/FINAL_REPORT.md) for the evidence
that motivates this new phase.
