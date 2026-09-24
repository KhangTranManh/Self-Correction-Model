# Phase 5 — guided repair and error-signal diagnosis

**Status: protected evaluation complete; no checkpoint promoted.**

The one protected opening completed after all prompts, probe choices and
controls were locked. V2 neutral review made 0/80 fixes and 2/80 harms; V3 made
4/80 fixes and 32/80 harms. With explicit error-status feedback, V2 made 5/80
fixes and V3 7/80, with no observed harms, but these are assisted rather than
autonomous results. Pre-hint probe balanced accuracy was 71.25% for the
original solver/V2 and 74.38% for V3. See [the final report](docs/FINAL_REPORT.md).

## What the result means

Phase 5 found **decodable correctness information**, not reliable autonomous
self-correction. The linear probe is an external diagnostic harness fitted on
frozen hidden states. It can classify correct versus wrong answers above the
controls, but the model's own neutral-review generation does not safely convert
that signal into a KEEP/REVISE decision and a successful repair.

Accordingly, the evidence supports three different claims:

- **Internal signal:** supported by protected pre-hint probe results.
- **Assisted repair:** supported only for V2 under explicit status feedback.
- **Autonomous self-correction:** not supported; neutral V2 made no fixes and
  neutral V3 caused substantially more harms than fixes.

A future model-plus-probe router would be an externally controlled correction
system unless the model itself learns to detect, preserve, and repair reliably
without being given the correctness label.

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
stopped at the first ordered prefix with 240 natural wrong answers. A
deterministic, source-disjoint 240/80/160 train/development/protected split is
now frozen under `data/splits/v1/`. The full initial audit and summary are
backed up under `../outputs/phase5_remote_v100/`.

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
behave. Their private Hub revisions, weight hashes, and parent metadata were
verified against the Phase 4 registry on 2026-09-24. A newer model would
require a separately frozen comparison.

Checkpoint sources, checked against the `Kxck` Hugging Face account on
2026-09-23:

| Phase 5 role | Exact source | Parent |
|---|---|---|
| Original solver | Hugging Face `Kxck/Self_Correction_v1` (full merged model) | — |
| Warm-start V2 | Private Hugging Face `Kxck/phase4_exploration_warmstart_v2` at revision `a8417cf1ed170881907910bba5ebb27c5a82bd6e` (LoRA) | `Kxck/Self_Correction_v1` |
| Correction SFT V3 | Private Hugging Face `Kxck/phase4_correction_sft_v3` at revision `2947ef2aeea951765434b2dd9db7f9bc7db8381d` (LoRA) | V2 merged into the original solver |

The two Phase 4 adapters were uploaded as private Hugging Face repos on
2026-09-23 and are now the canonical Phase 5 runtime sources. Their remote
weight SHA-256 hashes match the Phase 4 registry. Historical local paths are
retained only as provenance in `configs/experiments.yaml`; they are not required
for a new GPU. The older Hugging Face repos `Kxck/AGI_v2` and `Kxck/AGI_V3` are
different checkpoints and must not be substituted. The merged V2 parent must
be recreated on the GPU before loading V3. Exact revisions and hashes are
pinned in both Phase 5 configs.

The Phase 5 collector reads the exact environment variable `HF_TOKEN` from
the project-root `.env` on the machine where it runs. `.env` is gitignored and
excluded from transfer archives; place it separately on any new GPU if
authenticated Hugging Face access is needed. Never put token values in configs,
logs, or archives.

## Execution sequence

The source audit, natural-answer collection, and balanced 480-source split are
complete. The split contains 120 correct and 120 wrong sources in train, 40 and
40 in development, and 80 and 80 in protected test. It was selected by a
deterministic hash rule before any review or probe outcome. The 240 selected
wrong cases are all natural model errors; do not replace cases after seeing
hint coverage.

Hint audit V2 uses only literal, self-contained arithmetic equalities and does
not use reference answers or verifier traces for annotation. It found all 480
rows eligible for neutral/status, but only one wrong train row and zero wrong
development/protected rows eligible for location/type. Location/type is
therefore disabled rather than weakened or refilled. The V2/V3 Hub revisions,
parents, and hashes are verified. Prompts, FP16, greedy decoding, seed, metrics,
probe selection, and stop rules are frozen in `configs/review_protocol_v1.yaml`.
The initial collection's BF16 emulation remains historical provenance and is
not relabeled as FP16.

Paired neutral/status reviews, train/development probe selection, surface and
shuffle controls, and the single protected opening are complete. The evidence
shows linearly decodable correctness information without a safe generative
neutral-review policy. Phase 5 is therefore closed as a diagnostic rather than
a self-correction success. Further tuning requires a new preregistered
experiment; the protected set must not be reopened for selection.

## Repository layout

```text
phase5/
  README.md
  configs/experiments.yaml       # status and model identities
  configs/pilot_v1.yaml          # pilot protocol and pending decisions
  configs/review_protocol_v1.yaml # frozen prompts, decoding, metrics and probe rules
  data/splits/v1/                # frozen 240/80/160 source split
  data/hints/v2/                 # strict hint feasibility audit
  data/protocol/                 # immutable protocol lock and input hashes
  data/checkpoint_source_audit.json # metadata-only Hub verification
  data/README.md                 # source ownership and exclusions
  docs/PREREGISTRATION.md        # hypotheses, safeguards and decisions
  docs/EXECUTION_PLAN.md         # next steps and feasibility gates
  docs/CODEBASE.md               # code and evidence boundaries
```

CPU-only preparation and validation commands (neither loads model weights):

```bash
python phase5/scripts/validate_checkpoint_sources.py
python phase5/scripts/freeze_balanced_split.py  # refuses to overwrite the frozen split
```

See the [Phase 4 final report](../phase4/docs/FINAL_REPORT.md) for the evidence
that motivates this new phase.
