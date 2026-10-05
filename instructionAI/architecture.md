# Project Architecture

This document defines ownership and dependency boundaries for the current
repository. The root README routes current work. Phase 3 documentation owns
its closed results; Phase 4 documentation owns its closed correction pilots.

## Phase boundaries

```text
Phase 1 verified solver
        │
        ├── Kxck/Self_Correction_v1 (base solver/repair model)
        │
        ▼
Phase 3 immutable Base/V1 attempts
        │
        ├── verifier-labelled source inventory
        ├── router training datasets
        └── protected frozen evaluations
                │
                ▼
        Decision-Only V1 router
                │
                ├── DPO Pilot V1 (not promoted)
                └── DPO Semantic V2 (not promoted)
```

Phase 2 is a separate KTO exploration. Its data and scripts are not imported by
the canonical Phase 3 pipeline.

## Top-level ownership

| Path | Owner | Purpose |
|---|---|---|
| `phase1/` | Phase 1 | Verified solve/correct pipeline and historical results |
| `phase2/` | Phase 2 | KTO preference experiment |
| `phase3/` | Phase 3 | Closed discrimination/selective-repair research package |
| `phase4/` | Phase 4 | Closed selective-correction pilots and immutable evidence |
| `phase5/` | Phase 5 | Complete guided-repair and pre-hint probe diagnostic; no promotion |
| `phase6/` | Phase 6 | Closed signal-readout and harness diagnostic; no promotion |
| `phase7/` | Phase 7 | Completed paired blind/answer-visible re-solving diagnostic |
| `phase8/` | Phase 8 | Completed resampling, distractor, and probe-routed blind re-solve study |
| `phase9/` | Phase 9 | Closed voting and self-check study; voting passed, self-check failed |
| `phase10/` | Phase 10 | Closed judge-LoRA training study; negative, adapter not promoted |
| `phase11/` | Phase 11 | Closed DPO judge study; negative, adapter not promoted |
| `serving/` | Project | vLLM serving profiles for all checkpoints and the strategy client |
| `outputs/` | Runtime | Local adapters, logs, activations; excluded from Git |
| `instructionAI/` | Project | Cross-phase architecture and data invariants |

## Canonical Phase 3 layers

### `phase3/lib/`

Reusable implementation primitives only:

- provenance resolution;
- prompt construction;
- math/code verification;
- APPS sandbox execution.

Libraries must not select experiment samples or contain hard-coded run paths.

### `phase3/scripts/data/`

CPU-oriented acquisition and deterministic dataset builders. Builders may read
immutable attempts and verifier metadata, but may not read frozen model outputs
to choose training samples.

### `phase3/scripts/training/`

GPU training entry points. All hyperparameters and artifact locations come from
YAML configs. Training scripts do not silently rebuild datasets.

### `phase3/scripts/evaluation/`

Frozen behavioral evaluation, model comparison, activation extraction, and
linear probes. Evaluation outputs belong under `phase3/runs/` or a clearly named
runtime output directory.

### `phase3/scripts/serving/`

vLLM launchers only. Serving is operational and never changes model weights.

### `phase3/configs/`

- one YAML file per training run;
- `experiments.yaml` as the canonical model/status registry;
- no secrets or provider passwords.

### `phase3/data/`

```text
source/ and apps_pilot/candidates/    original problem records
attempts/ and apps_pilot/attempts/    immutable natural model generations
behavior/ and buckets/                compact provenance/index metadata
decision_only/                        Decision-Only V1 SFT data
revised_router/                       Router V3 experimental data
contrastive_pairs/                    first 120-pair DPO source
dpo_pilot/                            DPO Pilot V1 preferences
semantic_model_dpo/                   100 model-vs-model semantic pairs
two_stage_selective_repair/           protected frozen evaluation manifest
```

### `phase3/runs/`

Research evidence: raw decisions, model-visible rationales, verifier results,
metrics, and generated reports. These files are not training inputs.

## Model roles

| Model | Parent | Role | Status |
|---|---|---|---|
| `Kxck/Self_Correction_v1` | Phase 1 model | Solver and repair model | canonical solver |
| Decision-Only V1 | Self_Correction_v1 | KEEP/REVISE router | canonical router |
| Router V3 mini | Decision-Only V1 | REVISE-heavy router experiment | diagnostic only |
| DPO Pilot V1 | Decision-Only V1 | First decision-token DPO | not promoted |
| DPO Semantic V2 | Decision-Only V1 | Model-vs-model semantic DPO | not promoted |

LoRA adapters are not standalone models. Serving or extraction loads
`Kxck/Self_Correction_v1` first, then attaches the selected adapter.

## Artifact lifecycle

```text
source problem
  → natural Base/V1 attempt
  → deterministic verification
  → immutable inventory/bucket
  → deterministic dataset builder
  → reviewed YAML config
  → adapter training
  → immediate local backup + SHA-256
  → frozen behavioral evaluation
  → representation probe
  → promote or reject in experiments.yaml
```

No later result may retroactively change an earlier frozen manifest. A new
manifest or experiment version is required.

## Phase 5 boundary

Phase 5 has 1,200 checked, source-disjoint arithmetic candidates and a
completed original-solver pass over the first 815 (575 correct, 240 wrong).
The append-only initial audit, derived rollouts, and summary are backed up
locally under `outputs/phase5_remote_v100/` and on the V100 GPU. The run
recorded BF16 software emulation, not native BF16 or FP16. The balanced
240/80/160 source split is frozen and the private V2/V3 Hub sources are pinned
and hash-verified. Guided review and pre-hint probe evaluation are complete.
Protected neutral review found no safe autonomous gain: V2 had 0 fixes/2 harms
and V3 had 4 fixes/32 harms. Pre-hint correctness remained linearly decodable
(71.25% to 74.38% protected balanced accuracy), showing a policy-use problem
rather than absence of internal signal. No training ran and no checkpoint was
promoted.

The Phase 5 linear probe is an external harness over frozen activations. Its
result supports the existence of accessible correctness information, not a
claim that any standalone checkpoint performs self-correction. Keep these
layers distinct in future reports:

```text
frozen model representation -> external probe/controller -> optional repair
```

Only an end-to-end model that detects errors without a correctness label,
preserves correct answers, and produces verified repairs should be described as
autonomous self-correction. A probe-triggered pipeline is harness-controlled
unless separately proven otherwise.

Hint eligibility and the review/probe protocol remain frozen. Neutral/status cover
all 480 rows; location/type is disabled because strict wrong-case coverage is
1/120 train and 0 in development/protected.
Location/type hints have a documented timing deviation from the earlier
selection rule and remain disabled. `phase5/docs/FINAL_REPORT.md` owns the
result; Phase 3/4 evidence remains closed and unchanged.

## Phase 6 closure boundary

Phase 6 uses a deterministic 16-source subset of the already-open Phase 5
development split. It does not touch the protected set. The first pilot found
positive rank correlation between explicit confidence and frozen-probe scores,
but a large calibration gap, and found no wrong-to-correct benefit from an
explicit model-visible step-by-step review. These results are exploratory;
runtime generations remain outside training data and no model is promoted.

The follow-up fuzzy-hint experiment uses a separate eight-source development
subset and supplies identical 50--90% error-probability hints to correct and
wrong answers. It found no useful operating point: V3's first response at 90%
caused two verified harms and no fixes. This is evidence of harness sensitivity,
not autonomous detection.

The layer-path audit reuses frozen Phase 5 activations. At fixed probe
regularization, layer 14 outperforms final hidden layer 28 for all three
checkpoints (mean ROC-AUC delta -0.040), but the final layer stays above chance.
Document this as partial attenuation of linearly decodable correctness signal.
Do not call it signal disappearance or a logit-layer result: no vocabulary-logit
readout was measured.

CPU-only temperature scaling and isotonic regression were subsequently fitted
to the 16 stored verbal confidence values per checkpoint using verifier labels.
Large fitted temperatures confirm overconfidence, but held-out Brier scores do
not beat a constant 50% predictor and isotonic regression overfits. Treat this
as a rejection of verbal-confidence calibration as a sufficient solution on
current evidence. It is not a test of direct vocabulary logits, which were not
saved by the earlier vLLM generation.

The final decomposition distinguishes detection from repair. On the already
opened protected artifact, explicit wrong-status yields valid REVISE compliance
for V2/V3 but only 5/77 and 7/79 verified repairs respectively. The frozen
linear probe is a stronger detector than verbal confidence, but cached
true-positive routes still repair only 2/31 V2 and 5/32 V3 development rows.
This supports an honest external architecture (probe -> router -> repair ->
verifier), while also showing that repair is independently inadequate. It is
not a deployable end-to-end harness result because the prior cache has no
matching repair generations for probe false positives. Evaluate that only on a
new, frozen, non-protected split.

That fresh confirmation is complete: it used a newly filtered 160-source pool
to select a locked balanced 40-source holdout, and applied the same non-oracle
recheck prompt to both frozen-probe true positives and false positives. The
probe detects 65--75% of initial errors across original/V2/V3, but correct
preservation is 60--80% and end-to-end final accuracy is 50%. Fixed calibrated
verbal confidence is unstable (routes 1/40, 40/40, and 0/40 respectively).
Most importantly, oracle-known-wrong repair is only 1/20, 2/20, and 2/20.
Therefore detection is real but repair is the binding bottleneck. This is a
verified external harness result, not evidence of autonomous self-correction;
do not promote a threshold, router, or checkpoint from it.

Phase 6 is closed as of 2026-09-24. Preserve its frozen configs, manifests,
locks, scripts, and raw outputs as reproducible negative-result evidence.
Neither Phase 6 holdouts nor the Phase 5 protected set may be used for further
tuning. Any follow-up must open a new phase with a distinct repair hypothesis
and a newly frozen evaluation split. The canonical closure is
`phase6/docs/FINAL_REPORT.md`.

Phase 7 is a completed diagnostic. It compared a new single-turn solution
with only the original problem against the same re-solving task with the
earlier answer visible as an unverified candidate, on fresh Phase 1-6-disjoint
sources, with every final answer verified deterministically. Its protected set
is opened. `phase7/docs/FINAL_REPORT.md` owns the result.

## Phase 8 boundary

Phase 8 completed on 2026-09-30 under execution-lock amendment v2
(`phase8/data/execution_lock_v2.json`). Its 400-source protected pool is
opened. The only supported system claim is an external harness: the frozen
Phase 5 probe (read from `outputs/phase5_gpu_vllm/probe_v1/selection/`, hash
verified) flags first answers, and flagged answers are replaced by a blind
re-solve that never sees the old answer. Blind re-solving every answer is at
least as accurate, so no checkpoint, probe, or threshold is promoted. Phase 8
outputs are evidence only and must never enter training. See
`phase8/docs/FINAL_REPORT.md`.

## Phase 9 and Phase 10 boundaries

Phase 9 (closed 2026-10-01) compared voting over independent attempts with a
model self-check of two conflicting solutions on 400 fresh protected sources.
Voting over five attempts passed (+8 to +10.5 points); the self-check failed.
Phase 10 (closed 2026-10-01) trained a LoRA judge (`phase10-judge`) on the
original solver from the model's own verifier-correct judgments; on a
300-source holdout its choice between two solutions stayed at chance and it
trailed voting. Both protected sets are opened. The judge adapter is kept for
reproducibility only. See `phase9/docs/FINAL_REPORT.md` and
`phase10/docs/FINAL_REPORT.md`.

Phase 11 (closed 2026-10-02) trained a DPO LoRA (`phase11-dpo-judge`) on
contrasting judgments of the same pair; the preference was learned in
likelihood but did not change choices on a 234-source holdout, and the judge
favors the second-shown solution. See `phase11/docs/FINAL_REPORT.md`.

**Harness versus model.** Every accuracy gain established so far (probe
routing, blind re-solving, voting) is an external harness around an unchanged
model. No training intervention has yet changed the model's own ability to
recognize which of its answers is right. Report harness and model-level
results separately.

## Serving

`serving/serve.sh` serves one base model per GPU: profile `original`
(`original-solver` plus whichever of the `warmstart-v2`, `phase10-judge`, and
`phase11-dpo-judge` LoRAs exist on the host) or profile `v3`
(`warmstart-v2-merged`, `correction-sft-v3` LoRA). `serving/client.py` runs the
`single`, `vote`, and `self_check` strategies with the exact experimental
prompts. Phase 3 routers keep their own launcher in `phase3/scripts/serving/`.

Phase 3 is closed as of 2026-09-13. Its canonical and negative-result artifacts
remain immutable. New discrimination research must create a new phase rather
than adding rows, adapters, or tuned thresholds to Phase 3.

## Security and operational rules

Phase 4 is closed as of 2026-09-15. No reliable correction gain was established
and no compliant confirmation was run. Warm-start V2 remains an archived pilot
reference, not a promoted canonical model. Final dispositions live in
`phase4/configs/experiments.yaml`; ownership and closure rules live in
`phase4/docs/CODEBASE.md` and `phase4/docs/FINAL_REPORT.md`. Preserve historical
configs and generated evidence. Any new research must open a separate phase.

- Make and check implementation changes locally before uploading to the GPU.
- Use Luna or Terra for routine log monitoring and straightforward checks when
  delegation is available; retain experiment design and result decisions with
  the main agent, as requested by the user.
- Use CPython 3.10/SymPy 1.14.0 for Phase 4 verification on both machines;
  Python-version differences can change parsing of trailing currency symbols.

- Never inspect, copy, document, or commit `.env` values.
- Never store SSH passwords in scripts, reports, or configs.
- vLLM binds to `127.0.0.1` by default; public binding requires an explicit
  `PHASE3_HOST=0.0.0.0` choice and should be protected externally.
- Stop GPU processes or the rented instance when work is complete.
- Download adapters and small reports immediately after each stage; do not wait
  for all downstream evaluation to finish.
