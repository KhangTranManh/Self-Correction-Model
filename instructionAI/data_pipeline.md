# Verified Data Pipeline

## Phase 7 and Phase 8 boundaries

Phase 7 and Phase 8 are complete, and both protected sets are opened. Phase 8
drew 400 fresh GSM8K sources disjoint from every Phase 1-7 inventory and
regenerated a Phase 7 donor pool (`outputs/phase7_initials_regen_v2/`) for
distractor text only. No Phase 7 or Phase 8 source, answer, probe score, or
distractor may enter training or selection. See `phase7/docs/FINAL_REPORT.md`
and `phase8/docs/FINAL_REPORT.md`.

The Phase 7 blind re-solve experiment audited fresh problems against every recorded Phase 1-6 source,
including the entire Phase 5 selected candidate manifest and Phase 6
confirmation source pool. Natural first answers come from the original solver;
the paired second-pass blind prompt contains only the original problem and
ordinary solve instruction. The old answer stays in harness metadata for
comparison and deterministic verification, never in the blind prompt.

The answer-visible paired arm may show the old answer as an unverified
candidate. Both arms use the same source, checkpoint, decoding settings, and
verifier. A frozen probe may route answers only if its original artifact is
recovered and verified. No Phase 5 protected or Phase 6 holdout source may be
reused, and Phase 7 results cannot enter training. See
`phase7/docs/PREREGISTRATION.md` and `phase7/docs/EXECUTION_PLAN.md`.

> Phase 3 closed on 2026-09-13. The commands below document reproduction; they
> are not an active data-collection or training queue.

## Active Phase 5 data boundary

Phase 5's CPU audit selected 1,200 new GSM8K train candidates with fresh
reference verification and no recorded Phase 1-4 source overlap. The original
solver then generated one natural answer for each of the first 815 candidates:
575 correct and 240 wrong. The final append-only audit and derived rollouts
are backed up under `outputs/phase5_remote_v100/`; their SHA-256 hashes and
the recorded BF16-emulated runtime are in `phase5/configs/experiments.yaml`.

The deterministic, balanced 480-source split is frozen: 240 train, 80
development, and 160 protected test sources, each evenly divided between
initially correct and wrong answers. Exact IDs and hashes were fixed before
review/probe outcomes. A strict CPU audit found neutral/status eligible on all
480 rows, but only one wrong train row and no wrong development/protected rows
eligible for reliable location/type hints. Location/type is therefore disabled
for this pilot rather than refilled or weakened. The neutral/status review and
pre-hint probe protocol was frozen before model runs. The one protected opening
is complete, with its lock and receipt under `phase5/data/protocol/`. Raw
prompts, outputs and activations are backed up in `outputs/phase5_gpu_vllm/`;
none may be recycled into training or selection. See
`phase5/docs/FINAL_REPORT.md`.

The frozen linear probe is a diagnostic harness. It shows that correctness is
partly decodable before feedback, but it is not itself a self-correction model.
If a later pipeline routes repairs using probe predictions, record probe output
as an external controller decision and report detection, correct-answer
preservation, and verified repair separately. Do not label such a composition
autonomous model-only correction without a new end-to-end protected test.

## Objective

Phase 4 is closed, with its own archived protocol and diagnostics.
Its blind preference collection excluded verifier hints from model-visible
prompts, verifies real candidate outputs afterwards, balances action coverage
within source-disjoint splits, and freshly validates pairs against the audit
before training. Existing Phase 3 frozen artifacts remain unchanged. See
`phase4/docs/FINAL_REPORT.md` for the closure and final results.

Construct training and evaluation data for error discrimination without label
leakage. The router sees a problem, a previous answer, and neutral review text,
then emits exactly one decision tag:

```text
<decision>KEEP</decision>
<decision>REVISE</decision>
```

The verifier determines the label before the row is admitted.

## Source-of-truth hierarchy

1. Original problem and deterministic tests/reference value.
2. Immutable Base and Self-Correction V1 generations.
3. Fresh verifier outcome produced by current verifier code.
4. Compact inventory/bucket metadata.
5. Derived training dataset.
6. Frozen evaluation output.

A cached label is never stronger than a fresh verifier result.

## Canonical Phase 3 pipeline

### 1. Prepare sources

```bash
python phase3/scripts/data/prepare_source_pool.py
python phase3/scripts/data/prepare_apps_pilot.py
```

### 2. Collect natural model attempts

```bash
python phase3/scripts/data/collect_initial_attempts.py --help
python phase3/scripts/data/collect_apps_pilot.py --help
```

Collection is GPU/API dependent. Do not regenerate attempts merely to make a
desired label distribution.

### 3. Build buckets and provenance inventory

```bash
python phase3/scripts/data/bucket_initial_attempts.py
python phase3/scripts/data/bucket_apps_pilot.py
python phase3/scripts/data/build_source_inventory.py
python phase3/scripts/data/build_unified_source_inventory.py
```

Bucket convention:

- `CC`: Base correct, V1 correct
- `CW`: Base correct, V1 wrong
- `WC`: Base wrong, V1 correct
- `WW`: Base wrong, V1 wrong

The bucket is metadata, not a training label by itself.

### 4. Build router datasets

Decision-Only V1:

```bash
python phase3/scripts/data/build_decision_only.py
```

Router V3 diagnostic:

```bash
python phase3/scripts/data/build_revised_router_dataset.py
```

First contrastive/DPO pilot:

```bash
python phase3/scripts/data/build_contrastive_pairs.py
python phase3/scripts/data/build_dpo_pilot.py
```

Model-vs-model semantic DPO:

```bash
python phase3/scripts/data/build_semantic_model_dpo.py
```

The semantic builder produced 100 matched problems/200 preference rows. Both
answers are natural Base/V1 generations, labels are balanced 100/100, and fresh
verification passed for every correct member and failed for every wrong member.
It contains no reference answers in training prompts. The current historical
pool stores only one generation per model/problem, so same-model-origin
correct/wrong pairs could not be constructed without new sampling.

### 5. Train from explicit configs

```bash
python phase3/scripts/training/train_behavior_pilot.py --config phase3/configs/decision_only.yaml
python phase3/scripts/training/train_dpo_pilot.py --config phase3/configs/dpo_pilot.yaml
python phase3/scripts/training/train_dpo_pilot.py --config phase3/configs/dpo_semantic_v2.yaml
```

Both DPO runs start from the Decision-Only V1 LoRA, not from the bare base model.
The base checkpoint is loaded because a LoRA adapter cannot run independently.

### 6. Evaluate on protected data

`phase3/data/two_stage_selective_repair/frozen_eval.jsonl` contains 200 protected
rows. It is never an input to selection or training.

```bash
python phase3/scripts/evaluation/evaluate_two_stage_selective_repair.py --help
```

Each evaluation stores:

- scored decision generations;
- a separate non-scored rationale pass;
- unseen-template decisions;
- deterministic re-verification;
- selective versus always-repair comparisons.

The rationale is visible model output for audit, not hidden chain-of-thought.

### 7. Probe representations

```bash
python phase3/scripts/evaluation/extract_representation_probe.py --help
python phase3/scripts/evaluation/run_representation_probe.py --help
```

Activation extraction is forward-only. Logistic probes are fitted after model
weights are frozen. Layer and regularization selection use probe-dev only; test
is reported once.

## Pair construction requirements

For canonical semantic contrastive pairs:

1. Same problem.
2. Same model origin when multiple verified generations permit it; otherwise
   explicitly report the cross-origin limitation.
3. Comparable answer length.
4. Comparable formatting.
5. Identical code interface.
6. Wrong answer is executable/parseable, semantically wrong, and plausible.
7. Exactly balanced KEEP/REVISE labels.
8. Same neutral-review template within a pair and the same template distribution
   across labels.
9. No synthetic wrong answer and no reference answer used as a training member.
10. No frozen-evaluation overlap.

## Promotion gate

Phase 4 closed on 2026-09-15 without a confirmed gain. Its blind collection
failed the frozen coverage gate (28 train / four dev balanced pairs), so DPO V2
was not trained. Its three-round diagnostic was negative. Keep all raw attempts,
source splits and fixed diagnostics unchanged; do not reuse evaluation outputs
for training or lower thresholds after closure. New work requires a new phase.

An adapter is promoted only if the same frozen benchmark shows:

- balanced decision accuracy improves;
- REVISE recall improves without KEEP collapse;
- code accuracy improves;
- unseen neutral-template behavior does not regress materially; and
- representation probes improve, especially for code and REVISE.

DPO Pilot V1 and DPO Semantic V2 fail this gate. They remain reproducible
negative results and must not silently replace Decision-Only V1.

Router calibration and activation steering also failed their promotion gates.
No additional Phase 3 rows may be selected using the protected benchmark. A
continuation must define a new phase, new holdouts, and a new registry entry.
