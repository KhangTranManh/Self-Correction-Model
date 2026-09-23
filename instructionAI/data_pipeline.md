# Verified Data Pipeline

> Phase 3 closed on 2026-09-13. The commands below document reproduction; they
> are not an active data-collection or training queue.

## Active Phase 5 data boundary

Phase 5's CPU audit selected 1,200 new GSM8K train candidates with fresh
reference verification and no recorded Phase 1-4 source overlap. The original
solver then generated one natural answer for each of the first 815 candidates:
575 correct and 240 wrong. The final append-only audit and derived rollouts
are backed up under `outputs/phase5_remote_v100/`; their SHA-256 hashes and
the recorded BF16-emulated runtime are in `phase5/configs/experiments.yaml`.

The next CPU-only operation is a deterministic, balanced 480-source split:
240 train, 80 development, and 160 protected test sources, each split evenly
between initially correct and wrong answers. Freeze exact IDs and hashes
before review/probe outcomes. Independently validate error-location/type
hints; do not use the reference answer or verifier trace in model prompts,
or refill failed hint cases after seeing outcomes. The earlier rule to freeze
model-error alignment before the first GPU pass was missed, so current
location/type analyses are exploratory unless a new preregistered holdout
resolves that deviation. See `phase5/docs/EXECUTION_PLAN.md`.

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
