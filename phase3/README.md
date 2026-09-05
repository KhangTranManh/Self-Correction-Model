# Phase 3 — Selective correction / error discrimination

Phase 3 asks one question:

> **Can a 7B LLM learn to distinguish when its own output should be preserved
> versus revised, while retaining the correction capability learned earlier?**

The short thesis is:

> **Phase 3 tests whether correction can become selective: preserving correct
> outputs, rejecting misleading feedback, and revising only when an error is
> actually present.**

## Current status — 2026-09-05

The source-generation, objective verification, bucket construction, 860-row
selection manifest, two pilot datasets, two QLoRA runs, and a frozen 30-row
micro-evaluation have been completed.

The latest contract-focused pilot improved review formatting and preservation,
but did not learn autonomous repair and regressed on fresh tasks. It passed
**4/8** predefined smoke gates. **Do not scale to all 860 selected rows yet.**

No GPU service is currently running. The last rental was shut down after the V2
validation. Its V2 adapter and raw evaluation outputs were not uploaded or
downloaded, so they are not durable repository artifacts. The local data,
configuration, construction code, and captured aggregate results are sufficient
to reproduce the run.

## Required behaviors

| Condition | Initial state and review | Required behavior |
|---|---|---|
| Guided repair | wrong answer + true verifier feedback | `REVISE`, then produce a correct answer |
| False-feedback preservation | correct answer + misleading negative feedback | `KEEP`, preserving the correct answer |
| Neutral preservation | correct answer + neutral review | `KEEP`, preserving the correct answer |
| Autonomous repair | wrong answer + neutral review | detect the error, `REVISE`, and repair it |
| Normal solve | fresh task, no review | answer normally without correction scaffolding |
| Regression recovery | a CW source where Base was right and V1 was wrong | recover a correct fresh answer |

Review responses use one minimal contract:

```xml
<decision>KEEP</decision>
<answer>...</answer>
```

or:

```xml
<decision>REVISE</decision>
<answer>...</answer>
```

Decision accuracy and answer correctness are scored separately. A valid tag is
not evidence of successful repair.

## Model provenance

- Original foundation model: `Qwen/Qwen2.5-7B-Instruct`.
- Policy entering Phase 3: `Kxck/Self_Correction_v1`.
- The Hub repository for `Self_Correction_v1` is a merged full checkpoint
  (`config` plus four weight shards), not a standalone PEFT adapter repository.
- Both Phase 3 pilots therefore loaded the merged V1 checkpoint in 4-bit NF4 and
  attached a new Phase 3 LoRA directly to it.
- Pilot V2 did **not** continue from the Pilot V1 LoRA.

## Data pipeline completed so far

### 1. GSM8K + MBPP initial attempts

Base and `Self_Correction_v1` answered the same 1,474 training problems. Every
raw attempt was retained and scored with the existing deterministic math or
executable-code verifier.

| Model | Correct | Wrong | Accuracy | GSM8K | MBPP |
|---|---:|---:|---:|---:|---:|
| Base Qwen2.5-7B-Instruct | 769 | 705 | 52.17% | 734/1,000 | 35/474 |
| Self_Correction_v1 | 902 | 572 | 61.19% | 873/1,000 | 29/474 |

The paired problem buckets are:

| Bucket | Meaning | Count |
|---|---|---:|
| CC | Base correct, V1 correct | 716 |
| WW | Base wrong, V1 wrong | 519 |
| WC | Base wrong, V1 correct | 186 |
| CW | Base correct, V1 wrong | 53 |

### 2. APPS introductory pilot

A model-independent 300-problem APPS pilot was run through both policies and
executable tests:

| Bucket | Count |
|---|---:|
| CC | 120 |
| WW | 132 |
| WC | 21 |
| CW | 27 |

Base passed 147/300 (49%); V1 passed 141/300 (47%). Although V1 had a small net
regression, the source supplied 141 V1-correct code examples, both supported test
modes, and diverse source hosts. APPS was accepted as a useful Phase 3 source,
not as evidence that V1 improved over Base.

### 3. Unified inventory and 860-row selection

The unified inventory contains 1,774 source problems:

- 1,000 GSM8K, 474 MBPP, and 300 APPS;
- 836 CC, 651 WW, 207 WC, and 80 CW;
- 1,000 math and 774 code problems;
- no frozen P0 rows and no HumanEval rows.

The deterministic selection manifest contains 860 potential behavior rows from
706 unique sources:

| Condition | Rows |
|---|---:|
| Preserve false feedback | 185 |
| Preserve neutral | 185 |
| Repair neutral | 145 |
| Repair true feedback | 145 |
| Normal solve | 120 |
| Regression recovery | 80 |

It includes 120 counterfactual pairs and is balanced to 422 math / 438 code. CW
is used only for normal-solve anchors or regression recovery, never as ordinary
repair data.

## Pilot V1 — 150 rows, one epoch

The first pilot used 25 rows per condition and a source-disjoint split of 120
train / 30 dev. The dev set is frozen with SHA-256:

```text
29a90c9cc74747b158bd79f3662f68e220b296ef2a101f2be1d432cea9a9ef07
```

Training configuration:

- checkpoint: `Kxck/Self_Correction_v1`;
- QLoRA NF4, rank 32, alpha 64;
- effective batch size 8;
- one epoch, 15 optimizer steps, learning rate `5e-5`;
- 120 train rows, no truncation;
- no Hub upload.

Train loss was 0.6128 and dev loss moved from 1.0228 to 0.5570. The
micro-evaluation passed 3/8 gates:

- false-flip improved from 5/5 to 2/5;
- normal solve improved from 1/5 to 3/5;
- regression recovery improved from 1/5 to 3/5;
- guided repair reached 1/5;
- autonomous repair remained 0/5;
- every review response missed the exact decision contract.

This run learned some answer-level behavior but not the requested interface.

## Pilot V2 — 200 rows, two epochs

### Data redesign

Pilot V2 contains 200 rows: 170 train and the **same byte-identical 30-row dev
set**. Its data validation passed completely.

| Train condition | Rows |
|---|---:|
| Preserve neutral | 32 |
| Preserve false feedback | 32 |
| Repair neutral | 32 |
| Repair true feedback | 32 |
| Normal solve | 13 |
| Regression recovery | 13 |
| Format-only warmup | 16 |

Important properties:

- 164/200 rows (82%) have an exact KEEP/REVISE target contract, versus 100/150
  (66.7%) in Pilot V1;
- 144/170 training rows have the exact contract;
- format warmups are balanced 8 KEEP / 8 REVISE;
- the warmup decision is explicitly supplied, so its label is independent of
  answer correctness;
- legacy target prose such as “the previous response is correct” was removed;
- train and dev share no source IDs;
- one APPS source whose historical CC state failed fresh reverification was
  excluded and deterministically replaced.

### Training

Pilot V2 started directly from `Kxck/Self_Correction_v1`, not Pilot V1:

- QLoRA NF4, rank 32, alpha 64;
- 80,740,352 trainable parameters (1.821% of the quantized loaded model view);
- effective batch size 8;
- two epochs, 44 optimizer steps;
- learning rate `3e-5` with cosine decay and 10% warmup;
- 170 train rows, no truncation;
- runtime about 255 seconds on one RTX 3090;
- train loss 0.4269;
- frozen-dev loss 1.0228 → 0.4246;
- no Hub upload.

### Frozen micro-evaluation

Both V1 and Pilot V2 used the exact same 30 dev rows, prompts, seed, deterministic
decoding, and objective verifiers. Each condition has only five examples, so the
results are directional smoke tests, not significance claims.

| Metric | V1 baseline | Pilot V1 | Pilot V2 |
|---|---:|---:|---:|
| Exact review contract | 0/20 | 0/20 | **20/20** |
| False-feedback false-flip | 5/5 | 2/5 | **1/5** |
| Preserve-neutral final correct | 2/5 | 2/5 | **5/5** |
| Preserve-neutral correct decision + answer | 0/5 | 0/5 | 2/5 |
| Neutral repair | 0/5 | 0/5 | 0/5 |
| Guided repair | 0/5 | 1/5 | 1/5 |
| Normal-solve behavioral success | 1/5 | 3/5 | **0/5** |
| Regression-recovery behavioral success | 1/5 | 3/5 | **0/5** |

Pilot V2 emitted 9 KEEP and 11 REVISE decisions across the 20 review rows, so it
did not collapse to a single review action. However:

- on neutral wrong answers it chose REVISE in 3/5 cases but repaired 0/5;
- on guided wrong answers it chose REVISE in 5/5 but repaired only 1/5;
- fresh final correctness was 2/10 and correction scaffolding appeared in 4/10;
- fresh code formatting was valid, but executable pass rate was 0/6;
- several REVISE outputs wrapped or copied the old wrong answer rather than
  generating a genuine repair.

Pilot V2 passed these four gates:

1. preserve-neutral correctness at least 80% and no worse than baseline;
2. false-feedback false-flip improved / at most 20%;
3. guided repair did not drop by more than 20 percentage points;
4. review decisions were not all KEEP or all REVISE.

It failed the neutral-repair, normal-solve, recovery, and fresh-code gates. The
correct decision is therefore:

```text
do_not_scale_yet
```

## Interpretation

The two pilots expose a real trade-off:

- Pilot V1 retained more fresh-task capability but did not learn the contract.
- Pilot V2 learned the contract perfectly and became much less susceptible to
  false feedback, but over-applied review formatting and did not convert REVISE
  decisions into successful repairs.

This is evidence for **format acquisition plus partial preservation**, not yet
for selective correction. Low dev loss is not enough: the model can imitate the
contract while still copying a wrong answer or contaminating fresh responses.

## Next experiment

Do not construct or train the full 860 rows yet. The next pilot should isolate
the Pareto point between the two runs:

1. train the 200-row V2 set for one epoch and save an epoch-1 checkpoint;
2. evaluate epoch 1 before continuing to epoch 2;
3. increase normal-solve and regression-recovery anchors relative to Pilot V2;
4. reduce or ablate format-only warmups now that their effect is established;
5. oversample or improve wrong+neutral repair targets, while measuring decision
   accuracy separately from repair correctness;
6. keep the 30-row dev set, prompts, seed, decoding, and verifiers unchanged;
7. scale to 860 only when contract validity, preservation, neutral repair, and
   fresh-task behavior pass together.

## Reproduction

From the repository root:

```bash
python phase3/build_behavior_pilot_v2.py
```

On a CUDA machine with the Phase 3 environment:

```bash
cd phase3
python train_behavior_pilot.py --config configs/mini_train_v2.yaml
./serve_behavior_pilot_v2.sh
python evaluate_behavior_micro.py \
  --dev data/behavior/mini_train/dev.jsonl \
  --output-dir outputs/phase3_behavior_pilot_v2_2epoch/micro_eval \
  --baseline-model self-correction-v1 \
  --candidate-model phase3-pilot-v2-2epoch \
  --baseline-label self_correction_v1 \
  --candidate-label phase3_pilot_v2_2epoch
```

The evaluator deliberately points to the original frozen dev file.

## Important local artifacts

- `data/attempts/`: complete Base and V1 GSM8K/MBPP attempts;
- `data/apps_pilot/`: APPS source, raw attempts, executable-test results, buckets;
- `data/behavior/unified_source_inventory.jsonl`: 1,774-source inventory;
- `data/behavior/selection/behavior_selection_manifest.jsonl`: 860-row plan;
- `data/behavior/construction_pilot_v2/`: validated 200-row Pilot V2;
- `data/behavior/mini_train_v2/`: 170/30 split;
- `configs/mini_train_v2.yaml`: exact V2 training configuration;
- `outputs/phase3_behavior_pilot_1epoch/`: downloaded Pilot V1 report and raw eval.

The Pilot V2 adapter, training report, and raw micro-evaluation were left on the
terminated rental and are not present locally. Do not claim that checkpoint as a
durable result; reproduce it from the committed data/config when needed.
