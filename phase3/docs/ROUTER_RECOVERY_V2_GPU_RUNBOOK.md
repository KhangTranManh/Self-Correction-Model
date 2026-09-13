# Router Recovery V2 GPU runbook

**Archived:** Phase 3 closed before Router Recovery V2 reached its data gate.
Do not execute this runbook as a Phase 3 continuation. It is retained only to
document the planned protocol.

## Scope

This runbook defines every GPU allocation in Router Recovery V2. It does not
authorize a run. GPU work starts only after its input gate is present locally
and passes validation.

The four allocations are intentionally separate so a server can be stopped
between them. A single 24 GB RTX 3090/3090 Ti is sufficient.

## Common remote environment

```text
Python           3.10-3.12
Torch            2.7.1+cu126
Transformers     4.57.1
PEFT             0.17.1
bitsandbytes     0.47.0
vLLM             0.10.0
scikit-learn     1.7.2
```

Before every allocation, record `nvidia-smi`, free disk, package versions, the
base-model revision, adapter SHA-256, dataset SHA-256, and Git commit/worktree
state. Never copy `.env` into an evidence bundle or report its contents.

## GPU-A — optional natural failure generation

Run only if Stage 01 cannot build 200 verified hard pairs from existing data.
The manifest must already be frozen and source-disjoint.

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Kxck/Self_Correction_v1 \
  --served-model-name self-correction-v1 \
  --dtype half --max-model-len 4096 \
  --gpu-memory-utilization 0.88 --max-num-seqs 16 \
  --enforce-eager --disable-log-requests \
  --host 127.0.0.1 --port 8000
```

In a second shell:

```bash
python phase3/scripts/data/collect_same_origin_candidates.py \
  --manifest phase3/data/router_recovery_v2/generation/generation_manifest.jsonl \
  --output phase3/data/router_recovery_v2/generation/raw_candidates.jsonl \
  --origin self_correction_v1 --model self-correction-v1 \
  --base-url http://127.0.0.1:8000/v1 \
  --temperature 0.9 --top-p 0.95 \
  --math-max-tokens 1024 --code-max-tokens 2048 \
  --concurrency 8 --batch-size 32 --max-samples-per-task 4 \
  --seed 20260908
```

Download raw candidates and stop vLLM immediately. Fresh verification and pair
construction run on local CPU. Raw generation is resumable by `candidate_id`.

## GPU-B — frozen hidden-state extraction

Input gate: at least 200 validated pairs, protected overlap zero, no duplicate
source groups, and maximum rendered length at most 4096 tokens.

```bash
python phase3/scripts/evaluation/extract_representation_probe.py \
  --dataset phase3/data/router_recovery_v2/development_all.jsonl \
  --base-model Kxck/Self_Correction_v1 \
  --model decision_only_v1=outputs/phase3_decision_only_v1/final_adapter \
  --output-dir outputs/phase3_router_recovery_v2/02_hidden_states \
  --layers 14,21,28 --expected-rows 0
```

The current extractor produces the required final-prompt-token representation.
Answer-end pooling is an optional diagnostic and must not delay this pass. After
the `.npz`, manifest, summary, and hashes are downloaded, unload the model; head
training and calibration run on CPU.

## GPU-C — conditional contrastive LoRA

This allocation is prohibited unless the Stage 03 frozen head and Stage 04
policy both fail their gate. The trainer interface is reserved as:

```bash
python phase3/scripts/training/train_contrastive_router.py \
  --config phase3/configs/router_recovery_v2_contrastive_lora.yaml
```

The implementation must pass CPU unit tests for pair batching, decision-token
masking, contrastive-positive/negative construction, source leakage, and loss
composition before renting the GPU. Whole-response DPO is prohibited. Training
is one epoch from Decision-Only V1 and never uploads automatically.

Immediately download the adapter, config, tokenizer metadata, run log, metrics,
per-example dev predictions, and SHA-256 manifest. A training-loss decrease is
not a promotion result.

## GPU-D — sealed confirmation and serving

Input gate: the 200-row confirmatory manifest was sealed before candidate
selection and the exact threshold/policy was frozen. Serve baseline and
candidate together when a candidate adapter exists:

```bash
PHASE3_LORA_MODULES="phase3-decision-only-v1=outputs/phase3_decision_only_v1/final_adapter phase3-router-recovery-v2=outputs/phase3_router_recovery_v2/05_contrastive_lora/final_adapter" \
PHASE3_GPU_MEMORY_UTILIZATION=0.88 \
bash phase3/scripts/serving/serve_phase3.sh
```

Evaluation must save each raw response, parsed decision, visible reasoning,
score, verifier result, and source ID. It may not tune a threshold on these 200
rows. If Stage 05 was skipped, evaluate the selected external head/policy from
saved Decision-Only hidden states instead of inventing a candidate LoRA.

## Budget boundaries

| Allocation | Trigger | RTX 3090 Ti estimate | Stop point |
|---|---|---:|---|
| GPU-A generation | existing hard pool is insufficient | 0.5–3 h | raw candidates downloaded |
| GPU-B extraction | Stage 01 validation PASS | 20–60 min | activation artifacts downloaded |
| GPU-C LoRA | Stages 03–04 both FAIL | 1–3 h | adapter and evidence downloaded |
| GPU-D confirmation | sealed set and candidate ready | 30–90 min | raw results downloaded |

OS shutdown alone may not stop provider billing. After each stop point, verify
that artifacts are local, then terminate the rented instance in the provider
control panel.
