#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PHASE3_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd -- "${PHASE3_ROOT}/.." && pwd)"

BASE_MODEL="${PHASE3_BASE_MODEL:-Kxck/Self_Correction_v1}"
BASE_NAME="${PHASE3_BASE_NAME:-self-correction-v1}"
HOST="${PHASE3_HOST:-127.0.0.1}"
PORT="${PHASE3_PORT:-8000}"
MAX_MODEL_LEN="${PHASE3_MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${PHASE3_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${PHASE3_MAX_NUM_SEQS:-8}"
LORA_MODULES="${PHASE3_LORA_MODULES:-phase3-decision-only-v1=${PROJECT_ROOT}/outputs/phase3_decision_only_v1/final_adapter}"

# Space-separated name=path entries. Phase 3 paths must not contain spaces.
read -r -a LORA_ARGUMENTS <<< "${LORA_MODULES}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

exec vllm serve "${BASE_MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype half \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --enforce-eager \
  --enable-lora \
  --max-lora-rank 32 \
  --served-model-name "${BASE_NAME}" \
  --lora-modules "${LORA_ARGUMENTS[@]}"
