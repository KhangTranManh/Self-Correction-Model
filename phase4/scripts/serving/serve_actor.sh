#!/usr/bin/env bash
set -euo pipefail

MODEL="${PHASE4_MODEL:-Kxck/Self_Correction_v1}"
HOST="${PHASE4_HOST:-127.0.0.1}"
PORT="${PHASE4_PORT:-8999}"
MAX_LEN="${PHASE4_MAX_MODEL_LEN:-4096}"
PYTHON="${PHASE4_PYTHON:-python}"

args=(
  "$PYTHON" -m vllm.entrypoints.openai.api_server
  --model "$MODEL"
  --served-model-name phase4-actor
  --host "$HOST"
  --port "$PORT"
  --dtype bfloat16
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization 0.90
  --enable-lora
)

if [[ -n "${PHASE4_LORA_MODULES:-}" ]]; then
  args+=(--lora-modules "$PHASE4_LORA_MODULES")
fi

exec "${args[@]}"
