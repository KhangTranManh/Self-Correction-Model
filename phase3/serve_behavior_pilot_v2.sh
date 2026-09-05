#!/usr/bin/env bash
set -euo pipefail

# Serve the untouched merged V1 checkpoint and the local contract-focused V2
# adapter together. Neither artifact is uploaded by this script.
exec vllm serve Kxck/Self_Correction_v1 \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype half \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  --enforce-eager \
  --enable-lora \
  --max-lora-rank 32 \
  --served-model-name self-correction-v1 \
  --lora-modules \
    phase3-pilot-v2-2epoch=/root/phase3-mini/phase3/outputs/phase3_behavior_pilot_v2_2epoch/final_adapter \
  --disable-log-requests
