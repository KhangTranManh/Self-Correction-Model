#!/usr/bin/env bash
set -euo pipefail

# Serve the untouched merged V1 checkpoint and the locally trained pilot adapter together
# for paired micro-evaluation. This script never uploads either adapter.
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
    phase3-pilot-1epoch=/root/phase3-mini/phase3/outputs/phase3_behavior_pilot_1epoch/final_adapter \
  --disable-log-requests
