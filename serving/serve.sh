#!/usr/bin/env bash
# OpenAI-compatible vLLM server for the project's checkpoints (one GPU, one profile at a time).
#
#   bash serving/serve.sh original   # original solver + any present LoRAs: V2, Phase 10 judge, Phase 11/12 DPO judges, Phase 13 verdict judge
#   bash serving/serve.sh v3         # merged V2 base + V3 LoRA
#
# Binds to 127.0.0.1 by default; set SERVE_HOST=0.0.0.0 only on a network you
# trust. vLLM adds no authentication unless SERVE_API_KEY is set.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
profile="${1:-original}"
vllm="${VLLM_BIN:-$project_root/.venv-vllm/bin/vllm}"
host="${SERVE_HOST:-127.0.0.1}"
port="${SERVE_PORT:-8000}"
args=(--host "$host" --port "$port" --dtype half --max-model-len 4096
      --gpu-memory-utilization 0.85 --enforce-eager
      --enable-lora --max-loras 5 --max-lora-rank 16)
[[ -n "${SERVE_API_KEY:-}" ]] && args+=(--api-key "$SERVE_API_KEY")

case "$profile" in
  original)
    # Load only the adapters present on this host; a missing path stops vLLM.
    loras=()
    for entry in "warmstart-v2=outputs/phase4_exploration_warmstart_v2/final_adapter" \
                 "phase10-judge=outputs/phase10_v1/judge_lora/final_adapter" \
                 "phase11-dpo-judge=outputs/phase11_v1/dpo_lora/final_adapter" \
                 "phase12-dpo-judge=outputs/phase12_v1/dpo_lora/final_adapter" \
                 "phase13-verdict-judge=outputs/phase13_v1/dpo_b_lora/final_adapter"; do
      [[ -f "${entry#*=}/adapter_model.safetensors" ]] && loras+=("$entry")
    done
    lora_args=()
    (( ${#loras[@]} )) && lora_args=(--lora-modules "${loras[@]}")
    exec "$vllm" serve Kxck/Self_Correction_v1 \
      --revision 6437f947999168a0ce2a98a86e4252fc77160a33 \
      --served-model-name original-solver "${args[@]}" "${lora_args[@]}"
    ;;
  v3)
    exec "$vllm" serve models/phase7_v2_merged_fp16 \
      --served-model-name warmstart-v2-merged "${args[@]}" \
      --lora-modules "correction-sft-v3=outputs/phase4_correction_sft_v3/final_adapter"
    ;;
  *)
    echo "Unknown profile: $profile (use: original | v3)" >&2
    exit 2
    ;;
esac
