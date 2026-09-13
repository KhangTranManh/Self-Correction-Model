#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PHASE3_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd -- "${PHASE3_ROOT}/.." && pwd)"
ADAPTER_PATH="${PHASE3_DECISION_ADAPTER_PATH:-${PROJECT_ROOT}/outputs/phase3_decision_only_v1/final_adapter}"

export PHASE3_LORA_MODULES="phase3-decision-only-v1=${ADAPTER_PATH}"
exec "${SCRIPT_DIR}/serve_phase3.sh"
