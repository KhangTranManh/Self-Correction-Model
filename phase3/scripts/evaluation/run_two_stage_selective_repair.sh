#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PHASE3_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PHASE3_PYTHON:-python}"
OUTPUT_DIR="${PHASE3_TWO_STAGE_OUTPUT:-${PHASE3_ROOT}/runs/two_stage_selective_repair}"
mkdir -p "${OUTPUT_DIR}"
rm -f "${OUTPUT_DIR}/run.exit"

"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_two_stage_selective_repair.py" \
  --manifest "${PHASE3_ROOT}/data/two_stage_selective_repair/frozen_eval.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  > "${OUTPUT_DIR}/run.log" 2>&1
status=$?
printf '%s\n' "${status}" > "${OUTPUT_DIR}/run.exit"
exit "${status}"
