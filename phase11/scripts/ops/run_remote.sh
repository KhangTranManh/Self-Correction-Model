#!/usr/bin/env bash
# Phase 11 sequential GPU pipeline; every step resumes or is skipped when complete.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"
py="$project_root/.venv-vllm/bin/python"
out=outputs/phase11_v1
step() { echo "=== $(date -u +%FT%TZ) $*"; }

step verify lock
"$py" phase11/scripts/verify_lock.py
step holdout samples
"$py" phase11/scripts/generate.py --stage samples
step base judge on holdout pairs
"$py" phase11/scripts/generate.py --stage judge_base
step train dpo lora
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$py" phase11/scripts/train_dpo.py
step dpo judge on holdout pairs
"$py" phase11/scripts/generate.py --stage judge_dpo
step verify lock before protected opening
"$py" phase11/scripts/verify_lock.py
if [[ ! -f "$out/analysis/report.json" ]]; then
  step protected analysis
  rm -rf "$out/analysis"
  "$py" phase11/scripts/analyze.py > logs/phase11_analysis_stdout.txt
fi
echo PHASE11_PIPELINE_COMPLETE
