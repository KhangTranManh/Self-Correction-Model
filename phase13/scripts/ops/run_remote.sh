#!/usr/bin/env bash
# Phase 13 sequential GPU pipeline (A -> B -> C); every step resumes or is skipped when complete.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"
py="$project_root/.venv-vllm/bin/python"
out=outputs/phase13_v1
step() { echo "=== $(date -u +%FT%TZ) $*"; }

step verify lock
"$py" phase13/scripts/verify_lock.py
step "holdout samples"
"$py" phase13/scripts/generate.py --stage samples
step "A: untrained judge"
"$py" phase13/scripts/generate.py --stage judge --judge base
step "A: phase12 judge"
"$py" phase13/scripts/generate.py --stage judge --judge p12
step "B: untrained constrained judge"
"$py" phase13/scripts/generate.py --stage judge --judge base_c
step "B: train constrained dpo lora"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$py" phase13/scripts/train_dpo_b.py
step "B: constrained dpo judge"
"$py" phase13/scripts/generate.py --stage judge --judge p13b
step verify lock before protected opening
"$py" phase13/scripts/verify_lock.py
if [[ ! -f "$out/analysis/report.json" ]]; then
  step "protected analysis (A, B, C)"
  rm -rf "$out/analysis"
  "$py" phase13/scripts/analyze.py > logs/phase13_analysis_stdout.txt
fi
echo PHASE13_PIPELINE_COMPLETE
