#!/usr/bin/env bash
# Phase 10 sequential GPU pipeline; every step resumes or is skipped when complete.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"
py="$project_root/.venv-vllm/bin/python"
out=outputs/phase10_v1
step() { echo "=== $(date -u +%FT%TZ) $*"; }

step verify lock
"$py" phase10/scripts/verify_lock.py
step holdout samples and base judge
"$py" phase10/scripts/generate.py --stage holdout
step training samples
"$py" phase10/scripts/generate.py --stage train_samples
if [[ ! -f "$out/train_pairs.jsonl" ]]; then
  step build pairs
  "$py" phase10/scripts/build_data.py --pairs
fi
step judge candidates
"$py" phase10/scripts/generate.py --stage judge_candidates
if [[ ! -f "$out/sft/report.json" ]]; then
  step build sft data
  "$py" phase10/scripts/build_data.py --sft
fi
step train judge lora
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$py" phase10/scripts/train_judge.py
step trained judge on holdout
"$py" phase10/scripts/generate.py --stage holdout_trained
step verify lock before protected opening
"$py" phase10/scripts/verify_lock.py
if [[ ! -f "$out/analysis/report.json" ]]; then
  step protected analysis
  rm -rf "$out/analysis"
  "$py" phase10/scripts/analyze.py > logs/phase10_analysis_stdout.txt
fi
echo PHASE10_PIPELINE_COMPLETE
