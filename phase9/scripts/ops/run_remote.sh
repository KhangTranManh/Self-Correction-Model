#!/usr/bin/env bash
# Phase 9 sequential GPU pipeline; every step resumes or is skipped when complete.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"
py="$project_root/.venv-vllm/bin/python"
step() { echo "=== $(date -u +%FT%TZ) $*"; }

step verify lock
"$py" phase9/scripts/verify_lock.py
step first answers
"$py" phase9/scripts/collect_first.py
for checkpoint in original_solver warmstart_v2 correction_sft_v3; do
  if [[ ! -f "outputs/phase9_probe_scores_v1/$checkpoint/summary.json" ]]; then
    step "probe scores $checkpoint"
    "$py" phase9/scripts/score_probe.py --checkpoint "$checkpoint"
  fi
  step "attempts and judge $checkpoint"
  "$py" phase9/scripts/collect_checkpoint.py --checkpoint "$checkpoint"
done
step verify lock before protected opening
"$py" phase9/scripts/verify_lock.py
if [[ ! -f outputs/phase9_analysis_v1/report.json ]]; then
  step protected analysis
  rm -rf outputs/phase9_analysis_v1
  "$py" phase9/scripts/analyze.py > logs/phase9_analysis_stdout.txt
fi
echo PHASE9_PIPELINE_COMPLETE
