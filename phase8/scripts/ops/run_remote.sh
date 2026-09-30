#!/usr/bin/env bash
# Phase 8 amendment v2: full sequential pipeline on one V100 32 GB.
# Every step is resumable or skipped when its summary already exists, so the
# supervisor may rerun this script after any failure.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"
py="$project_root/.venv-vllm/bin/python"
step() { echo "=== $(date -u +%FT%TZ) $*"; }

step verify lock
"$py" phase8/scripts/verify_execution_lock.py
step fetch adapters
"$py" phase8/scripts/fetch_adapters.py
"$py" phase8/scripts/verify_adapters.py

# Merge V2 on CPU in parallel with GPU generation; a partial merge is discarded.
if [[ ! -f models/phase7_v2_merged_fp16/phase7_lineage.json ]]; then
  rm -rf models/phase7_v2_merged_fp16
fi
"$py" phase7/scripts/materialize_v2_local.py > logs/materialize_v2.log 2>&1 &
merge_pid=$!

step phase7 donor regeneration
"$py" phase8/scripts/regen_phase7_donors.py

for stage in initial sample_repeat greedy_same_prompt; do
  step "first pass $stage"
  "$py" phase8/scripts/collect_first_pass_vllm.py --stage "$stage"
done

if [[ ! -f phase8/data/distractors_v2/report.json ]]; then
  step freeze distractors
  rm -rf phase8/data/distractors_v2
  "$py" phase8/scripts/freeze_distractors.py
fi

for checkpoint in original_solver warmstart_v2 correction_sft_v3; do
  if [[ "$checkpoint" == warmstart_v2 ]]; then
    step wait for merged V2
    wait "$merge_pid"
  fi
  step "three arms $checkpoint"
  "$py" phase8/scripts/collect_three_arms_vllm.py --checkpoint "$checkpoint"
  if [[ ! -f "outputs/phase8_probe_scores_v2/$checkpoint/summary.json" ]]; then
    step "probe scores $checkpoint"
    "$py" phase8/scripts/score_probe.py --checkpoint "$checkpoint"
  fi
done

step verify lock before protected opening
"$py" phase8/scripts/verify_execution_lock.py
if [[ ! -f outputs/phase8_analysis_v2/report.json ]]; then
  step protected analysis
  rm -rf outputs/phase8_analysis_v2
  "$py" phase8/scripts/analyze_protected.py > logs/phase8_analysis_stdout.json
fi
echo PHASE8_PIPELINE_COMPLETE
