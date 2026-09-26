#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"
python_bin="$project_root/.venv-vllm/bin/python"
mkdir -p logs outputs/phase8_probe_v1/activations outputs/phase8_probe_v1/selection

"$python_bin" phase8/scripts/verify_execution_lock.py
"$python_bin" phase8/scripts/verify_adapters.py

while [[ ! -f outputs/phase8_first_pass_v1/initial/summary.json ]]; do
  if ! pgrep -f 'phase8/scripts/collect_first_pass_vllm.py --stage initial' >/dev/null; then
    echo "Initial collector stopped without complete summary" >&2
    exit 1
  fi
  sleep 30
done

"$python_bin" phase8/scripts/collect_first_pass_vllm.py --stage sample_repeat
"$python_bin" phase8/scripts/collect_first_pass_vllm.py --stage greedy_same_prompt
"$python_bin" phase8/scripts/reproduce_phase7_initials.py
"$python_bin" phase8/scripts/freeze_distractors.py
"$python_bin" phase8/scripts/collect_three_arms_vllm.py --checkpoint original_solver

"$python_bin" phase5/scripts/extract_prehint_activations.py \
  --checkpoint original_solver --split train --batch-size 1 \
  --output-dir outputs/phase8_probe_v1/activations
"$python_bin" phase5/scripts/extract_prehint_activations.py \
  --checkpoint original_solver --split development --batch-size 1 \
  --output-dir outputs/phase8_probe_v1/activations
"$python_bin" phase8/scripts/fit_locked_probe.py --checkpoint original_solver
"$python_bin" phase8/scripts/score_probe.py --checkpoint original_solver

"$python_bin" phase7/scripts/materialize_v2_local.py
"$python_bin" phase8/scripts/mark_phase5_lineage.py
for checkpoint in warmstart_v2 correction_sft_v3; do
  "$python_bin" phase8/scripts/collect_three_arms_vllm.py --checkpoint "$checkpoint"
  for split in train development; do
    "$python_bin" phase5/scripts/extract_prehint_activations.py \
      --checkpoint "$checkpoint" --split "$split" --batch-size 1 \
      --merged-v2 "$project_root/models/phase7_v2_merged_fp16" \
      --output-dir outputs/phase8_probe_v1/activations
  done
  "$python_bin" phase8/scripts/fit_locked_probe.py --checkpoint "$checkpoint"
  "$python_bin" phase8/scripts/score_probe.py --checkpoint "$checkpoint"
done

"$python_bin" phase8/scripts/verify_execution_lock.py
"$python_bin" phase8/scripts/analyze_protected.py
echo PHASE8_PIPELINE_COMPLETE
