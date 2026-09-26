#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"
python=.venv-vllm/bin/python

while [ ! -f outputs/phase7_initials_v1/summary.json ]; do
  if ! pgrep -f '[c]ollect_initials_vllm.py' >/dev/null; then
    echo "Initial-answer collector exited without a completed summary" >&2
    exit 1
  fi
  sleep 15
done

"$python" phase7/scripts/freeze_balanced_split.py
"$python" phase7/scripts/freeze_paired_protocol.py

for checkpoint in original_solver warmstart_v2 correction_sft_v3; do
  "$python" phase7/scripts/collect_paired_vllm.py \
    --split development --checkpoint "$checkpoint"
done
"$python" phase7/scripts/analyze_paired.py --split development

for checkpoint in original_solver warmstart_v2 correction_sft_v3; do
  "$python" phase7/scripts/collect_paired_vllm.py \
    --split protected --checkpoint "$checkpoint"
done
"$python" phase7/scripts/analyze_paired.py --split protected
echo "PHASE7_PIPELINE_COMPLETE"
