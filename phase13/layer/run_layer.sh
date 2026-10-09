#!/usr/bin/env bash
# Exploratory layer study: extract hidden states for every dataset/judge, then fit probes.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
py="$project_root/.venv-vllm/bin/python"
for spec in phase12:base phase12:p12 phase13:base phase13:p12 phase13:base_c phase13:p13b; do
  echo "=== $(date -u +%FT%TZ) extract ${spec}"
  "$py" phase13/layer/extract_hidden.py --dataset "${spec%%:*}" --judge "${spec##*:}"
done
echo "=== $(date -u +%FT%TZ) probe analysis"
"$py" phase13/layer/probe_analysis.py > logs/phase13_layer_probe.txt
echo LAYER_STUDY_COMPLETE
