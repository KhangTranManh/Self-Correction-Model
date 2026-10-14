#!/usr/bin/env bash
# Remainder of the cut main run for the last model(s), ordered to protect the endpoints
# under the GPU time limit (amendment 2026-10-13): all generations first, then the
# protected opening (CPU, in the background) while the hidden states are extracted;
# the probe analysis for models whose hidden states are complete runs early on CPU.
# Same scripts and arguments as run_sequential.sh main; only the order differs.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
py=.venv-vllm/bin/python
models="${*:-mistral_7b}"
step() { echo "=== $(date -u +%FT%TZ) $*"; }
gen() { "$py" phase14/scripts/ops/gen_batch.py "${GEN_BATCH:-256}" --model "$1" --dataset "$2" --split main --stage "$3" ${4:+--judge "$4"}; }
step "verify main lock"; "$py" phase14/scripts/verify_lock.py --stage main
step "early probe analysis (CPU, background; models with complete hidden states)"
( "$py" phase14/scripts/probe_analysis.py > logs/phase14_probe_early.txt 2>&1 || true ) &
for model in $models; do
  for dataset in asdiv math13; do
    step "main $model $dataset samples"; gen "$model" "$dataset" samples
    for judge in base base_c; do step "main $model $dataset judge $judge"; gen "$model" "$dataset" judge "$judge"; done
  done
done
step "endpoint generations complete; protected opening in the background (CPU)"
( "$py" phase14/scripts/verify_lock.py --stage main && \
  { [[ -f outputs/phase14_v1/main/analysis/report.json ]] || "$py" phase14/scripts/analyze.py > logs/phase14_analysis_stdout.txt 2>&1; } && \
  echo ANALYSIS_DONE >> logs/phase14_analysis_stdout.txt ) &
analysis_pid=$!
for model in $models; do
  for dataset in asdiv math13; do
    for site in answer judge direct; do
      step "hidden $model $dataset $site"
      "$py" phase14/scripts/extract_hidden.py --model "$model" --dataset "$dataset" --split main --site "$site"
    done
  done
done
wait "$analysis_pid" || true
step "probe analysis (all models)"; "$py" phase14/scripts/probe_analysis.py > logs/phase14_probe_stdout.txt
step "report tables"; "$py" phase14/scripts/report_tables.py > logs/phase14_report_tables.md
echo PHASE14_MAIN_COMPLETE
