#!/usr/bin/env bash
# Phase 14 GPU pipeline, model by model (for a host with a 100 GB disk).
# Usage: run_sequential.sh smoke | pilot | main | extra
#   smoke  per model: vLLM smoke test, then the Transformers hidden-state smoke test
#   pilot  verify the pilot lock; every model on the pilot sets; pilot report
#   main   verify the main lock; then ONE MODEL AT A TIME: all its generations
#          (endpoint datasets, GSM8K/SVAMP, and its comparison sets: the bridge set,
#          or for the original solver the report's regression sets), then all its
#          hidden states, then its weights are deleted from the Hugging Face cache
#          before the next model starts. After the last model: verify the lock,
#          open once (comparison cells that are ready are scored in the same
#          opening; endpoints E1/E2 use only the uniform cells), regression check,
#          probes, report tables.
# Same scripts, arguments and per-cell work as run_remote.sh; only the order differs,
# so that at most one model's weights and its hidden states share the disk.
# Every step resumes or is skipped when complete; a model is deleted only after
# all of its steps succeeded (set -e).
set -euo pipefail

mode="${1:?usage: run_sequential.sh smoke|pilot|main}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"
mkdir -p logs
py="$project_root/.venv-vllm/bin/python"
step() { echo "=== $(date -u +%FT%TZ) $*"; }
cfg() { "$py" -c "import sys; sys.path.insert(0, '.'); from phase14.scripts import config; print(' '.join(config.$1) if isinstance(config.$1, tuple) else config.$1)"; }
repo() { "$py" -c "import sys; sys.path.insert(0, '.'); from phase14.scripts import config; print(config.MODELS['$1']['repo'])"; }
models=$(cfg RUN_MODELS)
datasets=$(cfg DATASETS)
comparison=$(cfg COMPARISON)
bridge=$(cfg BRIDGE)
if [[ -f .env ]] && grep -q '^HF_TOKEN=' .env; then
  HF_TOKEN="$(grep '^HF_TOKEN=' .env | head -1 | cut -d= -f2- | tr -d '"'"'"' \r')"
  export HF_TOKEN
fi

gen() { "$py" phase14/scripts/generate.py --model "$1" --dataset "$2" --split "$3" --stage "$4" ${5:+--judge "$5"}; }
hidden() { for site in answer judge verdict direct; do
             step "hidden $1 $2 $site"
             "$py" phase14/scripts/extract_hidden.py --model "$1" --dataset "$2" --split main --site "$site"
           done; }
delete_weights() {
  local name; name="$(repo "$1")"
  local dir="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${name//\//--}"
  step "delete weights of $1 ($dir)"
  rm -rf "$dir"
  df -h "$project_root" | tail -1
}

case "$mode" in
smoke)
  for model in $models; do
    step "smoke $model (vLLM)"
    "$py" phase14/scripts/smoke_model.py --model "$model"
    step "smoke $model (Transformers hidden states)"
    "$py" phase14/scripts/smoke_hidden.py --model "$model"
  done
  echo PHASE14_SMOKE_COMPLETE
  ;;
pilot)
  step "verify pilot lock"
  "$py" phase14/scripts/verify_lock.py --stage pilot
  for model in $models; do
    for dataset in $datasets; do
      step "pilot $model $dataset samples"; gen "$model" "$dataset" pilot samples
      for judge in base base_c; do step "pilot $model $dataset judge $judge"; gen "$model" "$dataset" pilot judge "$judge"; done
    done
  done
  step "verify pilot lock before scoring"
  "$py" phase14/scripts/verify_lock.py --stage pilot
  step "pilot report"
  "$py" phase14/scripts/pilot_report.py > logs/phase14_pilot_report.txt
  echo PHASE14_PILOT_COMPLETE
  ;;
main)
  # Amendment 2026-10-13 (GPU time): endpoint cells first, for the two new models only;
  # hidden-state sites answer, judge and direct (the verdict site is an agreement check
  # only). The protected opening runs as soon as the endpoint cells exist.
  step "verify main lock"
  "$py" phase14/scripts/verify_lock.py --stage main
  for model in $models; do
    for dataset in $datasets; do
      step "main $model $dataset samples"; gen "$model" "$dataset" main samples
      for judge in base base_c; do step "main $model $dataset judge $judge"; gen "$model" "$dataset" main judge "$judge"; done
      for site in answer judge direct; do
        step "hidden $model $dataset $site"
        "$py" phase14/scripts/extract_hidden.py --model "$model" --dataset "$dataset" --split main --site "$site"
      done
    done
  done
  step "verify main lock before the protected opening"
  "$py" phase14/scripts/verify_lock.py --stage main
  if [[ ! -f outputs/phase14_v1/main/analysis/report.json ]]; then
    step "protected analysis (endpoints opened once)"
    rm -rf outputs/phase14_v1/main/analysis
    "$py" phase14/scripts/analyze.py > logs/phase14_analysis_stdout.txt
  fi
  step "exploratory probe analysis"
  "$py" phase14/scripts/probe_analysis.py > logs/phase14_probe_stdout.txt
  step "report tables"
  "$py" phase14/scripts/report_tables.py > logs/phase14_report_tables.md
  echo PHASE14_MAIN_COMPLETE
  ;;
extra)
  # If GPU time remains after main: GSM8K (the report's 561) for the new models, then the
  # comparison cells are added without touching the endpoints.
  for model in $models; do
    for dataset in gsm8k13; do
      step "extra $model $dataset samples"; gen "$model" "$dataset" main samples
      for judge in base base_c; do step "extra $model $dataset judge $judge"; gen "$model" "$dataset" main judge "$judge"; done
      for site in answer judge direct; do
        step "hidden $model $dataset $site"
        "$py" phase14/scripts/extract_hidden.py --model "$model" --dataset "$dataset" --split main --site "$site"
      done
    done
  done
  "$py" phase14/scripts/analyze.py --add-comparison > logs/phase14_comparison_stdout.txt
  "$py" phase14/scripts/probe_analysis.py > logs/phase14_probe_stdout.txt
  "$py" phase14/scripts/report_tables.py > logs/phase14_report_tables.md
  echo PHASE14_EXTRA_COMPLETE
  ;;
*) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
