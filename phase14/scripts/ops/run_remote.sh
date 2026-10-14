#!/usr/bin/env bash
# Phase 14 GPU pipeline. Usage: run_remote.sh smoke | pilot | main | comparison
#   smoke       technical check of every model, vLLM then Transformers hidden states (no project data)
#   comparison  after main, with the report's outputs uploaded: regression check on
#               Self_Correction's report outputs and the bridge set; never touches main cells
#   pilot       verify the pilot lock, generate pilot samples and judges, write the pilot report
#   main        verify the main lock; all four models, identical protocol, on ASDiv, MATH,
#               GSM8K and SVAMP; extract hidden states (label-free); verify again; open
#               once; probe analysis; report tables
# Every step resumes or is skipped when complete.
set -euo pipefail

mode="${1:?usage: run_remote.sh smoke|pilot|main|comparison}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"
mkdir -p logs
py="$project_root/.venv-vllm/bin/python"
step() { echo "=== $(date -u +%FT%TZ) $*"; }
cfg() { "$py" -c "import sys; sys.path.insert(0, '.'); from phase14.scripts import config; print(' '.join(config.$1) if isinstance(config.$1, tuple) else config.$1)"; }
models=$(cfg RUN_MODELS)
datasets=$(cfg DATASETS)
comparison=$(cfg COMPARISON)
bridge=$(cfg BRIDGE)

if [[ "$mode" == smoke ]]; then
  for model in $models; do
    step "smoke $model (vLLM)"
    "$py" phase14/scripts/smoke_model.py --model "$model"
    step "smoke $model (Transformers hidden states)"
    "$py" phase14/scripts/smoke_hidden.py --model "$model"
  done
  echo PHASE14_SMOKE_COMPLETE
  exit 0
fi

if [[ "$mode" == comparison ]]; then
  # After the main run, once outputs/phase13_v1 and outputs/phase12_v1 are uploaded:
  # Self_Correction on GSM8K/SVAMP (the report's own outputs), the bridge set, the
  # regression check against the report, then the comparison cells are added to the
  # analysis without touching the endpoints.
  [[ -f outputs/phase14_v1/main/analysis/report.json ]] || { echo "run main first" >&2; exit 2; }
  for stage in outputs/phase13_v1/samples outputs/phase13_v1/judge_base outputs/phase13_v1/judge_base_c                outputs/phase12_v1/samples outputs/phase12_v1/judge_base; do
    [[ -f "$stage/summary.json" ]] || { echo "missing $stage: upload outputs/phase13_v1 and outputs/phase12_v1 first" >&2; exit 3; }
  done
  step "verify main lock"
  "$py" phase14/scripts/verify_lock.py --stage main
  for dataset in gsm8k13_report svamp12_report; do
    for judge in base base_c; do   # base: imported from the report's run; base_c: re-run with log-probabilities
      step "regression original_solver $dataset judge $judge"
      "$py" phase14/scripts/generate.py --model original_solver --dataset "$dataset" --split main --stage judge --judge "$judge"
    done
  done
  for model in $models; do
    [[ "$model" == original_solver ]] && continue   # its bridge cell is its own GSM8K cell
    for judge in base base_c; do
      step "bridge $model judge $judge"
      "$py" phase14/scripts/generate.py --model "$model" --dataset "$bridge" --split main --stage judge --judge "$judge"
    done
  done
  for spec in $(for m in $models; do [[ $m != original_solver ]] && echo "$m:$bridge"; done); do
    for site in answer judge verdict direct; do
      step "hidden ${spec%%:*} ${spec##*:} $site"
      "$py" phase14/scripts/extract_hidden.py --model "${spec%%:*}" --dataset "${spec##*:}" --split main --site "$site"
    done
  done
  step "regression check against the report"
  "$py" phase14/scripts/regression.py > logs/phase14_regression.txt
  step "add comparison cells (endpoints unchanged)"
  "$py" phase14/scripts/analyze.py --add-comparison > logs/phase14_comparison_stdout.txt
  step "exploratory probe analysis (all cells)"
  "$py" phase14/scripts/probe_analysis.py > logs/phase14_probe_stdout.txt
  step "report tables"
  "$py" phase14/scripts/report_tables.py > logs/phase14_report_tables.md
  echo PHASE14_COMPARISON_COMPLETE
  exit 0
fi

split="$mode"
[[ "$split" == pilot || "$split" == main ]] || { echo "unknown mode: $mode" >&2; exit 2; }
step "verify $split lock"
"$py" phase14/scripts/verify_lock.py --stage "$split"
for model in $models; do
  for dataset in $datasets; do
    step "$split $model $dataset samples"
    "$py" phase14/scripts/generate.py --model "$model" --dataset "$dataset" --split "$split" --stage samples
    for judge in base base_c; do
      step "$split $model $dataset judge $judge"
      "$py" phase14/scripts/generate.py --model "$model" --dataset "$dataset" --split "$split" --stage judge --judge "$judge"
    done
  done
done

if [[ "$split" == pilot ]]; then
  step "verify pilot lock before scoring"
  "$py" phase14/scripts/verify_lock.py --stage pilot
  step "pilot report"
  "$py" phase14/scripts/pilot_report.py > logs/phase14_pilot_report.txt
  echo PHASE14_PILOT_COMPLETE
  exit 0
fi

# Main run: the uniform cells. All four models solve and judge GSM8K and SVAMP with the
# identical protocol (Self_Correction included); nothing from earlier phases is reused.
for model in $models; do
  for dataset in $comparison; do
    step "main $model $dataset samples"
    "$py" phase14/scripts/generate.py --model "$model" --dataset "$dataset" --split main --stage samples
    for judge in base base_c; do
      step "main $model $dataset judge $judge"
      "$py" phase14/scripts/generate.py --model "$model" --dataset "$dataset" --split main --stage judge --judge "$judge"
    done
  done
done
for model in $models; do
  for dataset in $datasets $comparison; do
    for site in answer judge verdict direct; do
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
