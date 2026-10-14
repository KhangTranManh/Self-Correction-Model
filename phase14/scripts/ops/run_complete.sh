#!/usr/bin/env bash
# Completion run after the protected opening (amendment 2026-10-13): fills the grid
# {Self_Correction, Qwen2.5-7B-Instruct, Llama-3.1-8B, Mistral-7B} x {GSM8K, SVAMP, ASDiv, MATH 1-3}
# with the identical protocol, plus the bridge set and the base_c regression re-run, one model
# at a time (weights fetched, used, deleted). Generation through gen_batch.py (256-request
# chunks). Hidden-state sites answer, judge, direct. Then analyze.py --add-comparison (scored
# cells, i.e. the endpoints, are never recomputed), regression check, probes, report tables.
# Every step resumes or is skipped when complete.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
py=.venv-vllm/bin/python
step() { echo "=== $(date -u +%FT%TZ) $*"; }
gen() { "$py" phase14/scripts/ops/gen_batch.py "${GEN_BATCH:-256}" --model "$1" --dataset "$2" --split main --stage "$3" ${4:+--judge "$4"}; }
hidden() {
  for site in answer judge direct; do
    step "hidden $1 $2 $site"
    "$py" phase14/scripts/extract_hidden.py --model "$1" --dataset "$2" --split main --site "$site"
  done
}
cell() {  # model dataset: samples, both judges, hidden states
  step "complete $1 $2 samples"; gen "$1" "$2" samples
  for judge in base base_c; do step "complete $1 $2 judge $judge"; gen "$1" "$2" judge "$judge"; done
  hidden "$1" "$2"
}
bridge() {
  for judge in base base_c; do step "bridge $1 judge $judge"; gen "$1" bridge13 judge "$judge"; done
  hidden "$1" bridge13
}
delete_weights() {
  local name
  name="$("$py" -c "import sys; sys.path.insert(0, '.'); from phase14.scripts.config import MODELS; print(MODELS['$1']['repo'])")"
  step "delete weights of $1"
  rm -rf "$HOME/.cache/huggingface/hub/models--${name//\//--}"
  df -h . | tail -1
}

step "verify completion lock"; "$py" phase14/scripts/verify_lock.py --stage completion
for model in llama31_8b mistral_7b; do
  [[ -f "outputs/phase14_v1/main/.complete_$model" ]] && continue
  step "fetch $model"; "$py" phase14/scripts/ops/fetch_model.py "$model"
  for dataset in gsm8k13 svamp12; do cell "$model" "$dataset"; done
  bridge "$model"
  touch "outputs/phase14_v1/main/.complete_$model"
  delete_weights "$model"
done
for model in original_solver qwen25_instruct; do
  [[ -f "outputs/phase14_v1/main/.complete_$model" ]] && continue
  step "fetch $model"; "$py" phase14/scripts/ops/fetch_model.py "$model"
  for dataset in asdiv math13 gsm8k13 svamp12; do cell "$model" "$dataset"; done
  if [[ "$model" == original_solver ]]; then
    for dataset in gsm8k13_report svamp12_report; do
      for judge in base base_c; do step "regression $model $dataset judge $judge"; gen "$model" "$dataset" judge "$judge"; done
    done
  else
    bridge "$model"
  fi
  touch "outputs/phase14_v1/main/.complete_$model"
  delete_weights "$model"
done
step "verify completion lock"; "$py" phase14/scripts/verify_lock.py --stage completion
step "add the completion cells (endpoints unchanged)"
"$py" phase14/scripts/analyze.py --add-comparison > logs/phase14_comparison_stdout.txt
step "regression check"
"$py" phase14/scripts/regression.py > logs/phase14_regression.txt || echo "regression reported a mismatch (see log)"
step "probe analysis (all models)"; "$py" phase14/scripts/probe_analysis.py > logs/phase14_probe_stdout.txt
step "report tables"; "$py" phase14/scripts/report_tables.py > logs/phase14_report_tables.md
echo PHASE14_COMPLETION_DONE
