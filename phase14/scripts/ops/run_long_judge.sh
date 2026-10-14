#!/usr/bin/env bash
# Queue the 1536-token judge sensitivity for Llama after the completion run (exploratory).
# Waits for PHASE14_COMPLETION_DONE (or for run_complete.sh to stop), re-fetches Llama, re-judges the
# endpoint pairs (ASDiv, MATH 1-3) with base and base_c at 1536 tokens, deletes the weights.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
py=.venv-vllm/bin/python
step() { echo "=== $(date -u +%FT%TZ) $*"; }
until grep -q PHASE14_COMPLETION_DONE logs/phase14_complete.log 2>/dev/null || ! pgrep -f ops/run_complete.sh >/dev/null; do
  sleep 60
done
step "fetch llama31_8b"; "$py" phase14/scripts/ops/fetch_model.py llama31_8b
for dataset in asdiv math13; do
  for judge in base base_c; do
    step "long judge llama31_8b $dataset $judge (1536)"
    "$py" phase14/scripts/ops/judge_long.py llama31_8b "$dataset" "$judge"
  done
done
step "delete weights of llama31_8b"
rm -rf "$HOME/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct"
echo PHASE14_LONG_JUDGE_DONE
