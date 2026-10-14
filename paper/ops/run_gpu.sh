#!/usr/bin/env bash
# Paper checklist GPU pipeline (sequential, resumable: every step skips finished outputs).
set -euo pipefail
cd /root/AGI
PY=/root/AGI/.venv-vllm/bin/python
export PYTHONPATH=/root/AGI HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false
mkdir -p logs outputs/paper
step() { echo "=== $(date -u +%FT%TZ) $*"; }
for ds in phase13 phase12; do
  for j in base p12 base_c p13b; do
    step "extract $ds $j"; $PY phase13/layer/extract_hidden.py --dataset $ds --judge $j
  done
done
step "c1_c4 + c2b_f1 (CPU, background)"
( $PY paper/scripts/c1_c4_probe.py > logs/c1_c4.log 2>&1; $PY paper/scripts/gpu/c2b_f1.py > logs/c2b_f1.log 2>&1; echo CPU_DONE >> logs/c2b_f1.log ) &
step "judge variants base"; $PY paper/scripts/gpu/judge_variants.py --model base
step "c3 greedy"; $PY paper/scripts/gpu/c3_greedy.py
step "judge variants instruct"; $PY paper/scripts/gpu/judge_variants.py --model instruct
step "c3 logprob"; $PY paper/scripts/gpu/c3_logprob.py
step "c5 logit lens"; $PY paper/scripts/gpu/c5_logit_lens.py
step "f2 pointwise"; $PY paper/scripts/gpu/f2_pointwise.py
wait
step "PIPELINE_DONE"
