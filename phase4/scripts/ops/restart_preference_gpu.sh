#!/usr/bin/env bash
set -euo pipefail

cd /root/agi
mkdir -p phase4/runs/preference_v1 /root/models
if ! command -v gcc >/dev/null 2>&1 || [[ ! -f /usr/include/python3.10/Python.h ]]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential python3.10-dev
fi
if [[ ! -x /root/.local/bin/uv ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
bash phase4/scripts/ops/bootstrap_gpu.sh

/root/venv-train/bin/python -u - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Kxck/Self_Correction_v1",
    local_dir="/root/models/Kxck_Self_Correction_v1",
    max_workers=4,
)
PY

if [[ ! -f /root/models/phase4_warmstart_v2_merged/phase4_merge_report.json ]]; then
  /root/venv-train/bin/python -u phase4/scripts/training/merge_adapter.py \
    --base-model /root/models/Kxck_Self_Correction_v1 \
    --adapter outputs/phase4_exploration_warmstart_v2/final_adapter \
    --output /root/models/phase4_warmstart_v2_merged
fi

/root/venv-train/bin/python -u phase4/scripts/training/train_preference_dpo.py \
  --config phase4/configs/preference_dpo_v1.yaml \
  >phase4/runs/preference_v1/train_preference_dpo_v1.log 2>&1
