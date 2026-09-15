#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/agi}
UV_BIN=${UV_BIN:-/root/.local/bin/uv}

"${UV_BIN}" venv --allow-existing --python python3.10 /root/venv-vllm
"${UV_BIN}" pip install \
  --python /root/venv-vllm/bin/python \
  -r "${PROJECT_ROOT}/phase4/requirements-vllm.txt"

"${UV_BIN}" venv --allow-existing --python python3.10 /root/venv-train
"${UV_BIN}" pip install \
  --python /root/venv-train/bin/python \
  -r "${PROJECT_ROOT}/phase4/requirements-train.txt"

/root/venv-vllm/bin/python - <<'PY'
import torch
import transformers
import vllm

print(
    "VLLM_OK",
    torch.__version__,
    transformers.__version__,
    vllm.__version__,
    torch.cuda.get_device_name(0),
)
PY

/root/venv-train/bin/python - <<'PY'
import bitsandbytes
import peft
import torch
import transformers
import trl

print(
    "TRAIN_OK",
    torch.__version__,
    transformers.__version__,
    peft.__version__,
    bitsandbytes.__version__,
    trl.__version__,
    torch.cuda.get_device_name(0),
)
PY
