#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu GPU host (needs >= 32 GB VRAM, CUDA driver, Python 3.10).
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"
mkdir -p logs
# vLLM's LoRA Triton kernels compile C code at runtime (Phase 8 lesson).
DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc g++ python3.10-dev python3.10-venv
python3.10 -m venv .venv-vllm
.venv-vllm/bin/python -m pip install -q --upgrade pip
.venv-vllm/bin/python -m pip install -q 'vllm==0.7.0' 'transformers==4.48.1' \
  'huggingface_hub==0.28.1' 'python-dotenv==1.0.1' 'PyYAML==6.0.2' 'peft==0.14.0' \
  'accelerate==1.2.1' 'scikit-learn==1.6.1' scipy joblib
.venv-vllm/bin/python -m pip install -q --no-deps 'sympy==1.14.0'
.venv-vllm/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Kxck/Self_Correction_v1', revision='6437f947999168a0ce2a98a86e4252fc77160a33')"
.venv-vllm/bin/python - <<'PY'
import torch, vllm, sympy, peft
assert torch.cuda.is_available() and vllm.__version__ == "0.7.0" and sympy.__version__ == "1.14.0"
print({"gpu": torch.cuda.get_device_name(0), "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)})
PY
.venv-vllm/bin/python phase11/scripts/verify_lock.py
echo BOOTSTRAP_DONE
