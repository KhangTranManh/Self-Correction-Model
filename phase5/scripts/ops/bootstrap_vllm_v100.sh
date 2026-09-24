#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"

# vLLM releases after the V100 support window require compute capability >= 7.5.
# The frozen Phase 5 host is a V100 (CC 7.0), so this environment is deliberately
# isolated and pinned.  FP16 is mandatory because V100 has no native BF16 path.
if ! command -v cc >/dev/null 2>&1 || [[ ! -f /usr/include/python3.10/Python.h ]]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential python3-dev
fi
python3 -m venv .venv-vllm
.venv-vllm/bin/python -m pip install --upgrade pip
.venv-vllm/bin/python -m pip install \
  'vllm==0.7.0' \
  'transformers==4.48.1' \
  'huggingface_hub==0.28.1' \
  'python-dotenv==1.0.1' \
  'sympy==1.13.1' \
  'PyYAML==6.0.2'

.venv-vllm/bin/python - <<'PY'
import torch
import vllm

assert torch.cuda.is_available(), "CUDA is not available"
major, minor = torch.cuda.get_device_capability(0)
assert (major, minor) == (7, 0), f"Expected frozen V100 CC 7.0, got {major}.{minor}"
print({
    "vllm": vllm.__version__,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": f"{major}.{minor}",
})
PY
