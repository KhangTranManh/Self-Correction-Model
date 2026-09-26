#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"

python3 -m venv .venv-vllm
.venv-vllm/bin/python -m pip install --upgrade pip
.venv-vllm/bin/python -m pip install \
  'vllm==0.7.0' \
  'transformers==4.48.1' \
  'huggingface_hub==0.28.1' \
  'python-dotenv==1.0.1' \
  'PyYAML==6.0.2'
# The Phase 7 math verifier is frozen to SymPy 1.14.0. This overrides the
# earlier vLLM environment's incidental SymPy dependency without changing
# model weights or decoding.
.venv-vllm/bin/python -m pip install --no-deps 'sympy==1.14.0'
.venv-vllm/bin/python - <<'PY'
import sympy, torch, vllm
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.get_device_capability(0)[0] == 7, "Expected V100 compute capability 7.x"
assert vllm.__version__ == "0.7.0"
assert sympy.__version__ == "1.14.0"
print({"vllm": vllm.__version__, "torch": torch.__version__,
       "sympy": sympy.__version__, "gpu": torch.cuda.get_device_name(0)})
PY
