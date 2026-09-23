#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu118
.venv/bin/python -m pip install \
  'transformers==4.57.1' 'accelerate==1.10.1' \
  'huggingface_hub==0.35.3' 'python-dotenv==1.1.1' 'sympy==1.14.0' \
  'safetensors==0.6.2' 'PyYAML==6.0.3'
.venv/bin/python phase5/scripts/gpu_preflight.py --output phase5/runs/gpu_preflight.json
