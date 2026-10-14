#!/usr/bin/env bash
# One-time setup of a Phase 14 GPU host (run from the project root after uploading the bundle).
#   bash phase14/scripts/ops/bootstrap.sh
# Installs the pinned environment of Phases 8-13 (.venv-vllm), checks GPU and disk,
# reads HF_TOKEN from .env without printing it, checks access to the gated model,
# downloads every model in config.RUN_MODELS at its pinned revision, and writes
# logs/phase14_bootstrap.json. Safe to re-run: finished steps are skipped.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"
mkdir -p logs

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free_gb=$(df -BG --output=avail "$project_root" | tail -1 | tr -dc '0-9')
echo "free disk: ${free_gb} GB"
need_gb=90; [[ -x .venv-vllm/bin/python ]] && need_gb=78   # environment (~11 GB) already installed
need_gb="${MIN_FREE_GB:-$need_gb}"   # override when some models are already in the cache
if (( free_gb < need_gb )); then
  echo "need at least ${need_gb} GB free (four models ~61 GB, environment ~11 GB, hidden states ~25-30 GB)" >&2
  exit 2
fi

if [[ -f .env ]] && grep -q '^HF_TOKEN=' .env; then
  HF_TOKEN="$(grep '^HF_TOKEN=' .env | head -1 | cut -d= -f2- | tr -d '"'"'"' \r')"
  export HF_TOKEN
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN missing: put it in .env or the environment" >&2; exit 3; }

if [[ ! -x .venv-vllm/bin/python ]]; then
  DEBIAN_FRONTEND=noninteractive apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc g++ python3.10-dev python3.10-venv
  python3.10 -m venv .venv-vllm
fi
py=.venv-vllm/bin/python
if ! "$py" -c "import vllm, sklearn, scipy, peft" 2>/dev/null; then
  "$py" -m pip install -q --upgrade pip
  "$py" -m pip install -q 'vllm==0.7.0' 'transformers==4.48.1' 'huggingface_hub==0.28.1' \
    'python-dotenv==1.0.1' 'PyYAML==6.0.2' 'peft==0.14.0' 'accelerate==1.2.1' \
    'scikit-learn==1.6.1' scipy joblib
  "$py" -m pip install -q --no-deps 'sympy==1.14.0'
fi

PYTHONPATH="$project_root" "$py" - <<'PY'
import json, os, sys
import torch, vllm, sympy, transformers
from huggingface_hub import hf_hub_download, snapshot_download
from phase14.scripts import config

assert sys.version_info[:2] == (3, 10), sys.version
assert vllm.__version__ == "0.7.0" and sympy.__version__ == "1.14.0", (vllm.__version__, sympy.__version__)
assert torch.cuda.is_available()
token = os.environ["HF_TOKEN"]
report = {"gpu": torch.cuda.get_device_name(0),
          "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
          "bf16_supported": torch.cuda.is_bf16_supported(), "python": sys.version.split()[0],
          "vllm": vllm.__version__, "transformers": transformers.__version__, "sympy": sympy.__version__,
          "models": {}}
for key in config.RUN_MODELS:
    spec = config.MODELS[key]
    try:  # a real file download: model metadata of a gated repo is public even without access
        hf_hub_download(spec["repo"], "config.json", revision=spec["revision"], token=token)
        path = snapshot_download(spec["repo"], revision=spec["revision"], token=token,
                                 allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"],
                                 ignore_patterns=["consolidated*", "original/*"])  # Mistral/Llama duplicate weights
    except Exception as exc:  # gated repo without an approved access request, or a bad token
        report["models"][key] = {"repo": spec["repo"], "status": f"NO ACCESS: {type(exc).__name__}"}
        print(json.dumps(report["models"][key]), flush=True)
        continue
    status = "downloaded"
    if spec.get("weight_sha256"):  # e.g. the Llama copy: shards must match Meta's hashes byte for byte
        import hashlib, pathlib
        for name, expected in spec["weight_sha256"].items():
            digest = hashlib.sha256()
            with open(pathlib.Path(path) / name, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 24), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                status = f"WEIGHT HASH MISMATCH: {name}"
        print(json.dumps({key: "weight sha256 " + ("verified" if status == "downloaded" else status)}), flush=True)
    if spec.get("tokenizer_repo"):  # tokenizer pinned separately (Llama: Meta-identical tokenizer files)
        snapshot_download(spec["tokenizer_repo"], revision=spec["tokenizer_revision"], token=token,
                          allow_patterns=["tokenizer*", "special_tokens_map.json"])
        print(json.dumps({key: "tokenizer from " + spec["tokenizer_repo"]}), flush=True)
    report["models"][key] = {"repo": spec["repo"], "revision": spec["revision"], "status": status, "path": path}
    print(json.dumps({key: "downloaded"}), flush=True)
with open("logs/phase14_bootstrap.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
missing = [k for k, v in report["models"].items() if v["status"] != "downloaded"]
print(json.dumps({"gpu": report["gpu"], "bf16": report["bf16_supported"], "missing": missing}))
if missing:
    sys.exit("Model access failed for " + ", ".join(missing) + " (replace with phi35_mini only per the preregistration)")
PY
echo PHASE14_BOOTSTRAP_DONE
