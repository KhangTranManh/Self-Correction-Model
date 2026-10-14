"""Download one Phase 14 model (pinned revision) into the Hugging Face cache; HF_TOKEN from .env.

Usage: fetch_model.py <model key>. Same file patterns and weight-hash check as bootstrap.sh.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from huggingface_hub import snapshot_download  # noqa: E402
from phase14.scripts.config import MODELS  # noqa: E402

key = sys.argv[1]
spec = MODELS[key]
env = ROOT / ".env"
token = os.environ.get("HF_TOKEN") or next((line.split("=", 1)[1].strip().strip('"').strip("'")
                                            for line in env.read_text(encoding="utf-8").splitlines()
                                            if line.startswith("HF_TOKEN=")), None)
path = snapshot_download(spec["repo"], revision=spec["revision"], token=token,
                         allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"],
                         ignore_patterns=["consolidated*", "original/*"])
for name, expected in spec.get("weight_sha256", {}).items():
    digest = hashlib.sha256()
    with open(Path(path) / name, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        sys.exit(f"weight hash mismatch: {name}")
if spec.get("tokenizer_repo"):
    snapshot_download(spec["tokenizer_repo"], revision=spec["tokenizer_revision"], token=token,
                      allow_patterns=["tokenizer*", "special_tokens_map.json"])
print(json.dumps({"model": key, "repo": spec["repo"], "revision": spec["revision"], "path": path}))
