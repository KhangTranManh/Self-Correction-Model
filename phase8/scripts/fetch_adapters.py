"""Download the pinned private Phase 4 adapters and verify their locked weight hashes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "phase5/configs/experiments.yaml"
TARGETS = {
    "phase4_warmstart_v2": "phase4_exploration_warmstart_v2",
    "phase4_correction_sft_v3": "phase4_correction_sft_v3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set")
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    observed = {}
    for key, folder in TARGETS.items():
        target = ROOT / "outputs" / folder / "final_adapter"
        snapshot_download(repo_id=registry["checkpoint_hub_repos"][key],
                          revision=registry["checkpoint_hub_revisions"][key],
                          local_dir=target, token=token)
        actual = sha256(target / "adapter_model.safetensors")
        if actual != registry["checkpoint_adapter_sha256"][key]:
            raise RuntimeError(f"Adapter hash mismatch for {key}: {actual}")
        observed[key] = actual
    print(json.dumps({"status": "verified", "adapter_sha256": observed}, indent=2))


if __name__ == "__main__":
    main()
