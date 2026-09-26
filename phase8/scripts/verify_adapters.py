"""Check local V2/V3 adapter weights before launching Phase 8 GPU stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    lineage = json.loads((ROOT / "phase8/data/model_lineage_v1.json").read_text(encoding="utf-8"))
    required = {
        "phase4_exploration_warmstart_v2": lineage["v2_merged_lineage"]["adapter_weight_sha256"],
        "phase4_correction_sft_v3": lineage["v3_adapter_sha256"],
    }
    observed = {}
    for name, expected in required.items():
        path = ROOT / "outputs" / name / "final_adapter/adapter_model.safetensors"
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Adapter hash mismatch for {name}: {actual}")
        observed[name] = actual
    print(json.dumps({"status": "verified", "adapter_sha256": observed}, indent=2))


if __name__ == "__main__":
    main()
