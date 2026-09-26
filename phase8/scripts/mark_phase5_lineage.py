"""Add the validated Phase 5 lineage marker to a Phase 7 V2 FP16 merge."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/phase7_v2_merged_fp16"
LOCK = ROOT / "phase8/data/model_lineage_v1.json"


def main() -> None:
    locked = json.loads(LOCK.read_text(encoding="utf-8"))["v2_merged_lineage"]
    observed = json.loads((MODEL / "phase7_lineage.json").read_text(encoding="utf-8"))
    if observed != locked:
        raise RuntimeError("Phase 7 V2 merged lineage mismatch")
    expected = {
        "base_repo": locked["original_repo"],
        "base_revision": locked["original_revision"],
        "adapter_repo": locked["adapter_repo"],
        "adapter_revision": locked["adapter_revision"],
        "adapter_weight_sha256": locked["adapter_weight_sha256"],
        "weight_dtype": "float16",
    }
    marker = MODEL / "phase5_lineage.json"
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("Existing Phase 5 lineage marker differs")
    else:
        marker.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "verified", "path": str(marker), "lineage": expected}, indent=2))


if __name__ == "__main__":
    main()
