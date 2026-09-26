"""Merge the hash-checked local V2 adapter into the pinned original FP16 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase7.scripts.collect_initials_4bit import sha256, write_atomic

REGISTRY = ROOT / "phase7/configs/experiments.yaml"
ADAPTER = ROOT / "outputs/phase4_exploration_warmstart_v2/final_adapter"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "models/phase7_v2_merged_fp16")
    args = parser.parse_args()
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    original = registry["models"]["original_solver"]
    warmstart = registry["models"]["warmstart_v2"]
    expected = {
        "schema_version": "phase7_v2_merged_lineage_v1",
        "original_repo": original["repo"],
        "original_revision": original["revision"],
        "adapter_repo": warmstart["repo"],
        "adapter_revision": warmstart["revision"],
        "adapter_weight_sha256": warmstart["adapter_sha256"],
        "dtype": "float16",
    }
    if sha256(ADAPTER / "adapter_model.safetensors") != expected["adapter_weight_sha256"]:
        raise ValueError("Local V2 adapter weight hash mismatch")
    marker = args.output / "phase7_lineage.json"
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError("Existing merged V2 has different lineage")
        print(json.dumps({"status": "already_materialized", "output": str(args.output)}))
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite non-empty merged V2 directory")
    tokenizer = AutoTokenizer.from_pretrained(original["repo"], revision=original["revision"])
    base = AutoModelForCausalLM.from_pretrained(
        original["repo"], revision=original["revision"], torch_dtype=torch.float16,
        device_map={"": "cpu"}, low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER), is_trainable=False,
                                    device_map={"": "cpu"})
    merged = model.merge_and_unload(safe_merge=True)
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(args.output)
    write_atomic(marker, json.dumps(expected, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output": str(args.output), "lineage": expected},
                     indent=2))


if __name__ == "__main__":
    main()
