"""Merge a Phase 4 LoRA adapter into a BF16 checkpoint for the next cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter_config.json in {args.adapter}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload(safe_merge=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="4GB")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    tokenizer.save_pretrained(args.output)
    report = {
        "base_model": args.base_model,
        "adapter": str(args.adapter.resolve()),
        "output": str(args.output.resolve()),
        "dtype": "bfloat16",
        "safe_merge": True,
    }
    (args.output / "phase4_merge_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
