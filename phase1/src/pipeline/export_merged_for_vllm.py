"""Merge a trained LoRA adapter into BF16 weights and publish a vLLM-ready model.

Example:
    python -m src.export_merged_for_vllm \
        --adapter-dir outputs/phase1_lora \
        --output-dir outputs/Self_Correction_v1_merged \
        --repo-id Kxck/Self_Correction_v1

The Hugging Face credential is read only from ``HF_TOKEN``. It is never accepted
as a command-line argument, which keeps it out of process listings and logs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-shard-size", default="4GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")

    adapter_dir = args.adapter_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"No adapter_config.json in {adapter_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty merged output directory: {output_dir}"
        )

    # Verify repository access before spending time and disk space on the merge.
    api = HfApi(token=token)
    api.repo_info(repo_id=args.repo_id, repo_type="model")
    print(f"[export] repository access verified: {args.repo_id}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the BF16 merge")

    # Load the original BF16 base explicitly, then apply the QLoRA adapter. This
    # is the standard QLoRA deployment path and avoids dequantizing a BNB model.
    # It also creates a checkpoint that native vLLM can load with no BNB plugin.
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
    print("[export] BF16 base and adapter loaded; merging LoRA weights", flush=True)

    merged_model = model.merge_and_unload(safe_merge=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(str(output_dir))

    model_card = f"""---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: transformers
pipeline_tag: text-generation
tags:
- vllm
- unsloth
- self-correction
---

# Self_Correction_v1

Qwen2.5-7B-Instruct fine-tuned with verified math and code correction examples.
The failed attempt and objective verifier feedback are context; training loss is
computed only on the verified corrected response. This repository contains merged
BF16 weights and can be loaded directly by vLLM.
"""
    (output_dir / "README.md").write_text(model_card, encoding="utf-8")

    print("[export] merged checkpoint saved; uploading", flush=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload merged Self_Correction_v1 checkpoint",
    )
    total_bytes = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file())
    print(
        f"[export] upload complete: {args.repo_id} ({total_bytes / 2**30:.2f} GiB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
