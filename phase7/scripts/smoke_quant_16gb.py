"""Measure 4-bit loading and one short generation on a single CUDA GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "phase7/configs/blind_resolve_v1.yaml"
REGISTRY = ROOT / "phase7/configs/experiments.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/phase7_quant_smoke/report.json")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    generation = config["generation_draft"]
    if generation["backend"] != "transformers_bitsandbytes_4bit":
        raise RuntimeError("Unexpected Phase 7 backend")
    model_info = registry["models"]["original_solver"]
    model_id, revision = model_info["repo"], model_info["revision"]
    gpu = torch.cuda.get_device_properties(0)
    free_before, total = torch.cuda.mem_get_info(0)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.monotonic()
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, quantization_config=quantization,
        device_map={"": 0}, torch_dtype=torch.float16,
    )
    model.eval()
    free_after_load, _ = torch.cuda.mem_get_info(0)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt = "Solve this arithmetic problem and give the final answer: What is 17 + 25?"
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt",
    ).to("cuda:0")
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded, do_sample=False, max_new_tokens=64,
            pad_token_id=tokenizer.pad_token_id,
        )
    output = tokenizer.decode(output_ids[0, encoded["input_ids"].shape[-1]:],
                              skip_special_tokens=True)
    free_after_generation, _ = torch.cuda.mem_get_info(0)
    report = {
        "schema_version": "phase7_quant_smoke_v1",
        "model": model_id, "revision": revision,
        "quantization": "bitsandbytes_nf4_double_quant",
        "compute_dtype": "float16", "gpu_name": gpu.name,
        "compute_capability": f"{gpu.major}.{gpu.minor}",
        "gpu_total_bytes": total,
        "gpu_free_before_bytes": free_before,
        "gpu_free_after_load_bytes": free_after_load,
        "gpu_free_after_generation_bytes": free_after_generation,
        "gpu_used_by_load_bytes": free_before - free_after_load,
        "gpu_used_by_generation_bytes": free_before - free_after_generation,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "smoke_output": output,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": sys.version.split()[0],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
