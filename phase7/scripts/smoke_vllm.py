"""Load the frozen original solver in FP16 vLLM and generate one short answer."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]


def gpu_used_mib() -> int | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return None
    return int(result.stdout.strip().splitlines()[0])


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if vllm.__version__ != "0.7.0" or not torch.cuda.is_available():
        raise RuntimeError("Pinned vLLM 0.7.0 and CUDA are required")
    registry = yaml.safe_load((ROOT / "phase7/configs/experiments.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "phase7/configs/blind_resolve_v1.yaml").read_text(encoding="utf-8"))
    model_info = registry["models"]["original_solver"]
    model_id, revision = model_info["repo"], model_info["revision"]
    before = gpu_used_mib()
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    llm = LLM(
        model=model_id, revision=revision,
        tokenizer=model_id, tokenizer_revision=revision,
        dtype="half", max_model_len=int(config["generation_draft"]["max_model_len"]),
        gpu_memory_utilization=float(config["generation_draft"]["gpu_memory_utilization"]),
        enforce_eager=True, trust_remote_code=False,
    )
    after_load = gpu_used_mib()
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Solve this arithmetic problem. What is 17 + 25?"}],
        tokenize=False, add_generation_prompt=True,
    )
    response = llm.generate([rendered], SamplingParams(temperature=0.0, max_tokens=64),
                            use_tqdm=False)[0].outputs[0].text
    after_generation = gpu_used_mib()
    report = {
        "schema_version": "phase7_vllm_smoke_v1",
        "model": model_id, "revision": revision, "vllm": vllm.__version__,
        "dtype": "float16", "quantization": "none",
        "gpu": torch.cuda.get_device_name(0),
        "total_memory_mib": torch.cuda.get_device_properties(0).total_memory // 2**20,
        "used_before_mib": before, "used_after_load_mib": after_load,
        "used_after_generation_mib": after_generation,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "smoke_response": response, "python": sys.version.split()[0],
    }
    output = ROOT / "outputs/phase7_vllm_smoke/report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
