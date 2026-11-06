"""Phase 16 E1: nine-task A0 GPU engineering smoke, not pilot evidence."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path("/content/phase16")
OUT = ROOT / "gpu_smoke_v1"
AUDIT = OUT / "audit.jsonl"
SUMMARY = OUT / "summary.json"
EXPECTED = {
    "tasks.json": "cabac7e679b2de020e9de2cefa9c8d01d0f4b3c4f6f364600508aaafb7154cde",
    "score_boundary.py": "4cabc19a099497ca5b9d5f388fb3d8f7be6102c71cbd82aaa6f9f5e86cf52b21",
    "verify_code.py": "bd897486d4a331b114820dbd4a9b655da8eed3eac3383beb1b994c6496512ef7",
    "check_colab_sandbox.py": "1839a7c9aa2a6866ebd0ed8d5090934a400d1f465c84c27be081d2ea19108d18",
    "verify_runner_smoke.py": "279fd5cfd614f8efb766451a832880b6fc28e5358fb42a03473af234a32da03e",
}
PACKAGES = {
    "transformers": "4.46.3",
    "accelerate": "1.0.1",
    "bitsandbytes": "0.50.2",
}
MODEL = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM = "Write the requested Python function. Return only Python code."
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 512
WALL_SECONDS = 3600


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append(row: dict) -> None:
    with AUDIT.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def stage(name: str, **details) -> None:
    append({"type": "stage", "name": name, "timestamp": time.time(), **details})
    print(f"stage: {name}", flush=True)


def ensure_packages() -> dict[str, str]:
    import importlib.metadata as metadata
    present = {}
    for name in PACKAGES:
        try:
            present[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            present[name] = None
    if present != PACKAGES:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade",
             *[f"{name}=={version}" for name, version in PACKAGES.items()]],
            check=True, timeout=600,
        )
    return {name: metadata.version(name) for name in PACKAGES}


def main() -> None:
    if AUDIT.exists() or SUMMARY.exists():
        raise RuntimeError("Refusing to overwrite an existing E1 audit")
    for filename, expected in EXPECTED.items():
        actual = sha256(ROOT / filename)
        if actual != expected:
            raise RuntimeError(f"Frozen input hash mismatch: {filename}: {actual}")
    manifest = json.loads((ROOT / "tasks.json").read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    if manifest["purpose"] != "mechanics_only_excluded_from_pilot" or len(tasks) != 9:
        raise RuntimeError("Unexpected E1 task pack")
    if len({task["task_id"] for task in tasks}) != 9:
        raise RuntimeError("Duplicate engineering task ID")

    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    append({
        "type": "metadata", "amendment": "P16-E1", "model": MODEL,
        "scope": "engineering_only", "evaluation_split_opened": False,
        "task_count": len(tasks), "input_sha256": EXPECTED,
        "packages": PACKAGES, "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS, "decoding": "greedy",
        "wall_seconds": WALL_SECONDS,
    })
    try:
        if subprocess.run(["which", "bwrap"], capture_output=True).returncode:
            subprocess.run(["apt-get", "update", "-qq"], check=True, timeout=180)
            subprocess.run(["apt-get", "install", "-y", "-qq", "bubblewrap"],
                           check=True, timeout=180)
        sys.path.insert(0, str(ROOT))
        from check_colab_sandbox import main as check_sandbox
        from verify_runner_smoke import trusted_code
        from verify_code import verify

        check_sandbox()
        for task in tasks:
            if verify(task, trusted_code(task, inclusive=True))["label"] != "correct":
                raise RuntimeError(f"Trusted correct fixture failed: {task['task_id']}")
            if verify(task, trusted_code(task, inclusive=False))["label"] != "upper_boundary_signature":
                raise RuntimeError(f"Trusted exclusive fixture failed: {task['task_id']}")
        stage("sandbox_and_oracle_ready", trusted_tasks=len(tasks))

        versions = ensure_packages()
        stage("packages_ready", versions=versions)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU unavailable")
        stage("model_loading", cuda_device=torch.cuda.get_device_name(0),
              torch_version=torch.__version__)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        torch.manual_seed(20261103)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, device_map={"": 0}, torch_dtype=torch.float16,
            quantization_config=quantization, low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        model.eval()
        stage("model_ready")

        scores = []
        for index, task in enumerate(tasks):
            if time.time() - started > WALL_SECONDS:
                stage("wall_stop", completed=len(scores))
                break
            prompt = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": task["prompt"]},
            ]
            inputs = tokenizer.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True,
                return_tensors="pt",
            )
            if inputs.shape[1] > MAX_INPUT_TOKENS:
                raise RuntimeError(f"Prompt too long: {task['task_id']}")
            seed = int(hashlib.sha256(task["task_id"].encode()).hexdigest()[:8], 16)
            torch.manual_seed(seed)
            tick = time.time()
            with torch.inference_mode():
                output = model.generate(
                    input_ids=inputs.to("cuda:0"),
                    attention_mask=torch.ones_like(inputs).to("cuda:0"),
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = output[0, inputs.shape[1]:]
            completion = tokenizer.decode(generated, skip_special_tokens=True)
            scored = verify(task, completion)
            scores.append(scored)
            append({
                "type": "completion", "task_id": task["task_id"],
                "index": index, "seed": seed, "prompt": task["prompt"],
                "output": completion, "generated_tokens": int(generated.numel()),
                "elapsed_seconds": time.time() - tick, "score": scored,
            })
            print(f"item {index + 1}/{len(tasks)} {task['task_id']}: {scored['label']}", flush=True)

        labels = dict(Counter(row["label"] for row in scores))
        stage("complete", completed=len(scores), labels=labels)
        summary = {
            "schema_version": "phase16_a0_engineering_smoke_v1",
            "scope": "engineering_only", "task_count": len(tasks),
            "completed": len(scores), "labels": labels,
            "audit_sha256": sha256(AUDIT),
            "evaluation_split_opened": False,
        }
        SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        stage("failure", traceback=traceback.format_exc()[-5000:])
        raise


if __name__ == "__main__":
    main()
