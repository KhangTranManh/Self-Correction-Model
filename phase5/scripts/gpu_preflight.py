"""Read-only GPU and Python runtime inventory for a Phase 5 host."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys


def module_version(name: str):
    try:
        module = importlib.import_module(name)
        return getattr(module, "__version__", "installed")
    except Exception as error:
        return f"unavailable: {type(error).__name__}: {error}"


def inspect():
    result = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "nvidia_smi": None,
        "modules": {name: module_version(name) for name in
                    ("torch", "transformers", "bitsandbytes", "peft", "sympy")},
        "cuda_available": False,
        "devices": [],
    }
    try:
        command = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=15,
            check=False,
        )
        result["nvidia_smi"] = command.stdout.strip() if command.returncode == 0 else command.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        result["nvidia_smi"] = f"unavailable: {type(error).__name__}: {error}"
    try:
        import torch
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                result["devices"].append({
                    "index": index,
                    "name": properties.name,
                    "memory_gib": round(properties.total_memory / 2**30, 2),
                    "compute_capability": [properties.major, properties.minor],
                    "native_bfloat16": properties.major >= 8,
                })
    except Exception as error:
        result["torch_cuda_error"] = f"{type(error).__name__}: {error}"
    result["phase5_initial_generation_ready"] = (
        result["cuda_available"]
        and bool(result["devices"])
        and not any(result["modules"][name].startswith("unavailable")
                    for name in ("torch", "transformers", "sympy"))
    )
    result["phase5_adapter_evaluation_ready"] = (
        result["phase5_initial_generation_ready"]
        and not result["modules"]["peft"].startswith("unavailable")
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect()
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
