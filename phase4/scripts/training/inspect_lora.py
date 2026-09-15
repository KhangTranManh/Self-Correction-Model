"""Report whether a saved LoRA adapter moved away from its zero-B initialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    args = parser.parse_args()
    tensors = load_file(args.adapter / "adapter_model.safetensors")
    result = {
        "tensor_count": len(tensors),
        "lora_a_abs_sum": sum(
            float(value.abs().sum()) for name, value in tensors.items() if "lora_A" in name
        ),
        "lora_b_abs_sum": sum(
            float(value.abs().sum()) for name, value in tensors.items() if "lora_B" in name
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
