"""Create a provenance-explicit copy of a generation manifest for another model origin."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MODEL_IDS = {
    "base": "Qwen/Qwen2.5-7B-Instruct",
    "self_correction_v1": "Kxck/Self_Correction_v1",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--origin", choices=sorted(MODEL_IDS), required=True)
    args = parser.parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    remapped = []
    for row in rows:
        item = dict(row)
        old_task = str(item["task_id"])
        parts = old_task.split("::")
        if len(parts) < 3:
            raise RuntimeError(f"Unexpected task ID: {old_task}")
        parts[-2] = args.origin
        item["task_id"] = "::".join(parts)
        item["model_origin_to_sample"] = args.origin
        item["model_id_to_sample"] = MODEL_IDS[args.origin]
        item["pair_id"] = f"router_recovery_v2::{item['source_id']}"
        remapped.append(item)
    if len({row["task_id"] for row in remapped}) != len(remapped):
        raise RuntimeError("Remapping created duplicate task IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in remapped:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "input": str(source), "output": str(output), "origin": args.origin,
        "tasks": len(remapped), "jobs": sum(int(row["samples_requested"]) for row in remapped),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
