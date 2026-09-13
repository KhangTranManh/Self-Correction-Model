"""Forward-only extraction of decision-router hidden representations."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-model", default="Kxck/Self_Correction_v1")
    parser.add_argument("--model", action="append", required=True, help="LABEL=ADAPTER_PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="7,14,21,28")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=256,
        help="Expected dataset size; use 0 to accept any non-empty unique-source dataset.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    activation_dir = output_dir / "activations"
    activation_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(dataset_path)
    layers = [int(value) for value in args.layers.split(",")]
    if not rows:
        raise RuntimeError("Probe dataset is empty")
    if args.expected_rows and len(rows) != args.expected_rows:
        raise RuntimeError(f"Expected {args.expected_rows} probe rows, found {len(rows)}")
    source_ids = [str(row["source_id"]) for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("Probe dataset contains duplicate source IDs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for activation extraction")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered = [
        tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        for row in rows
    ]
    encoded = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in rendered]
    lengths = [len(ids) for ids in encoded]
    if max(lengths) > args.max_length:
        raise RuntimeError(f"Refusing truncation: max tokens {max(lengths)} > {args.max_length}")

    manifest = []
    model_summaries = {}
    for specification in args.model:
        label, separator, adapter_text = specification.partition("=")
        if not separator:
            raise ValueError(f"Invalid --model value: {specification}")
        adapter = Path(adapter_text).resolve()
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
        model.eval()
        model.config.use_cache = False
        backbone = model.get_base_model().model
        values = []
        observed_hidden_count = None
        with torch.inference_mode():
            for index, input_ids in enumerate(encoded):
                tensor = torch.tensor([input_ids], dtype=torch.long, device="cuda")
                output = backbone(
                    input_ids=tensor,
                    attention_mask=torch.ones_like(tensor),
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                hidden_states = output.hidden_states
                observed_hidden_count = len(hidden_states)
                if any(layer >= len(hidden_states) for layer in layers):
                    raise RuntimeError(f"Requested layers {layers}, model has {len(hidden_states)} hidden-state entries")
                values.append(torch.stack([hidden_states[layer][0, -1] for layer in layers]).float().cpu().numpy())
                if (index + 1) % 25 == 0 or index + 1 == len(rows):
                    print(f"{label}: {index + 1}/{len(rows)}", flush=True)
        array = np.stack(values).astype(np.float16)
        activation_path = activation_dir / f"{label}_final_token.npz"
        np.savez_compressed(
            activation_path,
            activations=array,
            source_ids=np.asarray([row["source_id"] for row in rows]),
            layers=np.asarray(layers, dtype=np.int16),
            token_lengths=np.asarray(lengths, dtype=np.int16),
        )
        adapter_model = adapter / "adapter_model.safetensors"
        model_summaries[label] = {
            "adapter_path": str(adapter),
            "adapter_sha256": file_hash(adapter_model),
            "activation_path": str(activation_path),
            "activation_sha256": file_hash(activation_path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "hidden_state_entries": observed_hidden_count,
            "layers": layers,
            "pooling": "final_prompt_token_before_decision_generation",
        }
        for row_index, row in enumerate(rows):
            for layer_index, layer in enumerate(layers):
                manifest.append({
                    "source_id": row["source_id"],
                    "dataset": row["dataset"],
                    "domain": row["domain"],
                    "label": row["label"],
                    "class_id": row["class_id"],
                    "split": row.get("split", row.get("calibration_role", "unspecified")),
                    "model": label,
                    "layer": layer,
                    "representation_ref": f"activations/{activation_path.name}#activations[{row_index},{layer_index},:]",
                    "token_position": lengths[row_index] - 1,
                    "token_length": lengths[row_index],
                })
        del backbone, model, base, array, values
        gc.collect()
        torch.cuda.empty_cache()

    write_jsonl(output_dir / "activation_manifest.jsonl", manifest)
    summary = {
        "schema_version": "phase3_representation_extraction_v1",
        "forward_pass_only": True,
        "generation_performed": False,
        "weights_changed": False,
        "dataset": {"path": str(dataset_path), "sha256": file_hash(dataset_path), "rows": len(rows)},
        "token_lengths": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)},
        "models": model_summaries,
        "manifest_rows": len(manifest),
    }
    (output_dir / "activation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
