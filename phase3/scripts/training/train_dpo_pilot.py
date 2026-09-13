"""Train a one-epoch decision-token DPO adapter from Decision-Only V1."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[2]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def encode_preference(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    prompt = row["prompt_messages"]
    prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    encoded: dict[str, Any] = {key: row[key] for key in ("preference_id", "pair_id", "dataset", "domain", "answer_state")}
    encoded["prompt_length"] = len(prompt_ids)
    for field in ("chosen", "rejected"):
        full_ids = tokenizer.apply_chat_template(
            prompt + [{"role": "assistant", "content": row[field]}],
            tokenize=True,
            add_generation_prompt=False,
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(f"Chat-template prefix mismatch: {row['preference_id']} {field}")
        if len(full_ids) > max_length:
            raise RuntimeError(f"Refusing truncation at {row['preference_id']}: {len(full_ids)} > {max_length}")
        if len(full_ids) == len(prompt_ids):
            raise RuntimeError(f"Empty completion at {row['preference_id']} {field}")
        encoded[field + "_ids"] = full_ids
    return encoded


def preference_logps(model: Any, item: dict[str, Any], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [item["chosen_ids"], item["rejected_ids"]]
    max_length = max(map(len, sequences))
    input_ids = torch.full((2, max_length), pad_token_id, dtype=torch.long, device="cuda")
    attention = torch.zeros((2, max_length), dtype=torch.long, device="cuda")
    completion_mask = torch.zeros((2, max_length - 1), dtype=torch.bool, device="cuda")
    prompt_length = item["prompt_length"]
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[index, :length] = torch.tensor(sequence, dtype=torch.long, device="cuda")
        attention[index, :length] = 1
        completion_mask[index, prompt_length - 1 : length - 1] = True
    outputs = model(input_ids=input_ids, attention_mask=attention, use_cache=False, return_dict=True)
    token_logps = F.log_softmax(outputs.logits[:, :-1].float(), dim=-1).gather(
        -1, input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    sequence_logps = (token_logps * completion_mask).sum(dim=-1)
    return sequence_logps[0], sequence_logps[1]


def summarize_margins(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "preference_accuracy": sum(value > 0 for value in values) / len(values),
        "ties": sum(value == 0 for value in values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def evaluate_margins(model: Any, rows: list[dict[str, Any]], pad_token_id: int) -> tuple[list[float], list[dict[str, Any]]]:
    model.eval()
    margins: list[float] = []
    audit: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, item in enumerate(rows):
            chosen, rejected = preference_logps(model, item, pad_token_id)
            margin = float((chosen - rejected).cpu())
            margins.append(margin)
            audit.append({
                "preference_id": item["preference_id"],
                "pair_id": item["pair_id"],
                "dataset": item["dataset"],
                "domain": item["domain"],
                "answer_state": item["answer_state"],
                "chosen_logp": float(chosen.cpu()),
                "rejected_logp": float(rejected.cpu()),
                "chosen_margin": margin,
            })
            if (index + 1) % 20 == 0 or index + 1 == len(rows):
                print(f"preference_eval={index + 1}/{len(rows)}", flush=True)
    return margins, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if int(config["training"]["epochs"]) != 1:
        raise RuntimeError("This pilot is deliberately limited to one epoch")

    data_path = resolve(config["data"]["train_path"])
    adapter_path = resolve(config["model"]["initial_adapter"])
    output_root = resolve(config["output"]["root"])
    final_adapter = resolve(config["output"]["final_adapter"])
    train_report_path = resolve(config["output"]["train_report"])
    output_root.mkdir(parents=True, exist_ok=True)
    source_rows = read_jsonl(data_path)
    expected_rows = int(config["data"]["expected_rows"])
    if len(source_rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} preferences, found {len(source_rows)}")
    if len({row["pair_id"] for row in source_rows}) != int(config["data"]["expected_pairs"]):
        raise RuntimeError("Unexpected pair count")

    base_model = config["model"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode_preference(tokenizer, row, int(config["data"]["max_length"])) for row in source_rows]
    lengths = [max(len(row["chosen_ids"]), len(row["rejected_ids"])) for row in encoded]

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=config["model"].get("attn_implementation", "sdpa"),
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(
        trainable,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    optimizer_steps = math.ceil(len(encoded) / accumulation)
    warmup_steps = max(1, round(optimizer_steps * float(config["training"]["warmup_ratio"])))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, optimizer_steps)

    started = time.time()
    reference_margins, reference_audit = evaluate_margins(model, encoded, tokenizer.pad_token_id)
    reference_path = output_root / "reference_logps.jsonl"
    write_jsonl(reference_path, reference_audit)
    reference_lookup = {
        row["preference_id"]: (row["chosen_logp"], row["rejected_logp"]) for row in reference_audit
    }

    order = list(range(len(encoded)))
    random.Random(seed).shuffle(order)
    beta = float(config["training"]["beta"])
    max_grad_norm = float(config["training"]["max_grad_norm"])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    history: list[dict[str, Any]] = []
    accumulated_loss = 0.0
    completed_steps = 0
    for micro_step, row_index in enumerate(order, start=1):
        item = encoded[row_index]
        chosen, rejected = preference_logps(model, item, tokenizer.pad_token_id)
        ref_chosen, ref_rejected = reference_lookup[item["preference_id"]]
        logit = beta * ((chosen - rejected) - (ref_chosen - ref_rejected))
        loss = -F.logsigmoid(logit)
        (loss / accumulation).backward()
        accumulated_loss += float(loss.detach().cpu())
        boundary = micro_step % accumulation == 0 or micro_step == len(order)
        if boundary:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            completed_steps += 1
            rows_in_step = accumulation if micro_step % accumulation == 0 else micro_step % accumulation
            event = {
                "optimizer_step": completed_steps,
                "micro_step": micro_step,
                "loss": accumulated_loss / rows_in_step,
                "learning_rate": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "elapsed_seconds": time.time() - started,
            }
            history.append(event)
            accumulated_loss = 0.0
            print(json.dumps(event, sort_keys=True), flush=True)

    final_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_adapter, safe_serialization=True)
    tokenizer.save_pretrained(final_adapter)
    final_margins, final_audit = evaluate_margins(model, encoded, tokenizer.pad_token_id)
    write_jsonl(output_root / "final_preference_audit.jsonl", final_audit)
    adapter_weights = final_adapter / "adapter_model.safetensors"
    gpu = torch.cuda.get_device_properties(0)
    report = {
        "schema_version": "phase3_dpo_pilot_train_v1",
        "experiment": config["experiment"],
        "model": {
            "base_model": base_model,
            "initial_adapter": str(adapter_path),
            "initial_adapter_sha256": file_hash(adapter_path / "adapter_model.safetensors"),
            "final_adapter": str(final_adapter),
            "final_adapter_sha256": file_hash(adapter_weights),
            "load_in_4bit": True,
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        },
        "data": {
            "path": str(data_path),
            "sha256": file_hash(data_path),
            "rows": len(encoded),
            "pairs": len({row["pair_id"] for row in encoded}),
            "dataset_distribution": dict(Counter(row["dataset"] for row in encoded)),
            "domain_distribution": dict(Counter(row["domain"] for row in encoded)),
            "token_lengths": {
                "min": min(lengths),
                "max": max(lengths),
                "mean": statistics.fmean(lengths),
                "truncated": 0,
            },
        },
        "training": {
            **config["training"],
            "optimizer_steps": completed_steps,
            "warmup_steps": warmup_steps,
            "history": history,
            "runtime_seconds": time.time() - started,
        },
        "preference_metrics": {
            "decision_only_v1_before": summarize_margins(reference_margins),
            "dpo_after": summarize_margins(final_margins),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": gpu.name,
            "gpu_memory_bytes": gpu.total_memory,
        },
        "validation": {
            "one_epoch": True,
            "frozen_reference_precomputed_before_updates": True,
            "completion_only_log_prob": True,
            "no_truncation": True,
            "hub_upload": False,
        },
    }
    write_json(train_report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
