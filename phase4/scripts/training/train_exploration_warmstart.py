"""QLoRA SFT warm-start for the Phase 4 KEEP/REVISE policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

from datasets import Dataset
from peft import LoraConfig
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


class NonzeroGradientGuard(TrainerCallback):
    """Abort before the first optimizer step when every adapter gradient is zero."""

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if state.global_step != 0 or model is None:
            return control
        absolute_sum = sum(
            float(parameter.grad.detach().abs().sum())
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        if absolute_sum == 0.0:
            raise RuntimeError("QLoRA preflight found zero adapter gradients")
        print(f"QLORA_GRADIENT_PREFLIGHT_OK absolute_sum={absolute_sum:.6f}")
        return control


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "..":
        parts = parts[1:]
    return PROJECT_ROOT.joinpath(*parts)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_completion_dataset(rows: list[dict[str, Any]]) -> Dataset:
    values = []
    for row in rows:
        messages = row["messages"]
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            raise ValueError(f"Invalid warm-start row: {row.get('construction_id')}")
        values.append({"prompt": messages[:-1], "completion": [messages[-1]]})
    return Dataset.from_list(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    model_cfg = config["model"]
    data_cfg = config["data"]
    training_cfg = config["training"]
    output_cfg = config["output"]
    train_path = resolve_project_path(data_cfg["train_path"])
    dev_path = resolve_project_path(data_cfg["dev_path"])
    output_root = resolve_project_path(output_cfg["root"])
    adapter_path = resolve_project_path(output_cfg["final_adapter"])
    report_path = resolve_project_path(output_cfg["train_report"])
    train_rows = read_jsonl(train_path)
    dev_rows = read_jsonl(dev_path)
    if len(train_rows) != int(data_cfg["expected_train_rows"]):
        raise ValueError(f"Expected {data_cfg['expected_train_rows']} train rows, got {len(train_rows)}")
    if len(dev_rows) != int(data_cfg["expected_dev_rows"]):
        raise ValueError(f"Expected {data_cfg['expected_dev_rows']} dev rows, got {len(dev_rows)}")

    compute_dtype = torch.bfloat16
    quantization = BitsAndBytesConfig(
        load_in_4bit=bool(model_cfg["load_in_4bit"]),
        bnb_4bit_quant_type=str(model_cfg["quant_type"]),
        bnb_4bit_use_double_quant=bool(model_cfg["double_quant"]),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["training_checkpoint"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["training_checkpoint"],
        quantization_config=quantization,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=int(model_cfg["new_lora_r"]),
        lora_alpha=int(model_cfg["new_lora_alpha"]),
        lora_dropout=float(model_cfg["new_lora_dropout"]),
        target_modules=list(model_cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    sft_args = SFTConfig(
        output_dir=str(output_root),
        completion_only_loss=True,
        max_length=int(data_cfg["max_seq_length"]),
        num_train_epochs=float(training_cfg["epochs"]),
        per_device_train_batch_size=int(training_cfg["micro_batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        warmup_ratio=float(training_cfg["warmup_ratio"]),
        lr_scheduler_type=str(training_cfg["lr_scheduler_type"]),
        weight_decay=float(training_cfg["weight_decay"]),
        bf16=True,
        fp16=False,
        gradient_checkpointing=bool(training_cfg["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=str(training_cfg["optim"]),
        logging_steps=int(training_cfg["logging_steps"]),
        save_strategy=str(training_cfg["save_strategy"]),
        report_to="none",
        seed=seed,
        data_seed=seed,
    )
    trainer = SFTTrainer(
        model=model,
        peft_config=lora_config,
        args=sft_args,
        processing_class=tokenizer,
        train_dataset=prompt_completion_dataset(train_rows),
        eval_dataset=prompt_completion_dataset(dev_rows),
        callbacks=[NonzeroGradientGuard()],
    )
    # Let TRL prepare the quantized model and attach PEFT in its supported
    # order. Manual preparation can leave checkpointed QLoRA adapters with a
    # valid loss graph but zero adapter gradients.
    trainable, total = trainer.model.get_nb_trainable_parameters()
    before = trainer.evaluate(metric_key_prefix="eval_before")
    train_result = trainer.train()
    after = trainer.evaluate(metric_key_prefix="eval_after")
    adapter_path.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    report = {
        "experiment": config["experiment"],
        "config_path": str(args.config.resolve()),
        "data": {
            "train_path": str(train_path),
            "dev_path": str(dev_path),
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "train_sha256": sha256(train_path),
            "dev_sha256": sha256(dev_path),
        },
        "parameters": {"trainable": trainable, "total": total},
        "eval_before": before,
        "train": train_result.metrics,
        "eval_after": after,
        "adapter_path": str(adapter_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
