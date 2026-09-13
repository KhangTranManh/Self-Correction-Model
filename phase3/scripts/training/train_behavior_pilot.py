"""QLoRA-train Self_Correction_v1 with an explicit Phase 3 configuration.

The merged V1 checkpoint is loaded in 4-bit NF4 and a new Phase 3 LoRA is
attached. Only the final assistant turn contributes to the loss. The resulting
adapter is saved locally and is never uploaded by this script.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[2]
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
                raise ValueError(f"Invalid training row at {path}:{line_number}")
            rows.append(row)
    return rows


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FinalAssistantDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_seq_length: int,
    ) -> None:
        self.examples: list[dict[str, list[int]]] = []
        self.token_lengths: list[int] = []
        self.prompt_token_lengths: list[int] = []
        self.target_token_lengths: list[int] = []
        over_length: list[tuple[str, int]] = []
        for row in rows:
            messages = row["messages"]
            if len(messages) < 2 or messages[-1].get("role") != "assistant":
                raise ValueError(f"Invalid final assistant turn: {row['construction_id']}")
            prompt_text = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            target_text = str(messages[-1]["content"]) + tokenizer.eos_token
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
            input_ids = prompt_ids + target_ids
            if len(input_ids) > max_seq_length:
                over_length.append((row["construction_id"], len(input_ids)))
                continue
            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": [-100] * len(prompt_ids) + target_ids,
                }
            )
            self.token_lengths.append(len(input_ids))
            self.prompt_token_lengths.append(len(prompt_ids))
            self.target_token_lengths.append(len(target_ids))
        if over_length:
            raise RuntimeError(
                "Refusing to truncate pilot rows because that can remove task context: "
                + json.dumps(over_length)
            )
        if len(self.examples) != len(rows):
            raise RuntimeError("Tokenized row count mismatch")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]

    def stats(self) -> dict[str, Any]:
        def describe(values: list[int]) -> dict[str, float | int]:
            ordered = sorted(values)
            return {
                "min": min(values),
                "median": statistics.median(values),
                "p95": ordered[max(0, int(0.95 * len(ordered)) - 1)],
                "max": max(values),
            }

        return {
            "total": describe(self.token_lengths),
            "prompt": describe(self.prompt_token_lengths),
            "target": describe(self.target_token_lengths),
        }


@dataclass
class FinalAssistantCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(feature["input_ids"], dtype=torch.long) for feature in features]
        attention = [
            torch.tensor(feature["attention_mask"], dtype=torch.long) for feature in features
        ]
        labels = [torch.tensor(feature["labels"], dtype=torch.long) for feature in features]
        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id),
            "attention_mask": pad_sequence(attention, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
        }


def clean_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            cleaned[key] = value
        elif hasattr(value, "item"):
            cleaned[key] = value.item()
        else:
            cleaned[key] = str(value)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Explicit Phase 3 YAML config; no implicit experiment is selected.",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if config["experiment"].get("upload_to_hub"):
        raise RuntimeError("This pilot forbids Hub upload")
    epochs = float(config["training"]["epochs"])
    if not 1.0 <= epochs <= 2.0:
        raise RuntimeError("Pilot training is limited to one or two epochs")
    train_path = resolve(config["data"]["train_path"])
    dev_path = resolve(config["data"]["dev_path"])
    output_root = resolve(config["output"]["root"])
    final_adapter = resolve(config["output"]["final_adapter"])
    report_path = resolve(config["output"]["train_report"])
    train_rows = read_jsonl(train_path)
    dev_rows = read_jsonl(dev_path)
    if len(train_rows) != int(config["data"]["expected_train_rows"]):
        raise RuntimeError(
            f"Expected {config['data']['expected_train_rows']} train rows, found {len(train_rows)}"
        )
    if len(dev_rows) != int(config["data"]["expected_dev_rows"]):
        raise RuntimeError(
            f"Expected {config['data']['expected_dev_rows']} dev rows, found {len(dev_rows)}"
        )
    train_sources = {row["source_id"] for row in train_rows}
    dev_sources = {row["source_id"] for row in dev_rows}
    if train_sources & dev_sources:
        raise RuntimeError("Source leakage detected between train and dev")

    seed = int(config["experiment"]["seed"])
    set_seed(seed, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this QLoRA pilot")

    training_checkpoint = str(config["model"]["training_checkpoint"])
    original_base_model = str(config["model"]["original_base_model"])
    tokenizer = AutoTokenizer.from_pretrained(training_checkpoint, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = torch.bfloat16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(config["model"]["quant_type"]),
        bnb_4bit_use_double_quant=bool(config["model"]["double_quant"]),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        training_checkpoint,
        quantization_config=quantization,
        torch_dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=bool(config["training"]["gradient_checkpointing"]),
    )
    initial_adapter_value = config["model"].get("initial_adapter")
    if initial_adapter_value:
        initial_adapter = resolve(str(initial_adapter_value))
        if not (initial_adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Initial adapter not found: {initial_adapter}")
        model = PeftModel.from_pretrained(
            base_model,
            str(initial_adapter),
            is_trainable=True,
        )
        checkpoint_layout = "4-bit merged V1 base plus continued trainable decision-only LoRA"
    else:
        initial_adapter = None
        lora_config = LoraConfig(
            r=int(config["model"]["new_lora_r"]),
            lora_alpha=int(config["model"]["new_lora_alpha"]),
            lora_dropout=float(config["model"]["new_lora_dropout"]),
            target_modules=list(config["model"]["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        checkpoint_layout = "merged V1 full checkpoint; new Phase-3 LoRA attached"
    model.config.use_cache = False

    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    if trainable_params <= 0:
        raise RuntimeError("No trainable adapter parameters found")

    max_seq_length = int(config["data"]["max_seq_length"])
    train_dataset = FinalAssistantDataset(train_rows, tokenizer, max_seq_length)
    dev_dataset = FinalAssistantDataset(dev_rows, tokenizer, max_seq_length)
    output_root.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_root / "trainer"),
        num_train_epochs=float(config["training"]["epochs"]),
        per_device_train_batch_size=int(config["training"]["micro_batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_ratio=float(config["training"]["warmup_ratio"]),
        lr_scheduler_type=str(config["training"]["lr_scheduler_type"]),
        weight_decay=float(config["training"]["weight_decay"]),
        optim=str(config["training"]["optim"]),
        logging_steps=int(config["training"]["logging_steps"]),
        save_strategy=str(config["training"]["save_strategy"]),
        eval_strategy="no",
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=bool(config["training"]["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        prediction_loss_only=True,
        eval_accumulation_steps=1,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=seed,
        data_seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=FinalAssistantCollator(tokenizer.pad_token_id),
    )

    eval_before = clean_metrics(trainer.evaluate(metric_key_prefix="eval_before"))
    train_result = trainer.train()
    eval_after = clean_metrics(trainer.evaluate(metric_key_prefix="eval_after"))
    final_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_adapter, safe_serialization=True)
    tokenizer.save_pretrained(final_adapter)

    import bitsandbytes
    import peft
    import transformers

    gpu = torch.cuda.get_device_properties(0)
    report = {
        "experiment": config["experiment"],
        "model": {
            "original_base_model": original_base_model,
            "training_checkpoint": training_checkpoint,
            "checkpoint_layout": checkpoint_layout,
            "initial_adapter": str(initial_adapter) if initial_adapter else None,
            "quantization": "4-bit NF4 double-quant",
            "compute_dtype": "bfloat16",
            "trainable_parameters": trainable_params,
            "total_loaded_parameters": total_params,
            "trainable_percentage": 100 * trainable_params / total_params,
        },
        "data": {
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "train_unique_sources": len(train_sources),
            "dev_unique_sources": len(dev_sources),
            "source_overlap": [],
            "train_sha256": file_hash(train_path),
            "dev_sha256": file_hash(dev_path),
            "train_token_stats": train_dataset.stats(),
            "dev_token_stats": dev_dataset.stats(),
            "max_seq_length": max_seq_length,
            "truncated_rows": 0,
        },
        "training": config["training"],
        "eval_before": eval_before,
        "train_metrics": clean_metrics(train_result.metrics),
        "eval_after": eval_after,
        "log_history": trainer.state.log_history,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "gpu": gpu.name,
            "gpu_memory_bytes": gpu.total_memory,
        },
        "output": {
            "final_adapter": str(final_adapter),
            "uploaded_to_hub": False,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
