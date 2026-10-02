"""LoRA SFT of the original solver on its own verified judge outputs.

Loss is on assistant tokens only. FP16 base weights are frozen; LoRA weights
train in FP32 with FP16 autocast (V100 has no BF16). Hyperparameters are
fixed in advance: r=16, alpha=32, dropout 0.05, all attention and MLP
projections, learning rate 1e-4 cosine with 3% warmup, one epoch, effective
batch 16, maximum length 3072 tokens.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phase10.scripts.generate import MODEL, OUT, REVISION, read_jsonl  # noqa: E402

SFT = OUT / "sft"
ADAPTER_DIR = OUT / "judge_lora"
CONFIG = {"r": 16, "alpha": 32, "dropout": 0.05, "lr": 1e-4, "warmup_ratio": 0.03,
          "epochs": 1, "grad_accum": 16, "max_length": 3072, "seed": 20261001,
          "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}


def encode(tokenizer, example: dict) -> dict | None:
    prompt = tokenizer.apply_chat_template(example["messages"][:1], tokenize=False,
                                           add_generation_prompt=True)
    full = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if ids[:len(prompt_ids)] != prompt_ids or len(ids) > CONFIG["max_length"]:
        return None
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + ids[len(prompt_ids):]}


def main() -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    final = ADAPTER_DIR / "final_adapter"
    if (final / "adapter_model.safetensors").exists():
        print(json.dumps({"status": "already_trained"}))
        return
    random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    report = json.loads((SFT / "report.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    data = {}
    for split in ("train", "validation"):
        path = SFT / f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != report[f"{split}_sha256"]:
            raise RuntimeError(f"SFT {split} changed")
        encoded = [encode(tokenizer, ex) for ex in read_jsonl(path)]
        data[split] = [ex for ex in encoded if ex is not None]
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float16,
                                                 device_map={"": 0}, attn_implementation="sdpa")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=CONFIG["r"], lora_alpha=CONFIG["alpha"],
                                             lora_dropout=CONFIG["dropout"],
                                             target_modules=CONFIG["targets"], task_type="CAUSAL_LM"))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=CONFIG["lr"], weight_decay=0.0)
    order = list(range(len(data["train"])))
    random.shuffle(order)
    steps = math.ceil(len(order) * CONFIG["epochs"] / CONFIG["grad_accum"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(steps * CONFIG["warmup_ratio"])), steps)
    # init_scale 1024: the default 65536 overflowed and skipped updates in the V100 smoke test.
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)
    log = (ADAPTER_DIR / "train_log.jsonl")
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    def loss_of(example: dict):
        ids = torch.tensor([example["input_ids"]], device="cuda")
        labels = torch.tensor([example["labels"]], device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            return model(input_ids=ids, labels=labels).loss

    def validation_loss() -> float:
        model.eval()
        with torch.no_grad():
            values = [float(loss_of(ex)) for ex in data["validation"]]
        model.train()
        return sum(values) / max(1, len(values))

    model.train()
    start = validation_loss()
    with log.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"step": 0, "validation_loss": start}) + "\n")
    running, step = 0.0, 0
    for position, index in enumerate(order, start=1):
        loss = loss_of(data["train"][index]) / CONFIG["grad_accum"]
        scaler.scale(loss).backward()
        running += float(loss)
        if position % CONFIG["grad_accum"] == 0 or position == len(order):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            entry = {"step": step, "of": steps, "train_loss": running, "lr": scheduler.get_last_lr()[0]}
            running = 0.0
            if step % 20 == 0 or step == steps:
                entry["validation_loss"] = validation_loss()
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
    model.save_pretrained(final)
    weights = final / "adapter_model.safetensors"
    summary = {"status": "complete", "config": CONFIG, "train_rows": len(data["train"]),
               "validation_rows": len(data["validation"]),
               "dropped_too_long": {s: report[f"{s}_rows"] - len(data[s]) for s in data},
               "validation_loss_start": start, "validation_loss_end": validation_loss(),
               "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
    (ADAPTER_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
