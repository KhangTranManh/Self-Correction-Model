"""DPO LoRA on the original solver: prefer the correct judgment of a pair over
an incorrect judgment of the same pair.

Loss: -log sigmoid(beta * ((log pi(c) - log ref(c)) - (log pi(r) - log ref(r))))
with summed log-probabilities over response tokens. The reference is the same
base model with the adapter disabled; reference log-probabilities are
precomputed once. Hyperparameters fixed in advance: LoRA r=16, alpha=32,
dropout 0.05, all attention and MLP projections; beta 0.1; learning rate 2e-5
cosine with 5% warmup; two epochs; effective batch 16 pairs; max length 3072;
FP16 base, FP32 LoRA, gradient scaler starting at 1024; seed 20261002.
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
from phase11.scripts.generate import MODEL, OUT, REVISION, read_jsonl  # noqa: E402

DATA = OUT / "dpo"
REPORT = ROOT / "phase11/data/dpo_v1_report.json"
ADAPTER_DIR = OUT / "dpo_lora"
CONFIG = {"r": 16, "alpha": 32, "dropout": 0.05, "beta": 0.1, "lr": 2e-5, "warmup_ratio": 0.05,
          "epochs": 2, "grad_accum": 16, "max_length": 3072, "seed": 20261002,
          "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}


def encode(tokenizer, prompt: str, response: str) -> dict | None:
    head = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True)
    full = tokenizer.apply_chat_template([{"role": "user", "content": prompt},
                                          {"role": "assistant", "content": response}], tokenize=False)
    head_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if ids[:len(head_ids)] != head_ids or len(ids) > CONFIG["max_length"]:
        return None
    return {"ids": ids, "start": len(head_ids)}


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
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    data = {}
    for split in ("train", "validation"):
        path = DATA / f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != report[split]["sha256"]:
            raise RuntimeError(f"DPO {split} data changed")
        rows = []
        for ex in read_jsonl(path):
            chosen, rejected = encode(tokenizer, ex["prompt"], ex["chosen"]), encode(tokenizer, ex["prompt"], ex["rejected"])
            if chosen and rejected:
                rows.append({"chosen": chosen, "rejected": rejected})
        data[split] = rows
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

    causal_lm = model.base_model.model  # PEFT wrapper -> Qwen2ForCausalLM

    def logprob(sequence: dict):
        # Project only the response positions through lm_head: full-sequence FP32 logits
        # for two ~3k-token sequences would cost several GB on a 32 GB V100.
        ids = torch.tensor([sequence["ids"]], device="cuda")
        start = sequence["start"]
        with torch.autocast("cuda", dtype=torch.float16):
            hidden = causal_lm.model(input_ids=ids).last_hidden_state[:, start - 1:-1]
            logits = causal_lm.lm_head(hidden).float()
        return torch.log_softmax(logits, dim=-1).gather(2, ids[:, start:, None]).sum()

    # Reference log-probabilities: same weights with the adapter disabled.
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        for rows in data.values():
            for ex in rows:
                ex["ref_c"], ex["ref_r"] = float(logprob(ex["chosen"])), float(logprob(ex["rejected"]))

    def pair_loss(ex: dict):
        margin = (logprob(ex["chosen"]) - ex["ref_c"]) - (logprob(ex["rejected"]) - ex["ref_r"])
        return -torch.nn.functional.logsigmoid(CONFIG["beta"] * margin), float(margin)

    def validate() -> dict:
        model.eval()
        losses, wins = [], 0
        with torch.no_grad():
            for ex in data["validation"]:
                loss, margin = pair_loss(ex)
                losses.append(float(loss))
                wins += margin > 0
        model.train()
        return {"validation_loss": sum(losses) / max(1, len(losses)),
                "validation_reward_accuracy": wins / max(1, len(losses))}

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=CONFIG["lr"], weight_decay=0.0)
    order = [i for _ in range(CONFIG["epochs"]) for i in random.sample(range(len(data["train"])), len(data["train"]))]
    steps = math.ceil(len(order) / CONFIG["grad_accum"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(steps * CONFIG["warmup_ratio"])), steps)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    log = ADAPTER_DIR / "train_log.jsonl"
    start = validate()
    log.write_text(json.dumps({"step": 0, **start}) + "\n", encoding="utf-8")
    model.train()
    running, step = [], 0
    for position, index in enumerate(order, start=1):
        loss, margin = pair_loss(data["train"][index])
        scaler.scale(loss / CONFIG["grad_accum"]).backward()
        running.append((float(loss), margin > 0))
        if position % CONFIG["grad_accum"] == 0 or position == len(order):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            entry = {"step": step, "of": steps,
                     "train_loss": sum(l for l, _ in running) / len(running),
                     "train_reward_accuracy": sum(w for _, w in running) / len(running),
                     "lr": scheduler.get_last_lr()[0]}
            running = []
            if step % 10 == 0 or step == steps:
                entry.update(validate())
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
    model.save_pretrained(final)
    end = validate()
    summary = {"status": "complete", "config": CONFIG,
               "train_pairs": len(data["train"]), "validation_pairs": len(data["validation"]),
               "dropped_too_long": {s: report[s]["rows"] - len(data[s]) for s in data},
               "start": start, "end": end,
               "adapter_sha256": hashlib.sha256((final / "adapter_model.safetensors").read_bytes()).hexdigest()}
    (ADAPTER_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
