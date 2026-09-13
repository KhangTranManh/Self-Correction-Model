"""Small constrained GRPO-style pilot for the binary KEEP/REVISE router.

The action group contains both legal decision completions. Rewards come from
fresh verifier state encoded in each row's label: +1 for the correct action and
-1 for the other action. Only decision completion log-probabilities are used.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup


ACTIONS = ("<decision>KEEP</decision>", "<decision>REVISE</decision>")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    prompt = row["messages"][:-1]
    prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    sequences = []
    for action in ACTIONS:
        ids = tokenizer.apply_chat_template(prompt + [{"role": "assistant", "content": action}], tokenize=True, add_generation_prompt=False)
        if ids[: len(prompt_ids)] != prompt_ids or len(ids) > max_length:
            raise RuntimeError(f"Invalid/truncated action encoding: {row['construction_id']}")
        sequences.append(ids)
    return {"source_id": row["source_id"], "label": row["label"], "domain": row["domain"], "dataset": row["dataset"], "prompt_length": len(prompt_ids), "sequences": sequences}


def action_logps(model: Any, item: dict[str, Any], pad: int) -> torch.Tensor:
    sequences = item["sequences"]
    maximum = max(map(len, sequences))
    input_ids = torch.full((2, maximum), pad, dtype=torch.long, device="cuda")
    attention = torch.zeros_like(input_ids)
    mask = torch.zeros((2, maximum - 1), dtype=torch.bool, device="cuda")
    for i, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[i, :length] = torch.tensor(sequence, device="cuda")
        attention[i, :length] = 1
        mask[i, item["prompt_length"] - 1 : length - 1] = True
    output = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
    token = F.log_softmax(output.logits[:, :-1].float(), dim=-1).gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (token * mask).sum(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--base-model", default="Kxck/Self_Correction_v1")
    parser.add_argument("--initial-adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.epochs != 1:
        raise RuntimeError("The initial reward pilot is limited to one epoch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    data_path = Path(args.train).resolve(); adapter_path = Path(args.initial_adapter).resolve(); output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(data_path)
    if not rows or set(row["label"] for row in rows) != {"KEEP", "REVISE"}:
        raise RuntimeError("Training data must contain both verifier-backed labels")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode(tokenizer, row, args.max_length) for row in rows]
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=quant, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)
    model.enable_input_require_grads(); model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.config.use_cache = False
    reference = {}
    model.eval()
    with torch.inference_mode():
        for item in encoded:
            reference[item["source_id"] + "::" + item["label"]] = F.log_softmax(action_logps(model, item, tokenizer.pad_token_id), dim=0).cpu()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=args.learning_rate)
    steps = math.ceil(len(encoded) / args.gradient_accumulation)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, round(0.1 * steps)), steps)
    order = list(range(len(encoded))); random.shuffle(order); model.train(); optimizer.zero_grad(set_to_none=True)
    history=[]; started=time.time(); accumulated=0.0
    for micro, index in enumerate(order, 1):
        item=encoded[index]; log_probs=F.log_softmax(action_logps(model,item,tokenizer.pad_token_id),dim=0)
        correct=0 if item["label"]=="KEEP" else 1
        rewards=torch.tensor([1.0 if i==correct else -1.0 for i in range(2)],device="cuda")
        advantages=(rewards-rewards.mean())/(rewards.std(unbiased=False)+1e-6)
        policy_loss=-(advantages*log_probs).mean()
        ref=reference[item["source_id"]+"::"+item["label"]].to("cuda")
        probs=log_probs.exp(); kl=(probs*(log_probs-ref)).sum()
        loss=policy_loss+args.beta*kl
        (loss/args.gradient_accumulation).backward(); accumulated+=float(loss.detach().cpu())
        if micro%args.gradient_accumulation==0 or micro==len(order):
            grad=float(torch.nn.utils.clip_grad_norm_(trainable,1.0)); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            event={"optimizer_step":len(history)+1,"micro_step":micro,"mean_loss":accumulated/(args.gradient_accumulation if micro%args.gradient_accumulation==0 else micro%args.gradient_accumulation),"grad_norm":grad,"lr":scheduler.get_last_lr()[0]}
            history.append(event); accumulated=0.0; print(json.dumps(event),flush=True)
    final=output/"final_adapter"; final.mkdir(parents=True,exist_ok=True); model.save_pretrained(final,safe_serialization=True); tokenizer.save_pretrained(final)
    report={
        "schema_version":"phase3_binary_grpo_router_v1","method":"constrained two-action GRPO-style policy gradient",
        "reward":{"correct_decision":1.0,"wrong_decision":-1.0,"source":"fresh verifier-backed answer state","kl_beta":args.beta},
        "model":{"base_model":args.base_model,"initial_adapter":str(adapter_path),"initial_adapter_sha256":sha256(adapter_path/'adapter_model.safetensors'),"final_adapter":str(final),"final_adapter_sha256":sha256(final/'adapter_model.safetensors')},
        "data":{"path":str(data_path),"sha256":sha256(data_path),"rows":len(rows),"labels":dict(Counter(r['label'] for r in rows)),"domains":dict(Counter(r['domain'] for r in rows))},
        "training":{"epochs":args.epochs,"learning_rate":args.learning_rate,"optimizer_steps":len(history),"history":history,"runtime_seconds":time.time()-started},
        "limitations":["Binary constrained-action pilot, not free-form response GRPO","Frozen behavioral benchmark is required before promotion"]}
    (output/'train_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps(report,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
