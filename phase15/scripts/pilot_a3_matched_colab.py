"""P15-A3: paired-seed A0/A1/A2 diagnostic on the frozen S-format dev episodes.

Upload this file plus the P15-A2 manifest, spec, pilot_score, and verifier
vendor modules to /content/p15a3. Run only after PILOT_A3.md is recorded.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path("/content/p15a3")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))
from pilot_score import correct_answer, solve_errors, solve_prompt  # noqa: E402

SPEC = json.loads((ROOT / "continuation_spec.json").read_text(encoding="utf-8"))
MANIFEST_PATH = ROOT / "episodes.json"
OUT = ROOT / "run_a3"
AUDIT = OUT / "audit.jsonl"
SYSTEM = "You are a careful mathematical solver."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_for(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def append(row: dict) -> None:
    with AUDIT.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def adapter_sha(model, get_state) -> str:
    state = get_state(model)
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    if AUDIT.exists():
        raise RuntimeError(f"Refusing to overwrite existing audit: {AUDIT}")
    if sha256(MANIFEST_PATH) != SPEC["manifest_sha256"]:
        raise RuntimeError("Frozen manifest hash mismatch")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    episodes = manifest["episodes"]
    if manifest["amendment"] != "P15-A2" or len(episodes) != 40:
        raise RuntimeError("Expected the 40 frozen P15-A2 episodes")
    ids = [row["episode_id"] for row in episodes]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate episode ID")

    started = time.time()
    deadline = started + int(SPEC["wall_seconds"])
    needed = SPEC["packages"]
    import importlib.metadata as metadata
    present = {}
    for name in needed:
        try:
            present[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            present[name] = None
    if present != needed:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade",
             *[f"{name}=={version}" for name, version in needed.items()]],
            check=True,
        )
    import torch
    from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                      prepare_model_for_kbit_training, set_peft_model_state_dict)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    tokenizer = AutoTokenizer.from_pretrained(SPEC["model"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    torch.manual_seed(20261103)
    base = AutoModelForCausalLM.from_pretrained(
        SPEC["model"], device_map={"": 0}, torch_dtype=torch.float16,
        quantization_config=quantization, low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = get_peft_model(base, LoraConfig(
        r=int(SPEC["lora_r"]), lora_alpha=int(SPEC["lora_alpha"]),
        lora_dropout=float(SPEC["lora_dropout"]), bias="none", task_type="CAUSAL_LM",
        target_modules=list(SPEC["target_modules"]),
    ))
    get_state = get_peft_model_state_dict
    initial_state = {key: value.detach().cpu().clone()
                     for key, value in get_state(model).items()}

    def reset() -> str:
        set_peft_model_state_dict(model, copy.deepcopy(initial_state))
        return adapter_sha(model, get_state)

    initial_sha = reset()
    OUT.mkdir(parents=True, exist_ok=True)
    append({
        "type": "metadata", "amendment": "P15-A3", "model": SPEC["model"],
        "spec": SPEC, "manifest_sha256": sha256(MANIFEST_PATH),
        "initial_adapter_sha256": initial_sha,
        "torch_version": torch.__version__, "cuda_device": torch.cuda.get_device_name(0),
        "versions": {name: metadata.version(name) for name in needed},
        "shared_seed_rule": "seed_for(episode_id + '|a0') + item_index",
        "evaluation_split_opened": False,
    })

    def generate(prompt: str, adapter: bool, seed: int) -> tuple[str, int, bool, float]:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        if encoded.shape[1] > int(SPEC["max_input_tokens"]):
            raise RuntimeError("Generation prompt exceeds input cap")
        inputs = encoded.to("cuda:0")
        torch.manual_seed(seed)
        model.eval()
        model.config.use_cache = True
        context = contextlib.nullcontext() if adapter else model.disable_adapter()
        tick = time.time()
        with context, torch.inference_mode():
            output = model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                max_new_tokens=int(SPEC["max_new_tokens"]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True, temperature=float(SPEC["solution_temperature"]),
                top_p=1.0,
            )
        generated = output[0, inputs.shape[1]:]
        count = int(generated.numel())
        return (tokenizer.decode(generated, skip_special_tokens=True), count,
                count >= int(SPEC["max_new_tokens"]), time.time() - tick)

    def update(episode: dict) -> tuple[list[float], str]:
        if reset() != initial_sha:
            raise RuntimeError("Adapter reset mismatch before update")
        teaching = episode["teaching"]
        prompt = teaching.get("source_text", teaching["question"])
        user = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]
        full = [*user, {"role": "assistant", "content": episode["target"]}]
        prefix = tokenizer.apply_chat_template(
            user, tokenize=True, add_generation_prompt=True, return_tensors="pt")[0]
        tokens = tokenizer.apply_chat_template(
            full, tokenize=True, add_generation_prompt=False, return_tensors="pt")[0]
        if tokens.numel() > int(SPEC["max_input_tokens"]):
            raise RuntimeError("Training row exceeds input cap")
        labels = tokens.clone()
        labels[:prefix.numel()] = -100
        inputs = tokens.unsqueeze(0).to("cuda:0")
        labels = labels.unsqueeze(0).to("cuda:0")
        torch.manual_seed(seed_for(episode["episode_id"] + "|a2"))
        model.train()
        model.config.use_cache = False
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters, lr=float(SPEC["learning_rate"]), weight_decay=0.0)
        losses = []
        for _ in range(int(SPEC["train_steps"])):
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                         labels=labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        trained_sha = adapter_sha(model, get_state)
        del optimizer, inputs, labels
        torch.cuda.empty_cache()
        return losses, trained_sha

    completed = 0
    for episode in episodes:
        if time.time() >= deadline:
            break
        eid = episode["episode_id"]
        items = [*episode["probes"], episode["control"]]
        append({"type": "episode_start", "episode_id": eid})
        if reset() != initial_sha:
            raise RuntimeError(f"Reset mismatch at {eid}")
        harness = (
            "Verified teaching example:\n"
            f"{episode['teaching'].get('source_text', episode['teaching']['question'])}\n\n"
            f"Verified correction:\n{episode['target']}\n\n"
            "Apply the same behavioral rule to the new task. Do not copy the teaching answer."
        )
        for arm in ("a0_base", "a1_context", "a2_naive"):
            if arm == "a2_naive":
                losses, trained_sha = update(episode)
                append({"type": "update", "episode_id": eid, "arm": arm,
                        "losses": losses, "trained_adapter_sha256": trained_sha,
                        "initial_adapter_sha256": initial_sha})
            for index, item in enumerate(items):
                prompt = item.get("source_text") or solve_prompt(item["question"])
                if arm == "a1_context":
                    prompt = f"{harness}\n\nNew task:\n{prompt}"
                seed = seed_for(eid + "|a0") + index
                output, count, cap, elapsed = generate(
                    prompt, adapter=arm == "a2_naive", seed=seed)
                errors = solve_errors(output, item["question"], hit_768_cap=cap)
                correct = correct_answer(output, item["reference_answer"])
                append({"type": "completion", "episode_id": eid,
                        "arm": arm, "item_id": item["id"],
                        "item_kind": "control" if index == 2 else "probe",
                        "seed": seed, "output": output, "generated_tokens": count,
                        "hit_cap": cap, "elapsed_seconds": elapsed,
                        "correct": correct, "errors": errors,
                        "joint_correct": correct and not errors})
        final_sha = reset()
        append({"type": "episode_complete", "episode_id": eid,
                "reset_sha256": final_sha, "reset_ok": final_sha == initial_sha})
        if final_sha != initial_sha:
            raise RuntimeError(f"Final reset mismatch at {eid}")
        completed += 1
        print(f"P15-A3 episode {completed}/40 complete ({eid})", flush=True)
    summary = {"amendment": "P15-A3", "completed_episodes": completed,
               "frozen_episodes": len(episodes),
               "status": "complete" if completed == len(episodes) else "stopped_at_wall",
               "wall_seconds": time.time() - started,
               "audit_sha256": sha256(AUDIT),
               "evaluation_split_opened": False}
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        failure = {"status": "failed", "error_type": type(error).__name__,
                   "error": str(error), "traceback": traceback.format_exc()}
        (OUT / "failure.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, indent=2), file=sys.stderr, flush=True)
        raise
