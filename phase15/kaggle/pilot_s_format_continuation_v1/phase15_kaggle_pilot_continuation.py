"""Run the nine frozen S-format continuation episodes for amendment P15-A2."""

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
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))

from pilot_episodes import build_episodes, keep_types  # noqa: E402
from pilot_score import (  # noqa: E402
    correct_answer, judge_error, judge_prompt, parse_lenient, same, solve_errors, solve_prompt,
)
from self_edit_gate import number_values, review  # noqa: E402


WORK = Path("/kaggle/working/phase15_pilot_a2_v1")
if not Path("/kaggle/working").exists():
    WORK = HERE / "local_work"
AUDIT = WORK / "audit.jsonl"
SPEC = json.loads((HERE / "continuation_spec.json").read_text(encoding="utf-8"))
SYSTEM = "You are a careful mathematical solver."


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def install_runtime() -> None:
    required = SPEC["packages"]
    try:
        import importlib.metadata as metadata
        current = {name: metadata.version(name) for name in required}
    except Exception:
        current = {}
    if current != required:
        specs = [f"{name}=={version}" for name, version in required.items()]
        run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *specs])


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(row: dict[str, Any]) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with AUDIT.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_for(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def forbidden_answers(rows: list[dict]) -> set[int]:
    found = set()
    for row in rows:
        for token in number_values(str(row["reference_answer"])):
            if token.lstrip("-").isdigit():
                found.add(int(token))
    return found


def main() -> None:
    if AUDIT.exists():
        raise RuntimeError(f"Refusing to append a second run to existing {AUDIT}")
    started = time.time()
    deadline = started + int(SPEC["wall_seconds"])
    install_runtime()
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    manifest_path = HERE / "episodes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = manifest["episodes"]
    if manifest["amendment"] != SPEC["amendment"] or len(episodes) != int(SPEC["continuation_episodes"]):
        raise RuntimeError("Continuation manifest does not match continuation_spec.json")
    if sha256(manifest_path) != SPEC["manifest_sha256"]:
        raise RuntimeError("Continuation manifest hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(SPEC["model"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        SPEC["model"], device_map={"": 0}, torch_dtype=torch.float16, quantization_config=quantization,
        low_cpu_mem_usage=True, attn_implementation="sdpa",
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = get_peft_model(base, LoraConfig(
        r=int(SPEC["lora_r"]), lora_alpha=int(SPEC["lora_alpha"]), lora_dropout=float(SPEC["lora_dropout"]),
        bias="none", task_type="CAUSAL_LM", target_modules=list(SPEC["target_modules"]),
    ))
    initial_state = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()}
    append({
        "type": "metadata", "spec": SPEC, "manifest": {key: value for key, value in manifest.items() if key != "episodes"},
        "scientific_protocol": "P15-A2 Kaggle NF4 S-format continuation", "fp16_pilot": False,
    })

    def reset_adapter() -> str:
        set_peft_model_state_dict(model, copy.deepcopy(initial_state))
        state = get_peft_model_state_dict(model)
        digest = hashlib.sha256()
        for key in sorted(state):
            digest.update(key.encode())
            digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    initial_sha = reset_adapter()

    def over_deadline() -> bool:
        return time.time() >= deadline

    def generate(user_prompt: str, *, adapter: bool, seed: int, temperature: float, limit: int) -> tuple[str, int, bool, float]:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt}]
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        if encoded.shape[1] > int(SPEC["max_input_tokens"]):
            raise RuntimeError(f"Prompt exceeds cap: {int(encoded.shape[1])}")
        inputs = encoded.to("cuda:0")
        torch.manual_seed(seed)
        model.eval()
        model.config.use_cache = True
        kwargs = {"do_sample": temperature > 0}
        if temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = 1.0
        context = contextlib.nullcontext() if adapter else model.disable_adapter()
        start = time.time()
        with context, torch.inference_mode():
            output = model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=limit,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, **kwargs,
            )
        generated = output[0, inputs.shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        tokens = int(generated.numel())
        return text, tokens, tokens >= limit, time.time() - start

    def train_adapter(user_prompt: str, target: str, seed: int) -> tuple[list[float], float, str]:
        before = reset_adapter()
        if before != initial_sha:
            raise RuntimeError("Adapter reset does not match the frozen initial state")
        user_messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt}]
        full_messages = [*user_messages, {"role": "assistant", "content": target}]
        prefix = tokenizer.apply_chat_template(user_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")[0]
        full = tokenizer.apply_chat_template(full_messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")[0]
        if full.numel() > int(SPEC["max_input_tokens"]):
            raise RuntimeError(f"Training row exceeds cap: {int(full.numel())}")
        labels = full.clone()
        labels[: prefix.numel()] = -100
        input_ids = full.unsqueeze(0).to("cuda:0")
        label_ids = labels.unsqueeze(0).to("cuda:0")
        torch.manual_seed(seed)
        model.train()
        model.config.use_cache = False
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(SPEC["learning_rate"]), weight_decay=0.0,
        )
        losses = []
        start = time.time()
        for _ in range(int(SPEC["train_steps"])):
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=label_ids).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        state = get_peft_model_state_dict(model)
        digest = hashlib.sha256()
        for key in sorted(state):
            digest.update(key.encode())
            digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
        del optimizer, input_ids, label_ids
        torch.cuda.empty_cache()
        return losses, time.time() - start, digest.hexdigest()

    def solve_record(row: dict, sample: int) -> dict:
        prompt = solve_prompt(row["question"])
        text, tokens, hit_cap, elapsed = generate(
            prompt, adapter=False, seed=seed_for(f"{SPEC['seed']}|solve|{row['id']}|{sample}"),
            temperature=float(SPEC["solution_temperature"]), limit=int(SPEC["max_new_tokens"]),
        )
        record = {
            "type": "screen_solution", "problem_id": row["id"], "sample": sample, "output": text,
            "generated_tokens": tokens, "hit_cap": hit_cap, "elapsed_seconds": elapsed,
            "correct": correct_answer(text, row["reference_answer"]),
            "errors": solve_errors(text, row["question"], hit_768_cap=hit_cap),
        }
        append(record)
        return record

    ran = 0
    for episode in episodes:
        if over_deadline():
            break
        try:
            run_episode(episode, generate, train_adapter, reset_adapter, initial_sha)
        except RuntimeError as error:
            append({"type": "episode_failed", "episode_id": episode["episode_id"], "error": str(error)})
            reset_adapter()
            continue
        ran += 1
    append({"type": "type_complete", "error_type": "S-format", "episodes": ran, "available": len(episodes)})

    summary = {
        "amendment": SPEC["amendment"],
        "status": "complete" if not over_deadline() else "stopped_at_wall",
        "fp16_pilot": False,
        "evaluation_split_opened": False,
        "meta_training": False,
        "p15_a1_completed_for_provenance": len(manifest["completed_episode_ids"]),
        "frozen_episodes": len(episodes),
        "completed_episodes": ran,
        "ready_for_go_nogo": ran == 40,
        "wall_seconds": time.time() - started,
        "audit_sha256": sha256(AUDIT) if AUDIT.exists() else None,
    }
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2), flush=True)

def run_episode(episode, generate, train_adapter, reset_adapter, initial_sha) -> None:
    append({"type": "episode_start", "episode_id": episode["episode_id"], "error_type": episode["error_type"]})
    items = [*episode["probes"], episode["control"]]
    temperature = 0.0 if episode["error_type"].startswith("J-") else float(SPEC["solution_temperature"])

    def one(arm: str, prefix: str | None, adapter: bool, seed_base: int) -> None:
        for index, item in enumerate(items):
            prompt = item.get("source_text") or solve_prompt(item["question"])
            if prefix:
                prompt = f"{prefix}\n\nNew task:\n{prompt}"
            text, tokens, hit_cap, elapsed = generate(
                prompt, adapter=adapter, seed=seed_base + index, temperature=temperature, limit=int(SPEC["max_new_tokens"]),
            )
            append({
                "type": "completion", "episode_id": episode["episode_id"], "arm": arm, "item_id": item["id"],
                "output": text, "generated_tokens": tokens, "hit_cap": hit_cap, "elapsed_seconds": elapsed,
            })

    reset_adapter()
    one("a0_base", None, False, seed_for(episode["episode_id"] + "|a0"))
    harness = (
        "Verified teaching example:\n"
        f"{episode['teaching'].get('source_text', episode['teaching']['question'])}\n\n"
        f"Verified correction:\n{episode['target']}\n\n"
        "Apply the same behavioral rule to the new task. Do not copy the teaching answer."
    )
    one("a1_context", harness, False, seed_for(episode["episode_id"] + "|a1"))
    losses, elapsed, trained = train_adapter(episode["teaching"].get("source_text", episode["teaching"]["question"]), episode["target"], seed_for(episode["episode_id"] + "|a2"))
    append({"type": "update", "episode_id": episode["episode_id"], "arm": "a2_naive", "losses": losses, "elapsed_seconds": elapsed, "trained_adapter_sha256": trained, "initial_adapter_sha256": initial_sha})
    one("a2_naive", None, True, seed_for(episode["episode_id"] + "|a2e"))
    edit_prompt = (
        "A model made a behavior mistake and received a verified correction. "
        "Write a short general rule for avoiding this kind of mistake on different problems. "
        "Do not repeat any numbers or reveal the teaching answer. Include an ideal response template.\n\n"
        f"Mistake context:\n{episode['teaching'].get('source_text', episode['teaching']['question'])}\n\n"
        f"Verified correction:\n{episode['target']}"
    )
    reset_adapter()
    draft, *_rest = generate(edit_prompt, adapter=False, seed=seed_for(episode["episode_id"] + "|edit"), temperature=0.0, limit=int(SPEC["self_edit_max_new_tokens"]))
    others = [*episode["probes"], episode["control"]]
    gated = review(draft, forbidden_numbers=forbidden_answers([episode["teaching"], *others]), protected_texts=[row["question"] for row in others])
    if gated["status"] == "retry":
        retry_prompt = edit_prompt + "\n\nThe previous lesson was rejected. Write it again with no digits and no copied problem text."
        draft, *_rest = generate(retry_prompt, adapter=False, seed=seed_for(episode["episode_id"] + "|edit2"), temperature=0.0, limit=int(SPEC["self_edit_max_new_tokens"]))
        gated = review(draft, forbidden_numbers=forbidden_answers([episode["teaching"], *others]), protected_texts=[row["question"] for row in others])
    append({"type": "self_edit", "episode_id": episode["episode_id"], "status": gated["status"], "reasons": gated["reasons"]})
    if gated["text"]:
        losses, elapsed, trained = train_adapter("Behavior lesson:\n" + gated["text"], gated["text"], seed_for(episode["episode_id"] + "|a3"))
        append({"type": "update", "episode_id": episode["episode_id"], "arm": "a3_self_edit", "losses": losses, "elapsed_seconds": elapsed, "trained_adapter_sha256": trained})
        one("a3_self_edit", None, True, seed_for(episode["episode_id"] + "|a3e"))
    else:
        append({"type": "a3_skipped", "episode_id": episode["episode_id"]})
    final = reset_adapter()
    append({"type": "episode_complete", "episode_id": episode["episode_id"], "reset_sha256": final, "reset_ok": final == initial_sha})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        WORK.mkdir(parents=True, exist_ok=True)
        failure = {"status": "failed", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        (WORK / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(failure, indent=2), file=sys.stderr, flush=True)
        raise
