"""Phase 15 Kaggle engineering smoke: temporary QLoRA updates and resets.

This is pipeline validation, not scientific evidence. It uses eight synthetic,
source-disjoint episodes and never reads a protected Phase 15 evaluation set.
Every completion and update record is appended durably under
``/kaggle/working/phase15_smoke_v1``.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback
from typing import Any


WORK = Path("/kaggle/working/phase15_smoke_v1")
AUDIT = WORK / "audit.jsonl"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 20261027
MAX_INPUT_TOKENS = 768
MAX_NEW_TOKENS = 128
TRAIN_STEPS = 3
LEARNING_RATE = 5e-4
LORA_R = 8
LORA_ALPHA = 16
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SYSTEM = (
    "You are a careful mathematical reviewer. Follow the requested output format exactly. "
    "When comparing solutions, choose one of the shown solutions rather than inventing a third answer."
)


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def install_runtime() -> None:
    required = {
        "transformers": "4.46.3",
        "peft": "0.13.2",
        "accelerate": "1.0.1",
        "bitsandbytes": "0.50.2",
    }
    try:
        import importlib.metadata as metadata

        current = {name: metadata.version(name) for name in required}
    except Exception:
        current = {}
    if current != required:
        specs = [f"{name}=={version}" for name, version in required.items()]
        run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *specs])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append(row: dict[str, Any]) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with AUDIT.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def gpu_report() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        index, name, memory, driver = [part.strip() for part in line.split(",", 3)]
        rows.append({"index": int(index), "name": name, "memory_mib": int(memory), "driver": driver})
    return rows


def solution(label: str, left: int, right: int, claimed: int) -> str:
    return (
        f"Solution {label}: There are {left} groups with {right} items in each group. "
        f"Multiplying gives {left} × {right} = {claimed}. Therefore the answer is {claimed}."
    )


def judge_item(item_id: str, left: int, right: int, correct_position: str, behavior: str) -> dict[str, Any]:
    gold = left * right
    wrong = gold + (left if left % 2 else right)
    a_value, b_value = (gold, wrong) if correct_position == "A" else (wrong, gold)
    extra = (
        "Review concisely. Do not repeat a sentence or restart the calculation. "
        if behavior == "j_loop"
        else "Review the arithmetic and obey the final-line format. "
    )
    prompt = (
        f"Problem: A package contains {left} rows of {right} markers. How many markers are there?\n\n"
        f"{solution('A', left, right, a_value)}\n\n"
        f"{solution('B', left, right, b_value)}\n\n"
        f"{extra}End with exactly one line: Verdict: Solution A or Verdict: Solution B."
    )
    return {
        "item_id": item_id,
        "kind": "judge",
        "behavior": behavior,
        "prompt": prompt,
        "expected": correct_position,
        "numbers": [left, right, gold, wrong],
    }


def control_item(item_id: str, start: int, given: int) -> dict[str, Any]:
    answer = start - given
    return {
        "item_id": item_id,
        "kind": "solve",
        "behavior": "control",
        "prompt": (
            f"A library had {start} notebooks and gave away {given}. How many notebooks remain? "
            "Show the calculation and end with Answer: <number>."
        ),
        "expected": answer,
        "numbers": [start, given, answer],
    }


def build_episodes() -> list[dict[str, Any]]:
    episodes = []
    for index in range(8):
        behavior = "j_format" if index < 4 else "j_loop"
        offset = 10 * index
        teaching = judge_item(
            f"e{index:02d}_teach", 3 + offset, 4 + index, "A" if index % 2 == 0 else "B", behavior
        )
        probes = [
            judge_item(f"e{index:02d}_p1", 71 + offset, 23 + index, "B" if index % 2 == 0 else "A", behavior),
            judge_item(f"e{index:02d}_p2", 83 + offset, 2 + index, "A" if index % 2 == 0 else "B", behavior),
        ]
        controls = [control_item(f"e{index:02d}_c1", 401 + offset, 37 + index)]
        target = (
            "The multiplication in the other solution is incorrect.\n"
            f"Verdict: Solution {teaching['expected']}"
        )
        episodes.append(
            {
                "episode_id": f"smoke_{index:02d}",
                "behavior": behavior,
                "teaching": teaching,
                "target": target,
                "probes": probes,
                "controls": controls,
            }
        )
    return episodes


VERDICT = re.compile(r"Verdict\s*:\s*Solution\s*([AB])", flags=re.IGNORECASE)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
WORD = re.compile(r"\w+", flags=re.UNICODE)


def repeated(text: str) -> bool:
    words = [word.lower() for word in WORD.findall(text)]
    grams: dict[tuple[str, ...], int] = {}
    for index in range(max(0, len(words) - 9)):
        gram = tuple(words[index : index + 10])
        grams[gram] = grams.get(gram, 0) + 1
    if grams and max(grams.values()) >= 3:
        return True
    segments = [part.strip() for part in re.split(r"[\n.]", text) if len(part.strip()) >= 25]
    return any(segments.count(part) >= 3 for part in set(segments))


def score(item: dict[str, Any], text: str, hit_cap: bool) -> dict[str, Any]:
    if item["kind"] == "judge":
        matches = VERDICT.findall(text)
        picked = matches[-1].upper() if matches else None
        return {
            "valid": picked in ("A", "B"),
            "correct": picked == item["expected"],
            "picked": picked,
            "loop": bool(hit_cap and repeated(text)),
            "hit_cap": hit_cap,
        }
    matches = NUMBER.findall(text)
    parsed = int(matches[-1].replace(",", "")) if matches else None
    return {
        "valid": parsed is not None,
        "correct": parsed == item["expected"],
        "picked": parsed,
        "loop": bool(hit_cap and repeated(text)),
        "hit_cap": hit_cap,
    }


def leakage_audit(episode: dict[str, Any]) -> dict[str, Any]:
    teaching_numbers = set(episode["teaching"]["numbers"])
    probe_numbers = {number for item in episode["probes"] for number in item["numbers"]}
    control_numbers = {number for item in episode["controls"] for number in item["numbers"]}
    return {
        "teaching_probe_number_overlap": sorted(teaching_numbers & probe_numbers),
        "teaching_control_number_overlap": sorted(teaching_numbers & control_numbers),
        "probe_control_number_overlap": sorted(probe_numbers & control_numbers),
    }


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    if AUDIT.exists():
        raise RuntimeError(f"Refusing to append a second run to existing {AUDIT}")
    started = time.time()
    install_runtime()

    import torch
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        prepare_model_for_kbit_training,
        set_peft_model_state_dict,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    gpus = gpu_report()
    if not gpus:
        raise RuntimeError("No Kaggle GPU is available")

    metadata = {
        "type": "metadata",
        "schema_version": "phase15_kaggle_smoke_v1",
        "scientific_evidence": False,
        "purpose": "engineering smoke only",
        "model": MODEL_ID,
        "quantization": "bnb_nf4_double_quant_fp16_compute",
        "seed": SEED,
        "episodes": 8,
        "train_steps_per_adapter": TRAIN_STEPS,
        "learning_rate": LEARNING_RATE,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "target_modules": TARGET_MODULES,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "gpus": gpus,
        "python": sys.version,
        "torch": torch.__version__,
    }
    append(metadata)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map={"": 0},
        torch_dtype=torch.float16,
        quantization_config=quantization,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(TARGET_MODULES),
    )
    model = get_peft_model(base, config)
    model.print_trainable_parameters()
    initial_state = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()}
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    append(
        {
            "type": "model_loaded",
            "model_commit": getattr(model.config, "_commit_hash", None),
            "tokenizer_commit": tokenizer.init_kwargs.get("_commit_hash"),
            "trainable_parameters": trainable_count,
            "cuda_allocated": int(torch.cuda.memory_allocated()),
            "cuda_reserved": int(torch.cuda.memory_reserved()),
        }
    )

    def reset_adapter() -> str:
        set_peft_model_state_dict(model, copy.deepcopy(initial_state))
        state = get_peft_model_state_dict(model)
        digest = hashlib.sha256()
        for key in sorted(state):
            digest.update(key.encode())
            digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    initial_adapter_sha = reset_adapter()

    def format_prompt(user_prompt: str) -> torch.Tensor:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt}]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if encoded.shape[1] > MAX_INPUT_TOKENS:
            raise RuntimeError(f"Prompt exceeds cap: {encoded.shape[1]} > {MAX_INPUT_TOKENS}")
        return encoded.to("cuda:0")

    def generate(user_prompt: str, adapter: bool, seed: int) -> tuple[str, int, bool, float]:
        torch.manual_seed(seed)
        inputs = format_prompt(user_prompt)
        model.eval()
        model.config.use_cache = True
        start = time.time()
        context = contextlib.nullcontext() if adapter else model.disable_adapter()
        with context, torch.inference_mode():
            output = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output[0, inputs.shape[1] :]
        elapsed = time.time() - start
        text = tokenizer.decode(generated, skip_special_tokens=True)
        tokens = int(generated.numel())
        return text, tokens, tokens == MAX_NEW_TOKENS, elapsed

    def training_example(user_prompt: str, target: str) -> tuple[torch.Tensor, torch.Tensor]:
        user_messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt}]
        full_messages = [*user_messages, {"role": "assistant", "content": target}]
        prefix = tokenizer.apply_chat_template(
            user_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )[0]
        full = tokenizer.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
        )[0]
        if full.numel() > MAX_INPUT_TOKENS:
            raise RuntimeError(f"Training row exceeds cap: {full.numel()} > {MAX_INPUT_TOKENS}")
        labels = full.clone()
        labels[: prefix.numel()] = -100
        return full.unsqueeze(0).to("cuda:0"), labels.unsqueeze(0).to("cuda:0")

    def train_adapter(user_prompt: str, target: str, seed: int) -> tuple[list[float], float, str]:
        torch.manual_seed(seed)
        before = reset_adapter()
        if before != initial_adapter_sha:
            raise RuntimeError("Adapter reset hash does not match the frozen initial state")
        model.train()
        model.config.use_cache = False
        input_ids, labels = training_example(user_prompt, target)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=LEARNING_RATE,
            weight_decay=0.0,
        )
        losses = []
        start = time.time()
        for _ in range(TRAIN_STEPS):
            optimizer.zero_grad(set_to_none=True)
            result = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels)
            loss = result.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        elapsed = time.time() - start
        del optimizer, input_ids, labels
        torch.cuda.empty_cache()
        state = get_peft_model_state_dict(model)
        digest = hashlib.sha256()
        for key in sorted(state):
            digest.update(key.encode())
            digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
        return losses, elapsed, digest.hexdigest()

    def evaluate(episode: dict[str, Any], arm: str, prefix: str | None, adapter: bool, seed_base: int) -> None:
        for item_index, item in enumerate([*episode["probes"], *episode["controls"]]):
            prompt = item["prompt"] if prefix is None else f"{prefix}\n\nNew task:\n{item['prompt']}"
            text, tokens, hit_cap, elapsed = generate(prompt, adapter=adapter, seed=seed_base + item_index)
            append(
                {
                    "type": "completion",
                    "episode_id": episode["episode_id"],
                    "behavior": episode["behavior"],
                    "arm": arm,
                    "item_id": item["item_id"],
                    "item_kind": item["kind"],
                    "generated_tokens": tokens,
                    "elapsed_seconds": elapsed,
                    "output": text,
                    "score": score(item, text, hit_cap),
                }
            )

    episodes = build_episodes()
    for episode_index, episode in enumerate(episodes):
        leak = leakage_audit(episode)
        if any(leak.values()):
            raise RuntimeError(f"Synthetic number-overlap audit failed: {episode['episode_id']} {leak}")
        append({"type": "episode_start", "episode_id": episode["episode_id"], "leakage_audit": leak})
        seed_base = SEED + 1000 * episode_index

        reset_sha = reset_adapter()
        if reset_sha != initial_adapter_sha:
            raise RuntimeError("A0 adapter reset verification failed")
        evaluate(episode, "a0_base", None, adapter=False, seed_base=seed_base + 100)

        harness = (
            "Verified teaching example:\n"
            f"{episode['teaching']['prompt']}\n\nVerified correction:\n{episode['target']}\n\n"
            "Apply the same behavioral rule to the new task. Do not copy the teaching numbers."
        )
        evaluate(episode, "a1_context", harness, adapter=False, seed_base=seed_base + 200)

        losses, train_seconds, trained_sha = train_adapter(
            episode["teaching"]["prompt"], episode["target"], seed_base + 300
        )
        append(
            {
                "type": "update",
                "episode_id": episode["episode_id"],
                "arm": "a2_naive",
                "initial_adapter_sha256": initial_adapter_sha,
                "trained_adapter_sha256": trained_sha,
                "losses": losses,
                "elapsed_seconds": train_seconds,
            }
        )
        evaluate(episode, "a2_naive", None, adapter=True, seed_base=seed_base + 400)

        self_edit_prompt = (
            "A model made the following behavior mistake and then received a verified correction. "
            "Write a short general rule for avoiding this kind of mistake on different problems. "
            "Do not repeat any numbers or reveal the teaching answer. Include an ideal response template.\n\n"
            f"Mistake context:\n{episode['teaching']['prompt']}\n\nVerified correction:\n{episode['target']}"
        )
        reset_sha = reset_adapter()
        if reset_sha != initial_adapter_sha:
            raise RuntimeError("A3 pre-self-edit reset verification failed")
        self_edit, edit_tokens, edit_cap, edit_seconds = generate(
            self_edit_prompt, adapter=False, seed=seed_base + 500
        )
        append(
            {
                "type": "self_edit",
                "episode_id": episode["episode_id"],
                "output": self_edit,
                "generated_tokens": edit_tokens,
                "hit_cap": edit_cap,
                "elapsed_seconds": edit_seconds,
            }
        )
        a3_training_prompt = (
            f"Behavior lesson written from a verified correction:\n{self_edit}\n\n"
            f"Practice task:\n{episode['teaching']['prompt']}"
        )
        losses, train_seconds, trained_sha = train_adapter(
            a3_training_prompt, episode["target"], seed_base + 600
        )
        append(
            {
                "type": "update",
                "episode_id": episode["episode_id"],
                "arm": "a3_self_edit",
                "initial_adapter_sha256": initial_adapter_sha,
                "trained_adapter_sha256": trained_sha,
                "losses": losses,
                "elapsed_seconds": train_seconds,
            }
        )
        evaluate(episode, "a3_self_edit", None, adapter=True, seed_base=seed_base + 700)

        final_reset = reset_adapter()
        if final_reset != initial_adapter_sha:
            raise RuntimeError("Final adapter reset verification failed")
        append({"type": "episode_complete", "episode_id": episode["episode_id"], "reset_sha256": final_reset})
        print(f"completed {episode['episode_id']} ({episode_index + 1}/{len(episodes)})", flush=True)

    rows = [json.loads(line) for line in AUDIT.read_text(encoding="utf-8").splitlines() if line]
    completions = [row for row in rows if row.get("type") == "completion"]
    updates = [row for row in rows if row.get("type") == "update"]
    by_arm: dict[str, dict[str, int]] = {}
    for row in completions:
        arm = row["arm"]
        bucket = by_arm.setdefault(arm, {"rows": 0, "valid": 0, "correct": 0, "loops": 0, "caps": 0})
        bucket["rows"] += 1
        bucket["valid"] += int(row["score"]["valid"])
        bucket["correct"] += int(row["score"]["correct"])
        bucket["loops"] += int(row["score"]["loop"])
        bucket["caps"] += int(row["score"]["hit_cap"])
    summary = {
        "schema_version": "phase15_kaggle_smoke_v1",
        "status": "complete",
        "scientific_evidence": False,
        "episodes": len(episodes),
        "completion_rows": len(completions),
        "update_rows": len(updates),
        "expected_completion_rows": 8 * 4 * 3,
        "expected_update_rows": 8 * 2,
        "adapter_reset_verified": all(
            row.get("reset_sha256") == initial_adapter_sha for row in rows if row.get("type") == "episode_complete"
        ),
        "by_arm": by_arm,
        "wall_seconds": time.time() - started,
        "audit_sha256": sha256(AUDIT),
    }
    if summary["completion_rows"] != summary["expected_completion_rows"]:
        raise RuntimeError(f"Completion count mismatch: {summary}")
    if summary["update_rows"] != summary["expected_update_rows"]:
        raise RuntimeError(f"Update count mismatch: {summary}")
    (WORK / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        WORK.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        (WORK / "failure.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
        print(json.dumps(failure, indent=2), file=sys.stderr, flush=True)
        raise
