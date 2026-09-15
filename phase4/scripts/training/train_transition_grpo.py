"""Train one bounded Phase 4 GRPO cycle from freshly sampled initial answers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phase1"))

from phase4.lib.transition import (  # noqa: E402
    parse_review,
    parse_review_for_training,
    transition_reward,
)
from src.core.schema import Problem  # noqa: E402
from src.data.verifiers.code import CodeVerifier  # noqa: E402
from src.data.verifiers.math import MathVerifier  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and isinstance(value[-1], dict):
        return str(value[-1].get("content", ""))
    return str(value)


def verify_from_spec(
    domain: str,
    problem_id: str,
    question: str,
    spec: dict[str, Any],
    answer: str,
) -> bool:
    problem = Problem(
        id=problem_id,
        domain=domain,
        question=question,
        reference_answer=spec.get("reference_answer"),
        entry_point=spec.get("entry_point"),
        tests=[str(value) for value in spec.get("tests", [])],
        calc_steps=spec.get("calc_steps"),
    )
    verifier = MathVerifier() if domain == "math" else CodeVerifier()
    return bool(verifier.verify(problem, answer).passed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="/root/models/Kxck_Self_Correction_v1")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-completion-length", type=int)
    parser.add_argument("--num-generations", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--initial-state", choices=("all", "correct", "wrong"), default="all")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "phase4/configs/transition_rl_v1.yaml",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rollout_config = config["rollout"]
    train_config = config["training"]
    rows = read_jsonl(args.rollouts)
    if args.initial_state != "all":
        required_state = args.initial_state == "correct"
        rows = [row for row in rows if bool(row.get("initial_correct")) == required_state]
    if not rows:
        raise ValueError("Rollout file is empty")
    required = {
        "problem_id", "domain", "task_prompt", "initial_output",
        "initial_correct", "verifier_spec", "policy_checkpoint",
    }
    missing = [(i, sorted(required - set(row))) for i, row in enumerate(rows) if required - set(row)]
    if missing:
        raise ValueError(f"Rollouts lack required verifier metadata: {missing[:5]}")
    checkpoints = {str(row["policy_checkpoint"]) for row in rows}
    if len(checkpoints) != 1:
        raise ValueError(f"A GRPO cycle must use one initial-answer policy: {sorted(checkpoints)}")

    neutral_prompt = str(rollout_config["neutral_prompt"])
    review_system_prompt = str(rollout_config["review_system_prompt"])
    dataset_rows = []
    for row in rows:
        dataset_rows.append({
            "prompt": [
                {"role": "system", "content": review_system_prompt},
                {"role": "user", "content": str(row["task_prompt"])},
                {"role": "assistant", "content": str(row["initial_output"])},
                {"role": "user", "content": neutral_prompt},
            ],
            "problem_id": str(row["problem_id"]),
            "domain": str(row["domain"]),
            "question": str(row["task_prompt"]),
            "initial_output": str(row["initial_output"]),
            "initial_correct": bool(row["initial_correct"]),
            "verifier_spec": row["verifier_spec"],
        })
    train_dataset = Dataset.from_list(dataset_rows)
    rewards = config["reward"]
    completion_audit: list[dict[str, Any]] = []

    def objective_transition_reward(
        completions: list[Any],
        problem_id: list[str],
        domain: list[str],
        question: list[str],
        initial_output: list[str],
        initial_correct: list[bool],
        verifier_spec: list[dict[str, Any]],
        **_: Any,
    ) -> list[float]:
        values: list[float] = []
        for completion, pid, kind, prompt, initial, was_correct, spec in zip(
            completions,
            problem_id,
            domain,
            question,
            initial_output,
            initial_correct,
            verifier_spec,
            strict=True,
        ):
            text = completion_text(completion)
            strict_action = parse_review(text)
            action = parse_review_for_training(text)
            final_answer = initial if action.decision == "KEEP" else (action.revised_answer or "")
            final_correct = verify_from_spec(kind, pid, prompt, spec, final_answer)
            reward, reason = transition_reward(
                initial_correct=bool(was_correct),
                final_correct=final_correct,
                action=action,
                rewards=rewards,
            )
            if strict_action.valid:
                reward += float(rewards["exact_contract_bonus"])
            values.append(reward)
            completion_audit.append({
                "problem_id": pid,
                "initial_correct": bool(was_correct),
                "completion": text,
                "training_decision": action.decision,
                "strict_contract_valid": strict_action.valid,
                "final_correct": final_correct,
                "reward_reason": reason,
                "reward": reward,
            })
        return values

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    peft_config = LoraConfig(
        r=int(config["model"]["lora_r"]),
        lora_alpha=int(config["model"]["lora_alpha"]),
        lora_dropout=float(config["model"]["lora_dropout"]),
        target_modules=list(config["model"]["target_modules"]),
        task_type="CAUSAL_LM",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_generations = (
        args.num_generations
        if args.num_generations is not None
        else int(rollout_config["candidates_per_problem"])
    )
    micro_batch_size = max(int(train_config["micro_batch_size"]), num_generations)
    grpo_args = GRPOConfig(
        output_dir=str(args.output_dir),
        seed=int(config["experiment"]["seed"]),
        learning_rate=float(train_config["learning_rate"]),
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=int(train_config["gradient_accumulation_steps"]),
        max_steps=(
            args.max_steps if args.max_steps is not None else int(train_config["max_steps"])
        ),
        num_generations=num_generations,
        temperature=(
            args.temperature
            if args.temperature is not None
            else float(rollout_config["review_temperature"])
        ),
        max_prompt_length=3072,
        max_completion_length=(
            args.max_completion_length
            if args.max_completion_length is not None
            else int(rollout_config["review_max_tokens"])
        ),
        beta=float(train_config["beta"]),
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        log_completions=True,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=objective_transition_reward,
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    # TRL's k-bit preparation casts non-quantized parameters to FP32. Qwen's
    # untied output head must match the BF16 hidden state during generation.
    output_head = trainer.model.get_output_embeddings()
    output_head.to(dtype=torch.bfloat16)
    original_output_forward = output_head.forward

    def dtype_safe_output_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        return original_output_forward(hidden_states.to(dtype=output_head.weight.dtype))

    output_head.forward = dtype_safe_output_forward
    result = trainer.train()
    final_adapter = args.output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(final_adapter)
    (args.output_dir / "completion_audit.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in completion_audit
        ),
        encoding="utf-8",
    )
    report = {
        "schema_version": "phase4_transition_grpo_train_v1",
        "model": args.model,
        "initial_policy_checkpoint": next(iter(checkpoints)),
        "rollouts": str(args.rollouts.resolve()),
        "rollouts_sha256": sha256(args.rollouts),
        "rows": len(rows),
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "metrics": result.metrics,
        "final_adapter": str(final_adapter.resolve()),
    }
    (args.output_dir / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
