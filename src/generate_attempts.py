"""Chay small model (Unsloth, 4-bit) tren bo problem toan/code de thu thap attempt THAT
(khong phai loi DeepSeek tuong tuong) -> data/processed/attempts.jsonl.

YEU CAU PHAN CUNG: GPU compute capability >= 7.5 (xem ghi chu trong train_sft.py).

Chay tren may co GPU:
    python -m src.generate_attempts
"""
from __future__ import annotations

import json

# unsloth PHAI import truoc trl/transformers/peft (xem ghi chu trong train_sft.py)
from unsloth import FastLanguageModel

from src.config import load_config
from src.data.problem_sources import load_code_problems, load_math_problems
from src.data.schema import Problem
from src.model_loading import load_model_and_tokenizer

_PROMPT_TEMPLATES = {
    "math": (
        "Giải bài toán sau từng bước, sau đó ghi rõ đáp số cuối cùng theo định dạng "
        "'Đáp số: <giá trị>'.\n\nBài toán: {question}"
    ),
    "code": (
        "{question}\n\nChỉ trả về code Python hoàn chỉnh trong 1 code block "
        "(```python ... ```), không giải thích thêm."
    ),
}


def _build_prompt(problem: Problem) -> str:
    return _PROMPT_TEMPLATES[problem.domain].format(question=problem.question)


def main() -> None:
    cfg = load_config()

    problems: list[Problem] = load_math_problems(cfg.path("problems_math")) + load_code_problems(
        cfg.path("problems_code")
    )

    model, tokenizer = load_model_and_tokenizer(cfg, cfg.small_model["max_seq_length"])
    FastLanguageModel.for_inference(model)

    gen_cfg = cfg.generation
    out_path = cfg.path("attempts_out")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for problem in problems:
            prompt = _build_prompt(problem)
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)

            for attempt_idx in range(gen_cfg["num_attempts_per_problem"]):
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=gen_cfg["max_new_tokens"],
                    temperature=gen_cfg["temperature"],
                    top_p=gen_cfg["top_p"],
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
                completion = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
                )

                f.write(
                    json.dumps(
                        {
                            "problem_id": problem.id,
                            "attempt_idx": attempt_idx,
                            "text": completion,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                print(f"[{problem.domain}] {problem.id} attempt {attempt_idx}: done")

    print(f"Da ghi attempts vao: {out_path}")


if __name__ == "__main__":
    main()
