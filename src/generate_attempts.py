"""Chay small model bang vLLM (batch inference, nhanh hon nhieu so voi Unsloth/HF
generate() tung bai mot) tren bo problem toan/code de thu thap attempt THAT (khong
phai loi DeepSeek tuong tuong) -> data/processed/attempts.jsonl.

VI SAO VLLM O DAY: day la tac vu inference thuan tren model GOC (chua LoRA), khong
can cac toi uu train cua Unsloth -- phu hop nhat de doi sang vLLM (continuous
batching + PagedAttention xu ly nhieu prompt cung luc mot cach hieu qua). Loi ich
ro nhat khi so luong problem lon (~1000+, xem note.txt muc 9). KHONG doi
evaluate_self_correction.py (can LoRA adapter + luong tuan tu attempt->reflect,
loi ich vLLM it chac chan hon neu khong viet lai vong lap thanh batch) va
train_sft.py (can dung Unsloth QLoRA de train, vLLM khong phuc vu train).

CANH BAO MOI TRUONG: cai vllm CO THE keo theo ban torch khac voi ban Unsloth dang
dung trong cung 1 Colab session (2 thu vien co yeu cau torch/CUDA rieng). Neu sau
khi chay xong script nay ma train_sft.py (dung Unsloth) bao loi la, thu Runtime >
Restart session roi cai lai unsloth (pip install unsloth) truoc khi chay train_sft.

RESUMABLE: neu attempts.jsonl da co san, tu dong bo qua cac problem_id da co attempt,
chi sinh THEM cho problem moi (vd sau khi mo rong problems_math/problems_code qua
prepare_public_datasets.py --math-extra/--code-extra) -- khong ton GPU chay lai tu dau.

YEU CAU PHAN CUNG: GPU compute capability >= 7.5 (giong Unsloth).

Chay tren may co GPU:
    pip install vllm
    python -m src.generate_attempts
"""
from __future__ import annotations

import json

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.config import load_config
from src.data.problem_sources import load_code_problems, load_math_problems
from src.data.schema import Problem

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

    gen_cfg = cfg.generation
    out_path = cfg.path("attempts_out")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resumable: neu attempts.jsonl da co san (vd sau khi them problem moi qua
    # prepare_public_datasets.py --math-extra/--code-extra), bo qua cac problem
    # DA co attempt -- tranh chay lai GPU inference tu dau moi lan scale them.
    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["problem_id"])
        print(f"[generate_attempts] Da co {len(done_ids)} problem duoc attempt tu truoc -- bo qua, chi sinh moi.")

    problems = [p for p in problems if p.id not in done_ids]
    if not problems:
        print("[generate_attempts] Khong co problem moi nao can sinh attempt. Xong.")
        return

    model_name = cfg.small_model["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm_kwargs = dict(
        model=model_name,
        max_model_len=cfg.small_model["max_seq_length"],
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
    )
    if cfg.small_model["load_in_4bit"]:
        # Tuong duong load_in_4bit=True cua Unsloth/bitsandbytes -- can de fit model
        # 7B vao GPU nho (vd T4 16GB). Ten tham so nay dung voi vLLM ban >=0.5.x;
        # neu vLLM ban moi hon doi ten, sua lai 2 dong nay theo docs vLLM hien tai.
        llm_kwargs["quantization"] = "bitsandbytes"
        llm_kwargs["load_format"] = "bitsandbytes"

    llm = LLM(**llm_kwargs)

    # Xay tat ca prompt truoc (dang text da qua chat template), roi goi generate()
    # MOT LAN DUY NHAT cho toan bo -- day la cho vLLM phat huy continuous batching,
    # khac han vong lap tung bai mot cua ban Unsloth/HF generate() truoc day.
    prompt_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": _build_prompt(problem)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for problem in problems
    ]

    sampling_params = SamplingParams(
        n=gen_cfg["num_attempts_per_problem"],
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
        max_tokens=gen_cfg["max_new_tokens"],
    )

    print(f"[generate_attempts] Dang sinh attempt cho {len(problems)} problem moi qua vLLM...")
    request_outputs = llm.generate(prompt_texts, sampling_params)

    with open(out_path, "a", encoding="utf-8") as f:
        for problem, request_output in zip(problems, request_outputs):
            for attempt_idx, completion_output in enumerate(request_output.outputs):
                f.write(
                    json.dumps(
                        {
                            "problem_id": problem.id,
                            "attempt_idx": attempt_idx,
                            "text": completion_output.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(f"[{problem.domain}] {problem.id}: done ({gen_cfg['num_attempts_per_problem']} attempt(s))")

    print(f"Da ghi attempts vao: {out_path}")


if __name__ == "__main__":
    main()
