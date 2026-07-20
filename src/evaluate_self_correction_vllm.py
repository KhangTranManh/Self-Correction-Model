"""Ban vLLM cua evaluate_self_correction.py -- CUNG phep do, khac engine.

VI SAO CO BAN NAY: ban Unsloth chay tuan tu, moi problem 1-2 lan model.generate(),
tuc ~120 lan sinh noi tiep cho 60 bai. Phep do nay von co cau truc 2 pha ro rang:
  Pha 1: TAT CA problem tu giai  -> verify
  Pha 2: CHI nhung bai sai moi phan tu -> verify lai
Ca 2 pha deu la batch thuan, dung han cho continuous batching cua vLLM: 2 lan
generate() theo lo thay vi ~120 lan tuan tu.

Loi ich thuc te khong chi la nhanh hon: no lam cho viec eval tren 200-300 bai tro
nen re, ma do chinh la thu can de thu hep sai so. O muc 60 bai + temperature 0.7,
chenh lech 1 case da lam ty le nhay vai phan tram (xem README: 2/9 vs 3/9).

GIU NGUYEN so voi ban Unsloth de 2 ben so sanh duoc:
  - prompt luot 1 va template luot phan tu (deu lay tu src/prompts.py)
  - bo problem held-out (src/eval_common.py)
  - temperature 0.7 / top_p 0.9, max_new_tokens tu configs/phase1.yaml
  - cach xu ly khi thieu "### Sua lai": tinh la CHUA sua duoc, KHONG fallback ve
    text tho (fallback do tung tao ra "tu sua thanh cong" gia -- xem note.txt)

LUU Y KHI DOI CHIEU SO LIEU: doi engine thi kernel va RNG khac nhau, nen ket qua
khong bao gio trung tuyet doi voi ban Unsloth du cung adapter. Voi temperature 0.7,
ban than 2 lan chay cua CUNG mot engine cung da khac nhau. Muon so sanh chat che
thi tang so bai eval (--math-limit/--code-limit) chu dung ky vong 2 engine ra cung
mot con so.

Chay (trong venv co vLLM):
    python -m src.evaluate_self_correction_vllm \
        --math-offset 150 --math-limit 30 --code-offset 150 --code-limit 30
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# PHAI chay TRUOC khi import torch/vllm ben duoi: ham nay set CUDA_HOME, ma
# torch.utils.cpp_extension doc bien do ngay luc import. Xem giai thich day du trong
# src/generate_attempts.py. Import module do cung tu dong goi _ensure_cuda_home().
from src.generate_attempts import _ensure_build_toolchain, _needs_enforce_eager, _vllm_dtype

_ensure_build_toolchain()

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.lora.request import LoRARequest  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.schema import Problem, VerifierResult  # noqa: E402
from src.eval_common import (  # noqa: E402
    append_log,
    load_held_out_problems,
    open_log,
    print_report,
)
from src.prompts import CORRECTION_RE, REFLECT_ROLES, build_prompt, build_reflect_message
from src.verifier.code_verifier import CodeVerifier
from src.verifier.math_verifier import MathVerifier


def _resolve_adapter(adapter: str) -> str:
    """Tra ve duong dan LOCAL toi adapter.

    vLLM's LoRARequest chi nhan duong dan tren dia, khong nhan repo id nhu Unsloth
    -- neu duoc dua repo id thi tai ve truoc.
    """
    if Path(adapter).is_dir():
        return str(adapter)

    from huggingface_hub import snapshot_download

    print(f"[eval-vllm] '{adapter}' khong phai thu muc local -> tai tu HF Hub...")
    path = snapshot_download(repo_id=adapter, token=os.environ.get("HF_TOKEN") or None)
    print(f"[eval-vllm] Da tai ve: {path}")
    return path


def _load_tokenizer(adapter_path: str, base_model: str):
    """Uu tien tokenizer luu kem adapter; neu khong co thi dung cua base model.

    LoRA khong doi vocabulary nen 2 cai phai giong nhau -- fallback nay chi de xu ly
    truong hop thu muc adapter khong kem file tokenizer.
    """
    try:
        return AutoTokenizer.from_pretrained(adapter_path)
    except Exception as e:  # noqa: BLE001
        print(f"[eval-vllm] Khong doc duoc tokenizer tu adapter ({e}) -> dung cua {base_model}")
        return AutoTokenizer.from_pretrained(base_model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-offset", type=int, default=150)
    parser.add_argument("--math-limit", type=int, default=30)
    parser.add_argument("--code-offset", type=int, default=150)
    parser.add_argument("--code-limit", type=int, default=30)
    parser.add_argument("--log-file", type=str, default="outputs/eval_self_correction_log.jsonl")
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Repo id tren HF Hub hoac duong dan local. Mac dinh: paths.lora_out_dir.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Mac dinh lay small_model.name_or_path trong phase1.yaml. vLLM load base "
             "roi gan LoRA len tren, khac Unsloth (tu doc base tu adapter_config.json).",
    )
    parser.add_argument(
        "--reflect-role", type=str, default="tool", choices=list(REFLECT_ROLES),
        help="Khung boc verifier_detail. 'memory' duoc hien thuc bang role system + "
             "the <memory> (KHONG phai role 'memory' -- template Qwen se nuot im lang). "
             "Xem build_reflect_message() trong src/prompts.py.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Luot phan tu = prompt + attempt + toi 2048 token moi, de cham tran 4096. "
             "8192 cho du cho ma van thua VRAM tren card 24GB.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Co dinh RNG de chay lai ra cung ket qua. Mac dinh None = ngau nhien, "
             "giong hanh vi ban Unsloth.",
    )
    args = parser.parse_args()

    cfg = load_config()
    adapter_path = _resolve_adapter(str(args.adapter or cfg.path("lora_out_dir")))
    base_model = args.base_model or cfg.small_model["name_or_path"]
    log_path = open_log(args.log_file)

    gen_cfg = cfg.generation
    verifiers = {
        "math": MathVerifier(),
        "code": CodeVerifier(
            timeout_seconds=cfg.verifier["code_timeout_seconds"],
            memory_limit_mb=cfg.verifier["code_memory_limit_mb"],
        ),
    }

    problems = load_held_out_problems(
        args.math_offset, args.math_limit, args.code_offset, args.code_limit
    )
    print(f"[eval-vllm] {len(problems)} problem held-out "
          f"({sum(p.domain == 'math' for p in problems)} toan / "
          f"{sum(p.domain == 'code' for p in problems)} code)")

    tokenizer = _load_tokenizer(adapter_path, base_model)

    # max_lora_rank PHAI >= lora_r luc train. Mac dinh cua vLLM la 16, ma config dang
    # dung r=32 -> khong set thi vLLM bao loi luc load adapter.
    lora_rank = int(cfg.training["lora_r"])
    # Giong generate_attempts: thieu Python.h thi Triton khong bien dich duoc C
    # extension luc chay -> phai bo torch.compile. Xem _needs_enforce_eager().
    env_eager = os.environ.get("VLLM_ENFORCE_EAGER")
    if env_eager is not None:
        enforce_eager = env_eager.lower() in ("1", "true", "yes")
    else:
        enforce_eager = gen_cfg.get("enforce_eager")
        if enforce_eager is None:
            enforce_eager = _needs_enforce_eager()
    llm = LLM(
        model=base_model,
        dtype=_vllm_dtype(),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        enable_lora=True,
        max_lora_rank=lora_rank,
        enforce_eager=bool(enforce_eager),
    )
    lora_request = LoRARequest("phase1_adapter", 1, adapter_path)
    print(f"[eval-vllm] base={base_model} adapter={adapter_path} lora_rank={lora_rank}")

    def sample(max_tokens: int) -> SamplingParams:
        return SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_tokens,
            seed=args.seed,
        )

    def render(messages: list[dict]) -> str:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ---------------- PHA 1: tat ca problem tu giai (1 lan batch) ----------------
    first_messages = [[{"role": "user", "content": build_prompt(p)}] for p in problems]
    print(f"[eval-vllm] Pha 1: sinh {len(problems)} loi giai dau...")
    first_outputs = llm.generate(
        [render(m) for m in first_messages],
        sample(gen_cfg["max_new_tokens"]),
        lora_request=lora_request,
    )
    attempts = [o.outputs[0].text for o in first_outputs]

    # ---------------- Verify luot 1, tach ra nhung bai sai ----------------
    records: list[dict] = []
    wrong: list[tuple[int, Problem, str, str]] = []  # (idx, problem, attempt, detail)
    for i, (problem, attempt_text) in enumerate(zip(problems, attempts)):
        result = verifiers[problem.domain].verify(problem, attempt_text)
        if result.passed:
            rec = dict(
                problem_id=problem.id,
                domain=problem.domain,
                attempt_text=attempt_text,
                initial_passed=True,
                verifier_detail=result.detail,
            )
            records.append(rec)
            append_log(log_path, **rec)
        else:
            wrong.append((i, problem, attempt_text, result.detail))

    print(f"[eval-vllm] Luot 1: {len(problems) - len(wrong)} dung / {len(wrong)} sai")
    if not wrong:
        print("[eval-vllm] Khong co bai nao sai -- khong co gi de do ky nang tu sua.")
        print_report(records)
        return

    # ---------------- PHA 2: chi nhung bai sai moi phan tu (1 lan batch) ----------------
    reflect_messages = [
        first_messages[i]
        + [
            {"role": "assistant", "content": attempt_text},
            build_reflect_message(detail, args.reflect_role),
        ]
        for i, _p, attempt_text, detail in wrong
    ]
    correction_max_tokens = gen_cfg.get("max_new_tokens_correction", gen_cfg["max_new_tokens"])
    print(f"[eval-vllm] Pha 2: sinh {len(wrong)} lan tu sua (role={args.reflect_role})...")
    second_outputs = llm.generate(
        [render(m) for m in reflect_messages],
        sample(correction_max_tokens),
        lora_request=lora_request,
    )

    for (_i, problem, attempt_text, detail), out in zip(wrong, second_outputs):
        correction_text = out.outputs[0].text
        match = CORRECTION_RE.search(correction_text)
        if match:
            corrected_solution = match.group(1).strip()
            second_result = verifiers[problem.domain].verify(problem, corrected_solution)
        else:
            # Bi cat ngang truoc khi hoan tat dinh dang -> tinh la CHUA sua duoc.
            # KHONG verify correction_text tho: verifier se vo tinh bat trung so/code
            # con do dang trong <thinking>, tao ra ty le tu sua gia.
            corrected_solution = None
            second_result = VerifierResult(
                passed=False,
                detail="Chua hoan tat dinh dang '### Sua lai' (co the bi cat vi het max_new_tokens)",
            )

        status = "CORRECT" if second_result.passed else f"STILL WRONG ({second_result.detail})"
        print(f"[{problem.id}] WRONG -> tu sua: {status}")

        rec = dict(
            problem_id=problem.id,
            domain=problem.domain,
            attempt_text=attempt_text,
            initial_passed=False,
            verifier_detail=detail,
            correction_text=correction_text,
            corrected_solution=corrected_solution,
            format_incomplete=match is None,
            second_passed=second_result.passed,
            second_detail=second_result.detail,
            reflect_role=args.reflect_role,
        )
        records.append(rec)
        append_log(log_path, **rec)

    print_report(records)
    if log_path:
        print(f"\nLog chi tiet: {log_path}")


if __name__ == "__main__":
    main()
