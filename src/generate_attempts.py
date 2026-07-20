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
  - T4 (CC 7.5): dtype=float16 tu dong (bf16 can CC >= 8.0).
  - 7B bf16 ~15GB VRAM. Tren GPU >= 24GB de generation.load_in_4bit=false ->
    KHONG dung bitsandbytes, tranh han lop loi CUDA runtime (libnvJitLink.so.13).
  - Tren GPU < 20GB (T4 16GB) buoc phai de true, va khi do bitsandbytes phai khop
    CUDA runtime cua may -- day chinh la cho da fail thuc te tren Kaggle T4.

Chay tren may co GPU:
    pip install vllm
    python -m src.generate_attempts
"""
from __future__ import annotations

import json
import os
import sys
import sysconfig
from pathlib import Path


def _ensure_build_toolchain() -> None:
    """Vá 2 thu ma torch.compile/inductor can nhung khong tu tim thay khi goi python
    cua venv TRUC TIEP (khong `activate`).

    1) CUDA_HOME. Inductor can `nvcc`. May khong cai CUDA toolkit he thong se chet:
           RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
           doesn't exist
       Nhung nvcc VAN CO SAN -- goi pip `nvidia-cuda-nvcc` (vllm keo ve theo) dat no
       o site-packages/nvidia/cu13/, thu muc do co du layout bin/include/lib/nvvm.
       Chi la torch khong biet nhin vao day. Nen KHONG can cai CUDA toolkit ~3GB va
       khong can quyen root.

    2) PATH. `ninja` (build system cua torch cpp_extension) nam o <venv>/bin/ninja.
       Khi chay `<venv>/bin/python -m ...` ma khong activate venv, <venv>/bin KHONG
       nam trong PATH -> chet voi:
           FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
       du ninja da duoc cai san. Them thu muc chua interpreter vao dau PATH la du.

    PHAI goi TRUOC khi import torch: torch.utils.cpp_extension tinh CUDA_HOME ngay
    luc import module, khong phai luc bien dich. Do la ly do ham nay nam tren cac
    import ben duoi, trai voi thu tu import thong thuong.
    """
    bin_dir = str(Path(sys.executable).parent)
    if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

    if os.environ.get("CUDA_HOME") or Path("/usr/local/cuda/bin/nvcc").exists():
        return
    site_packages = Path(sysconfig.get_paths()["purelib"])
    for candidate in sorted(site_packages.glob("nvidia/cu*"), reverse=True):
        if (candidate / "bin" / "nvcc").exists():
            os.environ["CUDA_HOME"] = str(candidate)
            print(f"[toolchain] CUDA_HOME -> {candidate} (nvcc tu goi pip)")
            return


_ensure_build_toolchain()

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from src.config import load_config
from src.data.problem_sources import load_code_problems, load_math_problems
from src.data.schema import Problem
from src.prompts import build_prompt as _build_prompt  # noqa: E402
from src.run_lock import acquire_single_instance  # noqa: E402


def _needs_enforce_eager() -> bool:
    """True neu may THIEU header Python (Python.h) -> phai tat torch.compile.

    vLLM mac dinh dung torch.compile + Triton, ma Triton bien dich mot C extension
    NGAY LUC CHAY. Viec do can header phat trien cua Python. Neu thieu, engine chet
    o buoc khoi tao voi:
        fatal error: Python.h: No such file or directory
        torch._inductor.exc.InductorError: CalledProcessError ... /usr/bin/gcc ...
    -- sau khi da tai va nap xong 15GB weight, tuc mat vai phut moi biet.

    Cach sua dung la cai goi header (`apt install python3-dev`, can root). Khi khong
    co quyen root, enforce_eager=True cho phep chay tiep: bo torch.compile va CUDA
    graph, doi lai cham hon (voi batch lon thi khac biet vua phai vi da compute-bound).
    """
    return not any(
        (Path(p) / f"python{sys.version_info.major}.{sys.version_info.minor}" / "Python.h").exists()
        or (Path(p) / "Python.h").exists()
        for p in sysconfig.get_paths().get("include", "").split(os.pathsep) + ["/usr/include"]
    )


def _vllm_dtype() -> str:
    """T4/CC 7.5 khong ho tro bfloat16 (can CC >= 8.0). Auto chon half tren GPU cu.

    vLLM se crash voi: 'Bfloat16 is only supported on GPUs with compute capability
    of at least 8.0' neu hardcode bfloat16 tren Tesla T4.
    """
    if not torch.cuda.is_available():
        return "float16"
    major, _minor = torch.cuda.get_device_capability(0)
    # Ampere (8.0+) va moi hon: bf16 ok. Turing/T4 (7.5): phai float16.
    if major >= 8:
        return "bfloat16"
    return "float16"


def main() -> None:
    cfg = load_config()

    problems: list[Problem] = load_math_problems(cfg.path("problems_math")) + load_code_problems(
        cfg.path("problems_code")
    )

    gen_cfg = cfg.generation
    out_path = cfg.path("attempts_out")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 2 instance cung ghi attempts.jsonl se lam hong file (da tung xay ra: file bi cat
    # cut, ban ghi xen ke). O day con ton them ca GPU. Xem src/run_lock.py.
    acquire_single_instance("generate_attempts", out_path.parent)

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

    dtype = _vllm_dtype()
    # generation.load_in_4bit dieu khien RIENG vLLM o day; small_model.load_in_4bit
    # la cua Unsloth/QLoRA (train_sft, evaluate). Tach ra de tren GPU 24GB co the
    # chay vLLM bf16 (khong bitsandbytes) MA VAN giu QLoRA 4-bit khi train -- doi
    # quantization luc train se lam so lieu khong con so sanh duoc voi baseline cu.
    # Fallback ve khoa cu neu config chua co khoa moi (ban yaml cu).
    use_4bit = bool(gen_cfg.get("load_in_4bit", cfg.small_model["load_in_4bit"]))
    max_len = int(cfg.small_model["max_seq_length"])
    cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "n/a"
    print(
        f"[generate_attempts] vLLM dtype={dtype} CC={cc} "
        f"load_in_4bit={use_4bit} max_model_len={max_len}"
    )

    # Thu tu uu tien: bien moi truong VLLM_ENFORCE_EAGER -> generation.enforce_eager
    # trong yaml -> tu phat hien. Bien moi truong ton tai de script dieu phoi co the
    # CHAY LAI o che do eager khi lan dau that bai, ma khong phai sua file config.
    #
    # torch.compile cua vLLM keo theo ca mot chuoi cong cu bien dich (Python.h ->
    # nvcc -> ninja -> ...), moi may thieu mot thu khac nhau, va moi lan that bai deu
    # ton vai phut nap model moi biet. enforce_eager bo han torch.compile: cham hon
    # nhung khong phu thuoc cong cu nao ca.
    env_eager = os.environ.get("VLLM_ENFORCE_EAGER")
    if env_eager is not None:
        enforce_eager = env_eager.lower() in ("1", "true", "yes")
        print(f"[generate_attempts] enforce_eager={enforce_eager} (tu VLLM_ENFORCE_EAGER)")
    else:
        enforce_eager = gen_cfg.get("enforce_eager")
        if enforce_eager is None:
            enforce_eager = _needs_enforce_eager()
            if enforce_eager:
                print(
                    "[generate_attempts] KHONG tim thay Python.h -> bat enforce_eager "
                    "(bo torch.compile/CUDA graph). Cai 'python3-dev' roi chay lai de "
                    "co hieu nang day du."
                )

    llm_kwargs = dict(
        model=model_name,
        max_model_len=max_len,
        dtype=dtype,
        gpu_memory_utilization=0.85,
        enforce_eager=bool(enforce_eager),
    )
    if use_4bit:
        # Tuong duong load_in_4bit=True cua Unsloth/bitsandbytes -- can de fit model
        # 7B vao GPU nho (vd T4 16GB). Ten tham so nay dung voi vLLM ban >=0.5.x;
        # neu vLLM ban moi hon doi ten, sua lai 2 dong nay theo docs vLLM hien tai.
        # Luu y: bitsandbytes can CUDA runtime khop (vd libnvJitLink.so.13 tren bnb moi).
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
