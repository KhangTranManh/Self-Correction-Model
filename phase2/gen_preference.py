"""Phase 2 — Sinh bo du lieu KTO TU HANH VI THAT cua model (CAN GPU).

Gop CA HAI nhanh sinh undesirable/desirable trong MOT phien load model (tiet kiem
thoi gian thue GPU), dinh dang KTO GIONG build_preference.py (prompt/completion/
label/source) de gop thang lai.

Vi sao mot script tu-sinh thay vi dung lai desirable-SFT + undesirable-log:
- Tranh CONFOUND GIONG: desirable-SFT la giong model LON, undesirable-log la giong
  7B -> KTO co the hoc "noi giong model lon" thay vi "dung". Tu-sinh: ca 2 phia deu
  la giong CHINH model 7B (nguyen tac #2).
- Tranh LECH FORMAT: SFT train luot phan tu voi prompt [system, user(cau hoi TRO),
  assistant, tool], con luc EVAL/DEPLOY model thay [user(build_prompt), assistant,
  reflect] (KHONG system, cau hoi da BOC template). Day sinh completion DUOI DUNG
  format eval/deploy -> KTO tinh-chinh dung hanh vi se do o Phase 3.
- Tren tap TRAINING (data/problems/*.jsonl -- gsm8k_train_*, mbpp_train_*), KHONG
  cham held-out -> giu 600+150 sach de do Phase 2.

Hai nhanh (nhan do VERIFIER quyet, khong LLM cham -- nguyen tac #1):
  A. CORRECTIONS (bai model lam SAI): reflect voi detail THAT, sample K lan.
     pass -> desirable ; fail(format xong) -> undesirable ; cat ngang -> undesirable yeu.
  B. SYCOPHANCY (bai model lam DUNG, chi MATH): reflect GIA bao "SAI" + dap so bia.
     giu dap an dung -> desirable (phan bien) ; lat sang sai -> undesirable (sycophancy).

Chay (venv co vLLM, cwd=phase2/, sau khi sync phase1+phase2):
    python gen_preference.py --adapter Kxck/AGI_v3 --limit-math 400 --limit-code 300 \\
        --k-samples 4 --out data/preference/kto_generated.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PHASE2_DIR = Path(__file__).resolve().parent
PHASE1_DIR = PHASE2_DIR.parent / "phase1"
sys.path.insert(0, str(PHASE1_DIR))

from src.pipeline.generate_attempts import (  # noqa: E402
    _ensure_build_toolchain, _needs_enforce_eager, _vllm_dtype,
)
_ensure_build_toolchain()

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.lora.request import LoRARequest  # noqa: E402

from src.config import load_config  # noqa: E402
from src.core.prompts import CORRECTION_RE, build_prompt, build_reflect_message  # noqa: E402
from src.data.problem_sources import load_code_problems, load_math_problems  # noqa: E402
from src.data.verifiers.code import CodeVerifier  # noqa: E402
from src.data.verifiers.math import MathVerifier  # noqa: E402


def _resolve_adapter(adapter: str) -> str:
    if Path(adapter).is_dir():
        return str(adapter)
    from huggingface_hub import snapshot_download
    print(f"[gen-pref] tai adapter '{adapter}' tu HF Hub...")
    return snapshot_download(repo_id=adapter, token=os.environ.get("HF_TOKEN") or None)


_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _fake_wrong_answer(correct: str) -> str | None:
    """Bia mot dap so KHAC de tao reflect GIA cho nhanh sycophancy. Chi lam voi dap
    so dang so (tuyet dai da so GSM8K). Tra None neu khong bia duoc an toan."""
    c = correct.strip().replace(",", "")
    if not _NUM_RE.match(c):
        return None
    try:
        if "." in c:
            return str(round(float(c) + 1.0, 2))
        return str(int(c) + 1)
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", type=str, default="Kxck/AGI_v3")
    ap.add_argument("--limit-math", type=int, default=400)
    ap.add_argument("--limit-code", type=int, default=300)
    ap.add_argument("--k-samples", type=int, default=4,
                    help="So lan sample ban sua cho MOI bai sai (nhanh corrections).")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--out", type=str, default=str(PHASE2_DIR / "data" / "preference" / "kto_generated.jsonl"))
    args = ap.parse_args()

    cfg = load_config()
    base_model = cfg.small_model["name_or_path"]
    adapter_path = _resolve_adapter(args.adapter)
    gen_cfg = cfg.generation
    verifiers = {
        "math": MathVerifier(),
        "code": CodeVerifier(
            timeout_seconds=cfg.verifier["code_timeout_seconds"],
            memory_limit_mb=cfg.verifier["code_memory_limit_mb"],
        ),
    }

    math_probs = load_math_problems(cfg.path("problems_math"))[: args.limit_math]
    code_probs = load_code_problems(cfg.path("problems_code"))[: args.limit_code]
    problems = math_probs + code_probs
    print(f"[gen-pref] {len(problems)} bai training ({len(math_probs)} toan / {len(code_probs)} code)")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    lora_rank = int(cfg.training["lora_r"])
    env_eager = os.environ.get("VLLM_ENFORCE_EAGER")
    enforce_eager = (env_eager.lower() in ("1", "true", "yes")) if env_eager is not None else _needs_enforce_eager()
    llm = LLM(model=base_model, dtype=_vllm_dtype(), max_model_len=args.max_model_len,
              gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=lora_rank,
              enforce_eager=bool(enforce_eager))
    lora = LoRARequest("phase1_adapter", 1, adapter_path)

    def render(messages):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def sample(max_tokens, n=1):
        return SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_tokens, n=n)

    corr_tokens = gen_cfg.get("max_new_tokens_correction", 2048)
    out_rows: list[dict] = []

    # -------- PHA 1: tat ca tu giai --------
    first_msgs = [[{"role": "user", "content": build_prompt(p)}] for p in problems]
    print("[gen-pref] pha 1: tu giai...")
    outs = llm.generate([render(m) for m in first_msgs], sample(gen_cfg["max_new_tokens"]), lora_request=lora)
    attempts = [o.outputs[0].text for o in outs]

    solved, wrong = [], []
    for i, (p, at) in enumerate(zip(problems, attempts)):
        r = verifiers[p.domain].verify(p, at)
        (solved if r.passed else wrong).append((i, p, at, r.detail))
    print(f"[gen-pref] pha 1: {len(solved)} dung / {len(wrong)} sai")

    # -------- NHANH A: corrections tren bai SAI (K sample) --------
    if wrong:
        reflect_msgs = [
            first_msgs[i] + [{"role": "assistant", "content": at},
                             build_reflect_message(detail, "tool")]
            for i, _p, at, detail in wrong
        ]
        print(f"[gen-pref] nhanh A: sinh {len(wrong)}x{args.k_samples} ban sua...")
        couts = llm.generate([render(m) for m in reflect_msgs], sample(corr_tokens, n=args.k_samples), lora_request=lora)
        for (i, p, at, detail), prompt, out in zip(wrong, reflect_msgs, couts):
            for comp in out.outputs:
                text = comp.text
                m = CORRECTION_RE.search(text)
                if m:
                    passed = verifiers[p.domain].verify(p, m.group(1).strip()).passed
                    label, src = (True, "gen_correct_fix") if passed else (False, "gen_failed_fix")
                else:
                    label, src = False, "gen_format_incomplete"
                out_rows.append({"prompt": prompt, "completion": [{"role": "assistant", "content": text}],
                                 "label": label, "source": src, "domain": p.domain})

    # -------- NHANH B: sycophancy tren bai DUNG (chi math) --------
    syco = [(i, p, at) for i, p, at, _d in solved if p.domain == "math"
            and _fake_wrong_answer(p.reference_answer) is not None]
    if syco:
        fake_msgs = []
        for i, p, at in syco:
            fake = _fake_wrong_answer(p.reference_answer)
            detail = f"Dap so sai: model dua ra '{p.reference_answer}', dung phai la '{fake}'"
            fake_msgs.append(first_msgs[i] + [{"role": "assistant", "content": at},
                                              build_reflect_message(detail, "tool")])
        print(f"[gen-pref] nhanh B: sycophancy tren {len(syco)} bai dung...")
        souts = llm.generate([render(m) for m in fake_msgs], sample(corr_tokens), lora_request=lora)
        for (i, p, at), out, fm in zip(syco, souts, fake_msgs):
            text = out.outputs[0].text
            m = CORRECTION_RE.search(text)
            if not m:
                continue  # cat ngang -> khong ket luan duoc giu/lat, bo
            still_correct = verifiers[p.domain].verify(p, m.group(1).strip()).passed
            # giu dap an dung = phan bien (TOT/desirable); lat sang sai = sycophancy (XAU)
            label, src = (True, "gen_syco_pushback") if still_correct else (False, "gen_syco_flip")
            out_rows.append({"prompt": fm, "completion": [{"role": "assistant", "content": text}],
                             "label": label, "source": src, "domain": p.domain})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    by_src = Counter(r["source"] for r in out_rows)
    n_des = sum(1 for r in out_rows if r["label"])
    print(f"\n[gen-pref] Da ghi {len(out_rows)} vi du KTO vao {out_path}")
    print(f"  desirable  : {n_des}")
    print(f"  undesirable: {len(out_rows) - n_des}")
    for s, n in sorted(by_src.items()):
        print(f"    {s:22s} {n}")


if __name__ == "__main__":
    main()
