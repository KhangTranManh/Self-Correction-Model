"""Phase 2 — Buoc 0 (KHONG GPU): dung bo du lieu KTO tu output cua Phase 1.

Tai sao KTO (Kahneman-Tversky Optimization) chu khong phai DPO:
- DPO can CAP cho cung 1 prompt (chosen vs rejected). Du lieu Phase 1 KHONG co cap
  san: SFT chi co ban sua DUNG (chosen), log eval chi co 1 ban sua/bai. Muon co cap
  sach phai sinh nhieu mau (ton GPU).
- KTO nhan tung mau LE, gan nhan nhi phan {desirable, undesirable} -- khop CHINH XAC
  cai du lieu Phase 1 dang co: ban sua dung = desirable, ban sua hong = undesirable,
  khong can cap. KTO cung ref-free nhe hon (co ich khi sau nay len 9B, VRAM chat).

Nguon du lieu (xem README phase2):
- desirable  : (a) ban sua DUNG cua model lon trong phase1_sft.jsonl (631 mau),
               (b) lan model TU SUA DUNG trong log eval (second_passed=True).
- undesirable: lan model tu sua HONG (second_passed=False) hoac bo do format
               (format_incomplete=True) trong log eval.
- (GIAI DOAN sau, CAN GPU) them undesirable tu test sycophancy: bai model lam DUNG,
  bi bao gia "SAI", model lat thanh sai -> xem gen_sycophancy.py.

Nguyen tac giu nguyen tu Phase 1:
- #1 dung/sai do VERIFIER quyet: nhan desirable/undesirable o day lay TRUC TIEP tu
  ket qua verifier da luu trong log (second_passed) / tu viec mau da qua re-verify
  khi dung SFT -- KHONG co LLM nao cham lai o buoc nay.
- #2 loi phai do CHINH model sinh: undesirable lay tu log eval cua dung model 7B nay
  (khop phan bo loi that). Neu sau doi base model, PHAI sinh lai (log 7B khong dung
  duoc lam undesirable cho model khac).
- #3 prompt train == prompt infer, byte-for-byte: prompt cho vi du tu log eval duoc
  DUNG LAI bang build_reflect_message() + load_held_out_problems() CUA CHINH Phase 1
  (import ben duoi), khong hardcode lai -- lech 1 byte la hong.

Chay (khong GPU, chi can internet de load GSM8K/MBPP mot lan cho phan tu log eval):
    python build_preference.py \
        --sft ../phase1/backup_server/processed/phase1_sft.jsonl \
        --eval-log <duong_dan>/probe_baseline.jsonl \
        --out data/preference/kto_seed.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PHASE2_DIR = Path(__file__).resolve().parent
PHASE1_DIR = PHASE2_DIR.parent / "phase1"
# phase2 KHONG co package 'src' rieng -> them phase1/ vao path de dung DUNG cac
# primitive chia-se-mot-nguon cua Phase 1 (prompts, loader eval). Tranh sao chep
# lai -> khong bao gio lech giua train va infer (ly do src/core/prompts.py ton tai).
sys.path.insert(0, str(PHASE1_DIR))
from src.core.prompts import build_reflect_message  # noqa: E402


def load_sft_desirable(sft_path: Path) -> tuple[list[dict], str]:
    """Moi dong SFT = [system, user, assistant(attempt sai), tool(detail),
    assistant(ban sua DUNG)]. Prompt = 4 msg dau, completion desirable = msg cuoi.
    Tra ve (examples, system_prompt) -- system_prompt lay tu chinh du lieu de dam
    bao khop byte voi luc train."""
    rows = [json.loads(l) for l in open(sft_path, encoding="utf-8") if l.strip()]
    examples = []
    system_prompt = None
    for r in rows:
        msgs = r["messages"]
        if len(msgs) != 5:
            # Dinh dang SFT co dinh 5 msg; dong khac la du lieu la -> bo, khong doan.
            continue
        if system_prompt is None:
            system_prompt = msgs[0]["content"]
        examples.append({
            "prompt": msgs[:4],
            "completion": [msgs[4]],
            "label": True,
            "source": "sft_deepseek",
        })
    if system_prompt is None:
        raise ValueError(f"Khong doc duoc mau SFT hop le nao tu {sft_path}")
    return examples, system_prompt


# tach detail ra khoi tin nhan tool cua SFT de tu-kiem build_reflect_message() dung.
_SFT_TOOL_DETAIL_RE = re.compile(
    r"Chi tiết lỗi từ hệ thống kiểm tra:\s*(.*?)\nHãy tự", re.DOTALL
)


def _self_check_reflect_framing(sft_rows_sample: dict) -> None:
    """Bao dam build_reflect_message() (Phase 1) dung lai DUNG khung tool ma du lieu
    SFT da train. Neu Phase 1 doi template ma quen, buoc nay bat ngay -- thay vi am
    tham sinh prompt lech (vi pham nguyen tac #3)."""
    tool_msg = sft_rows_sample["messages"][3]
    m = _SFT_TOOL_DETAIL_RE.search(tool_msg["content"])
    if not m:
        return  # khong tach duoc detail -> bo qua self-check (khong chan build)
    detail = m.group(1)
    rebuilt = build_reflect_message(detail, tool_msg["role"])
    if rebuilt != tool_msg:
        raise AssertionError(
            "build_reflect_message() KHONG dung lai dung khung tool cua du lieu SFT "
            f"-- prompt se lech byte (vi pham #3).\n  SFT : {tool_msg!r}\n  rebuilt: {rebuilt!r}"
        )


def _questions_by_id(math_limit: int, code_limit: int) -> dict[str, str]:
    """Dung DUNG loader eval cua Phase 1 -> id->question khop byte voi luc eval.
    load_held_out_problems(math_offset, math_limit, code_offset, code_limit)."""
    from src.pipeline.eval_common import load_held_out_problems
    problems = load_held_out_problems(150, math_limit, 150, code_limit)
    return {p.id: p.question for p in problems}


def load_eval_examples(
    log_path: Path, questions: dict[str, str], system_prompt: str
) -> list[dict]:
    rows = [json.loads(l) for l in open(log_path, encoding="utf-8") if l.strip()]
    out = []
    for r in rows:
        if r.get("initial_passed"):
            continue  # dung ngay lan dau -> khong co buoc phe binh de hoc
        pid = r["problem_id"]
        question = questions.get(pid)
        if question is None:
            continue  # khong dung lai duoc prompt -> bo, khong doan cau hoi
        completion_text = r.get("correction_text")
        if not completion_text:
            continue
        # Dung lai prompt Y HET luc infer: system(tu SFT) + question(tu loader Phase 1)
        # + attempt_text(log) + tool(build_reflect_message, role tu log).
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": r["attempt_text"]},
            build_reflect_message(r["verifier_detail"], r.get("reflect_role", "tool")),
        ]
        completion = [{"role": "assistant", "content": completion_text}]

        if r.get("second_passed"):
            label, source = True, "eval_self_correct_pass"
        elif r.get("format_incomplete"):
            label, source = False, "eval_format_incomplete"
        else:
            label, source = False, "eval_self_correct_fail"

        out.append({"prompt": prompt, "completion": completion,
                    "label": label, "source": source})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sft", type=str,
        default=str(PHASE1_DIR / "backup_server" / "processed" / "phase1_sft.jsonl"),
        help="phase1_sft.jsonl -- nguon desirable chinh (ban sua DUNG cua model lon).",
    )
    parser.add_argument(
        "--eval-log", type=str, action="append", default=[],
        help="Log eval Phase 1 (co the lap lai nhieu lan). Nguon undesirable (model "
             "tu sua hong) + desirable phu (model tu sua dung). Khong bat buoc.",
    )
    parser.add_argument(
        "--math-limit", type=int, default=600,
        help="Pham vi GSM8K held-out de dung lai question cho vi du tu log (khop eval).",
    )
    parser.add_argument("--code-limit", type=int, default=150)
    parser.add_argument(
        "--out", type=str, default=str(PHASE2_DIR / "data" / "preference" / "kto_seed.jsonl"),
    )
    args = parser.parse_args()

    sft_examples, system_prompt = load_sft_desirable(Path(args.sft))
    # self-check khung tool tren mau SFT dau tien
    _self_check_reflect_framing(
        next(iter(json.loads(l) for l in open(args.sft, encoding="utf-8") if l.strip()))
    )

    all_examples = list(sft_examples)
    questions = None
    for log in args.eval_log:
        if questions is None:
            questions = _questions_by_id(args.math_limit, args.code_limit)
        all_examples.extend(load_eval_examples(Path(log), questions, system_prompt))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Thong ke
    from collections import Counter
    by_source = Counter(e["source"] for e in all_examples)
    n_desirable = sum(1 for e in all_examples if e["label"])
    n_undesirable = len(all_examples) - n_desirable
    print(f"Da ghi {len(all_examples)} vi du KTO vao {out_path}")
    print(f"  desirable  : {n_desirable}")
    print(f"  undesirable: {n_undesirable}")
    print("  theo nguon:")
    for src, n in sorted(by_source.items()):
        print(f"    {src:28s} {n}")
    if n_undesirable == 0:
        print("  [!] Chua co vi du undesirable nao -- can them --eval-log, hoac sinh "
              "sycophancy (gen_sycophancy.py, CAN GPU). KTO can ca 2 phia moi hoc duoc.")


if __name__ == "__main__":
    main()
