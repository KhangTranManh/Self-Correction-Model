"""Phase 2 — Render du lieu KTO conversational -> standard (chuoi) cho trl.

VI SAO CAN BUOC NAY (phat hien khi smoke train_kto 2026-07-23):
trl KTOTrainer o che do CONVERSATIONAL tu ap chat template va VALIDATE role cua
message CUOI trong prompt -- no BAO LOI "Invalid role in the last message: tool"
vi prompt cua ta ket thuc bang tin nhan reflect role "tool" (build_reflect_message).
Doi role sang "user" se LECH prompt so voi luc train/eval (vi pham #3). Cach dung:
tu render prompt+completion thanh CHUOI bang CHINH chat template (giong het luc
sinh: apply_chat_template(..., add_generation_prompt=True)), roi dua trl o che do
STANDARD -- trl khong dung template nua, khong validate role, prompt van byte-exact.

Vao : jsonl {prompt:[msgs], completion:[{assistant}], label:bool}  (tu build_preference/gen_preference)
Ra  : jsonl {prompt:"<chuoi da render>", completion:"<text>", label:bool}

Chay (can transformers + tokenizer cua base model; chay tren box hoac may co transformers):
    python render_kto.py --in data/preference/kto_train.jsonl --out data/preference/kto_train_std.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

_DEFAULT_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=_DEFAULT_TOKENIZER,
                    help="Phai la tokenizer/chat-template DUNG voi luc sinh (Qwen2.5-7B).")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    n = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for line in open(args.inp, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            # add_generation_prompt=True: prompt ket thuc bang '<|im_start|>assistant\\n'
            # -> completion (text assistant) noi vao la lien mach, byte-exact voi luc sinh.
            prompt_str = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
            comp_str = r["completion"][0]["content"]
            out.write(json.dumps({"prompt": prompt_str, "completion": comp_str, "label": r["label"]},
                                 ensure_ascii=False) + "\n")
            n += 1
    print(f"Da render {n} dong -> {out_path}")


if __name__ == "__main__":
    main()
