"""Doi KHUNG (role) cua luot phan tu trong phase1_sft.jsonl -- KHONG goi API lai.

VI SAO LAM DUOC MA KHONG TON TIEN: trong moi ban ghi train, phan dat tien la
messages[-1] (reasoning chain-of-thought + critique do model lon sinh, thuong
5000-25000 ky tu). Phan do HOAN TOAN DOC LAP voi lop boc role cua luot phan tu.
Doi khung chi la sua 1 message: role va lop boc quanh verifier_detail. Nen tao mot
bien the dataset cho thi nghiem H3 -> H4 chi ton vai giay CPU, thay vi ~500 lan goi
API (~15k VND) neu sinh lai tu dau.

Dung:
    python -m src.pipeline.reframe_dataset --to memory \
        --in data/processed/phase1_sft.jsonl \
        --out data/processed/phase1_sft_memory.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.prompts import REFLECT_PROMPT_TEMPLATE, REFLECT_ROLES, build_reflect_message

# Tien to nhan dien luot phan tu -- ON DINH hon la gia dinh no o index 3, vi so luong
# message co the doi neu sau nay them turn (vd nhieu vong sua).
_REFLECT_PREFIX = REFLECT_PROMPT_TEMPLATE.split("{")[0]  # "Kết quả kiểm tra: SAI.\n..."


def _find_reflect_index(messages: list[dict]) -> int | None:
    for i, m in enumerate(messages):
        content = m.get("content", "")
        # Ban da duoc reframe sang memory se co lop boc <memory> o ngoai -> strip roi so.
        stripped = content.replace("<memory>", "").replace("</memory>", "").strip()
        if stripped.startswith(_REFLECT_PREFIX.strip()):
            return i
    return None


def _extract_detail(content: str) -> str:
    """Lay lai verifier_detail tho tu noi dung da duoc dinh dang."""
    stripped = content.replace("<memory>", "").replace("</memory>", "").strip()
    head, sep, tail = REFLECT_PROMPT_TEMPLATE.partition("{verifier_detail}")
    if not sep:
        return stripped
    body = stripped[len(head.strip()) :] if stripped.startswith(head.strip()) else stripped
    if tail.strip() and body.endswith(tail.strip()):
        body = body[: -len(tail.strip())]
    return body.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", required=True, choices=list(REFLECT_ROLES))
    parser.add_argument("--in", dest="in_path", default="data/processed/phase1_sft.jsonl")
    parser.add_argument("--out", dest="out_path", required=True)
    args = parser.parse_args()

    in_path, out_path = Path(args.in_path), Path(args.out_path)
    if out_path.resolve() == in_path.resolve():
        raise SystemExit("--out phai khac --in (khong ghi de dataset goc).")

    rows_in = [json.loads(l) for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    converted = skipped = 0
    out_lines = []
    for rec in rows_in:
        messages = rec["messages"]
        idx = _find_reflect_index(messages)
        if idx is None:
            # Khong tim thay luot phan tu -> GIU NGUYEN ban ghi thay vi doan bua.
            skipped += 1
            out_lines.append(json.dumps(rec, ensure_ascii=False))
            continue
        detail = _extract_detail(messages[idx]["content"])
        messages[idx] = build_reflect_message(detail, args.to)
        converted += 1
        out_lines.append(json.dumps(rec, ensure_ascii=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print(f"Doc  : {in_path}  ({len(rows_in)} ban ghi)")
    print(f"Ghi  : {out_path}")
    print(f"Doi khung sang '{args.to}': {converted} ban ghi | khong tim thay luot phan tu: {skipped}")
    if converted:
        sample = json.loads(out_lines[0])["messages"]
        i = _find_reflect_index(sample)
        print(f"\nVi du luot phan tu sau khi doi (role={sample[i]['role']!r}):")
        print("  " + sample[i]["content"][:180].replace("\n", "\n  "))


if __name__ == "__main__":
    main()
