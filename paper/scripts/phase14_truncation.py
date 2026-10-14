"""Phase 14: what a model was doing when its output hit the token cap (CPU, descriptive, post-hoc).

Applies to every generated output of every model that hit its cap, in every split and cell present
under --root: attempts (s1..s5; primary 768-token view, `hit_primary_cap`) and judgments (base and
base_c, 768 tokens, `hit_token_cap`). Each capped output gets one label, checked in this order:

  c  off-topic / garbage / mixed language
       a run of >= 30 identical characters ("1111..."), or >= 2% characters from scripts that are
       neither Latin nor Vietnamese (CJK, Cyrillic, ...), or a language switch inside the text
       (the first and last thirds differ between Vietnamese and English). A text written wholly in
       English (Mistral's habit) is NOT label c; it is counted separately.
  b  looping
       literal repetition (some 10-word sequence >= 3 times, or some line / sentence of >= 25
       characters >= 3 times), or templated enumeration (after replacing every number by 0, some
       line occurs >= 5 times: "Bước 21: 2 + 1 + 8 = 11 không chia hết cho 9", "For x = 50: ...").
       The >= 5 threshold keeps a step-by-step re-walk of the two shown solutions (whose lines
       repeat about twice) out of this label.
  d  (judgments only) drops the two solutions and re-solves the problem from scratch at length:
       the last 60% of the text never refers to the shown solutions ("Solution A/B", "Lời giải A/B",
       "Giải pháp A/B", "cách A/B") and contains its own step-by-step working
  a  long but on track, unfinished: everything else (still checking or computing coherently;
       `wrapping_up` additionally marks texts whose last 400 characters already state a conclusion)

Thresholds were fixed before reading the label counts. Revision 2 (after a hand check of examples
of revision 1): language is detected from function words with LaTeX math removed (math-only text
was read as "English", sending equation loops to label c), and the shown solutions may also be
called "Phương pháp 1/2", "Method 1/2" (missed before, sending ordinary checks to label d). An
earlier pass (single "loop" class on
judgments) was spot-checked by hand: 12 random "loop" cases were about half degenerate and half
coherent re-walks of both solutions; the >= 5 templated threshold and label d come from that check.
Output: paper/results/phase14_truncation.{json,md}.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"llama31_8b": "Llama-3.1-8B", "mistral_7b": "Mistral-7B", "original_solver": "Self_Correction",
         "qwen25_instruct": "Qwen2.5-7B-Instruct"}
DS = {"asdiv": "ASDiv", "math13": "MATH 1–3", "gsm8k13": "GSM8K", "svamp12": "SVAMP", "bridge13": "bridge",
      "gsm8k13_report": "GSM8K (báo cáo)", "svamp12_report": "SVAMP (báo cáo)"}
LABELS = ("a", "b", "c", "d")
LABEL_NAMES = {"a": "dài nhưng đúng hướng, chưa xong", "b": "lặp vòng", "c": "lạc đề / rác / lẫn ngôn ngữ",
               "d": "bỏ hai lời giải, tự giải lại lan man"}
WORD = re.compile(r"\w+", re.UNICODE)
VIET = re.compile(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹáàãéèíìóòõúùýĂÂĐÊÔƠƯ]")
FOREIGN = re.compile(r"[Ѐ-ӿ֐-ۿ฀-๿぀-ヿ㐀-鿿가-힯]")
CONCLUDE = re.compile(r"(Vậy|vậy nên|Kết luận|kết luận|Đáp số|Verdict|Therefore|So the (final )?answer|"
                      r"In conclusion|the correct (answer|solution) is|lời giải đúng là|Answer\s*:|Final answer)",
                      re.IGNORECASE)
STEPS = re.compile(r"(Bước\s*\d|Step\s*\d|^\s*\d+[.)]\s)", re.MULTILINE)
SHOWN = re.compile(r"(Solution|solution|Lời giải|lời giải|Giải pháp|giải pháp|Cách|cách|Phương pháp|"
                   r"phương pháp|Method|method|Approach|approach)\s*(A|B|1|2)\b")


EN_WORDS = {"the", "is", "we", "and", "of", "to", "that", "this", "for", "so", "are", "with", "can", "by",
            "find", "now", "then", "therefore", "since", "be", "it", "not", "have", "which"}
VI_WORDS = {"ta", "là", "và", "của", "có", "cho", "được", "nên", "vì", "không", "này", "các", "một", "với",
            "khi", "thì", "để", "số", "bài", "giải", "vậy", "bước", "tính", "đó"}
MATH = re.compile(r"\$[^$]*\$|\\\[.*?\\\]|\\begin\{.*?\\end\{[^}]*\}", re.DOTALL)


def lang(text: str) -> str | None:
    """Language from function words, ignoring LaTeX math; None when there is too little prose."""
    words = [w.lower() for w in WORD.findall(MATH.sub(" ", text))]
    en, vi = sum(w in EN_WORDS for w in words), sum(w in VI_WORDS for w in words)
    if en + vi < 8:
        return None
    return "en" if en > 2 * vi else "vi" if vi > 2 * en else None


def literal_repeat(text: str) -> bool:
    words = [w.lower() for w in WORD.findall(text)]
    grams = Counter(tuple(words[i:i + 10]) for i in range(max(0, len(words) - 9)))
    if grams and grams.most_common(1)[0][1] >= 3:
        return True
    segs = [s.strip() for s in re.split(r"[\n.]", text) if len(s.strip()) >= 25]
    return bool(segs) and Counter(segs).most_common(1)[0][1] >= 3


def templated_repeat(text: str) -> bool:
    lines = [ln.strip() for ln in re.sub(r"\d+", "0", text).split("\n") if len(ln.strip()) >= 15]
    return bool(lines) and Counter(lines).most_common(1)[0][1] >= 5


def label(text: str, judgment: bool) -> str:
    third = max(1, len(text) // 3)
    l1, l3 = lang(text[:third]), lang(text[-third:])
    if (re.search(r"(.)\1{29,}", text) or len(FOREIGN.findall(text)) >= 0.02 * max(1, len(text))
            or (l1 and l3 and l1 != l3)):
        return "c"
    if literal_repeat(text) or templated_repeat(text):
        return "b"
    tail = text[int(0.4 * len(text)):]
    if judgment and not SHOWN.search(tail) and STEPS.search(tail):
        return "d"
    return "a"


def scan(folder: Path, kind: str) -> dict | None:
    path = folder / "answers.jsonl"
    if not (folder / "summary.json").exists() or not path.exists():
        return None
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()]
    flag = "hit_primary_cap" if kind == "attempts" else "hit_token_cap"
    capped = [r for r in rows if r.get(flag)]
    counts, wrap, eng, examples = Counter(), 0, 0, {k: [] for k in LABELS}
    for r in capped:
        text = r["output"]
        k = label(text, kind != "attempts")
        counts[k] += 1
        wrap += k == "a" and bool(CONCLUDE.search(text[-400:]))
        eng += lang(text) == "en"
        if len(examples[k]) < 3:
            examples[k].append(text[-300:].replace("\n", " / "))
    n = len(capped)
    return {"outputs": len(rows), "capped": n, "capped_share": n / len(rows) if rows else 0.0,
            "label_share_of_capped": {k: counts[k] / n if n else 0.0 for k in LABELS},
            "a_wrapping_up_share_of_capped": wrap / n if n else 0.0,
            "english_share_of_capped": eng / n if n else 0.0, "examples": examples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/phase14_remote/outputs/phase14_v1")
    args = parser.parse_args()
    report = {"schema_version": "paper_phase14_truncation_v2", "descriptive": True, "labels": LABEL_NAMES,
              "cells": {}}
    for split in ("pilot", "main"):
        for folder in sorted((args.root / split).glob("*/*/*")):
            stage = folder.name
            if stage == "samples":
                kind = "attempts"
            elif stage.startswith("judge_"):
                kind = stage[len("judge_"):]
            else:
                continue
            res = scan(folder, kind)
            if res:
                report["cells"][f"{split}/{folder.parent.parent.name}/{folder.parent.name}/{kind}"] = res
    out = ROOT / "paper/results/phase14_truncation"
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                                        newline="\n")
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    lines = ["# Phase 14 — output bị cắt ở trần token đang làm gì (4 mô hình)", "",
             "**Khám phá, mô tả, gán nhãn tự động.** Mỗi output chạm trần (lời giải s1–s5: góc nhìn 768 token; lời "
             "chấm: trần 768) nhận một nhãn, xét theo thứ tự c → b → d → a:", "",
             "- **(c) lạc đề / rác / lẫn ngôn ngữ:** chuỗi ≥ 30 ký tự giống nhau, ≥ 2% ký tự chữ khác Latin/tiếng Việt "
             "(Hán, Kirin…), hoặc đổi ngôn ngữ giữa 1/3 đầu và 1/3 cuối (Việt ↔ Anh). Viết hoàn toàn bằng tiếng Anh "
             "**không** tính là (c); đếm riêng ở cột cuối.",
             "- **(b) lặp vòng:** một chuỗi 10 từ hoặc một câu ≥ 25 ký tự lặp nguyên văn ≥ 3 lần, hoặc liệt kê theo "
             "khuôn (thay mọi số bằng 0 thì một dòng lặp ≥ 5 lần).",
             "- **(d) bỏ hai lời giải, tự giải lại lan man** (chỉ cho lời chấm): 60% cuối bài không còn nhắc tới lời "
             "giải A/B nào và đang viết lời giải riêng từng bước.",
             "- **(a) dài nhưng đúng hướng, chưa xong:** còn lại. Trong ngoặc: phần của (a) mà 400 ký tự cuối đã có "
             "câu kết luận (bị cắt đúng lúc kết luận).", "",
             "Ngưỡng cố định trước khi đọc số đếm; ngưỡng ≥ 5 cho liệt kê và nhãn (d) đặt ra sau khi kiểm tra tay 12 ca "
             "của một lần chạy trước (khoảng một nửa \"lặp\" khi đó thực ra là chép lại từng bước hai lời giải). Bản 2, "
             "sau khi kiểm tra tay ví dụ của bản 1: nhận diện ngôn ngữ bằng từ chức năng và bỏ qua công thức LaTeX "
             "(trước đó đoạn toàn công thức bị coi là tiếng Anh nên vòng lặp phương trình rơi vào (c)); nhận cả cách gọi "
             "\"Phương pháp 1/2\", \"Method 1/2\" cho hai lời giải (trước đó bị gán nhầm (d)).", "",
             "| Split | Mô hình | Dataset | Loại output | Số output | Chạm trần | (a) đúng hướng, chưa xong "
             "(đang kết luận) | (b) lặp vòng | (c) lạc đề / rác / lẫn ngôn ngữ | (d) tự giải lại lan man | "
             "Toàn tiếng Anh |",
             "|---|---|---|---|---:|---:|---|---:|---:|---:|---:|"]
    kinds = {"attempts": "lời giải s1–s5", "base": "lời chấm base", "base_c": "lời chấm base_c"}
    for key, e in report["cells"].items():
        split, model, dataset, kind = key.split("/")
        s = e["label_share_of_capped"]
        lines.append(f"| {split} | {NAMES.get(model, model)} | {DS.get(dataset, dataset)} | {kinds.get(kind, kind)} | "
                     f"{e['outputs']} | {e['capped']} ({pct(e['capped_share'])}) | {pct(s['a'])} "
                     f"({pct(e['a_wrapping_up_share_of_capped'])}) | {pct(s['b'])} | {pct(s['c'])} | "
                     f"{'—' if kind == 'attempts' else pct(s['d'])} | {pct(e['english_share_of_capped'])} |")
    lines += ["", "Tỷ lệ các nhãn tính trên số output chạm trần của dòng đó. Ví dụ từng nhãn (300 ký tự cuối) trong "
              "file .json để kiểm tra tay."]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines[16:]))


if __name__ == "__main__":
    main()
