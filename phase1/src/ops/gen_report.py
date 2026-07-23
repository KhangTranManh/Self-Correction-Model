"""Sinh bao cao HTML tu ket qua eval THAT (outputs/eval_v3.jsonl), so voi moc v2
da biet. Stdlib-only (giong notify.py) de chay duoc bang python3 he thong, khong
can venv -- goi tu shell dieu phoi ngay sau khi evaluate_vllm xong.

Dung: python3 src/ops/gen_report.py outputs/eval_v3.jsonl outputs/report_v3.html
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

# Moc v2 (adapter tool/tool, 631 mau) -- CO CHU DICH hardcode, khong doc tu file
# song song vi log goc cua v2 khong chac con tren may nay. Nguon: README.md Key
# Finding 4 + phan ra theo domain da tinh 2026-07-21 (n=249, 600 toan+150 code).
V2_BASELINE = {
    "math": {"n_wrong": 104, "fmt_done": 54, "fix": 33},
    "code": {"n_wrong": 145, "fmt_done": 91, "fix": 61},
}


def decompose(rows: list[dict]) -> dict:
    stats = {d: defaultdict(int) for d in ("math", "code", "total")}
    for r in rows:
        for key in (r["domain"], "total"):
            s = stats[key]
            s["n"] += 1
            if r.get("initial_passed"):
                s["init_ok"] += 1
            else:
                s["wrong"] += 1
                if r.get("format_incomplete"):
                    s["fmt_bad"] += 1
                else:
                    s["fmt_done"] += 1
                    if r.get("second_passed"):
                        s["fix"] += 1
    return stats


def pct(n: int, d: int) -> float:
    return (n / d * 100) if d else 0.0


def row_html(label: str, s: dict, baseline: dict | None) -> str:
    wrong, fmt_done, fix = s["wrong"], s["fmt_done"], s["fix"]
    fmt_rate = pct(fmt_done, wrong)
    fix_rate = pct(fix, fmt_done)
    head_rate = pct(fix, wrong)

    delta_html = ""
    if baseline:
        b_fmt = pct(baseline["fmt_done"], baseline["n_wrong"])
        b_fix = pct(baseline["fix"], baseline["fmt_done"])
        b_head = pct(baseline["fix"], baseline["n_wrong"])
        d_fmt, d_fix, d_head = fmt_rate - b_fmt, fix_rate - b_fix, head_rate - b_head
        delta_html = (
            f"<td class='d {'pos' if d_fmt >= 0 else 'neg'}'>{d_fmt:+.1f}</td>"
            f"<td class='d {'pos' if d_fix >= 0 else 'neg'}'>{d_fix:+.1f}</td>"
            f"<td class='d {'pos' if d_head >= 0 else 'neg'}'>{d_head:+.1f}</td>"
        )
    else:
        delta_html = "<td class='d'>-</td><td class='d'>-</td><td class='d'>-</td>"

    return (
        f"<tr><td class='lbl'>{label}</td>"
        f"<td>{wrong}</td>"
        f"<td>{fmt_rate:.1f}%</td><td>{fix_rate:.1f}%</td><td class='head'>{head_rate:.1f}%</td>"
        f"{delta_html}</tr>"
    )


def build_html(stats: dict) -> str:
    rows = "\n".join([
        row_html("Toán", stats["math"], V2_BASELINE["math"]),
        row_html("Code", stats["code"], V2_BASELINE["code"]),
        row_html("Tổng", stats["total"], {
            "n_wrong": V2_BASELINE["math"]["n_wrong"] + V2_BASELINE["code"]["n_wrong"],
            "fmt_done": V2_BASELINE["math"]["fmt_done"] + V2_BASELINE["code"]["fmt_done"],
            "fix": V2_BASELINE["math"]["fix"] + V2_BASELINE["code"]["fix"],
        }),
    ])
    total = stats["total"]
    return f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<title>Kết quả AGI_v3</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 32px auto; padding: 0 16px;
         color: #1a1a1a; background: #fff; }}
  h1 {{ font-size: 20px; }}
  .sub {{ color: #666; font-size: 13px; margin-bottom: 24px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ padding: 8px 10px; text-align: right; border-bottom: 1px solid #e5e5e5; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ font-size: 11px; text-transform: uppercase; color: #888; font-weight: 600; }}
  td.lbl {{ font-weight: 600; }}
  td.head {{ font-weight: 700; }}
  td.d.pos {{ color: #157f3b; }}
  td.d.neg {{ color: #b42318; }}
  .note {{ margin-top: 20px; font-size: 13px; color: #444; line-height: 1.5; }}
  .headline {{ font-size: 28px; font-weight: 700; margin: 16px 0 4px; }}
  code {{ background: #f2f2f2; padding: 1px 5px; border-radius: 4px; font-size: 12px; }}
</style></head>
<body>
  <h1>Kết quả AGI_v3 — thêm 61 mẫu MATH (level 2–3)</h1>
  <div class="sub">Held-out: 600 toán + 150 code (offset 150) · so với mốc AGI_v2 (631 mẫu, chỉ GSM8K+MBPP)</div>

  <div class="headline">{pct(total['fix'], total['wrong']):.1f}%</div>
  <div class="sub">tỉ lệ tự sửa tổng (v2: {pct(V2_BASELINE['math']['fix']+V2_BASELINE['code']['fix'], V2_BASELINE['math']['n_wrong']+V2_BASELINE['code']['n_wrong']):.1f}%)</div>

  <table>
    <thead><tr><th>Domain</th><th>n sai</th><th>ra format</th><th>sửa|format</th><th>headline</th>
      <th colspan="3">Δ so với v2</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>

  <div class="note">
    <b>Cách đọc:</b> <code>sửa|format</code> là kỹ năng tự sửa thuần (đã loại phần
    không viết xong định dạng). Δ dương (xanh) = tốt hơn v2. Đặc biệt chú ý dòng
    <b>Code</b> — nó KHÔNG có mẫu train mới nào (chỉ toán được thêm), nên Δ ở đây
    đo đúng ảnh hưởng chéo domain của việc train chung 1 adapter.<br><br>
    Lưu ý nhiễu: hai adapter train trên cùng dữ liệu từng lệch nhau ~3.5pp ở
    init_ok chỉ do random khi train — Δ nhỏ hơn mức đó chưa chắc là hiệu ứng thật.
  </div>
</body></html>"""


def main() -> None:
    if len(sys.argv) < 3:
        print("Dung: gen_report.py <eval.jsonl> <out.html>")
        sys.exit(1)
    in_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [json.loads(l) for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    stats = decompose(rows)
    out_path.write_text(build_html(stats), encoding="utf-8")
    print(f"Da ghi bao cao: {out_path} ({len(rows)} dong nguon)")
    for d in ("math", "code", "total"):
        s = stats[d]
        print(f"  {d}: wrong={s['wrong']} fmt_done={s['fmt_done']} fix={s['fix']} "
              f"headline={pct(s['fix'], s['wrong']):.1f}%")


if __name__ == "__main__":
    main()
