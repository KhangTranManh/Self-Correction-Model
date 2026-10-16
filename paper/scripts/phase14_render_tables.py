"""Render paper/results/phase14_paper_tables.json into Markdown tables and figures (no recomputation).
Run with any Python that has matplotlib:  python paper/scripts/phase14_render_tables.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "paper/results"
FIG = ROOT / "paper/figures"
data = json.loads((R / "phase14_paper_tables.json").read_text(encoding="utf-8"))["cells"]
NAMES = {"llama31_8b": "Llama-3.1-8B", "mistral_7b": "Mistral-7B", "original_solver": "Self_Correction",
         "qwen25_instruct": "Qwen2.5-7B-Instruct"}
DS = {"asdiv": "ASDiv", "math13": "MATH 1–3", "gsm8k13": "GSM8K", "svamp12": "SVAMP", "bridge13": "bridge",
      "gsm8k13_report": "GSM8K (báo cáo)", "svamp12_report": "SVAMP (báo cáo)"}
CLS = {"1": "hợp lệ", "2": "lỗi định dạng", "3": "cắt-lặp vòng", "4": "cắt-khác", "5": "tự giải lại",
       "6": "đáp số thứ ba", "7": "không có số"}
GRID = ("asdiv", "math13", "gsm8k13", "svamp12")
pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
pp = lambda x: f"{100 * x:+.1f}"  # noqa: E731
ci = lambda v: f"[{100 * v[0]:.1f}; {100 * v[1]:.1f}]"  # noqa: E731


def name(key):
    m, d = key.split("/")
    return NAMES[m], DS[d]


# ---- 1a
L = ["# Phase 14 — phân loại 7 kết cục của mỗi lần chấm một thứ tự (khám phá)", "",
     "Cặp đúng-một, cả hai thứ tự; tỷ lệ theo lần chấm (%). Quy tắc: `phase14/RUN_STATUS.md` (2026-10-15, i). "
     "Hai ô báo cáo không có cờ chạm trần nên loại 3–4 luôn bằng 0 ở đó.", ""]
for judge in ("base", "base_c"):
    L += [f"## Judge `{judge}`", "", "| Mô hình | Dataset | Lần chấm | " + " | ".join(f"{k} {v}" for k, v in CLS.items()) + " |",
          "|---|---|---:|" + "---:|" * 7]
    for key, e in data.items():
        c = e[judge]["classes"]
        n1, n2 = name(key)
        L.append(f"| {n1} | {n2} | {e[judge]['judgments']} | " + " | ".join(pct(c[k]["share"]) for k in CLS) + " |")
    L.append("")
L += ["KTC 95% từng tỷ lệ có trong `phase14_paper_tables.json` (khóa `classes.<loại>.ci`)."]
(R / "phase14_invalid_types.md").write_text("\n".join(L) + "\n", encoding="utf-8")

# ---- 1c + 2
L = ["# Phase 14 — phân rã 4 yếu tố và độ nhạy (khám phá)", "",
     "Khoảng cách R − một thứ tự (điểm %) chia Shapley cho: **bias** vị trí, **lỗi định dạng** (loại 2), **bị cắt** (3–4), "
     "**không lựa chọn dù viết xong** (5–7); trung bình qua 24 thứ tự gỡ. δ, β ước lượng từ lần chấm hợp lệ. "
     "KTC 95% bootstrap theo bài (500 lần). Số đứng trước là ước lượng điểm.", "",
     "## 1c. Phân rã (judge `base`)", "",
     "| Mô hình | Dataset | Một thứ tự | R | Khoảng cách [KTC] | Bias [KTC] | Định dạng [KTC] | Bị cắt [KTC] | Không lựa chọn [KTC] |",
     "|---|---|---:|---:|---|---|---|---|---|"]
for key, e in data.items():
    d, c = e["base"]["decomp"], e["base"]["decomp_ci"]
    n1, n2 = name(key)
    L.append(f"| {n1} | {n2} | {pct(d['single_order'])} | {pct(d['R'])} | {pct(d['gap'])} {ci(c['gap'])} | "
             f"{pct(d['phi_bias'])} {ci(c['phi_bias'])} | {pct(d['phi_format'])} {ci(c['phi_format'])} | "
             f"{pct(d['phi_cut'])} {ci(c['phi_cut'])} | {pct(d['phi_nochoice'])} {ci(c['phi_nochoice'])} |")
L += ["", "## Judge `base_c`", "", "| Mô hình | Dataset | Khoảng cách | Bias | Định dạng | Bị cắt | Không lựa chọn |",
      "|---|---|---:|---:|---:|---:|---:|"]
for key, e in data.items():
    d = e["base_c"]["decomp"]
    n1, n2 = name(key)
    L.append(f"| {n1} | {n2} | {pct(d['gap'])} | {pct(d['phi_bias'])} | {pct(d['phi_format'])} | {pct(d['phi_cut'])} | {pct(d['phi_nochoice'])} |")
L += ["", "## 2a–2c. Độ nhạy (judge `base`)", "",
      "Cận dưới phần do không hợp lệ = π·0,5; cận trên ghi trước = π·acc_AB; π·R là cận trên khi tính cả tương tác với bias. "
      "Shapley (không hợp lệ) = định dạng + bị cắt + không lựa chọn. Probit: phần do bias khi đổi link. "
      "Đẳng thức: một thứ tự − (1 − π)·acc_AB (phải bằng 0). π_A − π_B: tỷ lệ không hợp lệ khi lời giải đúng ở A trừ khi ở B.", "",
      "| Mô hình | Dataset | π | Cận dưới | Shapley | Cận trên (π·acc_AB) | π·R | Trong [dưới, π·R]? | Bias (logit) | Bias (probit) | Đẳng thức | π_A − π_B [KTC] |",
      "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|"]
viol_pre, viol_r, probit_max = [], [], 0.0
for key, e in data.items():
    d, pr, c = e["base"]["decomp"], e["base"]["decomp_probit"], e["base"]["decomp_ci"]
    inside = d["lower_invalid"] - 1e-9 <= d["invalid_part"] <= d["upper_invalid_R"] + 1e-9
    if d["invalid_part"] > d["upper_invalid"] + 1e-9:
        viol_pre.append(key)
    if not inside:
        viol_r.append(key)
    probit_max = max(probit_max, pr["phi_bias"])
    n1, n2 = name(key)
    L.append(f"| {n1} | {n2} | {pct(d['pi_invalid'])} | {pct(d['lower_invalid'])} | {pct(d['invalid_part'])} | "
             f"{pct(d['upper_invalid'])} | {pct(d['upper_invalid_R'])} | {'có' if inside else '**không**'} | "
             f"{pct(d['phi_bias'])} | {pct(pr['phi_bias'])} | {d['identity_gap']:.1e} | {pp(d['pi_A_minus_pi_B'])} {ci(c['pi_A_minus_pi_B'])} |")
over = max(e["base"]["decomp"]["invalid_part"] - e["base"]["decomp"]["upper_invalid"] for e in data.values())
L += ["", f"- Shapley vượt cận trên ghi trước (π·acc_AB) ở {len(viol_pre)}/{len(data)} ô, mức vượt lớn nhất "
      f"{100 * over:.2f} điểm (phần tương tác với bias được Shapley chia đôi); nằm trong [π·0,5; π·R] ở "
      f"{len(data) - len(viol_r)}/{len(data)} ô. Kết luận: phần do không hợp lệ ≈ cận trên ở mọi ô, tức gần như toàn bộ "
      "lượt không hợp lệ lẽ ra đã được chấm đúng.",
      f"- Phần do bias với probit lớn nhất: {pct(probit_max)} điểm (tiêu chí ghi trước: ≤ 3 điểm ở mọi ô → "
      f"{'đạt' if probit_max <= 0.03 else 'không đạt'})."]
(R / "phase14_decomp4.md").write_text("\n".join(L) + "\n", encoding="utf-8")

# ---- 3
ST = [("keep", "Giữ s1"), ("random", "Ngẫu nhiên s1/s2"), ("judge_lenient", "Judge→ngẫu nhiên"),
      ("judge_rescued", "Judge→ngẫu nhiên (rescued)"), ("selfcheck_lenient", "Self-check"),
      ("selfcheck_rescued", "Self-check (rescued)"), ("agree_gated", "Agreement-gated vote@3"),
      ("maj3", "vote@3"), ("maj5", "vote@5"), ("oracle", "Trần (s1 hoặc s2 đúng)")]
L = ["# Phase 14 — giá trị thực tế (khám phá)", "",
     "Accuracy theo bài (%), lượt gọi và token trung bình (token lời giải tính tối đa 768). Hiệu so với vote@3 / vote@5: "
     "điểm %, KTC 95% bootstrap ghép cặp theo bài.", ""]
for key, e in data.items():
    if "practical" not in e:
        continue
    p = e["practical"]
    n1, n2 = name(key)
    L += [f"## {n1} · {n2} ({p['problems']} bài; s1 ≠ s2 ở {pct(p['disagree_share'])}%; trong đó sai cả hai "
          f"{pct(p['both_wrong_share_of_disagreements'])}%)", "",
          "| Chiến lược | Accuracy | Lượt gọi | Token | − vote@3 [KTC] | − vote@5 [KTC] |", "|---|---:|---:|---:|---|---|"]
    for k, lab in ST:
        s = p["strategies"][k]
        v3 = f"{pp(s['vs_maj3'][0])} {ci(s['vs_maj3'][1])}" if "vs_maj3" in s else ""
        v5 = f"{pp(s['vs_maj5'][0])} {ci(s['vs_maj5'][1])}" if "vs_maj5" in s else ""
        L.append(f"| {lab} | {pct(s['accuracy'])} | {s['calls']:.2f} | {s['tokens']:.0f} | {v3} | {v5} |")
    L.append("")
(R / "phase14_practical.md").write_text("\n".join(L) + "\n", encoding="utf-8")

# ---- 5
L = ["# Phase 14 — số liệu sau lọc mỗi ô", "",
     "| Mô hình | Dataset | Bài | s1 ≠ s2 | Cặp được chấm | Cặp đúng-một | Cặp sai cả hai | Lời giải chạm 768 | Judge `base` chạm trần | Judge `base_c` chạm trần |",
     "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
f = lambda x: "—" if x is None else pct(x) + "%"  # noqa: E731
for key, e in data.items():
    c = e["counts"]
    n1, n2 = name(key)
    no_solution_flag = key.endswith(("bridge13", "_report"))  # Phase 13/12 solutions carry no cap flag
    L.append(f"| {n1} | {n2} | {c['problems']} | {c['s1_ne_s2']} | {c['judged_pairs']} | {c['one_right_pairs']} | "
             f"{c['both_wrong_pairs']} | {f(None if no_solution_flag else c['solution_cap_768'])} | {f(c['judge_cap_base'])} | {f(c['judge_cap_base_c'])} |")
L += ["", "Ô bridge dùng lời giải Phase 13 của Self_Correction (không có cờ chạm trần của lời giải); hai ô báo cáo không có "
      "cờ chạm trần của judge `base`. GSM8K sau audit (519 bài): xem `report.json` → `lenient_audited`."]
(R / "phase14_counts.md").write_text("\n".join(L) + "\n", encoding="utf-8")

# ---- figures
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG.mkdir(exist_ok=True)
EN = {"1": "valid choice", "2": "format error", "3": "cut (looping)", "4": "cut (other)", "5": "re-solved",
      "6": "third answer", "7": "no answer"}
SHORT = {"llama31_8b": "Llama", "mistral_7b": "Mistral", "original_solver": "SelfCorr", "qwen25_instruct": "Qwen"}
colors = ["#4c72b0", "#dd8452", "#c44e52", "#e8a0a0", "#8172b3", "#937860", "#8c8c8c"]
for judge in ("base", "base_c"):
    keys = [k for k in data if k.split("/")[1] in GRID]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    bottom = [0.0] * len(keys)
    for i, c in enumerate(CLS):
        vals = [100 * data[k][judge]["classes"][c]["share"] for k in keys]
        ax.bar(range(len(keys)), vals, bottom=bottom, color=colors[i], label=f"{c} {EN[c]}")
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{SHORT[k.split('/')[0]]}\n{DS[k.split('/')[1]]}" for k in keys], fontsize=7)
    ax.set_ylabel("% of single-order judgments")
    ax.set_ylim(0, 100)
    ax.set_title(f"Outcome of each single-order judgment ({judge} judge)")
    ax.legend(ncol=7, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"invalid_types_{judge}.{ext}", dpi=200)
    plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
LAB = {"keep": "keep s1", "random": "random s1/s2", "judge_lenient": "judge", "selfcheck_lenient": "self-check",
       "agree_gated": "agreement-gated", "maj3": "vote@3", "maj5": "vote@5"}
marks = {"keep": "o", "random": "v", "judge_lenient": "s", "selfcheck_lenient": "D", "agree_gated": "^",
         "maj3": "P", "maj5": "*"}
mcol = {"llama31_8b": "#4c72b0", "mistral_7b": "#dd8452", "original_solver": "#55a868", "qwen25_instruct": "#c44e52"}
for ax, ds in zip(axes.flat, GRID):
    for m in mcol:
        e = data.get(f"{m}/{ds}", {}).get("practical")
        if not e:
            continue
        pts = sorted((e["strategies"][s]["calls"], 100 * e["strategies"][s]["accuracy"], s) for s in marks)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=mcol[m], alpha=0.5)
        for x, y, s in pts:
            ax.plot(x, y, marks[s], color=mcol[m], ms=6)
    ax.set_title(DS[ds])
    ax.set_xlabel("mean model calls")
    ax.set_ylabel("accuracy (%)")
from matplotlib.lines import Line2D  # noqa: E402
handles = [Line2D([], [], color=c, label=NAMES[m]) for m, c in mcol.items()] + \
          [Line2D([], [], marker=mk, ls="", color="k", label=LAB[s]) for s, mk in marks.items()]
fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7)
fig.tight_layout(rect=(0, 0.06, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"practical_calls.{ext}", dpi=200)
print("rendered")

# ---- LaTeX tables for the manuscript (\input from paper/manuscript/tables/)
TEX = ROOT / "paper/manuscript/tables"
TEX.mkdir(parents=True, exist_ok=True)
EN_DS = {"asdiv": "ASDiv", "math13": "MATH 1--3", "gsm8k13": "GSM8K", "svamp12": "SVAMP", "bridge13": "Bridge",
         "gsm8k13_report": "GSM8K (report)", "svamp12_report": "SVAMP (report)"}
EN_M = {"llama31_8b": "Llama-3.1-8B", "mistral_7b": "Mistral-7B", "original_solver": r"Self\_Correction",
        "qwen25_instruct": "Qwen2.5-7B-Instr."}
t1 = lambda x: f"{100 * x:.1f}".replace("-", "$-$")  # noqa: E731
tci = lambda v: f"[{t1(v[0])}, {t1(v[1])}]"  # noqa: E731


def rows(keys, fn):
    out, last = [], None
    for k in keys:
        m, d = k.split("/")
        out.append(f"{EN_M[m] if m != last else ''} & {EN_DS[d]} & " + fn(data[k]) + r" \\")
        last = m
    return out


grid = [k for k in data if k.split("/")[1] in GRID]
T = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
     r"Model & Dataset & Valid & Single & Gap [CI] & Bias & Format & Cut & No choice \\", r"\midrule"]
T += rows(grid, lambda e: f"{t1(e['base']['classes']['1']['share'])} & {t1(e['base']['decomp']['single_order'])} & "
          f"{t1(e['base']['decomp']['gap'])} {tci(e['base']['decomp_ci']['gap'])} & {t1(e['base']['decomp']['phi_bias'])} & "
          f"{t1(e['base']['decomp']['phi_format'])} & {t1(e['base']['decomp']['phi_cut'])} & "
          f"{t1(e['base']['decomp']['phi_nochoice'])}")
T += [r"\bottomrule", r"\end{tabular}"]
(TEX / "decomp4.tex").write_text("\n".join(T) + "\n", encoding="utf-8")

T = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
     r"Model & Dataset & Keep $s_1$ & Judge & Self-check & vote@3 & vote@5 & Ceiling & Judge $-$ vote@3 [CI] \\",
     r"\midrule"]
T += rows(grid, lambda e: " & ".join(t1(e["practical"]["strategies"][s]["accuracy"]) for s in
                                    ("keep", "judge_lenient", "selfcheck_lenient", "maj3", "maj5", "oracle")) +
          f" & {t1(e['practical']['strategies']['judge_lenient']['vs_maj3'][0])} "
          f"{tci(e['practical']['strategies']['judge_lenient']['vs_maj3'][1])}")
T += [r"\bottomrule", r"\end{tabular}"]
(TEX / "practical.tex").write_text("\n".join(T) + "\n", encoding="utf-8")

ft = lambda x: "---" if x is None else t1(x)  # noqa: E731
T = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
     r"Model & Dataset & Problems & $s_1\neq s_2$ & Judged & One-right & Both wrong & Sol.\ cap & Judge cap & "
     r"Judge\textsubscript{c} cap \\", r"\midrule"]
last = None
for k, e in data.items():
    m, d = k.split("/")
    c = e["counts"]
    sol = None if k.endswith(("bridge13", "_report")) else c["solution_cap_768"]  # Phase 13/12 solutions: no cap flag
    T.append(f"{EN_M[m] if m != last else ''} & {EN_DS[d]} & {c['problems']} & {c['s1_ne_s2']} & {c['judged_pairs']} & "
             f"{c['one_right_pairs']} & {c['both_wrong_pairs']} & {ft(sol)} & {ft(c['judge_cap_base'])} & "
             f"{ft(c['judge_cap_base_c'])}" + r" \\")
    last = m
T += [r"\bottomrule", r"\end{tabular}"]
(TEX / "counts.tex").write_text("\n".join(T) + "\n", encoding="utf-8")
print("tex tables written")

