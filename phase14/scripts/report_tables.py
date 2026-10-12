"""Print Phase 14 results in the layout of the report template's tables (Markdown, Vietnamese).

Reads outputs/phase14_v1/main/analysis/report.json and, if present,
probe_report.json. Fills the report's Table 1 (picks), skill table, Table 2,
Table 3 (single order vs order-averaged), section 4.3 selectors, the section 4.6
generalization grid, and the section 5 strategy table. Numbers use the lenient
primary view; GSM8K cells use the audited subset when it exists.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase14.scripts.config import MODELS, OUT  # noqa: E402

ANALYSIS = OUT / "main" / "analysis"
NAMES = {"original_solver": "Qwen2.5-7B (Self_Correction)", "qwen25_instruct": "Qwen2.5-7B-Instruct",
         **{k: v["repo"].split("/")[-1] for k, v in MODELS.items() if k not in ("original_solver", "qwen25_instruct")}}
COLUMNS = {"gsm8k13": "GSM8K", "svamp12": "SVAMP", "asdiv": "ASDiv", "math13": "MATH 1–3",
           "bridge13": "GSM8K (cặp Phase 13)", "gsm8k13_report": "GSM8K (output báo cáo, regression)",
           "svamp12_report": "SVAMP (output báo cáo, regression)"}


def pct(x, digits=1):
    return "—" if x is None else f"{100 * x:.{digits}f}%".replace(".", ",")


def num(x, digits=3):
    return "—" if x is None else f"{x:+.{digits}f}".replace(".", ",") if x < 0 else f"{x:.{digits}f}".replace(".", ",")


def ci(pair, digits=3):
    return "" if not pair or pair[0] is None else f" [{num(pair[0], digits)}; {num(pair[1], digits)}]"


def cell_view(cells: dict, key: str) -> dict | None:
    entry = cells.get(key)
    if entry is None:
        return None
    return entry.get("lenient_audited") or entry["lenient"]


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    report = json.loads((ANALYSIS / "report.json").read_text(encoding="utf-8"))
    probe_path = ANALYSIS / "probe_report.json"
    probes = json.loads(probe_path.read_text(encoding="utf-8"))["models"] if probe_path.exists() else {}
    cells = report["cells"]
    keys = sorted(cells, key=lambda k: (list(NAMES).index(k.split("/")[0]) if k.split("/")[0] in NAMES else 99,
                                        list(COLUMNS).index(k.split("/")[1])))
    out = ["# Phase 14 — bảng theo bố cục báo cáo", ""]

    out += ["## Bảng 1. Lựa chọn của judge chưa train (prompt Phase 9), theo vị trí lời giải đúng", ""]
    for key in keys:
        c = cell_view(cells, key)
        b = c["judges"]["base"]
        if not b.get("one_right_pairs"):
            continue
        t = b["picks_by_right_position"]
        rows = []
        for pos in ("A", "B"):
            n = sum(t[pos].values())
            rows.append([pos, *[pct(t[pos][x] / n) for x in ("A", "B", "invented")]])
        model, dataset = key.split("/")
        out += [f"**{NAMES.get(model, model)} · {COLUMNS[dataset]}** ({b['one_right_pairs']} cặp đúng-một × 2 thứ tự). "
                f"Accuracy một thứ tự {pct(b['single_accuracy'])}, skill index {num(b['skill_index'])}{ci(b['skill_index_ci'])}.",
                "", table(["Lời giải đúng ở vị trí", "Chọn A", "Chọn B", "Bịa đáp số"], rows), ""]

    out += ["## Skill index, thiên lệch vị trí, đáp số bịa", ""]
    rows = []
    for key in keys:
        c = cell_view(cells, key)
        model, dataset = key.split("/")
        for j, label in (("base", "base"), ("base_c", "base_c (ép Verdict)")):
            b = c["judges"][j]
            if b.get("one_right_pairs"):
                rows.append([NAMES.get(model, model), label, COLUMNS[dataset], f"{num(b['skill_index'])}{ci(b['skill_index_ci'])}",
                             pct(b["position_2_rate"]), pct(b["invented_rate"])])
    out += [table(["Mô hình", "Judge", "Dataset", "Skill index [KTC 95%]", "Tỷ lệ chọn vị trí 2", "Bịa"], rows), ""]

    out += ["## Bảng 2. Tie-break, đúng khi nhất quán, coverage, đúng / toàn bộ cặp", ""]
    rows = []
    for key in keys:
        c = cell_view(cells, key)
        model, dataset = key.split("/")
        for j in ("base", "base_c"):
            b = c["judges"][j]
            if b.get("one_right_pairs"):
                rows.append([NAMES.get(model, model), j, COLUMNS[dataset], f"{num(b['tie_break'])}{ci(b['tie_break_ci'])}",
                             pct(b["consistent_accuracy"]), pct(b["coverage"]), pct(b["all_pairs_accuracy"])])
    out += [table(["Mô hình", "Judge", "Dataset", "Tie-break [KTC]", "Đúng khi nhất quán", "Coverage", "Đúng / toàn bộ cặp"], rows), ""]

    out += ["## Bảng 3. Một thứ tự so với trung bình qua hai thứ tự (cùng thang với Ren et al.)", "",
            "Ba cách khử bias đặt cạnh nhau. *Tie-break − một thứ tự* bằng đúng phần đáp số bịa "
            "(cột cuối, kiểm tra trong code), nên không cùng thang với Ren et al. *Log-prob verdict* "
            "(sau khi judge viết phần kiểm tra) và *Verdict ép ngay* (không viết kiểm tra) lấy trung bình "
            "xác suất qua hai thứ tự như Ren et al., ở coverage 100%.", ""]
    rows = []
    for key in keys:
        c = cell_view(cells, key)
        model, dataset = key.split("/")
        b, bc = c["judges"]["base"], c["judges"]["base_c"]
        if not b.get("one_right_pairs"):
            continue
        lp = bc.get("logprob") or {}
        fv = (probes.get(model, {}).get("forced_verdict") or {}).get(dataset) or {}
        rows.append([NAMES.get(model, model), COLUMNS[dataset],
                     f"{pct(b['single_accuracy'])} → {num(b['tie_break'])} ({num(b['tie_minus_single'])}; bịa {num(b['invented_artifact'])})",
                     f"{pct(lp.get('single_order_accuracy'))} → {pct(lp.get('debiased_accuracy'))}" if lp else "—",
                     f"{num(lp.get('bias_removal_gain'))} / {num(lp.get('two_readings_gain'))}" if lp else "—",
                     f"{pct(fv.get('single_order_accuracy'))} → {pct(fv.get('order_averaged_accuracy'))}" if fv else "—"])
    out += [table(["Mô hình", "Dataset", "Tie-break: một thứ tự → tie-break (chênh; phần bịa)",
                   "Log-prob verdict: một thứ tự → trung bình", "Khử bias / hai lần đọc",
                   "Verdict ép ngay: một thứ tự → trung bình"], rows), ""]

    if probes:
        out += ["## Mục 4.3. Bên trong mô hình (cặp đúng-một)", ""]
        for model, entry in probes.items():
            for train_key, sel in entry.items():
                if not train_key.startswith("selectors_train_"):
                    continue
                train = train_key.removeprefix("selectors_train_")
                for test, r in sel["tests"].items():
                    rows = [["Đặc trưng bề mặt (độ dài, số bước, đáp số)", pct(r["surface"]["accuracy"]), num(r["surface"]["auc"])],
                            ["Log-prob cao hơn", pct(r["likelihood"]["accuracy"]), "—"],
                            [f"Probe hidden state (layer {sel['judge_site_layer']})", pct(r["probe"]["accuracy"]), num(r["probe"]["auc"])],
                            ["Probe + bề mặt", pct(r["probe_plus_surface"]["accuracy"]), num(r["probe_plus_surface"]["auc"])],
                            [f"Probe theo từng lời giải (layer {sel['pointwise_layer']})", pct(r["pointwise_probe"]["accuracy"]),
                             num(r["pointwise_probe"]["auc"])]]
                    out += [f"**{NAMES.get(model, model)}**, train {COLUMNS[train]} → test {COLUMNS[test]} "
                            f"({r['pairs']} cặp; probe cặp bị thứ tự lật {pct(r['pair_probe_order_flip_rate'])})", "",
                            table(["Bộ chọn", "Accuracy", "AUC"], rows), ""]

    out += ["## Mục 4.6. Tổng quát hóa (mỗi ô: skill index [KTC]; khử bias log-prob; bias log-odds; coverage)", ""]
    models = list(dict.fromkeys(k.split("/")[0] for k in keys))
    rows = []
    for model in models:
        row = [NAMES.get(model, model)]
        for dataset in ("gsm8k13", "svamp12", "asdiv", "math13"):
            c = cell_view(cells, f"{model}/{dataset}")
            if c is None or not c["judges"]["base"].get("one_right_pairs"):
                row.append("—")
                continue
            b, lp = c["judges"]["base"], c["judges"]["base_c"].get("logprob") or {}
            row.append(f"{num(b['skill_index'], 2)}{ci(b['skill_index_ci'], 2)}; "
                       f"{num(lp.get('bias_removal_gain'), 3) if lp else '—'}; "
                       f"{num(lp.get('bias_logodds'), 2) if lp else '—'}; {pct(b['coverage'], 0)}")
        rows.append(row)
    out += [table(["Mô hình", "GSM8K", "SVAMP", "ASDiv", "MATH 1–3"], rows), ""]

    out += ["## Mục 5. Giá trị thực tế (accuracy theo chiến lược)", ""]
    rows = []
    for key in keys:
        c = cell_view(cells, key)
        model, dataset = key.split("/")
        a = c["accuracy"]
        rows.append([NAMES.get(model, model), COLUMNS[dataset], pct(a["keep"]), pct(a["random_s1_s2"]),
                     pct(a["judge_or_random_base"]), pct(a["self_check_base"]), pct(a["maj3"]), pct(a["maj5"])])
    out += [table(["Mô hình", "Dataset", "Giữ s1", "Ngẫu nhiên s1/s2", "Judge, không nhất quán thì ngẫu nhiên",
                   "Self-check, không nhất quán thì vote@3", "vote@3", "vote@5"], rows), ""]
    text = "\n".join(out)
    (ANALYSIS / "report_tables.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    sys.stdout.reconfigure(encoding="utf-8")  # Vietnamese text on any console
    print(text)


if __name__ == "__main__":
    main()
