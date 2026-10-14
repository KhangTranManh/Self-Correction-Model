"""Preliminary (descriptive) scoring of every Phase 14 cell already mirrored locally, with the same
metrics.cell() as analyze.py, before the host's --add-comparison step. Never touches the protected
report. Endpoint cells are copied from the opened report, not recomputed. Lenient (project parser);
GSM8K cells also on the audited 519. Output: paper/results/phase14_preliminary.{json,md}.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts import phase14_data as P  # noqa: E402  (points generate at the local mirror)
from phase14.scripts import generate as g  # noqa: E402
from phase14.scripts.config import GSM8K_AUDIT_EXCLUDED, REPORT_REFERENCE  # noqa: E402
from phase14.scripts.metrics import cell  # noqa: E402

OUT = ROOT / "paper/results/phase14_preliminary"
ORDER = ("llama31_8b", "mistral_7b", "original_solver", "qwen25_instruct")
DATASETS = ("asdiv", "math13", "gsm8k13", "svamp12", "bridge13", "gsm8k13_report", "svamp12_report")


def ready(model: str, dataset: str) -> bool:
    base = g.folder(model, dataset, "main")
    return all((base / f"judge_{j}" / "summary.json").exists() for j in ("base", "base_c"))


def main() -> None:
    opened = P.report()["cells"]
    excluded = set(json.loads(GSM8K_AUDIT_EXCLUDED.read_text(encoding="utf-8"))["excluded_ids"])
    out = {"schema_version": "paper_phase14_preliminary_v1", "descriptive": True, "cells": {}}
    if (OUT.with_suffix(".json")).exists():
        out = json.loads(OUT.with_suffix(".json").read_text(encoding="utf-8"))
    for model in ORDER:
        for dataset in DATASETS:
            key = f"{model}/{dataset}"
            if key in out["cells"] or not ready(model, dataset):
                continue
            if key in opened:
                out["cells"][key] = {"source": "opened report (endpoint)", "lenient": opened[key]["lenient"]}
            else:
                rows = list(P.problems(dataset))
                samples = P.samples(model, dataset)
                judges = {j: P.judgments(model, dataset, j) for j in ("base", "base_c")}
                entry = {"source": "computed locally (descriptive)", "lenient": cell(rows, samples, judges, 20261100, "lenient")}
                if dataset in ("gsm8k13", "gsm8k13_report", "bridge13"):
                    entry["lenient_audited"] = cell([r for r in rows if r["id"] not in excluded], samples, judges,
                                                    20261100, "lenient")
                out["cells"][key] = entry
            OUT.with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
            print("scored", key, flush=True)
    write_md(out)


def write_md(out: dict) -> None:
    f = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
    p = lambda x: "—" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    ci = lambda v: "—" if not v or v[0] is None else f"[{v[0]:.3f}; {v[1]:.3f}]"  # noqa: E731
    lines = ["# Phase 14 — đánh giá sơ bộ mọi ô (mô tả)", "",
             "Parser chuẩn (`lenient`), cặp đúng-một. Ô endpoint lấy nguyên từ báo cáo đã mở; ô khác tính tại local "
             "bằng cùng `metrics.cell`. Số đứng trước là ước lượng điểm, [a; b] là KTC 95% bootstrap theo bài.", "",
             "## Judge `base`", "",
             "| Mô hình | Dataset | Nguồn | Cặp | s1 đúng | Một thứ tự | Skill index [KTC] | Chọn vị trí 2 | Bịa | "
             "Tie-break [KTC] | Đúng khi nhất quán / coverage |",
             "|---|---|---|---:|---:|---:|---|---:|---:|---|---|"]
    for key, e in out["cells"].items():
        model, dataset = key.split("/")
        c = e["lenient"]
        b = c["judges"]["base"]
        lines.append(f"| {model} | {dataset} | {'endpoint' if 'opened' in e['source'] else 'mô tả'} | "
                     f"{c['one_right_pairs']} | {p(c['accuracy'].get('keep'))} | {p(b['single_accuracy'])} | "
                     f"{f(b['skill_index'])} {ci(b['skill_index_ci'])} | {p(b['position_2_rate'])} | {p(b['invented_rate'])} | "
                     f"{f(b['tie_break'])} {ci(b['tie_break_ci'])} | {p(b['consistent_accuracy'])} / {p(b['coverage'])} |")
    lines += ["", "## Judge `base_c`: log-prob ở token verdict", "",
              "| Mô hình | Dataset | Skill index `base_c` | Coverage log-prob | Bias log-odds (> 0 = A) | Một thứ tự → trung bình | "
              "Mức tăng [KTC] | Phần bỏ bias / hai lần đọc |", "|---|---|---|---:|---:|---|---|---|"]
    for key, e in out["cells"].items():
        model, dataset = key.split("/")
        bc = e["lenient"]["judges"]["base_c"]
        lp = bc.get("logprob") or {}
        lines.append(f"| {model} | {dataset} | {f(bc['skill_index'])} | {p(lp.get('logprob_coverage'))} | "
                     f"{f(lp.get('bias_logodds'))} | {p(lp.get('single_order_accuracy'))} → {p(lp.get('debiased_accuracy'))} | "
                     f"{f(lp.get('debias_gain'))} {ci(lp.get('debias_gain_ci'))} | "
                     f"{f(lp.get('bias_removal_gain'))} / {f(lp.get('two_readings_gain'))} |")
    lines += ["", "## Chiến lược (accuracy theo bài)", "",
              "| Mô hình | Dataset | Giữ s1 | Ngẫu nhiên s1/s2 | Judge hoặc ngẫu nhiên (base) | Self-check (base) | vote@3 | vote@5 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, e in out["cells"].items():
        model, dataset = key.split("/")
        a = e["lenient"]["accuracy"]
        lines.append(f"| {model} | {dataset} | {p(a.get('keep'))} | {p(a.get('random_s1_s2'))} | "
                     f"{p(a.get('judge_or_random_base'))} | {p(a.get('self_check_base'))} | {p(a.get('maj3'))} | {p(a.get('maj5'))} |")
    lines += ["", "## Kiểm tra hồi quy với số báo cáo (Phase 13/12)", "",
              "| Ô | Judge | Thước đo | Báo cáo | Tính lại | Chênh |", "|---|---|---|---:|---:|---:|"]
    for ref_key, judges in REPORT_REFERENCE.items():
        e = out["cells"].get(ref_key)
        if not e:
            continue
        for judge, ref in judges.items():
            blk = e["lenient"]["judges"][judge]
            for m, v in ref.items():
                got = e["lenient"]["one_right_pairs"] if m == "one_right_pairs" else blk.get(m)
                if got is None:
                    continue
                lines.append(f"| {ref_key} | {judge} | {m} | {v} | {got:.3f} | {got - v:+.3f} |")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT.with_suffix(".md"))


if __name__ == "__main__":
    main()
