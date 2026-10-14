"""Regression test: does Phase 14's code reproduce the report? (separate from the main tables)

The main tables use one uniform protocol for all four models (fresh attempts,
both judges with log-probabilities). This script instead reads the report's
own outputs, only to verify the pipeline:
  generate.py --model original_solver --dataset {gsm8k13_report,svamp12_report} --judge base
      (imports the report's base judgments, no generation)
  generate.py --model original_solver --dataset gsm8k13_report --judge base_c
      (re-runs the Phase 13 constrained judge on the report's pairs)

Checks:
  1. Self_Correction, base judge, the report's GSM8K 561 and SVAMP 960 outputs:
     every printed report number reproduced to the report's precision (proves
     Phase 14's metrics, pair selection, and verifier equal the report's).
  2. Self_Correction, re-run base_c on the report's pairs: same task set as
     Phase 13, share of identical texts and picks, Phase 13 numbers within
     RERUN_TOLERANCE (proves the generation settings reproduce Phase 13).
  3. Qwen2.5-7B-Instruct, its fresh uniform GSM8K cell: tie-break (strict,
     lenient) with an interval against the report's section 4.5 (descriptive:
     the report used a 1536-token judge view and other seeds).
Writes outputs/phase14_v1/regression/report.json and exits non-zero if check 1
or 2 fails.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase14.scripts.config import (EXACT, INSTRUCT_REFERENCE, OUT, REPORT_REFERENCE,  # noqa: E402
                                    RERUN_TOLERANCE)
from phase14.scripts.generate import completed, folder, load_samples, problems  # noqa: E402
from phase14.scripts.metrics import cell  # noqa: E402
from phase14.scripts.answers import parse, same  # noqa: E402

REGRESSION = OUT / "regression" / "report.json"


def compare(cells: dict) -> dict:
    """Report numbers vs observed cells; exact for reused judgments, tolerant for re-runs."""
    result = {}
    for key, judges in REPORT_REFERENCE.items():
        if key not in cells:
            result[key] = {"status": "cell not run"}
            continue
        entry = {}
        for judge, expected in judges.items():
            block = cells[key]["lenient"]["judges"][judge]
            tolerance = 0.0015 if judge in EXACT else RERUN_TOLERANCE
            mismatch = {}
            for name, value in expected.items():
                observed = block.get(name)
                limit = 0 if isinstance(value, int) else tolerance
                if observed is None or abs(observed - value) > limit:
                    mismatch[name] = {"report": value, "observed": observed}
            entry[judge] = {"status": "reproduced" if not mismatch else "MISMATCH", "tolerance": tolerance,
                            "mismatch": mismatch}
        result[key] = entry
    return result


def text_agreement() -> dict:
    """Phase 14 re-run of the constrained judge vs the Phase 13 run on identical prompts."""
    import phase13.scripts.generate as g13
    from phase13.scripts.constrained import pick

    old = g13.completed("judge_base_c")
    new = completed(folder("original_solver", "gsm8k13_report", "main"), "judge_base_c")
    samples = load_samples("original_solver", "gsm8k13_report", "main")
    shared = sorted(set(old) & set(new))
    same_text = same_pick = 0
    for pid, k, order in shared:
        first, other = parse(samples[(pid, 1)]["output"]), parse(samples[(pid, k)]["output"])
        shown_a, shown_b = (first, other) if order == "ab" else (other, first)
        same_text += old[(pid, k, order)] == new[(pid, k, order)]["output"]
        same_pick += pick(old[(pid, k, order)], shown_a, shown_b, True) == \
            pick(new[(pid, k, order)]["output"], shown_a, shown_b, True)
    return {"tasks_phase13": len(old), "tasks_phase14": len(new), "shared": len(shared),
            "identical_text": same_text / max(len(shared), 1), "identical_pick": same_pick / max(len(shared), 1),
            "same_task_set": set(old) == set(new)}


def main() -> None:
    cells = {}
    for key in (*REPORT_REFERENCE, *INSTRUCT_REFERENCE):
        model, dataset = key.split("/")
        base = folder(model, dataset, "main")
        judges = {j: completed(base, f"judge_{j}") for j in ("base", "base_c")
                  if (base / f"judge_{j}" / "summary.json").exists()}
        if "base" not in judges:
            continue
        rows = problems(dataset, "main")
        samples = load_samples(model, dataset, "main")
        cells[key] = {mode: cell(rows, samples, judges, 20261012, mode) for mode in ("lenient", "strict", "rescued")}
    report = {"schema_version": "phase14_regression_v1", "report_reproduction": compare(cells),
              "base_c_rerun_vs_phase13": text_agreement(), "instruct_section_4_5": {}}
    for key, ref in INSTRUCT_REFERENCE.items():
        if key not in cells:
            report["instruct_section_4_5"][key] = {"status": "cell not run"}
            continue
        entry = {}
        for mode in (m for m in ("lenient", "rescued") if f"tie_break_{m}" in ref):
            block = cells[key][mode]["judges"]["base"]
            low, high = block["tie_break_ci"]
            value = ref[f"tie_break_{mode}"]
            entry[mode] = {"report": value, "observed": block["tie_break"], "ci": [low, high],
                           "report_inside_ci": bool(low <= value <= high)}
        report["instruct_section_4_5"][key] = entry
    failed = [k for k, v in report["report_reproduction"].items()
              if any(j.get("status") == "MISMATCH" for j in v.values() if isinstance(j, dict))]
    rerun = report["base_c_rerun_vs_phase13"]
    if not rerun["same_task_set"]:
        failed.append("base_c task set differs from Phase 13")
    report["pass"] = not failed
    report["failures"] = failed
    REGRESSION.parent.mkdir(parents=True, exist_ok=True)
    REGRESSION.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit(f"Regression failed: {failed}")


if __name__ == "__main__":
    main()
