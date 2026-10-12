"""Open the Phase 14 main holdout once and score every model x dataset cell.

Endpoints (thresholds frozen in docs/PREREGISTRATION.md; repeated here):
  E1 skill      base judge (Phase 9 prompt), lenient scoring: skill index
                with a source-clustered 95% CI above 0.
  E2 debias     base_c judge (Phase 13 prompt): log-prob debias gain
                >= +3 points with a source-clustered 95% CI above 0.
Generality rule: a claim is called general only if it passes in a majority of
cells (at least 3 of 4); otherwise it is reported as conditional on the cells
where it passed. Endpoint cells are the selected models x {asdiv, math13}.
Uniform cells: all four models (Llama-3.1-8B, Mistral-7B, Self_Correction,
Qwen2.5-7B-Instruct) x {asdiv, math13, gsm8k13, svamp12}, every one generated
and judged with the identical protocol; only the two new models on asdiv/math13
count toward the endpoints. Later cells (added by --add-comparison, never in
the endpoint summary): the regression-only sets that read the report's own
outputs, and the bridge set (other models re-judging the Phase 13 pairs). With
data/gsm8k13_audit_excluded.json, GSM8K cells are also scored on the audited
subset (519 in the report).
Reported, not endpoints: strict scoring; 1536-token view; tie-break minus
single-order accuracy (with its invented-answer identity); position bias
direction and size; coverage; invented rate; accuracy of keep / vote / self-check;
and the across-cell relation between |bias| and the bias-removal part of the
debias gain (the debias gain = bias removal + two-readings averaging; see
metrics.logprob_block).

Two runs:
  (default)          the single protected opening: every endpoint cell (all
                     required) plus every comparison cell already generated.
  --add-comparison   later, after the cells that need the report's outputs
                     exist (regression-only sets, the bridge set):
                     adds only cells missing from report.json, refreshes the
                     report reproduction check, and logs the addition. The
                     endpoint cells, E1, E2, and their summary are never
                     recomputed; GSM8K/SVAMP labels were opened before Phase 14.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase14.scripts.config import (ANCHOR, BRIDGE, COMPARISON, DATASETS, GSM8K_AUDIT_EXCLUDED,  # noqa: E402
                                    OUT, REGRESSION_SETS, RUN_MODELS, SELECTED)
from phase14.scripts.generate import REUSED_MODEL, JUDGES, completed, folder, load_samples, problems  # noqa: E402
from phase14.scripts.metrics import cell  # noqa: E402
from phase14.scripts.regression import compare  # noqa: E402

REPORT = OUT / "main" / "analysis"
E2_MIN_GAIN = 0.03
MAJORITY = 3
ALL_DATASETS = (*DATASETS, *COMPARISON, BRIDGE, *REGRESSION_SETS)
LATER = (BRIDGE, *REGRESSION_SETS)  # cells that need the report's outputs; added by --add-comparison


def models_for(dataset: str) -> list[str]:
    if dataset in REGRESSION_SETS:
        return [REUSED_MODEL]
    return [m for m in RUN_MODELS if not (dataset == BRIDGE and m == REUSED_MODEL)]


def ready(model: str, dataset: str) -> bool:
    return all((folder(model, dataset, "main") / f"judge_{j}" / "summary.json").exists() for j in JUDGES)


def audit_excluded() -> set | None:
    if not GSM8K_AUDIT_EXCLUDED.exists():
        return None
    return set(json.loads(GSM8K_AUDIT_EXCLUDED.read_text(encoding="utf-8"))["excluded_ids"])


def score(model: str, dataset: str, excluded: set | None) -> dict:
    """Score one cell; the seed depends only on the cell, not on which cells are ready."""
    rows = problems(dataset, "main")  # main holdout gold is first read in cell()
    samples = load_samples(model, dataset, "main")
    judges = {j: completed(folder(model, dataset, "main"), f"judge_{j}") for j in JUDGES}
    seed = 20261100 + 10 * ALL_DATASETS.index(dataset) + RUN_MODELS.index(model)
    result = {mode: cell(rows, samples, judges, seed, mode) for mode in ("lenient", "strict")}
    if excluded is not None and dataset in ("gsm8k13", "gsm8k13_report", BRIDGE):  # report: GSM8K after audit (519)
        result["lenient_audited"] = cell([r for r in rows if r["id"] not in excluded], samples, judges, seed, "lenient")
    return result


def opening() -> None:
    if (REPORT / "report.json").exists():
        raise RuntimeError("Refusing a second protected opening")
    excluded = audit_excluded()
    cells, e1, e2, dose = {}, {}, {}, []
    for dataset in (*DATASETS, *COMPARISON):  # uniform cells: every model, every dataset, required
        for model in RUN_MODELS:
            if not ready(model, dataset):
                raise RuntimeError(f"Uniform cell missing: {model}/{dataset}")
    for dataset in ALL_DATASETS:
        for model in models_for(dataset):
            if not ready(model, dataset):
                continue  # comparison cell deferred to --add-comparison
            key = f"{model}/{dataset}"
            cells[key] = score(model, dataset, excluded)
            if model not in SELECTED or dataset not in DATASETS:
                continue  # anchor and comparison cells are not endpoint cells
            base_block = cells[key]["lenient"]["judges"]["base"]
            lp = cells[key]["lenient"]["judges"]["base_c"]["logprob"]
            e1[key] = {"skill_index": base_block["skill_index"], "ci": base_block["skill_index_ci"],
                       "pass": bool(base_block["skill_index_ci"][0] is not None and base_block["skill_index_ci"][0] > 0)}
            e2[key] = ({"debias_gain": lp["debias_gain"], "ci": lp["debias_gain_ci"],
                        "bias_removal_gain": lp["bias_removal_gain"], "two_readings_gain": lp["two_readings_gain"],
                        "logprob_coverage": lp["logprob_coverage"], "bias_logodds": lp["bias_logodds"],
                        "pass": bool(lp["debias_gain"] >= E2_MIN_GAIN and lp["debias_gain_ci"][0] is not None
                                     and lp["debias_gain_ci"][0] > 0)}
                       if lp else {"pass": False, "note": "no pairs with verdict log-probabilities"})
            if lp:
                dose.append((abs(lp["bias_logodds"]), lp["bias_removal_gain"], key))
    passed = lambda table: [k for k, v in table.items() if v["pass"]]  # noqa: E731
    summary = {}
    for name, table in (("E1_skill", e1), ("E2_debias", e2)):
        hits = passed(table)
        summary[name] = {"cells_passed": hits, "count": len(hits), "of": len(table),
                         "verdict": "general" if len(hits) >= MAJORITY else
                                    "conditional" if hits else "not supported"}
    rho = spearmanr([x[0] for x in dose], [x[1] for x in dose]) if len(dose) >= 3 else None
    report = {"schema_version": "phase14_analysis_v1", "models": list(SELECTED), "anchor": list(ANCHOR),
              "datasets": list(DATASETS), "opened_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "gsm8k_audit_excluded": None if excluded is None else len(excluded),
              "deferred_cells": [f"{m}/{d}" for d in ALL_DATASETS for m in models_for(d) if f"{m}/{d}" not in cells],
              "cells": cells, "E1": e1, "E2": e2, "summary": summary,
              "report_reproduction": compare(cells), "comparison_additions": [],
              "bias_vs_gain": {"points": [{"cell": k, "abs_bias_logodds": b, "bias_removal_gain": g}
                                          for b, g, k in dose],
                               "spearman": None if rho is None else {"rho": float(rho.correlation),
                                                                     "p": float(rho.pvalue)},
                               "note": "descriptive; four cells cannot establish a trend"}}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"deferred_cells": report["deferred_cells"], "report_reproduction": report["report_reproduction"],
                      "summary": summary, "E1": e1, "E2": e2, "bias_vs_gain": report["bias_vs_gain"]}, indent=2))


def add_comparison() -> None:
    path = REPORT / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    excluded = audit_excluded()
    added = []
    for dataset in LATER:  # never the uniform cells
        for model in models_for(dataset):
            key = f"{model}/{dataset}"
            if key in report["cells"] or not ready(model, dataset):
                continue
            report["cells"][key] = score(model, dataset, excluded)
            added.append(key)
    report["deferred_cells"] = [k for k in report["deferred_cells"] if k not in report["cells"]]
    report["report_reproduction"] = compare(report["cells"])
    report["comparison_additions"].append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                           "cells": added})
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"added": added, "still_deferred": report["deferred_cells"],
                      "report_reproduction": report["report_reproduction"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--add-comparison", action="store_true")
    args = parser.parse_args()
    add_comparison() if args.add_comparison else opening()


if __name__ == "__main__":
    main()
