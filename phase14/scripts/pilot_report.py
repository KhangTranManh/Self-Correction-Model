"""Score the Phase 14 pilot and propose the main holdout size per dataset.

The pilot problems are burned: they never enter the main holdout, and their
labels may be read here. The report only sizes the main run; no endpoint is
tested on the pilot and no prompt, model, or threshold may be changed in
response to pilot outcomes (technical failures excepted, recorded as
deviations).

Proposed main size per dataset (shared by all models, including the anchor, so
every model is judged on the same problems; sized by the selected models):
    N = ceil(1.15 x TARGET_ONE_RIGHT_PAIRS / min over models of
             one-right pairs per problem), capped at the main pool size.
Writing data/main_size.json is a separate, deliberate step (--write), recorded
as an amendment before any main generation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase14.scripts.config import DATASETS, OUT, RUN_MODELS, SELECTED, SOURCES, TARGET_ONE_RIGHT_PAIRS  # noqa: E402
from phase14.scripts.generate import JUDGES, MAIN_SIZE, completed, folder, problems  # noqa: E402
from phase14.scripts.metrics import cell  # noqa: E402

SAFETY = 1.15


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write data/main_size.json")
    args = parser.parse_args()
    pools = json.loads((SOURCES / "report.json").read_text(encoding="utf-8"))["datasets"]
    report = {"schema_version": "phase14_pilot_report_v1", "models": list(RUN_MODELS), "cells": {},
              "proposed_main_size": {}}
    for dataset in DATASETS:
        rates = []
        for model in RUN_MODELS:
            base = folder(model, dataset, "pilot")
            samples = completed(base, "samples")
            judges = {j: completed(base, f"judge_{j}") for j in JUDGES if (base / f"judge_{j}").exists()}
            rows = problems(dataset, "pilot")
            cells = {mode: cell(rows, samples, judges, 20261012, mode) for mode in ("lenient", "strict")}
            report["cells"][f"{model}/{dataset}"] = cells
            if model in SELECTED:  # the anchor does not size the main run
                rates.append(cells["lenient"]["one_right_pairs_per_problem"])
        rate = min(rates)
        pool = pools[dataset]["main_pool"]["rows"]
        needed = math.ceil(SAFETY * TARGET_ONE_RIGHT_PAIRS / rate) if rate > 0 else None
        report["proposed_main_size"][dataset] = {
            "min_one_right_pairs_per_problem": rate, "needed": needed, "main_pool_rows": pool,
            "main_size": min(needed, pool) if needed else pool,
            "target_reachable": bool(needed and needed <= pool)}
    out = OUT / "pilot" / "pilot_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["proposed_main_size"], indent=2))
    if args.write:
        if MAIN_SIZE.exists():
            raise RuntimeError("Main size already fixed")
        MAIN_SIZE.write_text(json.dumps({
            "schema_version": "phase14_main_size_v1", "status": "fixed_from_pilot_before_main_generation",
            "pilot_report": (str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out)).replace("\\", "/"),
            "main_size": {d: v["main_size"] for d, v in report["proposed_main_size"].items()}},
            indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"Wrote {MAIN_SIZE}")


if __name__ == "__main__":
    main()
