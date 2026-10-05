"""Freeze the Phase 11 holdout: every remaining never-used eligible GSM8K source.

Same builder as Phases 9-10 (Phase 8 reproduction check plus over-exclusion of
all Phase 1-8 JSON/JSONL), additionally excluding the Phase 9 pool and the
Phase 10 training pool and holdout. Phase 11 training data comes only from
Phase 10 training-pool outputs, so no new training sources are drawn.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase5.scripts.prepare_candidates import previous_sources  # noqa: E402
from phase9.scripts import prepare_sources as p9  # noqa: E402

PRIOR_POOLS = (ROOT / "phase9/data/source_pool_v1/development.jsonl",
               ROOT / "phase9/data/source_pool_v1/protected.jsonl",
               ROOT / "phase10/data/sources_v1/train_pool.jsonl",
               ROOT / "phase10/data/sources_v1/holdout.jsonl",
               ROOT / "phase4/data/expansion_v1/train_problems.jsonl")
OUT = ROOT / "phase11/data/sources_v1"
SEED = 20261002


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite: {OUT}")
    p8 = json.loads(p9.P8_REPORT.read_text(encoding="utf-8"))
    if p9.lf_sha256(p9.RAW) != p8["raw_dataset_sha256"]:
        raise RuntimeError("Raw GSM8K train file changed")
    listed = [Path(path.replace("\\", "/").replace("D:/AGI/", str(ROOT).replace("\\", "/") + "/"))
              for path in json.loads(p9.P5_REPORT.read_text(encoding="utf-8"))["exclusion_files_sha256"]]
    extra = [path for rel in p9.OVER_EXCLUDE_DIRS for path in p9.json_files(rel)]
    roots = [*[p for p in listed if p.exists()], *p9.P8_EXTRA, p9.P8_POOL, *PRIOR_POOLS, *extra]
    questions, indices, inventory = previous_sources(roots)
    rank = lambda i: hashlib.sha256(f"phase11_holdout_v1|{SEED}|{i}".encode()).hexdigest()
    holdout, reasons = p9.eligible(questions, indices, "phase11_gsm8k_train", rank)
    if len(holdout) < 200:
        raise RuntimeError(f"Only {len(holdout)} eligible holdout sources")
    OUT.mkdir(parents=True)
    path = OUT / "holdout.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in holdout),
                    encoding="utf-8", newline="\n")
    report = {"schema_version": "phase11_sources_v1",
              "status": "frozen_before_any_phase11_generation",
              "holdout": {"rows": len(holdout), "rule": "all remaining eligible never-used sources",
                          "exclusion_files_scanned": len(inventory),
                          "rejected_by_first_reason": dict(sorted(reasons.items())),
                          "order": "lexicographic SHA256(phase11_holdout_v1|seed|raw_index)",
                          "seed": SEED, "sha256": p9.lf_sha256(path)},
              "protected_labels_opened": False}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
