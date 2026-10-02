"""Freeze Phase 10 training sources and a fresh protected holdout.

Training pool: the Phase 4 expansion GSM8K train sources, which were only ever
used as Phase 4 training material, minus every Phase 4 development split.
The original solver was never trained on them.

Holdout: fresh GSM8K train sources never used by any phase, built with the
Phase 9 builder logic (Phase 8 reproduction check plus over-exclusion) with
the Phase 9 pool and the Phase 10 training pool also excluded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase5.scripts.prepare_candidates import previous_sources, question_key  # noqa: E402
from phase9.scripts import prepare_sources as p9  # noqa: E402

EXPANSION = ROOT / "phase4/data/expansion_v1/train_problems.jsonl"
PHASE4_DEV = (ROOT / "phase4/data/preferences_v1/dev.jsonl",
              ROOT / "phase4/data/verified_corrections_v1/warmstart_dev.jsonl",
              ROOT / "phase4/data/blind_preferences_v2/dev.jsonl")
PHASE9_POOL = (ROOT / "phase9/data/source_pool_v1/development.jsonl",
               ROOT / "phase9/data/source_pool_v1/protected.jsonl")
OUT = ROOT / "phase10/data/sources_v1"
SEED = 20261001
HOLDOUT = 300


def ids_in(path: Path) -> set[str]:
    found = set()
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            row = json.loads(line)
            found |= {row[k] for k in ("problem_id", "id", "source_id") if isinstance(row.get(k), str)}
    return found


def write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8", newline="\n")
    return p9.lf_sha256(path)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite: {OUT}")
    # Training pool.
    dev_ids = set().union(*(ids_in(path) for path in PHASE4_DEV))
    expansion = [json.loads(line) for line in EXPANSION.read_text(encoding="utf-8").split("\n")
                 if line.strip()]
    train = [dict(row, question_sha256=question_key(row["question"])) for row in expansion
             if row["id"] not in dev_ids]

    # Holdout: Phase 9 builder with everything used so far excluded.
    p8 = json.loads(p9.P8_REPORT.read_text(encoding="utf-8"))
    if p9.lf_sha256(p9.RAW) != p8["raw_dataset_sha256"]:
        raise RuntimeError("Raw GSM8K train file changed")
    extra = [path for rel in p9.OVER_EXCLUDE_DIRS for path in p9.json_files(rel)]
    listed = [Path(path.replace("\\", "/").replace("D:/AGI/", str(ROOT).replace("\\", "/") + "/"))
              for path in json.loads(p9.P5_REPORT.read_text(encoding="utf-8"))["exclusion_files_sha256"]]
    roots = [*[p for p in listed if p.exists()], *p9.P8_EXTRA, p9.P8_POOL, *PHASE9_POOL, EXPANSION, *extra]
    questions, indices, inventory = previous_sources(roots)
    rank = lambda i: hashlib.sha256(f"phase10_holdout_v1|{SEED}|{i}".encode()).hexdigest()
    accepted, reasons = p9.eligible(questions, indices, "phase10_gsm8k_train", rank)
    train_keys = {row["question_sha256"] for row in train}
    if any(row["question_sha256"] in train_keys for row in accepted):
        raise RuntimeError("Training source leaked into holdout eligibility")
    if len(accepted) < HOLDOUT:
        raise RuntimeError(f"Only {len(accepted)} eligible holdout sources")
    holdout = accepted[:HOLDOUT]

    OUT.mkdir(parents=True)
    report = {"schema_version": "phase10_sources_v1",
              "status": "frozen_before_any_phase10_generation",
              "train_pool": {"source": str(EXPANSION.relative_to(ROOT)),
                             "expansion_rows": len(expansion),
                             "excluded_phase4_dev_ids": len(dev_ids),
                             "rows": len(train), "sha256": write(OUT / "train_pool.jsonl", train)},
              "holdout": {"eligible": len(accepted), "rows": len(holdout),
                          "exclusion_files_scanned": len(inventory),
                          "rejected_by_first_reason": dict(sorted(reasons.items())),
                          "selection_rule": "lexicographic SHA256(phase10_holdout_v1|seed|raw_index)",
                          "seed": SEED, "sha256": write(OUT / "holdout.jsonl", holdout)},
              "protected_labels_opened": False}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
