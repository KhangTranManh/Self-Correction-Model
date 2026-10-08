"""Freeze the Phase 12 holdout from SVAMP (all 1,000 problems).

GSM8K has no never-used train source left, so Phase 12 evaluates on SVAMP
(arkilpatel/SVAMP, SVAMP.json). Each problem is Body + " " + Question with an
integer reference answer. A source is kept only if the Phase 1 verifier
accepts its own reference and its normalized question does not appear in any
Phase 1-11 inventory (same exclusion scan as Phases 9-11).
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase4.lib.rollout import as_problem, verify  # noqa: E402
from phase5.scripts.prepare_candidates import previous_sources, question_key  # noqa: E402
from phase9.scripts import prepare_sources as p9  # noqa: E402

RAW = ROOT / "phase12/data/raw/SVAMP.json"
RAW_SHA256 = "5be77703a6d891ae"  # prefix recorded at download; full hash stored in the report
PRIOR = (ROOT / "phase9/data/source_pool_v1/development.jsonl",
         ROOT / "phase9/data/source_pool_v1/protected.jsonl",
         ROOT / "phase10/data/sources_v1/train_pool.jsonl",
         ROOT / "phase10/data/sources_v1/holdout.jsonl",
         ROOT / "phase11/data/sources_v1/holdout.jsonl")
OUT = ROOT / "phase12/data/sources_v1"
SEED = 20261006


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite: {OUT}")
    raw_bytes = RAW.read_bytes()
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    if not raw_sha.startswith(RAW_SHA256):
        raise RuntimeError("SVAMP file differs from the downloaded copy")
    records = json.loads(raw_bytes)
    listed = [Path(path.replace("\\", "/").replace("D:/AGI/", str(ROOT).replace("\\", "/") + "/"))
              for path in json.loads(p9.P5_REPORT.read_text(encoding="utf-8"))["exclusion_files_sha256"]]
    extra = [path for rel in p9.OVER_EXCLUDE_DIRS for path in p9.json_files(rel)]
    roots = [*[p for p in listed if p.exists()], *p9.P8_EXTRA, p9.P8_POOL, *PRIOR, *extra]
    prior_questions, _, inventory = previous_sources(roots)
    kept, reasons, seen = [], Counter(), set()
    for record in records:
        question = f"{record['Body'].strip()} {record['Question'].strip()}"
        key = question_key(question)
        answer = float(record["Answer"])
        if key in prior_questions:
            reasons["prior_source"] += 1
            continue
        if key in seen:
            reasons["duplicate_question"] += 1
            continue
        seen.add(key)
        if answer != int(answer):
            reasons["non_integer_answer"] += 1
            continue
        row = {"id": f"phase12_svamp_{record['ID']}", "svamp_id": record["ID"], "domain": "math",
               "question": question, "question_sha256": key, "reference_answer": str(int(answer)),
               "svamp_type": record["Type"],
               "source_rank_sha256": hashlib.sha256(f"phase12_holdout_v1|{SEED}|{record['ID']}".encode()).hexdigest()}
        if not verify(as_problem(row), row["reference_answer"])["passed"]:
            reasons["reference_failed_verifier"] += 1
            continue
        kept.append(row)
    kept.sort(key=lambda r: (r["source_rank_sha256"], r["id"]))
    OUT.mkdir(parents=True)
    path = OUT / "holdout.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in kept),
                    encoding="utf-8", newline="\n")
    report = {"schema_version": "phase12_sources_v1", "status": "frozen_before_any_phase12_generation",
              "dataset": "SVAMP (github.com/arkilpatel/SVAMP, SVAMP.json)", "raw_sha256": raw_sha,
              "raw_rows": len(records), "exclusion_files_scanned": len(inventory),
              "rejected": dict(sorted(reasons.items())),
              "types": dict(Counter(r["svamp_type"] for r in kept)),
              "holdout": {"rows": len(kept), "order": "lexicographic SHA256(phase12_holdout_v1|seed|svamp_id)",
                          "seed": SEED, "sha256": p9.lf_sha256(path)},
              "protected_labels_opened": False}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
