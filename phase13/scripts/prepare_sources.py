"""Freeze the Phase 13 holdout: GSM8K test problems 750-1318.

GSM8K test indices 0-749 are excluded outright because Phase 1's held-out
evaluation used test problems from offset 150 (600 problems) and earlier
indices may have served as development data. The remaining problems are kept
only if their reference passes the Phase 1 verifier with the same step checks
as Phases 8-11, and their normalized question appears in no Phase 1-12 file.
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
from phase5.scripts.prepare_candidates import FINAL, STEP, previous_sources, question_key, value  # noqa: E402
from phase9.scripts import prepare_sources as p9  # noqa: E402

RAW = ROOT / "phase13/data/raw/gsm8k_test.jsonl"
RAW_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
FIRST_INDEX = 750
PRIOR = (ROOT / "phase9/data/source_pool_v1/development.jsonl",
         ROOT / "phase9/data/source_pool_v1/protected.jsonl",
         ROOT / "phase10/data/sources_v1/train_pool.jsonl",
         ROOT / "phase10/data/sources_v1/holdout.jsonl",
         ROOT / "phase11/data/sources_v1/holdout.jsonl",
         ROOT / "phase12/data/sources_v1/holdout.jsonl")
OUT = ROOT / "phase13/data/sources_v1"
SEED = 20261009


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite: {OUT}")
    raw = RAW.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RAW_SHA256:
        raise RuntimeError("GSM8K test file differs from the downloaded copy")
    records = [json.loads(line) for line in raw.decode("utf-8").split("\n") if line.strip()]
    listed = [Path(path.replace("\\", "/").replace("D:/AGI/", str(ROOT).replace("\\", "/") + "/"))
              for path in json.loads(p9.P5_REPORT.read_text(encoding="utf-8"))["exclusion_files_sha256"]]
    extra = [path for rel in (*p9.OVER_EXCLUDE_DIRS, "phase8/data", "phase9/data", "phase10/data",
                              "phase11/data", "phase12/data") for path in p9.json_files(rel)]
    roots = [*[p for p in listed if p.exists()], *p9.P8_EXTRA, p9.P8_POOL, *PRIOR, *extra]
    prior_questions, _, inventory = previous_sources(roots)
    kept, reasons, seen = [], Counter(), set()
    for index, record in enumerate(records):
        if index < FIRST_INDEX:
            reasons["index_below_750"] += 1
            continue
        question, answer = record["question"], record["answer"]
        key = question_key(question)
        if key in prior_questions:
            reasons["prior_source"] += 1
            continue
        if key in seen:
            reasons["duplicate_question"] += 1
            continue
        seen.add(key)
        steps = STEP.findall(answer)
        try:
            if any(value(expr) != value(declared) for expr, declared in steps):
                raise ValueError("Reference step mismatch")
            match = FINAL.search(answer)
            if match is None:
                raise ValueError("Missing final answer")
            reference = str(value(match.group(1)))
        except (SyntaxError, ValueError, ZeroDivisionError, TypeError):
            reasons["unverifiable_reference"] += 1
            continue
        row = {"id": f"phase13_gsm8k_test_{index}", "dataset_index": index, "domain": "math",
               "question": question, "question_sha256": key, "reference_answer": reference,
               "reference_step_count": len(steps),
               "source_rank_sha256": hashlib.sha256(f"phase13_holdout_v1|{SEED}|{index}".encode()).hexdigest()}
        if not verify(as_problem(row), reference)["passed"]:
            reasons["reference_failed_verifier"] += 1
            continue
        kept.append(row)
    kept.sort(key=lambda r: (r["source_rank_sha256"], r["id"]))
    OUT.mkdir(parents=True)
    path = OUT / "holdout.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in kept),
                    encoding="utf-8", newline="\n")
    report = {"schema_version": "phase13_sources_v1", "status": "frozen_before_any_phase13_generation",
              "dataset": "GSM8K test (github.com/openai/grade-school-math, test.jsonl)",
              "raw_sha256": RAW_SHA256, "raw_rows": len(records), "index_range": [FIRST_INDEX, len(records) - 1],
              "exclusion_files_scanned": len(inventory), "rejected": dict(sorted(reasons.items())),
              "holdout": {"rows": len(kept), "order": "lexicographic SHA256(phase13_holdout_v1|seed|index)",
                          "seed": SEED, "sha256": p9.lf_sha256(path)},
              "protected_labels_opened": False}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
