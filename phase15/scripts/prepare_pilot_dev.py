"""Draw the Phase 15 dev screen and reserve pool. No evaluation split.

The pool is the MATH training split, levels 1–3, after the same constant-answer
and verifier rules as the source audit. Ranking is SHA-256 of the pilot seed
and the question key. The first rows are the screen. The next rows are the
reserve. Everything else stays unallocated.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1"))

from phase5.scripts.prepare_candidates import question_key  # noqa: E402
from phase14.scripts.prepare_sources import canonical  # noqa: E402
from phase15.scripts.audit_sources import (  # noqa: E402
    LEVELS, MATH_URL, SUBJECTS, download, files_under, passes_verifier, question_keys,
)
from src.data.verifiers.math import extract_last_boxed  # noqa: E402


SPEC_PATH = ROOT / "phase15/kaggle/pilot_v1/pilot_spec.json"
OUT = ROOT / "phase15/data/pilot_v1"
KERNEL = ROOT / "phase15/kaggle/pilot_v1"
VENDOR_FILES = (
    ROOT / "phase1/src/core/schema.py",
    ROOT / "phase1/src/data/verifiers/math.py",
)


def eligible_rows(prior: set[str]) -> tuple[list[dict], dict]:
    import pandas as pd

    reasons = {"prior_source": 0, "non_constant_answer": 0, "reference_failed_verifier": 0, "level_4_or_5": 0}
    rows = []
    seen = set()
    cache = Path(tempfile.mkdtemp(prefix="phase15_pilot_pool_"))
    for subject in SUBJECTS:
        path = cache / f"{subject}_train.parquet"
        download(MATH_URL.format(subject=subject), path)
        frame = pd.read_parquet(path)
        for record in frame.to_dict(orient="records"):
            if record.get("level") not in LEVELS:
                reasons["level_4_or_5"] += 1
                continue
            question = str(record.get("problem") or "").strip()
            key = question_key(question)
            if key in prior or key in seen:
                reasons["prior_source"] += 1
                continue
            seen.add(key)
            raw = extract_last_boxed(str(record.get("solution") or ""))
            reference = canonical(raw) if raw is not None else None
            if reference is None:
                reasons["non_constant_answer"] += 1
                continue
            if not passes_verifier(question, reference):
                reasons["reference_failed_verifier"] += 1
                continue
            rows.append({
                "question": question,
                "question_sha256": key,
                "reference_answer": reference,
                "level": record.get("level"),
                "subject": subject,
            })
    return rows, reasons


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def copy_runtime() -> None:
    for name in ("pilot_score.py", "pilot_episodes.py", "self_edit_gate.py"):
        shutil.copy2(ROOT / "phase15/scripts" / name, KERNEL / name)
    for source in VENDOR_FILES:
        relative = source.relative_to(ROOT / "phase1")
        dest = KERNEL / "vendor" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        init = dest.parent / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")
    for package in ("src", "src/core", "src/data", "src/data/verifiers"):
        init = KERNEL / "vendor" / package / "__init__.py"
        init.parent.mkdir(parents=True, exist_ok=True)
        if not init.exists():
            init.write_text("", encoding="utf-8")


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    roots = [ROOT / f"phase{number}" for number in range(1, 15)]
    prior, inventory = question_keys(files_under(*roots, ROOT / "paper"))
    rows, reasons = eligible_rows(prior)
    rows.sort(key=lambda row: hashlib.sha256(
        f"{spec['seed']}|{row['question_sha256']}".encode()).hexdigest())
    for index, row in enumerate(rows):
        row["id"] = f"p15dev_{index:04d}"
        row["rank"] = index
    screen_n = int(spec["screen_rows"])
    reserve_n = int(spec["reserve_rows"])
    if len(rows) < screen_n + reserve_n:
        raise RuntimeError(f"Only {len(rows)} eligible problems")
    OUT.mkdir(parents=True, exist_ok=True)
    screen, reserve = rows[:screen_n], rows[screen_n:screen_n + reserve_n]
    manifest = {
        "amendment": spec["amendment"],
        "seed": spec["seed"],
        "eligible": len(rows),
        "exact_overlap_removed": reasons["prior_source"],
        "rejected": reasons,
        "screen_rows": len(screen),
        "reserve_rows": len(reserve),
        "unallocated_rows": len(rows) - len(screen) - len(reserve),
        "evaluation_split_opened": False,
        "files_scanned": len(inventory),
        "screen_sha256": write_jsonl(OUT / "screen.jsonl", screen),
        "reserve_sha256": write_jsonl(OUT / "reserve.jsonl", reserve),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    shutil.copy2(OUT / "screen.jsonl", KERNEL / "screen.jsonl")
    shutil.copy2(OUT / "reserve.jsonl", KERNEL / "reserve.jsonl")
    copy_runtime()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
