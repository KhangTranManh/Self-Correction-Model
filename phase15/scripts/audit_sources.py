"""Count Phase 15 candidate sources against every Phase 1-14 question.

This draws no split and writes no problem text. ASDiv is read from the local
Phase 14 raw file. The MATH training split and the reserved-set candidate are
read from their public files into a temporary directory. Exact overlap uses the Phase 5 question key: NFKC, casefold, collapsed
whitespace, then SHA-256. The key is taken from ``question``, from ``problem``
(Phase 3 stores its source text there), and from ``Body`` plus ``Question``.

ASDiv eligibility reuses the Phase 14 rules, including the SVAMP Jaccard cap.
MATH eligibility reuses the Phase 14 constant-answer rule and the Phase 1
verifier. A clean ASDiv problem that already sits in a Phase 14 source file is
not counted as fresh.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1"))

from src.core.schema import Problem  # noqa: E402
from src.data.verifiers.math import MathVerifier, extract_last_boxed  # noqa: E402
from phase5.scripts.prepare_candidates import digest, nested_objects, question_key, records  # noqa: E402
from phase14.scripts.config import SVAMP_JACCARD_MAX  # noqa: E402
from phase14.scripts.prepare_sources import asdiv_rows, canonical, tokens  # noqa: E402


OUT = ROOT / "phase15/data/preparation_v1/source_audit.json"
LEVELS = ("Level 1", "Level 2", "Level 3")
SUBJECTS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)
MATH_URL = "https://huggingface.co/api/datasets/EleutherAI/hendrycks_math/parquet/{subject}/train/0.parquet"
RESERVED_URLS = (
    ("ChilleD/MultiArith", "https://huggingface.co/api/datasets/ChilleD/MultiArith/parquet/default/train/0.parquet"),
)
SKIP_PARTS = {"__pycache__", ".git", "outputs", ".venv"}
VERIFIER = MathVerifier()


def files_under(*roots: Path) -> list[Path]:
    found = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            if any(part in SKIP_PARTS or part.startswith(".venv") for part in path.parts):
                continue
            found.append(path)
    return sorted(found)


def question_keys(paths: list[Path]) -> tuple[set[str], dict[str, str]]:
    keys, inventory = set(), {}
    for path in paths:
        inventory[str(path.resolve())] = digest(path)
        for record, _number in records(path):
            for obj in nested_objects(record):
                if not isinstance(obj, dict):
                    continue
                for field in ("question", "problem"):
                    value = obj.get(field)
                    if isinstance(value, str) and value.strip():
                        keys.add(question_key(value))
                body, ask = obj.get("Body"), obj.get("Question")
                if isinstance(body, str) and isinstance(ask, str) and body.strip() and ask.strip():
                    keys.add(question_key(f"{body.strip()} {ask.strip()}"))
    if not keys:
        raise ValueError("Exclusion inputs yielded no question or problem text")
    return keys, inventory


def ids(path: Path, limit: int | None = None) -> list[str]:
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        found.append(json.loads(line)["id"])
        if limit is not None and len(found) >= limit:
            break
    return found


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "phase15-source-audit"})
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as handle:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)


def passes_verifier(question: str, reference: str) -> bool:
    problem = Problem(id="phase15-audit", domain="math", question=question, reference_answer=reference)
    try:
        return bool(VERIFIER.verify(problem, "\\boxed{" + reference + "}").passed)
    except Exception:
        return False


def math_train(prior: set[str], cache: Path) -> dict:
    import pandas as pd

    reasons: Counter[str] = Counter()
    seen = set()
    eligible_levels: Counter[str] = Counter()
    rows_read = 0
    for subject in SUBJECTS:
        path = cache / f"{subject}_train.parquet"
        download(MATH_URL.format(subject=subject), path)
        frame = pd.read_parquet(path)
        for record in frame.to_dict(orient="records"):
            rows_read += 1
            level = record.get("level")
            if level not in LEVELS:
                reasons["level_4_or_5"] += 1
                continue
            question = str(record.get("problem") or "").strip()
            key = question_key(question)
            if key in prior:
                reasons["prior_source"] += 1
                continue
            if key in seen:
                reasons["duplicate_question"] += 1
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
            reasons["eligible"] += 1
            eligible_levels[level] += 1
    return {
        "source": "EleutherAI/hendrycks_math train",
        "rows_read": rows_read,
        "exact_question_overlap": reasons["prior_source"],
        "eligible_levels_1_to_3": reasons["eligible"],
        "eligible_by_level": dict(sorted(eligible_levels.items())),
        "rejected": {key: value for key, value in sorted(reasons.items()) if key != "eligible"},
        "raw_kept": False,
    }


def asdiv_token_sets() -> list[set[str]]:
    sets = []
    for problem in ET.parse(ROOT / "phase14/data/raw/ASDiv.xml").getroot().iter("Problem"):
        question = " ".join(
            f"{problem.findtext('Body').strip()} {problem.findtext('Question').strip()}".split())
        sets.append(tokens(question))
    return sets


def reserved_candidate(earlier: set[str], phase14: set[str], asdiv_keys: set[str],
                       math_keys: set[str], cache: Path) -> dict:
    import pandas as pd

    name, url = RESERVED_URLS[0]
    path = cache / "multiarith_train.parquet"
    try:
        download(url, path)
        frame = pd.read_parquet(path)
    except Exception as exc:
        return {"name": name, "status": "download_failed", "error": type(exc).__name__, "reserved": False}
    prior = earlier | phase14
    asdiv_sets = asdiv_token_sets()
    rows = duplicates = earlier_hits = phase14_hits = asdiv_hits = math_hits = near_asdiv = 0
    seen = set()
    for record in frame.to_dict(orient="records"):
        question = record.get("question") or record.get("problem") or record.get("sQuestion")
        if not isinstance(question, str) or not question.strip():
            body, ask = record.get("Body"), record.get("Question")
            if isinstance(body, str) and isinstance(ask, str):
                question = f"{body.strip()} {ask.strip()}"
            else:
                continue
        rows += 1
        key = question_key(question)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        if key not in prior:
            question_tokens = tokens(question)
            if question_tokens and max(
                len(question_tokens & other) / len(question_tokens | other) for other in asdiv_sets
            ) >= SVAMP_JACCARD_MAX:
                near_asdiv += 1
            continue
        if key in earlier:
            earlier_hits += 1
        else:
            phase14_hits += 1
        if key in asdiv_keys:
            asdiv_hits += 1
        if key in math_keys:
            math_hits += 1
    overlap = earlier_hits + phase14_hits
    return {
        "name": name,
        "rows": rows,
        "duplicate_rows": duplicates,
        "unique_rows": rows - duplicates,
        "exact_question_overlap": overlap,
        "overlap_before_phase14": earlier_hits,
        "overlap_phase14_only": phase14_hits,
        "overlap_asdiv_pool": asdiv_hits,
        "overlap_math_test_pool": math_hits,
        "absent": rows - duplicates - overlap,
        "absent_near_asdiv_jaccard": near_asdiv,
        "jaccard_max": SVAMP_JACCARD_MAX,
        "reserved": False,
        "status": "candidate_not_reserved",
    }


def inventory_sha256(inventory: dict[str, str]) -> str:
    lines = "".join(f"{digest} {path}\n" for path, digest in sorted(inventory.items()))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def main() -> None:
    phase_roots = [ROOT / f"phase{number}" for number in range(1, 15)]
    earlier_paths = files_under(*[root for root in phase_roots if root.name != "phase14"], ROOT / "paper")
    phase14_paths = files_under(ROOT / "phase14")
    earlier, earlier_files = question_keys(earlier_paths)
    phase14, phase14_files = question_keys(phase14_paths)
    prior = earlier | phase14
    inventory = {**earlier_files, **phase14_files}

    before = Counter()
    eligible_before = asdiv_rows(earlier, before)
    strict_reasons: Counter = Counter()
    eligible_strict = asdiv_rows(prior, strict_reasons)
    main_size = json.loads((ROOT / "phase14/data/main_size.json").read_text(encoding="utf-8"))["main_size"]["asdiv"]
    pilot_path = ROOT / "phase14/data/sources_v1/asdiv_pilot.jsonl"
    main_path = ROOT / "phase14/data/sources_v1/asdiv_main_pool.jsonl"
    pilot_ids = set(ids(pilot_path))
    main_ids = ids(main_path)
    generated_ids = pilot_ids | set(main_ids[:main_size])
    pool_ids = pilot_ids | set(main_ids)
    stored_not_generated = pool_ids - generated_ids

    with tempfile.TemporaryDirectory(prefix="phase15_source_audit_") as temporary:
        cache = Path(temporary)
        asdiv_keys = set()
        math_keys = set()
        for name, bucket in (("asdiv", asdiv_keys), ("math13", math_keys)):
            for split in ("pilot", "main_pool"):
                for line in (ROOT / "phase14/data/sources_v1" / f"{name}_{split}.jsonl").read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        bucket.add(question_key(json.loads(line)["question"]))
        math = math_train(prior, cache)
        reserved = reserved_candidate(earlier, phase14, asdiv_keys, math_keys, cache)

    report = {
        "schema_version": "phase15_source_audit_v1",
        "scientific_evidence": False,
        "splits_drawn": False,
        "key_fields": ["question", "problem", "Body+Question"],
        "files_scanned": len(inventory),
        "question_keys": len(prior),
        "question_keys_before_phase14": len(earlier),
        "inventory_sha256": inventory_sha256(inventory),
        "asdiv": {
            "raw": "phase14/data/raw/ASDiv.xml",
            "eligible_excluding_phase14_files": len(eligible_before),
            "rejected_excluding_phase14_files": dict(sorted(before.items())),
            "pool_rows": len(pool_ids),
            "pilot_rows": len(pilot_ids),
            "main_rows_generated": main_size,
            "stored_in_phase14_pool_but_not_generated": len(stored_not_generated),
            "strict_eligible_absent_from_phase1_14": len(eligible_strict),
            "strict_rejected": dict(sorted(strict_reasons.items())),
            "fresh_pool": len(eligible_strict) > 0,
        },
        "math_train": math,
        "reserved_candidate": reserved,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
