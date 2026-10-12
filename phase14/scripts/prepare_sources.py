"""Freeze the Phase 14 source pools: ASDiv and MATH test levels 1-3.

For each dataset the eligible problems are ranked by
SHA256(phase14_sources_v1|dataset|seed|raw_id). The first PILOT_SIZE form the
pilot (burned after the pilot report); the rest form the ordered main pool,
whose first N rows become the main holdout once N is fixed from the pilot.

Eligibility:
  asdiv   numeric answer (units in parentheses dropped) that the Phase 1
          verifier accepts against itself; a simple "expr=value" formula, when
          present, must evaluate to the answer; token Jaccard to every SVAMP
          problem below SVAMP_JACCARD_MAX, because SVAMP (opened in Phase 12)
          was built by editing ASDiv-A problems.
  math13  MATH test, levels 1-3; reference = last \\boxed{} of the solution,
          parsed by the Phase 1 verifier to a constant (no free symbols, no
          tuples/intervals/sets) and accepted against itself.
Both: normalized question absent from every Phase 1-13 JSON/JSONL file
(same evidence set as Phase 13 plus Phase 13's own data), no duplicates.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.data.verifiers.math import _to_sympy, extract_last_boxed  # noqa: E402
from phase4.lib.rollout import as_problem, verify  # noqa: E402
from phase5.scripts.prepare_candidates import previous_sources, question_key, value  # noqa: E402
from phase9.scripts import prepare_sources as p9  # noqa: E402
from phase13.scripts import prepare_sources as p13  # noqa: E402
from phase14.scripts.config import PILOT_SIZE, RAW, SEED, SOURCES, SVAMP_JACCARD_MAX  # noqa: E402

SVAMP = ROOT / "phase12/data/raw/SVAMP.json"
PRIOR_DIRS = ("phase8/data", "phase9/data", "phase10/data", "phase11/data", "phase12/data", "phase13/data")
LEVELS = ("Level 1", "Level 2", "Level 3")
_TOKENS = re.compile(r"[a-z]+|\d+(?:\.\d+)?")
_NUMERIC = re.compile(r"-?\d+(?:\.\d+)?")


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_raw() -> None:
    manifest = json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["sha256"].items():
        if hashlib.sha256((RAW / rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Raw file changed since manifest: {rel}")


def prior_questions() -> tuple[set, int]:
    listed = [Path(path.replace("\\", "/").replace("D:/AGI/", str(ROOT).replace("\\", "/") + "/"))
              for path in json.loads(p9.P5_REPORT.read_text(encoding="utf-8"))["exclusion_files_sha256"]]
    extra = [path for rel in (*p9.OVER_EXCLUDE_DIRS, *PRIOR_DIRS) for path in p9.json_files(rel)]
    roots = [*[p for p in listed if p.exists()], *p9.P8_EXTRA, p9.P8_POOL, *p13.PRIOR, *extra]
    questions, _, inventory = previous_sources(roots)
    return questions, len(inventory)


def tokens(text: str) -> set[str]:
    return set(_TOKENS.findall(text.lower()))


_UNIT_TEXT = re.compile(r"\\(?:text|mbox)\{[^{}]*\}(?:\^\{?\d\}?)?")  # "\text{ units}^2" is a unit
_SPACING = re.compile(r"\\[!,;]|\\ ")
_THOUSANDS = re.compile(r"-?\d{1,3}(,\d{3})+(\.\d+)?")
_NUMERIC_MACROS = re.compile(r"\\(?:d?frac|sqrt|pi|cdot|times|left|right)")
_NUMERIC_CHARS = re.compile(r"[0-9.+\-*/^(){}\[\] ]+")
_MIXED_NUMBER = re.compile(r"\d\s*\\d?frac")


def canonical(reference: str) -> str | None:
    """The reference as one numeric value the Phase 1 verifier reads exactly, else None.

    The Phase 1 parser falls back to a leading number when a full parse fails,
    so "500a^7" reads as 500, "1,3" (two answers) as 1.3, "443_5" (base 5) as
    4435, "10{,}000" as 10, and "45 \\text{ units}^2" as 47. Such references
    would score silently wrong. Units in \\text{...} (with a trailing exponent),
    LaTeX spacing, and thousands separators are removed; anything else that is
    not a plain numeric expression is excluded. The cleaned string is stored as
    the reference so the verifier never sees the raw form.
    """
    clean = _SPACING.sub("", _UNIT_TEXT.sub("", reference)).replace("{,}", ",").strip().rstrip(".")
    clean = re.sub(r"\s+", "", clean)  # "15 \pi" reads as 15 ("pi" taken as a unit); "15\pi" as 15*pi
    if "," in clean:
        if not _THOUSANDS.fullmatch(clean):
            return None
        clean = clean.replace(",", "")
    if _MIXED_NUMBER.search(clean):  # "12\frac{3}{5}" parses as 12*3/5, not 12 + 3/5
        return None
    bare = _NUMERIC_MACROS.sub("", clean)
    if not bare or not _NUMERIC_CHARS.fullmatch(bare):
        return None
    try:
        parsed = _to_sympy(clean)
    except Exception:
        return None
    if getattr(parsed, "free_symbols", None) != set() or not getattr(parsed, "is_number", False):
        return None
    return clean


def asdiv_rows(prior: set, reasons: Counter) -> list[dict]:
    svamp = [tokens(r["Body"] + " " + r["Question"]) for r in json.loads(SVAMP.read_text(encoding="utf-8"))]
    rows, seen = [], set()
    for problem in ET.parse(RAW / "ASDiv.xml").getroot().iter("Problem"):
        raw_id = problem.get("ID")
        question = " ".join(f"{problem.findtext('Body').strip()} {problem.findtext('Question').strip()}".split())
        key = question_key(question)
        if key in prior:
            reasons["prior_source"] += 1
            continue
        if key in seen:
            reasons["duplicate_question"] += 1
            continue
        seen.add(key)
        answer = problem.findtext("Answer").split("(")[0].strip()
        if not _NUMERIC.fullmatch(answer):
            reasons["non_numeric_answer"] += 1
            continue
        formula = (problem.findtext("Formula") or "").strip()
        if formula.count("=") == 1:
            left, right = formula.split("=")
            try:
                if value(left) != value(right) or value(right) != value(answer):
                    reasons["formula_mismatch"] += 1
                    continue
            except (SyntaxError, ValueError, ZeroDivisionError, TypeError):
                pass  # non-arithmetic formula (e.g. with x); the answer check below still applies
        q = tokens(question)
        if max(len(q & s) / len(q | s) for s in svamp) >= SVAMP_JACCARD_MAX:
            reasons["near_svamp"] += 1
            continue
        rows.append({"id": f"phase14_asdiv_{raw_id}", "dataset": "asdiv", "raw_id": raw_id, "domain": "math",
                     "question": question, "question_sha256": key, "reference_answer": answer,
                     "grade": problem.get("Grade"), "solution_type": problem.findtext("Solution-Type")})
    return rows


def math_rows(prior: set, reasons: Counter) -> list[dict]:
    rows, seen = [], set()
    for line in (RAW / "math_test.jsonl").read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        if record["level"] not in LEVELS:
            reasons["level_4_or_5"] += 1
            continue
        question = record["problem"].strip()
        key = question_key(question)
        if key in prior:
            reasons["prior_source"] += 1
            continue
        if key in seen:
            reasons["duplicate_question"] += 1
            continue
        seen.add(key)
        raw_reference = extract_last_boxed(record["solution"])
        reference = canonical(raw_reference) if raw_reference is not None else None
        if reference is None:
            reasons["non_constant_answer"] += 1
            continue
        raw_id = f"{record['subject']}_{record['subject_index']}"
        rows.append({"id": f"phase14_math13_{raw_id}", "dataset": "math13", "raw_id": raw_id, "domain": "math",
                     "question": question, "question_sha256": key, "reference_answer": reference,
                     "reference_raw": raw_reference, "level": record["level"], "type": record["type"]})
    return rows


def main() -> None:
    if SOURCES.exists():
        raise RuntimeError(f"Refusing to overwrite: {SOURCES}")
    check_raw()
    prior, scanned = prior_questions()
    report = {"schema_version": "phase14_sources_v1", "status": "frozen_before_any_phase14_generation",
              "seed": SEED, "pilot_size": PILOT_SIZE, "exclusion_files_scanned": scanned,
              "svamp_jaccard_max": SVAMP_JACCARD_MAX, "datasets": {}, "protected_labels_opened": False}
    SOURCES.mkdir(parents=True)
    for dataset, build in (("asdiv", asdiv_rows), ("math13", math_rows)):
        reasons = Counter()
        rows = []
        for row in build(prior, reasons):
            # Boxed so LaTeX references (e.g. \frac{1}{2}) are read whole, not as the last number.
            if not verify(as_problem(row), "\\boxed{" + row["reference_answer"] + "}")["passed"]:
                reasons["reference_failed_verifier"] += 1
                continue
            row["source_rank_sha256"] = hashlib.sha256(
                f"phase14_sources_v1|{dataset}|{SEED}|{row['raw_id']}".encode()).hexdigest()
            rows.append(row)
        rows.sort(key=lambda r: (r["source_rank_sha256"], r["id"]))
        if len(rows) <= PILOT_SIZE:
            raise RuntimeError(f"{dataset}: only {len(rows)} eligible problems")
        files = {}
        for split, part in (("pilot", rows[:PILOT_SIZE]), ("main_pool", rows[PILOT_SIZE:])):
            path = SOURCES / f"{dataset}_{split}.jsonl"
            path.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in part),
                            encoding="utf-8", newline="\n")
            files[split] = {"rows": len(part), "sha256": lf_sha256(path)}
        mix = Counter(r.get("level") or f"grade {r.get('grade')}" for r in rows[PILOT_SIZE:])
        report["datasets"][dataset] = {"eligible": len(rows), "rejected": dict(sorted(reasons.items())),
                                       "main_pool_mix": dict(sorted(mix.items())), **files}
    (SOURCES / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
