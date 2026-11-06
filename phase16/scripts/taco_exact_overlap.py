"""Conservative exact overlap audit against local Phase 1–15 JSON artifacts.

This pass is not a substitute for source-ID reconstruction, near-duplicate
review, or eligibility adjudication. It never reads TACO raw questions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from taco_inventory import canonical_url, normalized_text, sha256_file


TEXT_KEYS = {"question", "problem", "prompt", "problem_statement"}
URL_KEYS = {"url", "source_url", "problem_url", "problem_link"}


def collect(obj: object, path: str, texts: dict, urls: dict) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in TEXT_KEYS and isinstance(value, str) and value.strip():
                digest = hashlib.sha256(normalized_text(value).encode("utf-8")).hexdigest()
                texts[digest].add(path)
            elif key in URL_KEYS:
                url = canonical_url(value)
                if url:
                    urls[url].add(path)
            if isinstance(value, (dict, list)):
                collect(value, path, texts, urls)
    elif isinstance(obj, list):
        for value in obj:
            if isinstance(value, (dict, list)):
                collect(value, path, texts, urls)


def scan_json(path: Path, root: Path, texts: dict, urls: dict, errors: list) -> None:
    label = path.relative_to(root).as_posix()
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8-sig", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    collect(json.loads(line), label, texts, urls)
                except json.JSONDecodeError:
                    errors.append({"file": label, "line": line_number})
    else:
        try:
            with path.open(encoding="utf-8-sig", errors="replace") as stream:
                collect(json.load(stream), label, texts, urls)
        except json.JSONDecodeError:
            errors.append({"file": label})


def audit(workspace: Path, metadata_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise SystemExit(f"output already exists: {output_path}")
    texts, urls = defaultdict(set), defaultdict(set)
    errors = []
    files = sorted(
        path for phase in range(1, 16)
        for path in (workspace / f"phase{phase}").rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    )
    engineering = workspace / "phase16/data/engineering_smoke_v1/tasks.json"
    if engineering.exists():
        files.append(engineering)
    for path in files:
        scan_json(path, workspace, texts, urls, errors)
    hits = []
    counts = Counter()
    within_text = defaultdict(list)
    within_url = defaultdict(list)
    with metadata_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            counts["taco_rows"] += 1
            row_id = row["id"]
            digest = row["normalized_question_sha256"]
            url = row["canonical_url"]
            within_text[digest].append(row_id)
            if url:
                within_url[url].append(row_id)
            matched_texts = texts.get(digest, set())
            matched_urls = urls.get(url, set()) if url else set()
            if matched_texts or matched_urls:
                counts["historical_exact_overlap_rows"] += 1
                if matched_texts:
                    counts["historical_text_overlap_rows"] += 1
                if matched_urls:
                    counts["historical_url_overlap_rows"] += 1
                hits.append({
                    "id": row_id,
                    "text_files": sorted(matched_texts),
                    "url_files": sorted(matched_urls),
                })
    duplicate_text = [ids for ids in within_text.values() if len(ids) > 1]
    duplicate_url = [ids for ids in within_url.values() if len(ids) > 1]
    result = {
        "schema_version": "p16_taco_exact_overlap_v1",
        "metadata_sha256": sha256_file(metadata_path),
        "historical_files_scanned": len(files),
        "historical_text_hashes": len(texts),
        "historical_urls": len(urls),
        "parse_errors": errors,
        "counts": dict(counts),
        "historical_exact_hits": hits,
        "within_taco_exact_text_groups": duplicate_text,
        "within_taco_exact_url_groups": duplicate_url,
        "limits": [
            "Scans local JSON/JSONL only; other artifact formats and remote source registries not covered.",
            "Matches normalized text and canonical URL exactly; near duplicates not covered.",
            "A matching row is flagged, not silently removed; independent source review remains required.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "files": len(files),
        "exact_overlap_rows": counts["historical_exact_overlap_rows"],
        "duplicate_text_groups": len(duplicate_text),
        "duplicate_url_groups": len(duplicate_url),
        "parse_errors": len(errors),
    }, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("metadata_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    audit(args.workspace, args.metadata_path, args.output_path)
