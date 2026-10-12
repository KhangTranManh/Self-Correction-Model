"""Record (and, if missing, download) the Phase 14 raw datasets.

Sources:
  ASDiv     github.com/chaochun/nlu-asdiv-dataset, dataset/ASDiv.xml
  MATH test huggingface.co/datasets/EleutherAI/hendrycks_math, test parquet
            files for the seven subjects

The MATH parquet files are converted once into data/raw/math_test.jsonl
(subject order fixed, original row order kept) so the GPU host needs no
pyarrow. data/raw/manifest.json stores the SHA-256 of every raw file; the
source builder refuses to run if a file no longer matches.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phase14.scripts.config import RAW  # noqa: E402

ASDIV_URL = "https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/master/dataset/ASDiv.xml"
MATH_URL = "https://huggingface.co/api/datasets/EleutherAI/hendrycks_math/parquet/{subject}/test/0.parquet"
SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus")
MANIFEST = RAW / "manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)


def main() -> None:
    if MANIFEST.exists():
        raise RuntimeError("Raw manifest already recorded; verify instead of refetching")
    import pandas as pd

    fetch(ASDIV_URL, RAW / "ASDiv.xml")
    rows = []
    for subject in SUBJECTS:
        path = RAW / "math_test" / f"{subject}.parquet"
        fetch(MATH_URL.format(subject=subject), path)
        for index, record in enumerate(pd.read_parquet(path).to_dict("records")):
            rows.append({"subject": subject, "subject_index": index, **{k: str(v) for k, v in record.items()}})
    math_jsonl = RAW / "math_test.jsonl"
    math_jsonl.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
                          encoding="utf-8", newline="\n")
    files = [RAW / "ASDiv.xml", math_jsonl, *sorted((RAW / "math_test").glob("*.parquet"))]
    manifest = {"schema_version": "phase14_raw_manifest_v1", "asdiv_url": ASDIV_URL,
                "math_url_template": MATH_URL, "math_rows": len(rows),
                "sha256": {str(p.relative_to(RAW)).replace("\\", "/"): sha256(p) for p in files}}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
