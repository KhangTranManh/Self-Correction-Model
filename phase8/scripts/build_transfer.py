"""Package checked Phase 8 code, frozen data, and historical audits for a GPU host."""

from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/phase8_transfer"
INCLUDE = (
    "phase1/src", "phase4/lib", "phase5/configs", "phase5/data/splits/v1",
    "phase5/data/protocol", "phase5/scripts", "phase7/configs", "phase7/data",
    "phase7/scripts", "phase8", "outputs/phase7_initials_v1",
    "outputs/phase7_paired_v1",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="v1")
    args = parser.parse_args()
    paths = []
    for relative in INCLUDE:
        base = ROOT / relative
        paths.extend(path for path in base.rglob("*") if path.is_file()
                     and "__pycache__" not in path.parts and path.suffix != ".pyc")
    paths.sort(key=lambda path: path.relative_to(ROOT).as_posix())
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / f"phase8_protocol_{args.label}.zip"
    if archive.exists():
        raise RuntimeError(f"Refusing to overwrite: {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path, path.relative_to(ROOT).as_posix())
    report = {"schema_version": "phase8_transfer_v1", "label": args.label, "files": len(paths),
              "archive_sha256": sha256(archive), "archive_bytes": archive.stat().st_size,
              "includes_secrets": False}
    (OUT / f"phase8_protocol_{args.label}.report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
