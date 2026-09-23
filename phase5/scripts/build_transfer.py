"""Create a small checked Phase 5 Windows transfer archive without secrets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
INCLUDE = (Path("phase5"), Path("phase1/src"), Path("phase4/lib"),
           Path("phase4/configs/cycle000_rollout.yaml"))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def include(path: Path) -> bool:
    names = path.parts
    return ("__pycache__" not in names and "runs" not in names
            and path.suffix not in (".pyc", ".log")
            and not any(part == ".env" or part.startswith(".env.") for part in names))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/phase5_transfer/phase5_v1_windows.zip")
    args = parser.parse_args()
    files = []
    for item in INCLUDE:
        absolute = ROOT / item
        if not absolute.exists():
            raise FileNotFoundError(absolute)
        files.extend([absolute] if absolute.is_file() else absolute.rglob("*"))
    files = sorted(path for path in set(files) if path.is_file() and
                   include(path.relative_to(ROOT)))
    if not any(path.name == "candidate_problems.jsonl" for path in files):
        raise ValueError("Candidate manifest missing from package")
    manifest = {
        "schema_version": "phase5_transfer_v1",
        "purpose": "code_and_cpu_data_only_no_model_weights_or_credentials",
        "files": {path.relative_to(ROOT).as_posix(): sha256(path.read_bytes()) for path in files},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(ROOT).as_posix())
        archive.writestr("TRANSFER_MANIFEST.json",
                         json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with zipfile.ZipFile(args.output) as archive:
        if set(archive.namelist()) != set(manifest["files"]) | {"TRANSFER_MANIFEST.json"}:
            raise ValueError("Archive members do not match manifest")
        for name, expected in manifest["files"].items():
            if sha256(archive.read(name)) != expected:
                raise ValueError(f"Archive corruption: {name}")
    report = {"archive": str(args.output), "archive_sha256": sha256(args.output.read_bytes()),
              "bytes": args.output.stat().st_size, "files": len(files),
              "manifest_sha256": sha256(json.dumps(manifest, indent=2, sort_keys=True).encode()+b"\n"),
              "model_weights_included": False, "credentials_included": False}
    sidecar = args.output.with_suffix(".report.json")
    sidecar.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
