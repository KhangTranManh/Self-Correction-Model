"""Build a checked Phase 7 Windows transfer ZIP without secrets or weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
INCLUDE = (
    Path("phase7"),
    Path("phase1/src"),
    Path("phase4/lib"),
    Path("phase5/scripts/materialize_v2_merged.py"),
    Path("phase5/data/checkpoint_source_audit.json"),
    Path("phase5/configs/experiments.yaml"),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def include(path: Path) -> bool:
    return ("__pycache__" not in path.parts
            and "runs" not in path.parts
            and path.suffix.lower() not in {".pyc", ".log", ".zip", ".safetensors"}
            and not any(part == ".env" or part.startswith(".env.") for part in path.parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/phase7_transfer/phase7_v1_windows.zip")
    args = parser.parse_args()
    files: set[Path] = set()
    for item in INCLUDE:
        source = ROOT / item
        if not source.exists():
            raise FileNotFoundError(source)
        files.update([source] if source.is_file() else source.rglob("*"))
    selected = sorted(path for path in files if path.is_file()
                      and include(path.relative_to(ROOT)))
    if not any(path.name == "candidate_problems.jsonl" for path in selected):
        raise RuntimeError("Frozen Phase 7 candidates are missing")
    manifest = {
        "schema_version": "phase7_transfer_v1",
        "purpose": "code_and_cpu_data_only_no_model_weights_or_credentials",
        "files": {path.relative_to(ROOT).as_posix(): sha256(path.read_bytes())
                  for path in selected},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for path in selected:
            archive.write(path, arcname=path.relative_to(ROOT).as_posix())
        archive.writestr("TRANSFER_MANIFEST.json",
                         json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with zipfile.ZipFile(args.output) as archive:
        if set(archive.namelist()) != set(manifest["files"]) | {"TRANSFER_MANIFEST.json"}:
            raise RuntimeError("Archive member list mismatch")
        for name, expected in manifest["files"].items():
            if sha256(archive.read(name)) != expected:
                raise RuntimeError(f"Archive corruption: {name}")
    report = {
        "archive": str(args.output), "archive_sha256": sha256(args.output.read_bytes()),
        "bytes": args.output.stat().st_size, "files": len(selected),
        "model_weights_included": False, "credentials_included": False,
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
