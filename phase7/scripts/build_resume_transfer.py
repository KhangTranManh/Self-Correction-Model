"""Bundle checked Phase 7 resume state for a replacement Linux GPU."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/phase7_transfer/phase7_resume_v1.zip"
FILES = [
    "outputs/phase7_transfer/phase7_v4_linux.zip",
    "outputs/phase7_initials_v1/initial_rollouts.audit.jsonl",
    "outputs/phase7_initials_v1/initial_rollouts.jsonl",
    "outputs/phase7_initials_v1/summary.json",
    "outputs/phase7_paired_v1/development/original_solver/paired_outputs.audit.jsonl",
    "outputs/phase7_paired_v1/development/original_solver/paired_outputs.jsonl",
    "outputs/phase7_paired_v1/development/original_solver/summary.json",
    "outputs/phase7_paired_v1/development/warmstart_v2/paired_outputs.audit.jsonl",
    "outputs/phase4_exploration_warmstart_v2/final_adapter/adapter_config.json",
    "outputs/phase4_exploration_warmstart_v2/final_adapter/adapter_model.safetensors",
    "outputs/phase4_correction_sft_v3/final_adapter/adapter_config.json",
    "outputs/phase4_correction_sft_v3/final_adapter/adapter_model.safetensors",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not all((ROOT / name).is_file() for name in FILES):
        raise FileNotFoundError("Resume bundle input missing")
    audit = ROOT / "outputs/phase7_paired_v1/development/warmstart_v2/paired_outputs.audit.jsonl"
    records = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    if not records or records[0].get("type") != "metadata":
        raise ValueError("V2 resume audit missing metadata")
    for index, record in enumerate(records[1:]):
        if record.get("type") != "completion" or record.get("index") != index:
            raise ValueError("V2 resume audit is not an ordered prefix")
    manifest = {
        "schema_version": "phase7_resume_transfer_v1",
        "v2_development_completions": len(records) - 1,
        "files": {name: sha256(ROOT / name) for name in FILES},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=3, allowZip64=True) as archive:
        for name in FILES:
            archive.write(ROOT / name, arcname=name)
        archive.writestr("RESUME_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(OUTPUT) as archive:
        if set(archive.namelist()) != set(FILES) | {"RESUME_MANIFEST.json"}:
            raise ValueError("Resume archive file list mismatch")
    print(json.dumps({"archive": str(OUTPUT), "bytes": OUTPUT.stat().st_size,
                      "sha256": sha256(OUTPUT), "v2_completed": len(records) - 1}, indent=2))


if __name__ == "__main__":
    main()
