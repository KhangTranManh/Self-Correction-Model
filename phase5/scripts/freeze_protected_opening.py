"""Freeze all train/development choices before the one protected opening."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ("original_solver", "warmstart_v2", "correction_sft_v3")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path,
                        default=ROOT / "outputs/phase5_gpu_vllm/review_v1")
    parser.add_argument("--probe-dir", type=Path,
                        default=ROOT / "outputs/phase5_gpu_vllm/probe_v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "phase5/data/protocol/protected_opening_v1_lock.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace protected-opening lock: {args.output}")

    files: dict[str, dict[str, str]] = {}
    protocol_lock = ROOT / "phase5/data/protocol/review_protocol_v1_lock.json"
    files["review_protocol_lock"] = {"path": relative(protocol_lock), "sha256": sha256(protocol_lock)}
    selected: dict[str, dict] = {}
    for checkpoint in CHECKPOINTS:
        for split in ("train", "development"):
            path = args.review_dir / f"{checkpoint}_{split}_summary.json"
            files[f"review_{checkpoint}_{split}"] = {"path": relative(path), "sha256": sha256(path)}
        selection_path = args.probe_dir / "selection" / f"{checkpoint}_probe_selection.json"
        report = json.loads(selection_path.read_text(encoding="utf-8"))
        if report.get("protected_test_opened") is not False:
            raise RuntimeError(f"Probe selection is not pre-protected: {checkpoint}")
        selected[checkpoint] = {
            "layer": report["selected"]["layer"],
            "layer_index": report["selected"]["layer_index"],
            "c": report["selected"]["c"],
        }
        files[f"probe_selection_{checkpoint}"] = {
            "path": relative(selection_path), "sha256": sha256(selection_path),
        }
    controls_path = args.probe_dir / "probe_controls.json"
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    if controls.get("protected_test_opened") is not False:
        raise RuntimeError("Probe controls are not pre-protected")
    files["probe_controls"] = {"path": relative(controls_path), "sha256": sha256(controls_path)}

    lock = {
        "schema_version": "phase5_protected_opening_lock_v1",
        "status": "ready_for_single_protected_opening",
        "protected_results_read": False,
        "protected_open_count": 0,
        "allowed_open_count": 1,
        "review_checkpoints": list(CHECKPOINTS),
        "review_conditions": ["neutral", "status"],
        "probe_selection": selected,
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
