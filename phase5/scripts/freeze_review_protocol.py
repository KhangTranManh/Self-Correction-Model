"""Validate and lock the CPU-only Phase 5 review/probe protocol.

The command reads configs and manifests only. It never imports Transformers or
PEFT, downloads model weights, opens review/probe outcomes, or runs inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "phase5/configs/review_protocol_v1.yaml"
DEFAULT_OUTPUT = ROOT / "phase5/data/protocol/review_protocol_v1_lock.json"


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the lock only before review/probe generation has started.",
    )
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite frozen protocol: {args.output}")
    if args.output.exists() and args.force:
        previous = load_json(args.output)
        if previous.get("review_generation_started") or previous.get("probe_extraction_started"):
            raise RuntimeError("Cannot replace protocol after review/probe generation started")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    registry_path = ROOT / config["models"]["registry"]
    split_manifest_path = ROOT / config["input"]["split_manifest"]
    hint_summary_path = ROOT / config["input"]["hint_summary"]
    checkpoint_audit_path = ROOT / "phase5/data/checkpoint_source_audit.json"
    prompt_source_path = ROOT / config["conversation"]["original_user_prompt_file"]
    strict_parser_path = ROOT / config["generation"]["strict_parser_file"]
    smoke_manifest_path = ROOT / config["input"]["smoke"]
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    split_manifest = load_json(split_manifest_path)
    hint_summary = load_json(hint_summary_path)
    checkpoint_audit = load_json(checkpoint_audit_path)
    with smoke_manifest_path.open(encoding="utf-8") as stream:
        smoke_rows = [json.loads(line) for line in stream if line.strip()]
    failures: list[str] = []

    if config["scope"]["primary_conditions"] != ["neutral", "status"]:
        failures.append("primary conditions must be neutral/status")
    if config["scope"]["disabled_conditions"] != ["location", "type"]:
        failures.append("location/type must remain disabled for this split")
    wrong_counts = hint_summary["counts"]
    if any(wrong_counts[split].get("wrong_location_type_eligible", 0) != expected
           for split, expected in (("train", 1), ("development", 0), ("protected_test", 0))):
        failures.append("hint coverage differs from the frozen infeasibility decision")
    if split_manifest.get("total_selected") != 480 or not split_manifest.get("source_disjoint"):
        failures.append("frozen split invariant failed")
    if not checkpoint_audit.get("all_passed") or not checkpoint_audit.get("metadata_only"):
        failures.append("checkpoint metadata audit did not pass")
    if not registry.get("checkpoint_hashes_verified_for_phase5"):
        failures.append("checkpoint verification flag is false")
    if config["generation"]["weight_dtype"] != "float16":
        failures.append("V100 review dtype must be explicitly float16")
    if config["generation"]["do_sample"] is not False:
        failures.append("review decoding must remain deterministic")
    if (len(smoke_rows) != 8 or len({row["problem_id"] for row in smoke_rows}) != 8
            or sum(row["initial_correct"] is True for row in smoke_rows) != 4):
        failures.append("smoke manifest must contain 4 correct and 4 wrong sources")
    conversation = config["conversation"]
    if set(conversation["condition_messages"]["neutral"]) != {"all"}:
        failures.append("neutral prompt must be identical across labels")
    prompt_text = "\n".join((
        conversation["system_prompt"], conversation["contract_instruction"],
        conversation["condition_messages"]["neutral"]["all"],
        conversation["condition_messages"]["status"]["initially_correct"],
        conversation["condition_messages"]["status"]["initially_wrong"],
    )).casefold()
    for banned in ("reference answer", "verifier detail", "correct answer is"):
        if banned in prompt_text:
            failures.append(f"model-visible prompt contains banned phrase: {banned}")

    lock = {
        "schema_version": "phase5_review_protocol_lock_v1",
        "status": "frozen_before_review_or_probe_generation" if not failures else "invalid",
        "frozen_on": config["frozen_on"],
        "cpu_only": True,
        "model_loaded": False,
        "model_weights_downloaded": False,
        "review_generation_started": False,
        "probe_extraction_started": False,
        "protected_evaluation_opened": False,
        "active_conditions": config["scope"]["primary_conditions"],
        "disabled_conditions": config["scope"]["disabled_conditions"],
        "files": {
            "protocol_config": {"path": str(args.config.relative_to(ROOT)).replace("\\", "/"), "sha256_lf": lf_sha256(args.config)},
            "experiment_registry": {"path": str(registry_path.relative_to(ROOT)).replace("\\", "/"), "sha256_lf": lf_sha256(registry_path)},
            "split_manifest": {"path": str(split_manifest_path.relative_to(ROOT)).replace("\\", "/"), "sha256_lf": lf_sha256(split_manifest_path)},
            "hint_summary": {"path": str(hint_summary_path.relative_to(ROOT)).replace("\\", "/"), "sha256_lf": lf_sha256(hint_summary_path)},
            "checkpoint_audit": {"path": str(checkpoint_audit_path.relative_to(ROOT)).replace("\\", "/"), "sha256_lf": lf_sha256(checkpoint_audit_path)},
            "original_prompt_source": {"path": str(prompt_source_path.relative_to(ROOT)).replace("\\", "/"), "sha256_lf": lf_sha256(prompt_source_path)},
            "strict_review_parser": {"path": str(strict_parser_path.relative_to(ROOT)).replace("\\", "/"), "sha256_lf": lf_sha256(strict_parser_path)},
            "smoke_manifest": {"path": str(smoke_manifest_path.relative_to(ROOT)).replace("\\", "/"), "sha256_lf": lf_sha256(smoke_manifest_path)},
        },
        "failures": failures,
        "all_passed": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(lock, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(lock, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
