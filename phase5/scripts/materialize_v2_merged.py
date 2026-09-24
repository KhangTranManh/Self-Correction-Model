"""Materialize the exact Phase 4 V2 parent required by the V3 LoRA adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "phase5/configs/experiments.yaml"
AUDIT = ROOT / "phase5/data/checkpoint_source_audit.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("/root/models/phase4_warmstart_v2_merged"))
    parser.add_argument("--report", type=Path,
                        default=ROOT / "phase5/runs/materialize_v2_report.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=False)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not audit.get("all_passed"):
        raise RuntimeError("Checkpoint source audit is not passing")

    repos = registry["checkpoint_hub_repos"]
    revisions = registry["checkpoint_hub_revisions"]
    hashes = registry["checkpoint_adapter_sha256"]
    base_repo = repos["original_solver"]
    base_revision = revisions["original_solver"]
    adapter_repo = repos["phase4_warmstart_v2"]
    adapter_revision = revisions["phase4_warmstart_v2"]

    marker = args.output / "phase5_lineage.json"
    expected_lineage = {
        "base_repo": base_repo,
        "base_revision": base_revision,
        "adapter_repo": adapter_repo,
        "adapter_revision": adapter_revision,
        "adapter_weight_sha256": hashes["phase4_warmstart_v2"],
        "weight_dtype": "float16",
    }
    if marker.exists() and not args.force:
        observed = json.loads(marker.read_text(encoding="utf-8"))
        if observed == expected_lineage:
            print(json.dumps({"status": "already_materialized", **observed}, indent=2))
            return
        raise RuntimeError(f"Existing output has different lineage: {args.output}")
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        raise RuntimeError(f"Refusing to overwrite non-empty output: {args.output}")

    adapter_path = Path(snapshot_download(
        repo_id=adapter_repo, revision=adapter_revision, token=token,
    ))
    weight_path = adapter_path / "adapter_model.safetensors"
    observed_hash = file_sha256(weight_path)
    if observed_hash != hashes["phase4_warmstart_v2"]:
        raise RuntimeError(f"V2 adapter hash mismatch: {observed_hash}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_repo, revision=base_revision, token=token,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_repo,
        revision=base_revision,
        token=token,
        torch_dtype="float16",
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        is_trainable=False,
        device_map={"": "cpu"},
    )
    model = model.merge_and_unload(safe_merge=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(args.output)
    marker.write_text(json.dumps(expected_lineage, indent=2) + "\n", encoding="utf-8")

    report = {
        "schema_version": "phase5_materialized_v2_v1",
        "status": "complete",
        "output": str(args.output),
        "lineage": expected_lineage,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
