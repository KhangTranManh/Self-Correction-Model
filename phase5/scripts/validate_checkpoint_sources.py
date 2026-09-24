"""Validate pinned Phase 5 Hugging Face sources without loading model weights.

This command queries repository metadata and downloads only the small
``adapter_config.json`` files. It never downloads safetensor weights and never
constructs a Transformers or PEFT model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "phase5/configs/experiments.yaml"

EXPECTED_PARENTS = {
    "phase4_warmstart_v2": "/root/models/Kxck_Self_Correction_v1",
    "phase4_correction_sft_v3": "/root/models/phase4_warmstart_v2_merged",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    repos = registry["checkpoint_hub_repos"]
    revisions = registry["checkpoint_hub_revisions"]
    expected_hashes = registry["checkpoint_adapter_sha256"]
    api = HfApi(token=True)

    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    solver_repo = repos["original_solver"]
    solver_revision = revisions["original_solver"]
    solver = api.model_info(solver_repo, revision=solver_revision)
    solver_ok = solver.sha == solver_revision
    checks.append({
        "role": "original_solver",
        "repo_id": solver_repo,
        "revision": solver.sha,
        "revision_matches": solver_ok,
    })
    if not solver_ok:
        failures.append("original solver revision mismatch")

    for role in ("phase4_warmstart_v2", "phase4_correction_sft_v3"):
        repo_id = repos[role]
        revision = revisions[role]
        info = api.model_info(repo_id, revision=revision, files_metadata=True)
        files = {item.rfilename: item for item in info.siblings}
        weights = files.get("adapter_model.safetensors")
        remote_hash = (weights.lfs or {}).get("sha256") if weights else None
        config_path = hf_hub_download(
            repo_id, "adapter_config.json", revision=revision, token=True
        )
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        item = {
            "role": role,
            "repo_id": repo_id,
            "private": info.private,
            "revision": info.sha,
            "revision_matches": info.sha == revision,
            "weight_bytes": weights.size if weights else None,
            "weight_sha256": remote_hash,
            "weight_hash_matches": remote_hash == expected_hashes[role],
            "peft_type": config.get("peft_type"),
            "task_type": config.get("task_type"),
            "base_model_name_or_path": config.get("base_model_name_or_path"),
            "parent_matches": config.get("base_model_name_or_path") == EXPECTED_PARENTS[role],
        }
        checks.append(item)
        if not all((
            item["private"], item["revision_matches"],
            item["weight_hash_matches"], item["parent_matches"],
            item["peft_type"] == "LORA", item["task_type"] == "CAUSAL_LM",
        )):
            failures.append(f"checkpoint invariant failed: {role}")

    report = {
        "schema_version": "phase5_checkpoint_source_audit_v1",
        "metadata_only": True,
        "model_weights_downloaded": False,
        "checks": checks,
        "failures": failures,
        "all_passed": not failures,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload.encode("utf-8"))
    print(payload, end="")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
