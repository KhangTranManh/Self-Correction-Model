"""Prepare GPU generation jobs for missing source-disjoint GSM8K REVISE rows."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase3.lib.behavior import ProvenanceResolver, build_prompt, problem_for_prompt


ROOT = Path(__file__).resolve().parents[2]
SEED = 20260907


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_ids(path: Path) -> set[str]:
    return {str(row.get("source_id", row.get("id"))) for row in read_jsonl(path)}


def rank(row: dict) -> tuple:
    # WC is a harder source: Base failed while V1 passed. Prefer it without
    # leaking any frozen/probe outcome into selection.
    bucket_priority = 0 if row["bucket"] == "WC" else 1
    digest = hashlib.sha256(f"{SEED}|calibration-generation|{row['id']}".encode()).hexdigest()
    return bucket_priority, digest, str(row["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default=str(ROOT / "data/behavior/unified_source_inventory.jsonl"))
    parser.add_argument("--probe", default=str(ROOT / "runs/representation_probe/probe_dataset.jsonl"))
    parser.add_argument("--frozen", default=str(ROOT / "data/two_stage_selective_repair/frozen_eval.jsonl"))
    parser.add_argument("--decision-train", default=str(ROOT / "data/decision_only/decision_only_train.jsonl"))
    parser.add_argument("--decision-dev", default=str(ROOT / "data/decision_only/decision_only_dev.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "data/router_calibration_v1/generation"))
    parser.add_argument("--sources", type=int, default=150)
    parser.add_argument("--samples-per-source", type=int, default=4)
    args = parser.parse_args()

    inventory_path = Path(args.inventory).resolve()
    protected = source_ids(Path(args.probe)) | source_ids(Path(args.frozen))
    decision_ids = source_ids(Path(args.decision_train)) | source_ids(Path(args.decision_dev))
    candidates = [
        row for row in read_jsonl(inventory_path)
        if row["dataset"] == "gsm8k"
        and bool(row["v1_correct"])
        and row["bucket"] in {"CC", "WC"}
        and str(row["id"]) not in protected
        and str(row["id"]) not in decision_ids
    ]
    candidates.sort(key=rank)
    selected = candidates[: args.sources]
    if len(selected) != args.sources:
        raise RuntimeError(f"Need {args.sources} eligible unseen sources, found {len(selected)}")
    resolver = ProvenanceResolver()
    manifest = []
    for meta in selected:
        source_id = str(meta["id"])
        source = resolver.resolve(meta["source_ref"], source_id)
        manifest.append({
            "schema_version": "phase3_calibration_generation_manifest_v1",
            "task_id": f"calibration_gsm8k::{source_id}",
            "pair_id": f"calibration_gsm8k::{source_id}",
            "source_id": source_id,
            "problem_ref": str(meta["source_ref"]),
            "dataset": "gsm8k",
            "domain": "math",
            "bucket": str(meta["bucket"]),
            "model_origin_to_sample": "self_correction_v1",
            "samples_requested": args.samples_per_source,
            "problem_message": {"role": "user", "content": build_prompt(problem_for_prompt(source))},
            "selection_reason": "source-disjoint hard-source stochastic resampling for natural REVISE mining",
        })
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "generation_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": "phase3_calibration_generation_summary_v1",
        "sources": len(manifest),
        "samples_per_source": args.samples_per_source,
        "total_generation_jobs": len(manifest) * args.samples_per_source,
        "dataset": "gsm8k",
        "model_origin": "self_correction_v1",
        "selection_bucket": dict(sorted(Counter(row["bucket"] for row in manifest).items())),
        "probe_256_overlap": len({row["source_id"] for row in manifest} & source_ids(Path(args.probe))),
        "frozen_200_overlap": len({row["source_id"] for row in manifest} & source_ids(Path(args.frozen))),
        "decision_only_sft_overlap": len({row["source_id"] for row in manifest} & decision_ids),
        "recommended_generation": {"temperature": 0.9, "top_p": 0.95, "max_tokens": 1024, "seed": SEED},
        "acceptance": "retain at most one fresh-verifier-failing answer per source; never synthesize or mutate an answer",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "gpu_required_for_manifest": False,
        "gpu_required_for_generation": True,
        "validation": "PASS",
    }
    (output / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
