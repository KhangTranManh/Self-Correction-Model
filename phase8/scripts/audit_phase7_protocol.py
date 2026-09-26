"""Audit the as-run Phase 7 initial and paired protocol before Phase 8 generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402
sys.path.insert(0, str(ROOT))
from phase7.scripts.paired_prompts import paired_prompt  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = yaml.safe_load((ROOT / "phase7/configs/blind_resolve_v1.yaml").read_text(encoding="utf-8"))
    initial_summary = json.loads((ROOT / "outputs/phase7_initials_v1/summary.json").read_text(encoding="utf-8"))
    initial_path = ROOT / "outputs/phase7_initials_v1/initial_rollouts.jsonl"
    if sha256(initial_path) != initial_summary["rollouts_sha256"]:
        raise RuntimeError("Initial answer hash changed")
    initial = read_jsonl(initial_path)
    protected_ids = json.loads((ROOT / "phase7/data/split_v1/protected_ids.json").read_text(encoding="utf-8"))
    source_path = ROOT / "phase7/data/candidates_v1/candidate_problems.jsonl"
    sources = {row["id"]: row for row in read_jsonl(source_path)}
    if len(initial) != 334 or len(protected_ids) != 160:
        raise RuntimeError("Phase 7 count changed")
    initial_by_id = {row["problem_id"]: row for row in initial}
    if len(initial_by_id) != len(initial):
        raise RuntimeError("Duplicate initial ID")
    wrong = sum(not initial_by_id[pid]["initial_correct"] for pid in protected_ids)
    if wrong != 80:
        raise RuntimeError(f"Expected 80 protected wrong, got {wrong}")
    same, suffix_only = 0, 0
    for pid in initial_by_id:
        source = sources[pid]
        problem = Problem(id=pid, domain="math", question=source["question"],
                          reference_answer=source["reference_answer"])
        first = build_prompt(problem)
        blind = paired_prompt(problem, "blind_resolve")
        same += first == blind
        suffix_only += blind.startswith(first) and blind[len(first):] == "\n\nGive a complete solution from scratch."
    initial_decoding = initial_summary["settings"]
    paired_decoding = config["generation_draft"]
    paired_summaries = {}
    for checkpoint in ("original_solver", "warmstart_v2", "correction_sft_v3"):
        path = ROOT / "outputs/phase7_paired_v1/protected" / checkpoint / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["tasks"] != 320:
            raise RuntimeError(f"Incomplete protected pairs: {checkpoint}")
        paired_summaries[checkpoint] = {
            "summary_sha256": sha256(path),
            "temperature": summary["settings"]["temperature"],
            "max_tokens": summary["settings"]["max_tokens"],
            "gpu": "RTX 3090 24 GB (per Phase 7 final report)",
        }
    report = {
        "schema_version": "phase8_phase7_as_run_protocol_audit_v1",
        "status": "completed_from_locked_local_artifacts",
        "initial_count": len(initial), "initial_correct": sum(row["initial_correct"] for row in initial),
        "initial_wrong": sum(not row["initial_correct"] for row in initial),
        "protected_count": len(protected_ids), "protected_wrong": wrong,
        "initial_decoding": {key: initial_decoding[key] for key in
                             ("temperature", "top_p", "top_k", "seed", "max_new_tokens", "max_model_len")},
        "paired_decoding": {key: paired_decoding.get(key) for key in
                            ("temperature", "top_p", "top_k", "seed", "max_new_tokens", "max_model_len")},
        "prompt_comparison_across_334": {
            "exact_same_user_text": same,
            "blind_equals_initial_plus_fixed_suffix": suffix_only,
            "fixed_suffix": "\n\nGive a complete solution from scratch.",
            "chat_template_calls_both_single_user_and_add_generation_prompt": True,
            "rendered_token_ids_compared": False,
        },
        "protected_pair_summaries": paired_summaries,
        "interpretation": "Initial-vs-second-pass comparisons mix prompt and decoding changes; within-Phase-7 blind-vs-visible comparison keeps paired decoding fixed.",
    }
    out = ROOT / "phase8/data/phase7_protocol_audit_v1.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
