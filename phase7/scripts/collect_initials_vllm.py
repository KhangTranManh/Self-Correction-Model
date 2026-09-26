"""Collect Phase 7 natural first answers with vLLM 0.7.0 on a V100 32 GB."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase7.scripts.collect_initials_4bit import (
    append_durable, build_prompt, MathVerifier, Problem,
    read_jsonl, read_or_create_audit, sha256, write_atomic,
)


CONFIG = ROOT / "phase7/configs/blind_resolve_v1.yaml"
REGISTRY = ROOT / "phase7/configs/experiments.yaml"
LOCK = ROOT / "phase7/data/protocol/initial_collection_v1_lock.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/phase7_initials_v1")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use CPython 3.10 for Phase 7 math verification")

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    import sympy
    if sympy.__version__ != "1.14.0":
        raise RuntimeError("Use SymPy 1.14.0 for the frozen Phase 7 verifier")
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for relative, expected in lock["files"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Frozen initial-collection protocol hash mismatch: {relative}")
    if config["status"] != "initial_collection_frozen":
        raise ValueError("Initial-collection protocol is not frozen")
    if config["generation_draft"]["backend"] != "vllm_0_7_0" or vllm.__version__ != "0.7.0":
        raise RuntimeError("Phase 7 requires pinned vLLM 0.7.0")
    source_plan = config["source_plan"]
    candidates_path = ROOT / source_plan["candidate_manifest"]
    if sha256(candidates_path) != source_plan["candidate_manifest_sha256"]:
        raise ValueError("Frozen Phase 7 candidate manifest hash mismatch")
    candidates = read_jsonl(candidates_path)
    ceiling = int(source_plan["initial_generation_ceiling"])
    quota = int(source_plan["stop_at_first_prefix_with_each_class_at_least"])
    ids = [str(row["id"]) for row in candidates]
    if len(candidates) != ceiling or len(set(ids)) != ceiling:
        raise ValueError("Candidate count or IDs differ from frozen pool")
    model_info = registry["models"]["original_solver"]
    model_id, revision = model_info["repo"], model_info["revision"]
    seed = int(config["seed"])
    decoding = config["initial_answer"]["decoding_draft"]
    max_new_tokens = int(decoding["max_new_tokens"])
    max_model_len = int(decoding["max_model_len"])
    settings = {
        "schema_version": "phase7_initials_v1", "candidate_manifest_sha256": sha256(candidates_path),
        "model": model_id, "revision": revision, "backend": "vllm_0_7_0",
        "dtype": "float16", "quantization": "none", "seed": seed,
        "temperature": float(decoding["temperature"]), "top_p": float(decoding["top_p"]),
        "top_k": int(decoding["top_k"]), "max_new_tokens": max_new_tokens,
        "max_model_len": max_model_len, "stop_correct_and_wrong": quota,
        "ceiling": ceiling, "python": sys.version.split()[0],
        "sympy": sympy.__version__, "torch": torch.__version__,
        "vllm": vllm.__version__, "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "prompt_source": "phase1/src/core/prompts.py::build_prompt(math)",
        "verifier_feedback_visible": False,
        "protocol_lock_sha256": sha256(LOCK),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "initial_rollouts.audit.jsonl"
    completed = read_or_create_audit(audit_path, settings, ids)
    counts = Counter()
    for index, row in enumerate(completed):
        if min(counts["correct"], counts["wrong"]) >= quota:
            raise ValueError(f"Existing audit continues beyond first quota prefix at {index}")
        counts["correct" if row["initial_correct"] else "wrong"] += 1
    if min(counts["correct"], counts["wrong"]) < quota and len(completed) < ceiling:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        llm = LLM(
            model=model_id, revision=revision,
            tokenizer=model_id, tokenizer_revision=revision,
            dtype="half", max_model_len=max_model_len,
            gpu_memory_utilization=float(config["generation_draft"]["gpu_memory_utilization"]),
            enforce_eager=True, trust_remote_code=False,
        )
        verifier = MathVerifier()
        for index in range(len(completed), ceiling):
            source = candidates[index]
            problem = Problem(id=source["id"], domain="math",
                              question=source["question"],
                              reference_answer=source["reference_answer"])
            messages = [{"role": "user", "content": build_prompt(problem)}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True)
            if len(tokenizer.encode(rendered, add_special_tokens=False)) + max_new_tokens > max_model_len:
                raise ValueError(f"Context budget exceeded by {problem.id}")
            request_seed = seed + index
            sampling = SamplingParams(
                temperature=float(decoding["temperature"]),
                top_p=float(decoding["top_p"]), top_k=int(decoding["top_k"]),
                max_tokens=max_new_tokens, seed=request_seed,
            )
            result = llm.generate([rendered], sampling, use_tqdm=False)[0].outputs[0]
            output = result.text
            verdict = verifier.verify(problem, output)
            row = {
                "problem_id": problem.id, "index": index, "request_seed": request_seed,
                "initial_output": output, "initial_correct": bool(verdict.passed),
                "initial_verifier_detail": str(verdict.detail),
                "generated_tokens": len(result.token_ids),
                "hit_token_cap": len(result.token_ids) >= max_new_tokens,
            }
            append_durable(audit_path, {"type": "completion", "index": index, "row": row})
            completed.append(row)
            counts["correct" if row["initial_correct"] else "wrong"] += 1
            if len(completed) % args.progress_every == 0:
                print(f"SAVED {len(completed)}/{ceiling} correct={counts['correct']} wrong={counts['wrong']}", flush=True)
            if min(counts["correct"], counts["wrong"]) >= quota:
                break

    if min(counts["correct"], counts["wrong"]) < quota:
        raise RuntimeError(f"Quota infeasible after {len(completed)} candidates: {dict(counts)}")
    rollout_path = args.output_dir / "initial_rollouts.jsonl"
    write_atomic(rollout_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                   for row in completed))
    summary = {
        "schema_version": "phase7_initials_summary_v1", "status": "quota_met",
        "completed": len(completed), "correct": counts["correct"],
        "wrong": counts["wrong"], "audit_sha256": sha256(audit_path),
        "rollouts_sha256": sha256(rollout_path), "settings": settings,
    }
    write_atomic(args.output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("status", "completed", "correct", "wrong", "audit_sha256")}, indent=2))


if __name__ == "__main__":
    main()
