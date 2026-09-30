"""Regenerate Phase 7 natural first answers as the Phase 8 distractor donor pool.

Amendment v2: the historical outputs/phase7_initials_v1 rollouts were lost, so
Phase 8 regenerates them with the frozen Phase 7 prompt, candidate order,
per-request seeds (20260924 + index), temperature 0.7, and 100/100 first-prefix
stop rule, batched. These rows are donor text only; they are NOT the historical
Phase 7 answers and must never be reported as Phase 7 results.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase7.scripts.collect_initials_4bit import (  # noqa: E402
    build_prompt, MathVerifier, Problem, read_jsonl, sha256, write_atomic,
)
from phase8.scripts.batched import BATCH_SIZE, generate_ordered, open_audit  # noqa: E402

CONFIG = ROOT / "phase7/configs/blind_resolve_v1.yaml"
REGISTRY = ROOT / "phase7/configs/experiments.yaml"
OUT = ROOT / "outputs/phase7_initials_regen_v2"


def main() -> None:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use CPython 3.10")
    import sympy
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if sympy.__version__ != "1.14.0" or vllm.__version__ != "0.7.0":
        raise RuntimeError("Pinned SymPy 1.14.0 and vLLM 0.7.0 required")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    plan = config["source_plan"]
    candidates_path = ROOT / plan["candidate_manifest"]
    if sha256(candidates_path) != plan["candidate_manifest_sha256"]:
        raise RuntimeError("Frozen Phase 7 candidate manifest changed")
    candidates = read_jsonl(candidates_path)
    quota = int(plan["stop_at_first_prefix_with_each_class_at_least"])
    model = registry["models"]["original_solver"]
    decoding = config["initial_answer"]["decoding_draft"]
    seed = int(config["seed"])
    settings = {
        "schema_version": "phase8_phase7_donor_regen_v2",
        "candidate_manifest_sha256": sha256(candidates_path),
        "model": model["repo"], "revision": model["revision"], "dtype": "float16",
        "seed": seed, "temperature": float(decoding["temperature"]),
        "top_p": float(decoding["top_p"]), "top_k": int(decoding["top_k"]),
        "max_new_tokens": int(decoding["max_new_tokens"]),
        "max_model_len": int(decoding["max_model_len"]),
        "stop_correct_and_wrong": quota, "batch_size": BATCH_SIZE,
        "gpu": torch.cuda.get_device_name(0),
    }
    audit = OUT / "initial_rollouts.audit.jsonl"
    ids = [row["id"] for row in candidates]
    done = open_audit(audit, settings, ids, lambda row: row["problem_id"])

    def met(rows: list[dict]) -> bool:
        counts = Counter(row["initial_correct"] for row in rows)
        return min(counts[True], counts[False]) >= quota

    if not met(done):
        tokenizer = AutoTokenizer.from_pretrained(model["repo"], revision=model["revision"])
        llm = LLM(model=model["repo"], revision=model["revision"], tokenizer=model["repo"],
                  tokenizer_revision=model["revision"], dtype="half",
                  max_model_len=settings["max_model_len"], gpu_memory_utilization=0.85,
                  enforce_eager=True, trust_remote_code=False)
        verifier = MathVerifier()
        problems, requests = [], []
        for index, source in enumerate(candidates):
            problem = Problem(id=source["id"], domain="math", question=source["question"],
                              reference_answer=source["reference_answer"])
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": build_prompt(problem)}], tokenize=False,
                add_generation_prompt=True)
            problems.append(problem)
            requests.append((rendered, SamplingParams(
                temperature=settings["temperature"], top_p=settings["top_p"],
                top_k=settings["top_k"], max_tokens=settings["max_new_tokens"],
                seed=seed + index)))

        def make_row(index: int, output) -> dict:
            verdict = verifier.verify(problems[index], output.text)
            return {"problem_id": problems[index].id, "index": index,
                    "request_seed": seed + index, "initial_output": output.text,
                    "initial_correct": bool(verdict.passed),
                    "initial_verifier_detail": str(verdict.detail),
                    "generated_tokens": len(output.token_ids),
                    "hit_token_cap": len(output.token_ids) >= settings["max_new_tokens"]}

        done = generate_ordered(llm, requests, len(done), audit, make_row, "phase7_donors",
                                stop=met, done=done)
    if not met(done):
        raise RuntimeError(f"Quota infeasible after {len(done)} candidates")
    counts = Counter(row["initial_correct"] for row in done)
    rollouts = OUT / "initial_rollouts.jsonl"
    write_atomic(rollouts, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                   for row in done))
    summary = {"schema_version": "phase8_phase7_donor_regen_summary_v2", "status": "quota_met",
               "completed": len(done), "correct": counts[True], "wrong": counts[False],
               "audit_sha256": sha256(audit), "rollouts_sha256": sha256(rollouts),
               "settings": settings}
    write_atomic(OUT / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("status", "completed", "correct", "wrong")}))


if __name__ == "__main__":
    main()
