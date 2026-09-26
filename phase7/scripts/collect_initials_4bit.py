"""Collect Phase 7 natural initial answers on one 16 GB-class CUDA GPU.

The run is append-only and resumable. It stops at the first ordered prefix
containing at least 100 verifier-correct and 100 verifier-wrong answers.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
from src.core.prompts import build_prompt
from src.core.schema import Problem
from src.data.verifiers.math import MathVerifier


CONFIG = ROOT / "phase7/configs/blind_resolve_v1.yaml"
REGISTRY = ROOT / "phase7/configs/experiments.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_durable(path: Path, record: dict) -> None:
    with path.open("ab") as stream:
        stream.write((json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def read_or_create_audit(path: Path, settings: dict, ids: list[str]) -> list[dict]:
    if not path.exists():
        append_durable(path, {"type": "metadata", "settings": settings})
        return []
    audit = read_jsonl(path)
    if not audit or audit[0] != {"type": "metadata", "settings": settings}:
        raise ValueError("Existing audit has different run settings")
    completed = []
    for index, record in enumerate(audit[1:]):
        row = record.get("row")
        if (index >= len(ids) or record.get("type") != "completion"
                or record.get("index") != index or not isinstance(row, dict)
                or row.get("problem_id") != ids[index]):
            raise ValueError("Existing audit is not a valid ordered prefix")
        completed.append(row)
    return completed


def write_atomic(path: Path, data: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(data, encoding="utf-8", newline="\n")
    temporary.replace(path)


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
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("This collector expects exactly one visible CUDA GPU")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    source_plan = config["source_plan"]
    generation = config["generation_draft"]
    if generation["backend"] != "transformers_bitsandbytes_4bit":
        raise ValueError("Unexpected Phase 7 generation backend")
    candidates_path = ROOT / source_plan["candidate_manifest"]
    if sha256(candidates_path) != source_plan["candidate_manifest_sha256"]:
        raise ValueError("Frozen Phase 7 candidate manifest hash mismatch")
    candidates = read_jsonl(candidates_path)
    ceiling = int(source_plan["initial_generation_ceiling"])
    quota = int(source_plan["stop_at_first_prefix_with_each_class_at_least"])
    ids = [str(row["id"]) for row in candidates]
    if len(candidates) != ceiling or len(set(ids)) != ceiling:
        raise ValueError("Candidate count or IDs differ from the frozen pool")
    model_info = registry["models"]["original_solver"]
    model_id, revision = model_info["repo"], model_info["revision"]
    seed = int(config["seed"])
    initial_decoding = config["initial_answer"]["decoding_draft"]
    max_new_tokens = int(initial_decoding["max_new_tokens"])
    max_model_len = int(initial_decoding["max_model_len"])
    temperature = float(initial_decoding["temperature"])
    top_p = float(initial_decoding["top_p"])
    top_k = int(initial_decoding["top_k"])
    settings = {
        "schema_version": "phase7_initials_v1",
        "candidate_manifest_sha256": sha256(candidates_path),
        "model": model_id, "revision": revision,
        "quantization": "bitsandbytes_nf4_double_quant",
        "compute_dtype": "float16", "seed": seed,
        "temperature": temperature, "top_p": top_p, "top_k": top_k,
        "max_new_tokens": max_new_tokens, "max_model_len": max_model_len,
        "stop_correct_and_wrong": quota, "ceiling": ceiling,
        "prompt_source": "phase1/src/core/prompts.py::build_prompt(math)",
        "python": sys.version.split()[0], "sympy": sympy.__version__,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "verifier_feedback_visible": False,
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
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, quantization_config=quantization,
            device_map={"": 0}, torch_dtype=torch.float16,
        )
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        verifier = MathVerifier()
        for index in range(len(completed), ceiling):
            source = candidates[index]
            problem = Problem(id=source["id"], domain="math",
                              question=source["question"],
                              reference_answer=source["reference_answer"])
            prompt = build_prompt(problem)
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt",
            ).to("cuda:0")
            input_tokens = int(encoded["input_ids"].shape[-1])
            if input_tokens + max_new_tokens > max_model_len:
                raise ValueError(f"Context budget exceeded by {problem.id}")
            request_seed = seed + index
            torch.manual_seed(request_seed)
            torch.cuda.manual_seed_all(request_seed)
            with torch.inference_mode():
                result = model.generate(
                    **encoded, do_sample=True, temperature=temperature,
                    top_p=top_p, top_k=top_k, max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_ids = result[0, input_tokens:]
            output = tokenizer.decode(new_ids, skip_special_tokens=True)
            verdict = verifier.verify(problem, output)
            row = {
                "problem_id": problem.id, "index": index, "request_seed": request_seed,
                "initial_output": output, "initial_correct": bool(verdict.passed),
                "initial_verifier_detail": str(verdict.detail),
                "generated_tokens": int(len(new_ids)),
                "hit_token_cap": len(new_ids) >= max_new_tokens,
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
