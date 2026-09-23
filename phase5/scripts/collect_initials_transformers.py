"""Resume-safe Phase 5 initial-answer collection on a single CUDA GPU.

Uses the frozen Phase 1 math prompt. No hint or verifier outcome is shown to
the model. Designed for a Windows CUDA GPU after gpu_preflight.py passes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
from src.core.prompts import build_prompt
from src.core.schema import Problem
from src.data.verifiers.math import MathVerifier


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_durable(path: Path, record: dict):
    with path.open("ab") as stream:
        stream.write((json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def read_audit(path: Path, expected: dict, candidate_ids: list[str]):
    if not path.exists():
        append_durable(path, {"type": "metadata", "settings": expected})
        return []
    records = read_jsonl(path)
    if not records or records[0] != {"type": "metadata", "settings": expected}:
        raise ValueError("Existing audit settings differ from this run")
    rows = []
    for index, entry in enumerate(records[1:]):
        if index >= len(candidate_ids):
            raise ValueError("Audit exceeds frozen candidate count")
        if entry.get("type") != "completion" or entry.get("index") != index:
            raise ValueError("Audit is not a complete ordered prefix")
        row = entry.get("row")
        if not isinstance(row, dict) or row.get("problem_id") != candidate_ids[index]:
            raise ValueError("Audit problem order differs from frozen candidates")
        rows.append(row)
    return rows


def resolve_model_revision(name: str, revision: str | None) -> str:
    if revision:
        return revision
    from huggingface_hub import model_info
    return str(model_info(name).sha)


def main():
    # The project-root .env is machine-local and is never part of transfer archives.
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path,
                        default=Path("phase5/data/candidates_v1/candidate_problems.jsonl"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("phase5/data/initials_v1"))
    parser.add_argument("--model", default="Kxck/Self_Correction_v1")
    parser.add_argument("--revision", help="Exact Hugging Face commit; resolved before generation if omitted")
    parser.add_argument("--quantization", choices=("auto", "none", "four_bit"), default="auto")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--min-per-class", type=int, default=240)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use CPython 3.10 to match the frozen math verifier")
    import sympy
    if sympy.__version__ != "1.14.0":
        raise RuntimeError("Use SymPy 1.14.0 to match the frozen math verifier")
    if args.max_new_tokens <= 0 or args.min_per_class <= 0 or args.temperature <= 0:
        raise ValueError("Generation budget, class floor and temperature must be positive")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import transformers
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    rows = read_jsonl(args.candidates)
    ids = [str(row["id"]) for row in rows]
    if len(rows) != 1200 or len(set(ids)) != len(ids):
        raise ValueError("Expected the frozen 1,200 unique candidates")
    selected_hash = sha256(args.candidates)
    if selected_hash != "3dc103e168ea149f51d16e5d22fad66e97b1af9e956fba1d8914b714120e954a":
        raise ValueError("Frozen candidate manifest hash mismatch")
    revision = resolve_model_revision(args.model, args.revision)
    memory_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    quantization = ("four_bit" if memory_gib < 22 else "none") if args.quantization == "auto" else args.quantization
    # PyTorch's default BF16 check includes software emulation on Volta.
    weight_dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    settings = {
        "schema_version": "phase5_initials_v1", "candidates_sha256": selected_hash,
        "model": args.model, "model_revision": revision,
        "quantization": quantization, "weight_dtype": str(weight_dtype).removeprefix("torch."), "seed": args.seed,
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "top_p": 1.0, "top_k": 0, "min_per_class": args.min_per_class,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "sympy": sympy.__version__, "python": sys.version.split()[0],
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_gib": round(memory_gib, 2),
        "prompt_source": "phase1/src/core/prompts.py::build_prompt(math)",
        "verifier_feedback_visible": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = args.output_dir / "initial_rollouts.audit.jsonl"
    completed = read_audit(audit, settings, ids)
    counts = Counter("correct" if row["initial_correct"] else "wrong" for row in completed)
    if len(completed) == len(rows) or min(counts["correct"], counts["wrong"]) >= args.min_per_class:
        print("Existing audit already satisfies frozen stop rule", flush=True)
    else:
        quant_config = (BitsAndBytesConfig(load_in_4bit=True,
                         bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=weight_dtype)
                        if quantization == "four_bit" else None)
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision,
                                                   use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=revision, device_map="auto", torch_dtype=weight_dtype,
            quantization_config=quant_config,
        )
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        verifier = MathVerifier()
        for index in range(len(completed), len(rows)):
            source = rows[index]
            problem = Problem(id=source["id"], domain="math", question=source["question"],
                              reference_answer=source["reference_answer"])
            prompt = build_prompt(problem)
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt",
            ).to(model.device)
            request_seed = args.seed + index
            torch.manual_seed(request_seed)
            torch.cuda.manual_seed_all(request_seed)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, do_sample=True, temperature=args.temperature,
                    top_p=1.0, top_k=0, max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_ids = generated[0, encoded["input_ids"].shape[-1]:]
            output = tokenizer.decode(new_ids, skip_special_tokens=True)
            verdict = verifier.verify(problem, output)
            row = {
                "problem_id": problem.id, "index": index, "request_seed": request_seed,
                "initial_output": output, "initial_correct": bool(verdict.passed),
                "initial_verifier_detail": str(verdict.detail),
                "generated_tokens": int(len(new_ids)),
                "hit_token_cap": len(new_ids) >= args.max_new_tokens,
            }
            append_durable(audit, {"type": "completion", "index": index, "row": row})
            completed.append(row)
            counts["correct" if row["initial_correct"] else "wrong"] += 1
            if len(completed) % args.progress_every == 0:
                print(f"SAVED {len(completed)}/{len(rows)} correct={counts['correct']} wrong={counts['wrong']}", flush=True)
            if min(counts["correct"], counts["wrong"]) >= args.min_per_class:
                print("STOP class quotas met", flush=True)
                break
    output = args.output_dir / "initial_rollouts.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                              for row in completed), encoding="utf-8", newline="\n")
    report = {"settings": settings, "rows": len(completed),
              "correct": counts["correct"], "wrong": counts["wrong"],
              "quota_met": min(counts["correct"], counts["wrong"]) >= args.min_per_class,
              "audit_sha256": sha256(audit), "initial_rollouts_sha256": sha256(output),
              "training_or_review_performed": False}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
