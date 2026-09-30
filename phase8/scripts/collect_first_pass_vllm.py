"""Collect Phase 8 original-solver first passes without opening gold labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402
sys.path.insert(0, str(ROOT))
from phase8.scripts.batched import BATCH_SIZE, generate_ordered, open_audit  # noqa: E402


SOURCE = ROOT / "phase8/data/fresh_source_pool_v1/candidate_problems.jsonl"
REPORT = SOURCE.with_name("candidate_report.json")
MODEL = "Kxck/Self_Correction_v1"
REVISION = "6437f947999168a0ce2a98a86e4252fc77160a33"
BASE_SEED = 20260925
STAGES = {
    "initial": (0.7, 0),
    "sample_repeat": (0.7, 100000),
    "greedy_same_prompt": (0.0, 200000),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/phase8_first_pass_v2")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use Python 3.10")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available() or vllm.__version__ != "0.7.0":
        raise RuntimeError("Pinned vLLM 0.7.0 and CUDA required")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if sha256(SOURCE) != report["candidate_manifest_sha256"]:
        raise RuntimeError("Frozen Phase 8 source manifest changed")
    sources = read_jsonl(SOURCE)
    if len(sources) != 400 or len({row["id"] for row in sources}) != 400:
        raise RuntimeError("Expected 400 unique protected source IDs")
    temperature, seed_offset = STAGES[args.stage]
    settings = {
        "schema_version": "phase8_first_pass_v2", "stage": args.stage,
        "batch_size": BATCH_SIZE,
        "source_sha256": sha256(SOURCE), "model": MODEL, "revision": REVISION,
        "backend": "vllm_0_7_0", "dtype": "float16", "temperature": temperature,
        "top_p": 1.0, "top_k": -1, "base_seed": BASE_SEED,
        "seed_offset": seed_offset, "max_new_tokens": 768,
        "max_model_len": 4096, "gpu": torch.cuda.get_device_name(0),
        "gold_labels_opened": False,
    }
    output_dir = args.output_root / args.stage
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = output_dir / "audit.jsonl"
    completed = open_audit(audit, settings, [row["id"] for row in sources],
                           lambda row: row["problem_id"])
    if len(completed) < len(sources):
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        llm = LLM(model=MODEL, revision=REVISION, tokenizer=MODEL,
                  tokenizer_revision=REVISION, dtype="half", max_model_len=4096,
                  gpu_memory_utilization=0.85, enforce_eager=True,
                  trust_remote_code=False)
        requests = []
        for index, source in enumerate(sources):
            problem = Problem(id=source["id"], domain="math", question=source["question"],
                              reference_answer=None)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": build_prompt(problem)}], tokenize=False,
                add_generation_prompt=True)
            if len(tokenizer.encode(rendered, add_special_tokens=False)) + 768 > 4096:
                raise RuntimeError(f"Context exceeds budget: {source['id']}")
            requests.append((rendered, SamplingParams(
                temperature=temperature, top_p=1.0, top_k=-1, max_tokens=768,
                seed=BASE_SEED + seed_offset + index)))

        def make_row(index: int, result) -> dict:
            return {"problem_id": sources[index]["id"], "index": index,
                    "request_seed": BASE_SEED + seed_offset + index, "output": result.text,
                    "generated_tokens": len(result.token_ids),
                    "finish_reason": result.finish_reason,
                    "hit_token_cap": len(result.token_ids) >= 768}

        completed = generate_ordered(llm, requests, len(completed), audit, make_row,
                                     args.stage, done=completed)
    output = output_dir / "answers.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                              for row in completed), encoding="utf-8", newline="\n")
    summary = {"status": "complete", "stage": args.stage, "rows": len(completed),
               "audit_sha256": sha256(audit), "answers_sha256": sha256(output),
               "settings": settings}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                             encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
