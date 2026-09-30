"""Generate the shared natural first answer (original solver, temperature 0.7)."""

from __future__ import annotations

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

POOL = ROOT / "phase9/data/source_pool_v1"
OUT = ROOT / "outputs/phase9_first_v1"
MODEL = "Kxck/Self_Correction_v1"
REVISION = "6437f947999168a0ce2a98a86e4252fc77160a33"
BASE_SEED = 20260930


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources() -> list[dict]:
    """Development rows first, then protected rows, each checked against the frozen report."""
    report = json.loads((POOL / "report.json").read_text(encoding="utf-8"))
    rows = []
    for split in ("development", "protected"):
        path = POOL / f"{split}.jsonl"
        if sha256(path) != report[f"{split}_sha256"]:
            raise RuntimeError(f"Frozen Phase 9 {split} pool changed")
        rows += [dict(json.loads(line), split=split)
                 for line in path.read_text(encoding="utf-8").splitlines() if line]
    return rows


def main() -> None:
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if sys.version_info[:2] != (3, 10) or vllm.__version__ != "0.7.0":
        raise RuntimeError("Python 3.10 and vLLM 0.7.0 required")
    rows = sources()
    settings = {"schema_version": "phase9_first_v1", "model": MODEL, "revision": REVISION,
                "temperature": 0.7, "top_p": 1.0, "top_k": -1, "max_new_tokens": 768,
                "max_model_len": 4096, "base_seed": BASE_SEED, "batch_size": BATCH_SIZE,
                "gpu": torch.cuda.get_device_name(0), "gold_labels_opened": False}
    audit = OUT / "audit.jsonl"
    done = open_audit(audit, settings, [row["id"] for row in rows], lambda row: row["problem_id"])
    if len(done) < len(rows):
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        llm = LLM(model=MODEL, revision=REVISION, tokenizer=MODEL, tokenizer_revision=REVISION,
                  dtype="half", max_model_len=4096, gpu_memory_utilization=0.85,
                  enforce_eager=True, trust_remote_code=False)
        requests = []
        for index, row in enumerate(rows):
            problem = Problem(id=row["id"], domain="math", question=row["question"])
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": build_prompt(problem)}], tokenize=False,
                add_generation_prompt=True)
            requests.append((rendered, SamplingParams(temperature=0.7, top_p=1.0, top_k=-1,
                                                      max_tokens=768, seed=BASE_SEED + index)))

        def make_row(index: int, result) -> dict:
            return {"problem_id": rows[index]["id"], "split": rows[index]["split"],
                    "index": index, "request_seed": BASE_SEED + index, "output": result.text,
                    "generated_tokens": len(result.token_ids),
                    "hit_token_cap": len(result.token_ids) >= 768}

        done = generate_ordered(llm, requests, len(done), audit, make_row, "first", done=done)
    answers = OUT / "answers.jsonl"
    answers.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                               for row in done), encoding="utf-8", newline="\n")
    summary = {"status": "complete", "rows": len(done), "audit_sha256": sha256(audit),
               "answers_sha256": sha256(answers), "settings": settings}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                      encoding="utf-8", newline="\n")
    print(json.dumps({k: summary[k] for k in ("status", "rows")}))


if __name__ == "__main__":
    main()
