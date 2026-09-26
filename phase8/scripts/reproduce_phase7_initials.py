"""Exploratory same-seed replay of Phase 7's 80 protected wrong first answers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402
from src.data.verifiers.math import _extract_answer  # noqa: E402
from src.data.verifiers.math import MathVerifier  # noqa: E402


BASE = "Kxck/Self_Correction_v1"
REVISION = "6437f947999168a0ce2a98a86e4252fc77160a33"
OUT = ROOT / "outputs/phase8_phase7_replay_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
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
    old_path = ROOT / "outputs/phase7_initials_v1/initial_rollouts.jsonl"
    old_summary = json.loads(old_path.with_name("summary.json").read_text(encoding="utf-8"))
    if sha256(old_path) != old_summary["rollouts_sha256"]:
        raise RuntimeError("Historical initial answers changed")
    old = {row["problem_id"]: row for row in read_jsonl(old_path)}
    source_path = ROOT / "phase7/data/candidates_v1/candidate_problems.jsonl"
    sources = {row["id"]: row for row in read_jsonl(source_path)}
    ids_path = ROOT / "phase7/data/split_v1/protected_ids.json"
    split_report = json.loads((ROOT / "phase7/data/split_v1/split_report.json").read_text(encoding="utf-8"))
    if sha256(ids_path) != split_report["protected_ids_sha256"]:
        raise RuntimeError("Protected IDs changed")
    ids = [pid for pid in json.loads(ids_path.read_text(encoding="utf-8"))
           if not old[pid]["initial_correct"]]
    if len(ids) != 80:
        raise RuntimeError("Expected exactly 80 historically wrong protected IDs")
    settings = {"schema_version": "phase8_phase7_same_seed_replay_v1",
                "historical_answers_sha256": sha256(old_path),
                "protected_ids_sha256": sha256(ids_path),
                "temperature": 0.7, "top_p": 1.0, "top_k": -1,
                "max_new_tokens": 768, "max_model_len": 4096,
                "model": BASE, "revision": REVISION, "gpu": torch.cuda.get_device_name(0),
                "analysis_status": "exploratory_prior_protected_already_open"}
    OUT.mkdir(parents=True, exist_ok=True)
    audit = OUT / "audit.jsonl"
    if audit.exists():
        entries = read_jsonl(audit)
        if not entries or entries[0] != {"type": "metadata", "settings": settings}:
            raise RuntimeError("Replay audit settings differ")
        done = entries[1:]
        for index, entry in enumerate(done):
            if entry["index"] != index or entry["row"]["problem_id"] != ids[index]:
                raise RuntimeError("Replay audit is not an ordered prefix")
    else:
        append(audit, {"type": "metadata", "settings": settings})
        done = []
    if len(done) < len(ids):
        tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
        llm = LLM(model=BASE, revision=REVISION, tokenizer=BASE,
                  tokenizer_revision=REVISION, dtype="half", max_model_len=4096,
                  gpu_memory_utilization=0.85, enforce_eager=True, trust_remote_code=False)
        for index in range(len(done), len(ids)):
            pid = ids[index]
            problem = Problem(id=pid, domain="math", question=sources[pid]["question"])
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": build_prompt(problem)}],
                tokenize=False, add_generation_prompt=True)
            params = SamplingParams(temperature=0.7, top_p=1.0, top_k=-1,
                                    max_tokens=768, seed=old[pid]["request_seed"])
            result = llm.generate([rendered], params, use_tqdm=False)[0].outputs[0]
            row = {"problem_id": pid, "index": index,
                   "request_seed": old[pid]["request_seed"], "output": result.text,
                   "generated_tokens": len(result.token_ids)}
            append(audit, {"type": "completion", "index": index, "row": row})
            done.append({"row": row})
            if len(done) % 20 == 0:
                print(f"replay: {len(done)}/80", flush=True)
    exact = sum(entry["row"]["output"] == old[entry["row"]["problem_id"]]["initial_output"]
                for entry in done)
    final = sum((parsed := _extract_answer(entry["row"]["output"])) is not None and
                parsed == _extract_answer(old[entry["row"]["problem_id"]]["initial_output"])
                for entry in done)
    verifier = MathVerifier()
    replay_correct = sum(verifier.verify(Problem(
        id=entry["row"]["problem_id"], domain="math",
        question=sources[entry["row"]["problem_id"]]["question"],
        reference_answer=sources[entry["row"]["problem_id"]]["reference_answer"]),
        entry["row"]["output"]).passed for entry in done)
    answers = OUT / "answers.jsonl"
    answers.write_text("".join(json.dumps(entry["row"], ensure_ascii=False, sort_keys=True) + "\n"
                               for entry in done), encoding="utf-8", newline="\n")
    summary = {"status": "complete", "rows": len(done), "exact_full_text_matches": exact,
               "parsed_final_answer_matches": final,
               "replay_correct_among_historically_wrong": replay_correct,
               "audit_sha256": sha256(audit), "answers_sha256": sha256(answers),
               "settings": settings}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                       encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
