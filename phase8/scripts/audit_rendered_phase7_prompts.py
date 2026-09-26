"""Compare pinned-tokenizer rendered Phase 7 initial and blind prompts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402
sys.path.insert(0, str(ROOT))
from phase7.scripts.paired_prompts import paired_prompt  # noqa: E402


def main() -> None:
    revision = "6437f947999168a0ce2a98a86e4252fc77160a33"
    tokenizer = AutoTokenizer.from_pretrained("Kxck/Self_Correction_v1", revision=revision)
    source = ROOT / "phase7/data/candidates_v1/candidate_problems.jsonl"
    initial = ROOT / "outputs/phase7_initials_v1/initial_rollouts.jsonl"
    sources = {row["id"]: row for row in
               (json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line)}
    rows = [json.loads(line) for line in initial.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 334:
        raise RuntimeError("Expected 334 initial answers")
    same_rendered, same_tokens = 0, 0
    initial_digest, blind_digest = hashlib.sha256(), hashlib.sha256()
    for row in rows:
        source_row = sources[row["problem_id"]]
        problem = Problem(id=row["problem_id"], domain="math", question=source_row["question"])
        first = tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt(problem)}],
            tokenize=False, add_generation_prompt=True)
        blind = tokenizer.apply_chat_template(
            [{"role": "user", "content": paired_prompt(problem, "blind_resolve")}],
            tokenize=False, add_generation_prompt=True)
        same_rendered += first == blind
        same_tokens += tokenizer.encode(first, add_special_tokens=False) == tokenizer.encode(
            blind, add_special_tokens=False)
        initial_digest.update(first.encode("utf-8") + b"\0")
        blind_digest.update(blind.encode("utf-8") + b"\0")
    report = {"schema_version": "phase8_phase7_rendered_prompt_audit_v1",
              "tokenizer_model": "Kxck/Self_Correction_v1", "revision": revision,
              "rows": len(rows), "same_rendered_prompt": same_rendered,
              "same_token_ids": same_tokens,
              "initial_rendered_sequence_sha256": initial_digest.hexdigest(),
              "blind_rendered_sequence_sha256": blind_digest.hexdigest()}
    output = ROOT / "phase8/data/phase7_rendered_prompt_audit_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
