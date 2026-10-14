"""C3 (part 1): one fresh greedy solve per problem with the original solver (vLLM).

Phase 1 math prompt, temperature 0, 768-token cap, one per problem for SVAMP
(phase12) and GSM8K test (phase13). Output: outputs/paper/c3_greedy_<ds>.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402

MODEL = "Kxck/Self_Correction_v1"
REVISION = "6437f947999168a0ce2a98a86e4252fc77160a33"
OUT = ROOT / "outputs/paper"


def main() -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from phase12.scripts import generate as g12
    from phase13.scripts import generate as g13

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    llm = LLM(model=MODEL, revision=REVISION, tokenizer=MODEL, tokenizer_revision=REVISION, dtype="half",
              max_model_len=4096, gpu_memory_utilization=0.85, enforce_eager=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, g in (("phase12", g12), ("phase13", g13)):
        path = OUT / f"c3_greedy_{name}.json"
        if path.exists():
            continue
        rows = g.holdout()
        prompts = [tok.apply_chat_template([{"role": "user", "content": build_prompt(
            Problem(id=r["id"], domain="math", question=r["question"]))}], tokenize=False,
            add_generation_prompt=True) for r in rows]
        results = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=768, seed=20261011))
        path.write_text(json.dumps({r["id"]: res.outputs[0].text for r, res in zip(rows, results)},
                                   ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"dataset": name, "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
