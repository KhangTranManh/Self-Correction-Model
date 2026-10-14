"""Token-cap sensitivity (exploratory, after the protected opening): re-judge the same pairs with a
1536-token judge cap instead of 768, written to separate stages judge_<judge>_1536.

Identical to generate.py's judge stage (same tasks, prompts, seeds, greedy, log-probabilities for
base_c) except max_tokens = 1536 and the stage name. Tasks whose prompt + 1536 tokens would exceed
the 4096-token context are skipped and counted in skipped_1536.json. Uses 256-request vLLM chunks.
No locked file is changed; E1/E2 are not recomputed.
Usage: judge_long.py <model> <dataset> <base|base_c>
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
import phase8.scripts.batched as batched  # noqa: E402
from phase14.scripts import generate as g  # noqa: E402
from src.core.schema import Problem  # noqa: E402

LONG_CAP = 1536
batched.BATCH_SIZE = 256
g.BATCH_SIZE = 256


def main() -> None:
    model, dataset, judge = sys.argv[1:4]
    from transformers import AutoTokenizer
    from phase14.scripts.config import MAX_MODEL_LEN, MODELS, tokenizer_source
    rows = g.problems(dataset, "main")
    base = g.folder(model, dataset, "main")
    stage = f"judge_{judge}_1536"
    if (base / stage / "summary.json").exists():
        print(json.dumps({"stage": stage, "status": "exists"}))
        return
    samples = g.load_samples(model, dataset, "main")
    by_id = {r["id"]: r for r in rows}
    make = g.JUDGES[judge]
    tok_repo, tok_rev = tokenizer_source(MODELS[model])
    tok = AutoTokenizer.from_pretrained(tok_repo, revision=tok_rev)
    tasks, prompts, seeds, skipped = [], [], [], []
    for pid, k, order in g.judge_tasks(rows, samples):
        problem = Problem(id=pid, domain="math", question=by_id[pid]["question"])
        first, other = samples[(pid, 1)]["output"], samples[(pid, k)]["output"]
        prompt = make(problem, first, other) if order == "ab" else make(problem, other, first)
        rendered = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True)
        if len(tok.encode(rendered, add_special_tokens=False)) + LONG_CAP > MAX_MODEL_LEN:
            skipped.append([pid, k, order])
            continue
        tasks.append((pid, k, order))
        prompts.append(prompt)
        seeds.append(g.seed(judge, model, dataset, "main", pid, k, order))
    (base / stage).mkdir(parents=True, exist_ok=True)
    (base / stage / "skipped_1536.json").write_text(json.dumps({"skipped": skipped}) + "\n", encoding="utf-8")
    g.run(model, base, stage, tasks, prompts, 0.0, LONG_CAP, seeds, with_logprobs=judge == "base_c")


if __name__ == "__main__":
    main()
