"""Exploratory layer study: save judge-prompt hidden states with Transformers.

For every judged task (source, k, order) of one dataset, the judge prompt is
rendered exactly as in generation (chat template + generation prompt) and one
forward pass records the hidden state of the LAST prompt token at every layer
(embeddings + 28 decoder layers = 29 vectors of size 3584), i.e. after the
model has read both solutions and before it writes anything.

Datasets: phase12 (SVAMP; judges base, p12) and phase13 (GSM8K test; judges
base, p12, base_c, p13b). Output: outputs/phase13_v1/layer/<dataset>_<judge>.npz
with "tasks" ("pid|k|order") and "hidden" float16 [N, 29, 3584].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.core.schema import Problem  # noqa: E402
from phase9.scripts.collect_checkpoint import judge_prompt  # noqa: E402
from phase13.scripts.constrained import constrained_prompt  # noqa: E402

MODEL = "Kxck/Self_Correction_v1"
REVISION = "6437f947999168a0ce2a98a86e4252fc77160a33"
ADAPTERS = {"p12": ROOT / "outputs/phase12_v1/dpo_lora/final_adapter",
            "p13b": ROOT / "outputs/phase13_v1/dpo_b_lora/final_adapter"}
CONSTRAINED = {"base_c", "p13b"}
OUT = ROOT / "outputs/phase13_v1/layer"


def dataset_inputs(dataset: str):
    if dataset == "phase12":
        from phase12.scripts import generate as g
    else:
        from phase13.scripts import generate as g
    samples = g.completed("samples")
    tasks = sorted(g.completed("judge_base"))  # identical task set for every judge
    rows = {row["id"]: row for row in g.holdout()}
    return samples, tasks, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("phase12", "phase13"))
    parser.add_argument("--judge", required=True, choices=("base", "p12", "base_c", "p13b"))
    args = parser.parse_args()
    out = OUT / f"{args.dataset}_{args.judge}.npz"
    if out.exists():
        print(json.dumps({"status": "exists", "path": str(out)}))
        return
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    samples, tasks, rows = dataset_inputs(args.dataset)
    make = constrained_prompt if args.judge in CONSTRAINED else judge_prompt
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float16,
                                                 device_map={"": 0}, attn_implementation="sdpa")
    if args.judge in ADAPTERS:
        model = PeftModel.from_pretrained(model, str(ADAPTERS[args.judge]), is_trainable=False)
    model.eval()
    hidden = np.zeros((len(tasks), 29, 3584), dtype=np.float16)
    with torch.inference_mode():
        for i, (pid, k, order) in enumerate(tasks):
            problem = Problem(id=pid, domain="math", question=rows[pid]["question"])
            first, other = samples[(pid, 1)], samples[(pid, k)]
            prompt = make(problem, first, other) if order == "ab" else make(problem, other, first)
            text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                 tokenize=False, add_generation_prompt=True)
            ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
            states = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
            hidden[i] = torch.stack([h[0, -1] for h in states]).float().cpu().numpy().astype(np.float16)
            if (i + 1) % 100 == 0:
                print(f"{args.dataset}/{args.judge}: {i + 1}/{len(tasks)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, tasks=np.array(["|".join(map(str, t)) for t in tasks]), hidden=hidden)
    print(json.dumps({"status": "complete", "path": str(out), "rows": len(tasks)}))


if __name__ == "__main__":
    main()
