"""Hidden states with Transformers over the exact Phase 14 text (after vLLM generation).

Generation stays in vLLM (as in Phases 8-13); this pass re-reads the same text
with Transformers and saves every layer (embeddings + all decoder layers,
float16), as Phases 5 and 13 did. Sites:

  answer   for (source, k), k in 1..3: solve prompt (Phase 1 for k = 1, blind
           for k > 1, add_generation_prompt) + the primary-view answer; the
           state of the last answer token, i.e. when the model finished
           writing. Also the mean log-probability of the answer tokens
           (likelihood baseline) and whether the prompt tokens are an exact
           prefix of the full tokenization.
  judge    for every judged task (base and base_c prompts): the state of the
           last prompt token, after reading both solutions and before writing.
  verdict  base_c only: teacher-forced log P("A") and log P("B") after the
           last "Verdict: Solution" in the generated judgment, with vLLM's
           values stored alongside for the agreement check.
  direct   base_c prompt + "Verdict: Solution" forced immediately (no written
           check): final-layer log P(A), log P(B) and the logit lens at every
           layer, for every judged task in both orders.

Output: outputs/phase14_v1/hidden/<split>/<model>/<dataset>/<site>.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402
from phase7.scripts.paired_prompts import paired_prompt  # noqa: E402
from phase14.scripts.config import BRIDGE, COMPARISON, DATASETS, EAGER_ATTENTION, JUDGED, MODELS, OUT, REGRESSION_SETS  # noqa: E402
from phase14.scripts.generate import JUDGES, completed, folder, load_samples, problems  # noqa: E402

_VERDICT_PREFIX = re.compile(r"Verdict:\s*Solution")


def destination(split: str, model: str, dataset: str, site: str) -> Path:
    return OUT / "hidden" / split / model / dataset / f"{site}.npz"


def load_model(model: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    attention = "eager" if spec["family"] in EAGER_ATTENTION else "sdpa"
    network = AutoModelForCausalLM.from_pretrained(spec["repo"], revision=spec["revision"],
                                                   torch_dtype=torch.float16, device_map={"": 0},
                                                   attn_implementation=attention, low_cpu_mem_usage=True)
    network.eval()
    network.config.use_cache = False
    return tokenizer, network


def render(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True)


def forward(network, ids: list[int]):
    import torch

    tensor = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        output = network(input_ids=tensor, output_hidden_states=True, use_cache=False, return_dict=True)
    return output


def last_states(output) -> np.ndarray:
    import torch

    return torch.stack([h[0, -1] for h in output.hidden_states]).float().cpu().numpy().astype(np.float16)


def answer_site(tokenizer, network, rows: list[dict], samples: dict) -> dict:
    import torch

    tasks, hidden, mean_lp, n_tokens, prefix_ok = [], [], [], [], []
    for row in rows:
        problem = Problem(id=row["id"], domain="math", question=row["question"])
        for k in (1, *JUDGED):
            prompt = render(tokenizer, build_prompt(problem) if k == 1 else paired_prompt(problem, "blind_resolve"))
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            ids = tokenizer(prompt + samples[(row["id"], k)]["output"], add_special_tokens=False)["input_ids"]
            ok = ids[:len(prompt_ids)] == prompt_ids and len(ids) > len(prompt_ids)
            output = forward(network, ids)
            start = len(prompt_ids) if ok else max(len(ids) - 1, 1)
            logits = output.logits[0, start - 1:len(ids) - 1].float()
            targets = torch.tensor(ids[start:], device=logits.device)
            lp = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None]).mean().item() if len(targets) else float("nan")
            tasks.append(f"{row['id']}|{k}")
            hidden.append(last_states(output))
            mean_lp.append(lp)
            n_tokens.append(len(ids) - len(prompt_ids))
            prefix_ok.append(ok)
    return {"tasks": np.array(tasks), "hidden": np.stack(hidden), "mean_logprob": np.array(mean_lp),
            "answer_tokens": np.array(n_tokens), "prefix_ok": np.array(prefix_ok)}


def judge_prompts(rows: list[dict], samples: dict, judge: str, judged: dict) -> list[tuple[str, str]]:
    by_id = {row["id"]: row for row in rows}
    make = JUDGES[judge]
    out = []
    for pid, k, order in sorted(judged):
        problem = Problem(id=pid, domain="math", question=by_id[pid]["question"])
        first, other = samples[(pid, 1)]["output"], samples[(pid, k)]["output"]
        prompt = make(problem, first, other) if order == "ab" else make(problem, other, first)
        out.append((f"{pid}|{k}|{order}", prompt))
    return out


def judge_site(tokenizer, network, rows, samples, judges: dict) -> dict:
    result = {}
    for judge, judged in judges.items():
        tasks, hidden = [], []
        for task, prompt in judge_prompts(rows, samples, judge, judged):
            tasks.append(task)
            hidden.append(last_states(forward(network, tokenizer(render(tokenizer, prompt),
                                                                 add_special_tokens=False)["input_ids"])))
        result[f"{judge}_tasks"] = np.array(tasks)
        result[f"{judge}_hidden"] = np.stack(hidden) if hidden else np.zeros((0,), dtype=np.float16)
    return result


def verdict_site(tokenizer, network, rows, samples, judged: dict) -> dict:
    import torch

    letters = letter_ids(tokenizer)
    tasks, ours, theirs = [], [], []
    for task, prompt in judge_prompts(rows, samples, "base_c", judged):
        pid, k, order = task.split("|")
        row = judged[(pid, int(k), order)]
        matches = list(_VERDICT_PREFIX.finditer(row["output"]))
        if not matches:
            continue
        text = render(tokenizer, prompt) + row["output"][:matches[-1].end()]
        output = forward(network, tokenizer(text, add_special_tokens=False)["input_ids"])
        logprobs = torch.log_softmax(output.logits[0, -1].float(), dim=-1)
        tasks.append(task)
        ours.append([max(logprobs[i].item() for i in letters[x]) for x in ("A", "B")])
        vllm = row.get("verdict_logprobs") or {}
        theirs.append([vllm.get("A") if vllm.get("A") is not None else np.nan,
                       vllm.get("B") if vllm.get("B") is not None else np.nan])
    return {"tasks": np.array(tasks), "transformers_logprobs": np.array(ours, dtype=float),
            "vllm_logprobs": np.array(theirs, dtype=float)}


def letter_ids(tokenizer) -> dict[str, list[int]]:
    out = {}
    for letter in ("A", "B"):
        out[letter] = sorted({tokenizer.encode(v, add_special_tokens=False)[0] for v in (letter, " " + letter)
                              if tokenizer.encode(v, add_special_tokens=False)})
    return out


def direct_site(tokenizer, network, rows, samples, judged: dict) -> dict:
    """Forced verdict right after the constrained judge prompt, no written check.

    Final layer: log P(A), log P(B). Logit lens: every layer's last-position
    state through the final norm and LM head (the last state already carries
    the final norm in Hugging Face models), giving logit(A) - logit(B).
    """
    import torch

    letters = letter_ids(tokenizer)
    norm, head = network.model.norm, network.get_output_embeddings()
    tasks, final, lens = [], [], []
    for task, prompt in judge_prompts(rows, samples, "base_c", judged):
        output = forward(network, tokenizer(render(tokenizer, prompt) + "Verdict: Solution",
                                            add_special_tokens=False)["input_ids"])
        logprobs = torch.log_softmax(output.logits[0, -1].float(), dim=-1)
        final.append([max(logprobs[i].item() for i in letters[x]) for x in ("A", "B")])
        margins = []
        with torch.inference_mode():
            last = len(output.hidden_states) - 1
            for layer, state in enumerate(output.hidden_states):
                h = state[0, -1]
                logits = head(h if layer == last else norm(h[None])[0]).float()
                margins.append(max(logits[i].item() for i in letters["A"]) - max(logits[i].item() for i in letters["B"]))
        tasks.append(task)
        lens.append(margins)
    return {"tasks": np.array(tasks), "final_logprobs": np.array(final, dtype=float),
            "lens_margin": np.array(lens, dtype=np.float32)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--dataset", required=True, choices=(*DATASETS, *COMPARISON, *REGRESSION_SETS, BRIDGE))
    parser.add_argument("--split", required=True, choices=("pilot", "main"))
    parser.add_argument("--site", required=True, choices=("answer", "judge", "verdict", "direct"))
    args = parser.parse_args()
    out = destination(args.split, args.model, args.dataset, args.site)
    if out.exists():
        print(json.dumps({"status": "exists", "path": str(out)}))
        return
    rows = problems(args.dataset, args.split)
    samples = load_samples(args.model, args.dataset, args.split)
    base = folder(args.model, args.dataset, args.split)
    tokenizer, network = load_model(args.model)
    if args.site == "answer":
        data = answer_site(tokenizer, network, rows, samples)
    elif args.site == "judge":
        data = judge_site(tokenizer, network, rows, samples, {j: completed(base, f"judge_{j}") for j in JUDGES})
    elif args.site == "verdict":
        data = verdict_site(tokenizer, network, rows, samples, completed(base, "judge_base_c"))
    else:
        data = direct_site(tokenizer, network, rows, samples, completed(base, "judge_base_c"))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    finite = {k: bool(np.isfinite(v.astype(np.float32)).all()) for k, v in data.items() if "hidden" in k and v.size}
    summary = {"status": "complete", "path": str(out), "all_finite": finite,
               "layers": int(next((v.shape[1] for k, v in data.items() if "hidden" in k and v.ndim == 3), 0))}
    if args.site == "verdict" and len(data["tasks"]):
        both = np.isfinite(data["vllm_logprobs"]).all(axis=1)
        diff = np.abs(data["transformers_logprobs"][both] - data["vllm_logprobs"][both])
        summary["verdict_agreement"] = {"tasks": int(len(data["tasks"])), "with_vllm_values": int(both.sum()),
                                        "mean_abs_logprob_diff": float(diff.mean()) if both.any() else None,
                                        "same_argmax": float((np.argmax(data["transformers_logprobs"][both], 1)
                                                              == np.argmax(data["vllm_logprobs"][both], 1)).mean())
                                        if both.any() else None}
    if args.site == "answer":
        summary["prefix_ok_rate"] = float(data["prefix_ok"].mean())
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
