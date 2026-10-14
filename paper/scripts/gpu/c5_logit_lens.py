"""C5: logit lens on the verdict token for the constrained judges (base_c, p13b), GSM8K.

For every judged task (pid, k, order) two readout points:
  forced  rendered judge prompt + assistant prefix "Verdict: Solution" (decide now,
          no reasoning written yet)
  written rendered judge prompt + the judge's own Phase 13 output cut just before
          the letter of its last "Verdict: Solution X" line (tasks with such a line)
At every layer l the hidden state of the last token is passed through the final
norm (except the last layer, already normed) and lm_head; d_l = logit(" A") - logit(" B").
Bias curve: mean d_l over all tasks (> 0 = leans to the first shown solution).
Skill curve: mean of d_l * (+1 if the right solution is shown as A else -1) on
one-right pairs; accuracy curve: share of one-right tasks with sign(d_l) on the right one.
Output: outputs/paper/c5_logit_lens.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.core.schema import Problem  # noqa: E402
from phase13.layer.probe_analysis import load  # noqa: E402
from phase13.scripts.constrained import constrained_prompt  # noqa: E402

MODEL = "Kxck/Self_Correction_v1"
REVISION = "6437f947999168a0ce2a98a86e4252fc77160a33"
ADAPTER = ROOT / "outputs/phase13_v1/dpo_b_lora/final_adapter"
OUT = ROOT / "outputs/paper"
_LAST = re.compile(r"Verdict:\s*Solution\s*(?=[AB]\b)")


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    g, samples, rows, correct = load("phase13")
    tasks = sorted(g.completed("judge_base"))
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    id_a, id_b = tok.encode(" A", add_special_tokens=False), tok.encode(" B", add_special_tokens=False)
    assert len(id_a) == len(id_b) == 1
    id_a, id_b = id_a[0], id_b[0]
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float16,
                                                device_map={"": 0}, attn_implementation="sdpa").eval()
    report = {"schema_version": "paper_c5_v1", "post_hoc": True, "tokens": [" A", " B"], "judges": {}}
    model = base
    for judge in ("base_c", "p13b"):
        if judge == "p13b":
            model = PeftModel.from_pretrained(base, str(ADAPTER), is_trainable=False).eval()
        inner = model.get_base_model() if judge == "p13b" else model
        norm, head = inner.model.norm, inner.lm_head
        outputs = g.completed(f"judge_{judge}")
        res = {}
        for point in ("forced", "written"):
            d_all, d_signed, groups = [], [], []
            with torch.inference_mode():
                for pid, k, order in tasks:
                    problem = Problem(id=pid, domain="math", question=rows[pid]["question"])
                    first, other = samples[(pid, 1)], samples[(pid, k)]
                    prompt = constrained_prompt(problem, first, other) if order == "ab" else \
                        constrained_prompt(problem, other, first)
                    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                   add_generation_prompt=True)
                    if point == "forced":
                        text += "Verdict: Solution"
                    else:
                        out = outputs[(pid, k, order)]
                        cut = list(_LAST.finditer(out))
                        if not cut:
                            continue
                        text += out[:cut[-1].end()].rstrip()
                    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
                    hs = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
                    last = torch.stack([h[0, -1] for h in hs])                     # [29, d]
                    normed = torch.cat([norm(last[:-1]), last[-1:]])
                    logits = head(normed).float()
                    d = (logits[:, id_a] - logits[:, id_b]).cpu().numpy()
                    d_all.append(d)
                    if correct[(pid, 1)] != correct[(pid, k)]:
                        right_is_a = (order == "ab") == correct[(pid, 1)]
                        d_signed.append(d if right_is_a else -d)
                        groups.append(pid)
            d_all, d_signed = np.array(d_all), np.array(d_signed)
            res[point] = {"tasks": int(len(d_all)), "one_right_tasks": int(len(d_signed)),
                          "bias_mean_logit_A_minus_B_by_layer": d_all.mean(0).tolist(),
                          "share_prefer_A_by_layer": (d_all > 0).mean(0).tolist(),
                          "skill_mean_signed_by_layer": d_signed.mean(0).tolist(),
                          "accuracy_one_right_by_layer": (d_signed > 0).mean(0).tolist()}
            print(judge, point, json.dumps({"n": len(d_all), "final_bias": float(d_all.mean(0)[-1]),
                                            "final_acc": float((d_signed > 0).mean(0)[-1]),
                                            "best_acc_layer": int((d_signed > 0).mean(0).argmax())}), flush=True)
        report["judges"][judge] = res
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c5_logit_lens.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
