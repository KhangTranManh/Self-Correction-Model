"""F2: pointwise correctness probe and probe-weighted voting.

Extraction (Transformers, original solver, no adapter): for every sample s1..s5 of
Phase 12 SVAMP and Phase 13 GSM8K, the pointwise prompt
  Phase 1 math prompt + "A solution to this problem is shown below. It may be wrong.
  Check it step by step.\n\nSolution:\n<text>\n\nIs the final answer of this solution correct?"
is rendered with the chat template + generation prompt and the last-token hidden
state is kept at every layer (outputs/paper/pointwise_<ds>.npz).
Analysis: label = verifier-correct; probe as in Phase 13 (StandardScaler + L2
logistic, C=0.01), layer chosen by 5-fold problem-grouped CV on SVAMP, test on GSM8K.
Strategies on GSM8K (problem accuracy): keep (s1), vote@5, probe best-of-5,
probe-weighted vote@5 (sum of P(correct) per answer cluster), and on one-right
(s1, sk) pairs the pointwise pick (higher P(correct)) vs the pairwise judge probe.
Output: outputs/paper/f2_pointwise.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.core.prompts import build_prompt  # noqa: E402
from src.core.schema import Problem  # noqa: E402
from phase9.scripts.analyze import ci  # noqa: E402
from phase9.scripts.answers import parse, same, vote  # noqa: E402
from phase13.layer.probe_analysis import LAYERS, load, probe  # noqa: E402

MODEL = "Kxck/Self_Correction_v1"
REVISION = "6437f947999168a0ce2a98a86e4252fc77160a33"
OUT = ROOT / "outputs/paper"
POINT = ("\n\nA solution to this problem is shown below. It may be wrong. Check it step by step."
         "\n\nSolution:\n{text}\n\nIs the final answer of this solution correct?")


def extract(name: str) -> None:
    path = OUT / f"pointwise_{name}.npz"
    if path.exists():
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _, samples, rows, _ = load(name)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float16,
                                                 device_map={"": 0}, attn_implementation="sdpa").eval()
    keys = sorted(samples)
    hidden = np.zeros((len(keys), 29, 3584), dtype=np.float16)
    with torch.inference_mode():
        for i, (pid, k) in enumerate(keys):
            prompt = build_prompt(Problem(id=pid, domain="math", question=rows[pid]["question"])) + \
                POINT.format(text=samples[(pid, k)])
            text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                           add_generation_prompt=True)
            ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
            hs = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
            hidden[i] = torch.stack([h[0, -1] for h in hs]).float().cpu().numpy().astype(np.float16)
            if (i + 1) % 500 == 0:
                print(f"pointwise {name}: {i + 1}/{len(keys)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(path, keys=np.array([f"{p}|{k}" for p, k in keys]), hidden=hidden)
    del model
    torch.cuda.empty_cache()


def _fold(x, y, tr, te):
    return float((probe().fit(x[tr], y[tr]).predict(x[te]) == y[te]).mean())


def cv_curve(x, y, groups):
    """Same as phase13.layer.probe_analysis.cv_curve, parallel over (layer, fold)."""
    folds = list(GroupKFold(n_splits=5).split(x, y, groups))
    hits = Parallel(n_jobs=30)(delayed(_fold)(np.ascontiguousarray(x[:, l]), y, tr, te)
                               for l in LAYERS for tr, te in folds)
    return [float(np.mean(hits[i * 5:(i + 1) * 5])) for i in range(len(LAYERS))]


def _transfer(xs, ys, xg, yg):
    pr = probe().fit(xs, ys).predict_proba(xg)[:, 1]
    return [float(((pr > 0.5) == yg).mean()), float(roc_auc_score(yg, pr))]


def arrays(name: str, correct: dict):
    data = np.load(OUT / f"pointwise_{name}.npz")
    keys = [(p, int(k)) for p, k in (s.split("|") for s in data["keys"])]
    y = np.array([int(correct[key]) for key in keys])
    return data["hidden"].astype(np.float32), y, np.array([p for p, _ in keys]), keys


def main() -> None:
    for name in ("phase12", "phase13"):
        extract(name)
    _, _, _, c12 = load("phase12")
    g13, s13, rows13, c13 = load("phase13")
    xs, ys, gs, _ = arrays("phase12", c12)
    xg, yg, gg, kg = arrays("phase13", c13)
    svamp_cv = cv_curve(xs, ys, gs)
    layer = int(np.argmax(svamp_cv))
    by_layer = Parallel(n_jobs=29)(delayed(_transfer)(np.ascontiguousarray(xs[:, l]), ys,
                                                      np.ascontiguousarray(xg[:, l]), yg) for l in LAYERS)
    model = probe().fit(xs[:, layer], ys)
    prob = dict(zip(kg, model.predict_proba(xg[:, layer])[:, 1]))
    acc = {"keep": [], "vote5": [], "probe_best_of_5": [], "probe_weighted_vote5": []}
    for pid in rows13:
        texts = [s13[(pid, k)] for k in range(1, 6)]
        vals = [parse(t) for t in texts]
        p = [prob[(pid, k)] for k in range(1, 6)]
        acc["keep"].append(c13[(pid, 1)])
        acc["vote5"].append(c13[(pid, 1 + vote(texts, [1, 2, 3, 4, 0]))])
        acc["probe_best_of_5"].append(c13[(pid, 1 + int(np.argmax(p)))])
        weight = [sum(p[j] for j in range(5) if vals[i] is not None and same(vals[i], vals[j])) for i in range(5)]
        acc["probe_weighted_vote5"].append(c13[(pid, 1 + int(np.argmax(weight)))])
    a = {k: np.array(v, dtype=float) for k, v in acc.items()}
    pair_hits = []
    for pid, k, order in g13.completed("judge_base"):
        if order == "ab" and c13[(pid, 1)] != c13[(pid, k)]:
            pair_hits.append((prob[(pid, 1)] > prob[(pid, k)]) == c13[(pid, 1)])
    report = {"schema_version": "paper_f2_v1", "post_hoc": True, "prompt_suffix": POINT,
              "svamp_rows": int(len(ys)), "gsm8k_rows": int(len(yg)),
              "svamp_base_rate_correct": float(ys.mean()), "gsm8k_base_rate_correct": float(yg.mean()),
              "svamp_cv_accuracy_by_layer": svamp_cv, "chosen_layer": layer,
              "gsm8k_accuracy_auc_by_layer": by_layer,
              "gsm8k_at_chosen_layer": {"accuracy": by_layer[layer][0], "auc": by_layer[layer][1]},
              "strategy_accuracy": {k: float(v.mean()) for k, v in a.items()},
              "probe_weighted_vs_vote5": {"difference": float(a["probe_weighted_vote5"].mean() - a["vote5"].mean()),
                                          "ci": ci(a["probe_weighted_vote5"], a["vote5"], 20261040)},
              "best_of_5_vs_vote5": {"difference": float(a["probe_best_of_5"].mean() - a["vote5"].mean()),
                                     "ci": ci(a["probe_best_of_5"], a["vote5"], 20261041)},
              "one_right_pairs_pointwise_pick": {"n": len(pair_hits), "accuracy": float(np.mean(pair_hits))}}
    (OUT / "f2_pointwise.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if "by_layer" not in k}, indent=1))


if __name__ == "__main__":
    main()
