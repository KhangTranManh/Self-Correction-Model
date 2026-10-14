"""Transformers smoke test for one Phase 14 model (run after smoke_model.py, in its own process).

Re-reads the toy attempts and judgments saved by smoke_model.py with the exact
functions of scripts/extract_hidden.py (answer, judge, verdict and direct sites),
so the hidden-state path is checked on the real model before any data run:
  1. the model loads in FP16 with Transformers (sdpa, or eager for Gemma);
  2. every site returns [rows, layers + 1, hidden_size] matching the model
     config, with all values finite (no FP16 overflow);
  3. the tokenized prompt is an exact prefix of prompt + answer (answer site);
  4. teacher-forced log P(A), log P(B) after "Verdict: Solution" agree with
     vLLM's values: the same choice on every judgment where vLLM's margin
     |log P(A) - log P(B)| is at least 1 nat (near-ties may flip under FP16 and
     re-tokenization); overall agreement and mean absolute difference reported;
  5. the logit lens at the last layer equals the final-layer A/B margin;
  6. timing per forward pass and peak GPU memory, to size the extraction.
Output: outputs/phase14_v1/smoke/<model>_hidden.json. No project data is read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from phase14.scripts.config import JUDGED, MODELS, OUT  # noqa: E402
from phase14.scripts import extract_hidden as eh  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    args = parser.parse_args()
    import torch

    toy = json.loads((OUT / "smoke" / f"{args.model}_toy.json").read_text(encoding="utf-8"))
    rows = toy["problems"]
    samples = {}
    for pid, attempts in toy["attempts"].items():
        samples[(pid, 1)] = {"output": attempts["1"]}
        for k in JUDGED:  # one blind attempt per toy problem stands in for s2 and s3
            samples[(pid, k)] = {"output": attempts["2"]}
    judged = {}
    for task, row in toy["judgments"].items():
        pid, k, order = task.split("|")
        judged[(pid, int(k), order)] = row

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    tokenizer, network = eh.load_model(args.model)
    load_seconds = time.time() - start
    layers, width = network.config.num_hidden_layers + 1, network.config.hidden_size
    timings, data = {}, {}
    for site, call in (("answer", lambda: eh.answer_site(tokenizer, network, rows, samples)),
                       ("judge", lambda: eh.judge_site(tokenizer, network, rows, samples,
                                                       {"base": judged, "base_c": judged})),
                       ("verdict", lambda: eh.verdict_site(tokenizer, network, rows, samples, judged)),
                       ("direct", lambda: eh.direct_site(tokenizer, network, rows, samples, judged))):
        torch.cuda.synchronize()
        t = time.time()
        data[site] = call()
        torch.cuda.synchronize()
        timings[site] = time.time() - t

    def shape_ok(array) -> bool:
        return array.ndim == 3 and array.shape[1] == layers and array.shape[2] == width

    answer, judge, verdict, direct = data["answer"], data["judge"], data["verdict"], data["direct"]
    checks = {
        "answer_shape_ok": shape_ok(answer["hidden"]),
        "answer_finite": bool(np.isfinite(answer["hidden"].astype(np.float32)).all()),
        "answer_prefix_ok_rate": float(answer["prefix_ok"].mean()),
        "answer_mean_logprob_finite": bool(np.isfinite(answer["mean_logprob"]).all()),
        "judge_shape_ok": all(shape_ok(judge[f"{j}_hidden"]) for j in ("base", "base_c")),
        "judge_finite": all(bool(np.isfinite(judge[f"{j}_hidden"].astype(np.float32)).all())
                            for j in ("base", "base_c")),
        "direct_finite": bool(np.isfinite(direct["final_logprobs"]).all() and np.isfinite(direct["lens_margin"]).all()),
        "lens_layers": int(direct["lens_margin"].shape[1]) if direct["lens_margin"].ndim == 2 else 0,
    }
    final_margin = direct["final_logprobs"][:, 0] - direct["final_logprobs"][:, 1]
    checks["lens_last_layer_max_abs_diff"] = float(np.abs(direct["lens_margin"][:, -1] - final_margin).max())
    both = np.isfinite(verdict["vllm_logprobs"]).all(axis=1) if len(verdict["tasks"]) else np.zeros(0, bool)
    if both.any():
        ours, theirs = verdict["transformers_logprobs"][both], verdict["vllm_logprobs"][both]
        checks["verdict_tasks_compared"] = int(both.sum())
        checks["verdict_mean_abs_logprob_diff"] = float(np.abs(ours - theirs).mean())
        checks["verdict_same_argmax"] = float((ours.argmax(1) == theirs.argmax(1)).mean())
        clear = np.abs(theirs[:, 0] - theirs[:, 1]) >= 1.0
        checks["verdict_clear_tasks"] = int(clear.sum())
        checks["verdict_same_argmax_clear"] = float((ours.argmax(1) == theirs.argmax(1))[clear].mean()) if clear.any() else None
    else:
        checks["verdict_tasks_compared"] = 0
    n_forward = {"answer": len(answer["tasks"]), "judge": 2 * len(judged), "verdict": len(verdict["tasks"]),
                 "direct": len(direct["tasks"])}
    report = {"model": MODELS[args.model]["repo"], "gpu": torch.cuda.get_device_name(0),
              "layers_plus_embeddings": layers, "hidden_size": width, "load_seconds": round(load_seconds, 1),
              "seconds_per_forward": {s: round(timings[s] / max(1, n_forward[s]), 3) for s in timings},
              "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "checks": checks}
    report["pass"] = bool(checks["answer_shape_ok"] and checks["answer_finite"] and checks["judge_shape_ok"]
                          and checks["judge_finite"] and checks["direct_finite"]
                          and checks["lens_layers"] == layers and checks["answer_prefix_ok_rate"] >= 0.8
                          and checks["lens_last_layer_max_abs_diff"] < 0.05
                          and checks["verdict_tasks_compared"] >= 4 and checks.get("verdict_clear_tasks", 0) >= 3
                          and checks.get("verdict_same_argmax_clear") == 1.0)
    out = OUT / "smoke" / f"{args.model}_hidden.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        sys.exit(f"Transformers smoke test failed for {args.model}")


if __name__ == "__main__":
    main()
