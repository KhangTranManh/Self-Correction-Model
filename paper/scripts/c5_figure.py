"""C5 figure: logit-lens bias and skill curves by layer (from outputs/paper_3090/c5_logit_lens.json)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
data = json.loads((ROOT / "outputs/paper_3090/c5_logit_lens.json").read_text(encoding="utf-8"))["judges"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
styles = {("base_c", "forced"): ("tab:blue", "--"), ("base_c", "written"): ("tab:blue", "-"),
          ("p13b", "forced"): ("tab:red", "--"), ("p13b", "written"): ("tab:red", "-")}
for (judge, point), (color, ls) in styles.items():
    v = data[judge][point]
    label = f"{judge}, {'forced verdict after prompt' if point == 'forced' else 'before its own verdict'}"
    axes[0].plot(v["share_prefer_A_by_layer"], color=color, ls=ls, label=label)
    axes[1].plot(v["accuracy_one_right_by_layer"], color=color, ls=ls, label=label)
axes[0].axhline(0.5, color="gray", lw=0.8)
axes[1].axhline(0.5, color="gray", lw=0.8)
axes[0].set(title="Bias: share of tasks preferring Solution A", xlabel="layer", ylabel="share")
axes[1].set(title="Skill: accuracy on one-right pairs", xlabel="layer", ylabel="accuracy")
axes[1].legend(fontsize=8, loc="upper left")
fig.tight_layout()
out = ROOT / "paper/figures/c5_logit_lens.png"
fig.savefig(out, dpi=160)
print(out)
