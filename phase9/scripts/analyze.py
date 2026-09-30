"""Open Phase 9 labels once and score every preregistered strategy per checkpoint.

Strategies (A0 = shared first answer; B1..B4 = this checkpoint's blind attempts;
J = judge output; agreement uses gold-free answer comparison):
  keep          A0
  blind1        B1
  maj3          plurality of (A0, B1, B2); ties -> B1, B2, A0
  maj5          plurality of (A0, B1..B4); ties -> B1, B2, B3, B4, A0
  agree_gated   A0 if A0 == B1, else maj3
  self_check    A0 if A0 == B1, else J
  probe_blind   B1 if the frozen probe flags A0, else A0
Primary families (Holm over three checkpoints): maj5 vs keep; self_check vs keep.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
from src.core.schema import Problem  # noqa: E402
from src.data.verifiers.math import MathVerifier  # noqa: E402
sys.path.insert(0, str(ROOT))
from phase8.scripts.batched import read_jsonl  # noqa: E402
from phase9.scripts.answers import parse, same, vote  # noqa: E402
from phase9.scripts.collect_first import sha256, sources  # noqa: E402

FIRST = ROOT / "outputs/phase9_first_v1"
CKPT = ROOT / "outputs/phase9_checkpoints_v1"
PROBE = ROOT / "outputs/phase9_probe_scores_v1"
OUT = ROOT / "outputs/phase9_analysis_v1"
CHECKPOINTS = ("original_solver", "warmstart_v2", "correction_sft_v3")
STRATEGIES = ("keep", "blind1", "maj3", "maj5", "agree_gated", "self_check", "probe_blind")
PRIMARY = {"maj5_vs_keep": ("maj5", "keep"), "self_check_vs_keep": ("self_check", "keep")}
SECONDARY = {"maj5_vs_blind1": ("maj5", "blind1"), "maj3_vs_keep": ("maj3", "keep"),
             "agree_gated_vs_keep": ("agree_gated", "keep"),
             "self_check_vs_agree_gated": ("self_check", "agree_gated"),
             "self_check_vs_probe_blind": ("self_check", "probe_blind"),
             "maj5_vs_probe_blind": ("maj5", "probe_blind")}
GENERATIONS = {"keep": 1, "blind1": 2, "maj3": 3, "maj5": 5}


def checked(folder: Path, name: str, key: str) -> list[dict]:
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or sha256(folder / name) != summary[key]:
        raise RuntimeError(f"Incomplete or changed: {folder / name}")
    return read_jsonl(folder / name)


def ci(left: np.ndarray, right: np.ndarray, seed: int) -> list[float]:
    delta = left.astype(float) - right.astype(float)
    rng = np.random.default_rng(seed)
    values = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def paired_p(left: np.ndarray, right: np.ndarray) -> float:
    better, worse = int(np.sum(left & ~right)), int(np.sum(~left & right))
    return float(binomtest(min(better, worse), better + worse, 0.5).pvalue) if better + worse else 1.0


def holm(raw: dict[str, float]) -> dict[str, float]:
    result, maximum = {}, 0.0
    for rank, key in enumerate(sorted(raw, key=raw.get)):
        maximum = max(maximum, min(1.0, raw[key] * (len(raw) - rank)))
        result[key] = maximum
    return result


def main() -> None:
    if OUT.exists():
        raise RuntimeError("Refusing a second protected opening")
    rows = sources()
    first = {row["problem_id"]: row["output"] for row in checked(FIRST, "answers.jsonl", "answers_sha256")}
    outputs, scores = {}, {}
    for checkpoint in CHECKPOINTS:
        blind = checked(CKPT / checkpoint, "blind_answers.jsonl", "blind_answers_sha256")
        judge = checked(CKPT / checkpoint, "judge_answers.jsonl", "judge_answers_sha256")
        outputs[checkpoint] = {
            "blind": {(row["problem_id"], row["attempt"]): row["output"] for row in blind},
            "judge": {row["problem_id"]: row["output"] for row in judge}}
        scores[checkpoint] = {row["problem_id"]: row for row in
                              checked(PROBE / checkpoint, "scores.jsonl", "scores_sha256")}

    # First read of gold answers.
    verifier = MathVerifier()
    report = {"schema_version": "phase9_analysis_v1", "splits": {}}
    verdicts = []
    for split in ("protected", "development"):
        split_rows = [row for row in rows if row["split"] == split]
        problems = {row["id"]: Problem(id=row["id"], domain="math", question=row["question"],
                                       reference_answer=row["reference_answer"]) for row in split_rows}
        ids = [row["id"] for row in split_rows]
        correct = lambda pid, text: bool(verifier.verify(problems[pid], text).passed)
        keep = np.array([correct(pid, first[pid]) for pid in ids])
        block = {"rows": len(ids), "initial_correct": int(keep.sum()), "checkpoints": {}}
        raw_primary = {name: {} for name in PRIMARY}
        raw_secondary = {name: {} for name in SECONDARY}
        for m, checkpoint in enumerate(CHECKPOINTS):
            chosen = {name: [] for name in STRATEGIES}
            agree_flags, probe_flags, judge_kind, cost = [], [], [], {"agree_gated": 0, "self_check": 0}
            for pid in ids:
                a0 = first[pid]
                b = [outputs[checkpoint]["blind"][(pid, k)] for k in range(1, 5)]
                judge = outputs[checkpoint]["judge"][pid]
                agree = same(parse(a0), parse(b[0]))
                flag = bool(scores[checkpoint][pid]["flag"])
                pool3, pool5 = [a0, b[0], b[1]], [a0, *b]
                maj3 = pool3[vote(pool3, [1, 2, 0])]
                maj5 = pool5[vote(pool5, [1, 2, 3, 4, 0])]
                chosen["keep"].append(a0)
                chosen["blind1"].append(b[0])
                chosen["maj3"].append(maj3)
                chosen["maj5"].append(maj5)
                chosen["agree_gated"].append(a0 if agree else maj3)
                chosen["self_check"].append(a0 if agree else judge)
                chosen["probe_blind"].append(b[0] if flag else a0)
                cost["agree_gated"] += 2 if agree else 3
                cost["self_check"] += 2 if agree else 3
                agree_flags.append(agree)
                probe_flags.append(flag)
                j = parse(judge)
                judge_kind.append("agree" if agree else
                                  "A0" if same(j, parse(a0)) else "B1" if same(j, parse(b[0])) else "other")
            ok = {name: np.array([correct(pid, text) for pid, text in zip(ids, texts)])
                  for name, texts in chosen.items()}
            disagree, flagged, wrong = ~np.array(agree_flags), np.array(probe_flags), ~keep
            detectors = {}
            for name, flags in (("disagreement", disagree), ("probe", flagged)):
                detectors[name] = {"flagged": int(flags.sum()),
                                   "wrong_recall": float(flags[wrong].mean()) if wrong.any() else None,
                                   "precision": float(wrong[flags].mean()) if flags.any() else None,
                                   "correct_preservation": float((~flags[keep]).mean()) if keep.any() else None}
            kinds = np.array(judge_kind)
            judge_stats = {kind: {"n": int((kinds == kind).sum()),
                                  "correct": int(ok["self_check"][kinds == kind].sum())}
                           for kind in ("A0", "B1", "other")}
            mean_cost = dict(GENERATIONS, agree_gated=cost["agree_gated"] / len(ids),
                             self_check=cost["self_check"] / len(ids),
                             probe_blind=1 + float(flagged.mean()))
            entry = {"accuracy": {name: float(v.mean()) for name, v in ok.items()},
                     "correct": {name: int(v.sum()) for name, v in ok.items()},
                     "fixes": {name: int((v & wrong).sum()) for name, v in ok.items()},
                     "harms": {name: int((~v & keep).sum()) for name, v in ok.items()},
                     "mean_generations": mean_cost, "detectors": detectors,
                     "judge_on_disagreement": judge_stats, "contrasts": {}}
            for family, table, raw in (("primary", PRIMARY, raw_primary),
                                       ("secondary", SECONDARY, raw_secondary)):
                for c, (name, (left, right)) in enumerate(table.items()):
                    entry["contrasts"][name] = {
                        "family": family,
                        "difference": float(ok[left].mean() - ok[right].mean()),
                        "ci": ci(ok[left], ok[right], 20260930 + 10 * m + c),
                        "p_raw": paired_p(ok[left], ok[right])}
                    raw[name][checkpoint] = entry["contrasts"][name]["p_raw"]
            block["checkpoints"][checkpoint] = entry
            for i, pid in enumerate(ids):
                verdicts.append({"split": split, "checkpoint": checkpoint, "problem_id": pid,
                                 "agree": bool(agree_flags[i]), "probe_flag": bool(probe_flags[i]),
                                 "judge_kind": judge_kind[i],
                                 **{f"{name}_correct": bool(v[i]) for name, v in ok.items()}})
        for raw in (raw_primary, raw_secondary):
            for name, values in raw.items():
                for checkpoint, p in holm(values).items():
                    block["checkpoints"][checkpoint]["contrasts"][name]["p_holm"] = p
        report["splits"][split] = block
    OUT.mkdir(parents=True)
    path = OUT / "verdicts.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in verdicts),
                    encoding="utf-8", newline="\n")
    report["verdict_rows_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["splits"]["protected"]["checkpoints"], indent=1)[:4000])


if __name__ == "__main__":
    main()
