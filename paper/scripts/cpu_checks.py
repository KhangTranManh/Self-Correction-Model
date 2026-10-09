"""CPU-only pre-paper checks (checklist A1-A3, B1-B5, C2, F3) on saved outputs.

Inputs are the hash-verified local mirrors of Phase 12 (SVAMP) and Phase 13
(GSM8K test 750-1318). Run with Python 3.10 + SymPy 1.14 (.venv-verify310),
the same verifier versions as the GPU runs. Output: paper/results/cpu_checks.json.
All numbers here are post-hoc unless they reproduce a preregistered gate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.core.schema import Problem  # noqa: E402
from src.data.verifiers.math import MathVerifier, _extract_answer  # noqa: E402
from phase5.scripts.prepare_candidates import question_key, records  # noqa: E402
from phase9.scripts.answers import parse, same, vote  # noqa: E402
from phase13.scripts.constrained import _VERDICT, pick  # noqa: E402

OUT = ROOT / "paper/results/cpu_checks.json"
DATA = {
    "gsm8k": {"mirror": ROOT / "outputs/phase13_remote_v100/outputs/phase13_v1",
              "holdout": ROOT / "phase13/data/sources_v1/holdout.jsonl",
              "judges": {"base": "judge_base", "p12": "judge_p12", "base_c": "judge_base_c", "p13b": "judge_p13b"}},
    "svamp": {"mirror": ROOT / "outputs/phase12_remote_v100/outputs/phase12_v1",
              "holdout": ROOT / "phase12/data/sources_v1/holdout.jsonl",
              "judges": {"base": "judge_base", "p12": "judge_dpo"}},
}
CONSTRAINED = {"base_c", "p13b"}
RNG_SEED = 20261010


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def load(name: str) -> dict:
    spec = DATA[name]
    rows = {r["id"]: r for r in read_jsonl(spec["holdout"])}
    samples = {tuple(r["task"]): r["output"] for r in read_jsonl(spec["mirror"] / "samples/answers.jsonl")}
    judges, meta = {}, {}
    for judge, folder in spec["judges"].items():
        recs = read_jsonl(spec["mirror"] / folder / "answers.jsonl")
        judges[judge] = {tuple(r["task"]): r["output"] for r in recs}
        meta[judge] = {tuple(r["task"]): r for r in recs}
    verifier = MathVerifier()
    correct = {}
    for (pid, k), text in samples.items():
        r = rows[pid]
        correct[(pid, k)] = bool(verifier.verify(Problem(id=pid, domain="math", question=r["question"],
                                                         reference_answer=r["reference_answer"]), text).passed)
    return {"rows": rows, "samples": samples, "judges": judges, "meta": meta, "correct": correct,
            "parsed": {key: parse(text) for key, text in samples.items()}}


def one_right_pairs(d: dict) -> list[tuple[str, int]]:
    pairs = sorted({(pid, k) for (pid, k, _) in d["judges"]["base"]})
    return [(pid, k) for pid, k in pairs if d["correct"][(pid, 1)] != d["correct"][(pid, k)]]


def picks_for(d: dict, judge: str, pid: str, k: int) -> dict:
    """Single-order picks mapped to 'first' / 'other' / 'invented' for both orders."""
    p1, pk = d["parsed"][(pid, 1)], d["parsed"][(pid, k)]
    c = judge in CONSTRAINED
    ab = pick(d["judges"][judge][(pid, k, "ab")], p1, pk, c)   # s1 shown as A
    ba = pick(d["judges"][judge][(pid, k, "ba")], pk, p1, c)   # s1 shown as B
    return {"ab": {"A": "first", "B": "other"}.get(ab, "invented"),
            "ba": {"A": "other", "B": "first"}.get(ba, "invented"),
            "ab_raw": ab, "ba_raw": ba}


def cluster_boot(values: np.ndarray, groups: np.ndarray, reps: int = 10000, seed: int = RNG_SEED) -> list[float]:
    ids = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(seed)
    means = [values[np.concatenate([idx[g] for g in rng.choice(ids, len(ids))])].mean() for _ in range(reps)]
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def cluster_boot_p(diff: np.ndarray, groups: np.ndarray, seed: int) -> tuple[list[float], float]:
    ids = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(seed)
    means = np.array([diff[np.concatenate([idx[g] for g in rng.choice(ids, len(ids))])].mean() for _ in range(10000)])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return [float(x) for x in np.quantile(means, [0.025, 0.975])], float(min(1.0, p))


def holm(raw: dict) -> dict:
    out, running = {}, 0.0
    for rank, key in enumerate(sorted(raw, key=raw.get)):
        running = max(running, min(1.0, raw[key] * (len(raw) - rank)))
        out[key] = running
    return out


# ---------------------------------------------------------------- A1
_WORD = re.compile(r"[a-z0-9]+")


def shingles(text: str, n: int = 8) -> set:
    words = _WORD.findall(text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def a1_audit() -> dict:
    holdout = read_jsonl(DATA["gsm8k"]["holdout"])
    target_keys = {question_key(r["question"]): r["id"] for r in holdout}
    target_sh = {r["id"]: shingles(r["question"]) for r in holdout}
    index = defaultdict(set)
    for rid, sh in target_sh.items():
        for s in sh:
            index[s].add(rid)
    excluded = {(ROOT / "phase13/data/sources_v1/holdout.jsonl").resolve(),
                (ROOT / "phase13/data/raw/gsm8k_test.jsonl").resolve()}
    scan_roots = [ROOT / f"phase{i}" for i in range(1, 14)] + [ROOT / "outputs"]
    files, exact, near = 0, Counter(), {}
    for base in scan_roots:
        for path in base.rglob("*"):
            if path.suffix not in (".json", ".jsonl") or not path.is_file():
                continue
            if path.resolve() in excluded or "phase13_v1" in path.parts or "phase13_remote_v100" in path.parts:
                continue
            if path.stat().st_size > 400_000_000:
                continue
            files += 1
            try:
                recs = records(path)
                for rec, _ in recs:
                    stack = [rec]
                    while stack:
                        obj = stack.pop()
                        if isinstance(obj, dict):
                            stack.extend(obj.values())
                            q = obj.get("question")
                            if isinstance(q, str) and q.strip():
                                key = question_key(q)
                                if key in target_keys:
                                    exact[target_keys[key]] += 1
                                sh = shingles(q)
                                hits = Counter(rid for s in sh for rid in index.get(s, ()))
                                for rid, n in hits.items():
                                    jac = n / len(sh | target_sh[rid])
                                    if jac >= 0.5 and jac > near.get(rid, (0, ""))[0]:
                                        near[rid] = (round(jac, 3), str(path.relative_to(ROOT)))
                        elif isinstance(obj, list):
                            stack.extend(obj)
            except Exception:
                continue
    # GSM8K train raw file is a sampling frame skipped above by "raw" folders; check it explicitly.
    train_hits = {}
    for rec, _ in records(ROOT / "phase5/data/raw/gsm8k_train.jsonl"):
        sh = shingles(rec["question"])
        hits = Counter(rid for s in sh for rid in index.get(s, ()))
        for rid, n in hits.items():
            jac = n / len(sh | target_sh[rid])
            if jac >= 0.5 and jac > train_hits.get(rid, 0):
                train_hits[rid] = round(jac, 3)
    return {"holdout_rows": len(holdout), "files_scanned": files,
            "exact_question_matches": len(exact),
            "near_duplicates_jaccard_8gram_ge_0.5": {rid: v for rid, v in sorted(near.items())},
            "near_duplicates_in_gsm8k_train_ge_0.5": dict(sorted(train_hits.items())),
            "near_duplicates_in_gsm8k_train_ge_0.8": {k: v for k, v in train_hits.items() if v >= 0.8}}


# ---------------------------------------------------------------- A2
def a2_splits() -> dict:
    out = {}
    for name, path in (("phase11_dpo", ROOT / "outputs/phase11_v1/dpo"),
                       ("phase12_dpo", ROOT / "outputs/phase12_v1/dpo"),
                       ("phase13_dpo_b", ROOT / "outputs/phase13_v1/dpo_b")):
        tr = {r["problem_id"] for r in read_jsonl(path / "train.jsonl")}
        va = {r["problem_id"] for r in read_jsonl(path / "validation.jsonl")}
        out[name] = {"train_problems": len(tr), "validation_problems": len(va), "overlap": len(tr & va)}
    out["probe_cv"] = ("phase13/layer/probe_analysis.py uses GroupKFold(groups=problem id): both orders and "
                       "both pairs (s1,s2),(s1,s3) of a problem fall in the same fold; cross-dataset probes "
                       "train on SVAMP and test on GSM8K (disjoint by construction).")
    return out


# ---------------------------------------------------------------- A3
def a3_parse(d: dict) -> dict:
    out = {}
    for judge, texts in d["judges"].items():
        n = len(texts)
        no_answer = sum(_extract_answer(t) is None for t in texts.values())
        unparsed = sum(parse(t) is None for t in texts.values())
        capped = sum(bool(m.get("hit_token_cap")) for m in d["meta"][judge].values())
        entry = {"judgments": n, "no_extractable_answer": no_answer / n,
                 "unparseable_answer": unparsed / n, "hit_768_token_cap": capped / n}
        if judge in CONSTRAINED:
            entry["missing_verdict_line"] = sum(not _VERDICT.findall(t) for t in texts.values()) / n
        out[judge] = entry
    return out


# ---------------------------------------------------------------- B1, B2, B4
def judge_tables(d: dict) -> dict:
    pairs = one_right_pairs(d)
    out = {}
    for judge in d["judges"]:
        t = Counter()
        single, single_g, pair_rows = [], [], []
        for pid, k in pairs:
            right = "first" if d["correct"][(pid, 1)] else "other"
            pk = picks_for(d, judge, pid, k)
            for order in ("ab", "ba"):
                right_pos = "A" if (order == "ab") == (right == "first") else "B"
                raw = pk[f"{order}_raw"]
                t[(right_pos, raw)] += 1
                single.append(float(raw == right_pos)); single_g.append(pid)
            consistent = pk["ab"] == pk["ba"] and pk["ab"] != "invented"
            pair_rows.append({"pid": pid, "consistent": consistent,
                              "right": consistent and pk["ab"] == right,
                              "tie": 1.0 if (consistent and pk["ab"] == right) else 0.5 if not consistent else 0.0})
        n_a = sum(t[("A", x)] for x in ("A", "B", "invented"))
        n_b = sum(t[("B", x)] for x in ("A", "B", "invented"))
        pa_given_a, pa_given_b = t[("A", "A")] / n_a, t[("B", "A")] / n_b
        # Skill index with a problem-clustered bootstrap over single-order judgments.
        per = []
        for pid, k in pairs:
            right = "first" if d["correct"][(pid, 1)] else "other"
            pk = picks_for(d, judge, pid, k)
            for order in ("ab", "ba"):
                right_pos = "A" if (order == "ab") == (right == "first") else "B"
                per.append((pid, right_pos, pk[f"{order}_raw"] == "A"))
        groups = np.array([p for p, _, _ in per])
        ids = np.unique(groups)
        rng = np.random.default_rng(RNG_SEED)
        by = {g: [i for i, x in enumerate(per) if x[0] == g] for g in ids}
        boot = []
        for _ in range(5000):
            take = [i for g in rng.choice(ids, len(ids)) for i in by[g]]
            a = [per[i][2] for i in take if per[i][1] == "A"]
            b = [per[i][2] for i in take if per[i][1] == "B"]
            boot.append(np.mean(a) - np.mean(b))
        cons = [r for r in pair_rows if r["consistent"]]
        sv, sg = np.array(single), np.array(single_g)
        tie = np.array([r["tie"] for r in pair_rows]); tg = np.array([r["pid"] for r in pair_rows])
        out[judge] = {
            "B1_position_table": {f"right_is_{pos}": {"n": n, **{f"picks_{x}": t[(pos, x)] / n for x in ("A", "B", "invented")}}
                                  for pos, n in (("A", n_a), ("B", n_b))},
            "B1_skill_index": pa_given_a - pa_given_b,
            "B1_skill_index_ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "B1_position2_rate_among_AB": (t[("A", "B")] + t[("B", "B")]) /
                                          max(1, sum(t[(p, x)] for p in "AB" for x in "AB")),
            "B2_tie_break": float(tie.mean()), "B2_tie_break_ci": cluster_boot(tie, tg),
            "B2_consistent_accuracy": (sum(r["right"] for r in cons) / len(cons)) if cons else None,
            "B2_coverage": len(cons) / len(pair_rows),
            "B2_right_over_all_pairs": sum(r["right"] for r in pair_rows) / len(pair_rows),
            "B4_single_order_accuracy": float(sv.mean()), "B4_single_order_ci": cluster_boot(sv, sg),
            "_tie": tie, "_tg": tg}
    out["_pairs"] = len(pairs)
    out["_problems"] = len({p for p, _ in pairs})
    return out


def b3_paired(tables: dict) -> dict:
    comps = [("base", "p12"), ("base", "p13b"), ("base_c", "p13b"), ("p12", "p13b")]
    raw, res = {}, {}
    for i, (a, b) in enumerate(comps):
        diff = tables[b]["_tie"] - tables[a]["_tie"]
        ci95, p = cluster_boot_p(diff, tables[a]["_tg"], RNG_SEED + i)
        res[f"{b}_minus_{a}"] = {"tie_break_difference": float(diff.mean()), "ci": ci95, "p_boot": p}
        raw[f"{b}_minus_{a}"] = p
    for key, p in holm(raw).items():
        res[key]["p_holm"] = p
    return res


# ---------------------------------------------------------------- B5, F3
def b5_f3(d: dict) -> dict:
    verdicts = {}
    for judge in d["judges"]:
        for (pid, k, order) in d["judges"][judge]:
            if k != 2 or order != "ab":
                continue
            pk = picks_for(d, judge, pid, 2)
            verdicts[(judge, pid)] = pk["ab"] if pk["ab"] == pk["ba"] and pk["ab"] != "invented" else None
    acc = defaultdict(list)
    both_wrong = one_right = disagree = 0
    for pid in d["rows"]:
        c = {k: d["correct"][(pid, k)] for k in range(1, 6)}
        p = {k: d["parsed"][(pid, k)] for k in range(1, 6)}
        s = {k: d["samples"][(pid, k)] for k in range(1, 6)}
        maj3_k = [1, 2, 3][vote([s[1], s[2], s[3]], [1, 2, 0])]
        maj5_k = [1, 2, 3, 4, 5][vote([s[k] for k in range(1, 6)], [1, 2, 3, 4, 0])]
        agree = same(p[1], p[2])
        acc["keep"].append(c[1]); acc["vote3"].append(c[maj3_k]); acc["vote5"].append(c[maj5_k])
        acc["agree_gated"].append(c[1] if agree else c[maj3_k])
        acc["random_s1_s2"].append(c[1] if agree else 0.5 * (c[1] + c[2]))
        if not agree:
            disagree += 1
            if c[1] != c[2]:
                one_right += 1
            elif not c[1]:
                both_wrong += 1
        acc["oracle_s1_s2"].append(c[1] if agree else max(c[1], c[2]))
        acc["oracle_on_one_right_else_vote3"].append(c[1] if agree else (True if c[1] != c[2] else c[maj3_k]))
        for judge in d["judges"]:
            v = None if agree else verdicts.get((judge, pid))
            if agree:
                acc[f"self_check_{judge}"].append(c[1]); acc[f"judge_or_random_{judge}"].append(c[1])
            else:
                chosen = c[1] if v == "first" else c[2] if v == "other" else None
                acc[f"self_check_{judge}"].append(chosen if chosen is not None else c[maj3_k])
                acc[f"judge_or_random_{judge}"].append(chosen if chosen is not None else 0.5 * (c[1] + c[2]))
    res = {k: float(np.mean(v)) for k, v in acc.items()}
    return {"accuracy": res,
            "F3": {"problems": len(d["rows"]), "s1_s2_disagree": disagree,
                   "disagree_one_right": one_right, "disagree_both_wrong": both_wrong,
                   "both_wrong_share_of_disagreements": both_wrong / max(1, disagree),
                   "ceiling_perfect_choice_s1_s2": res["oracle_s1_s2"],
                   "ceiling_perfect_on_one_right_else_vote3": res["oracle_on_one_right_else_vote3"]}}


# ---------------------------------------------------------------- C2
def surface(text: str) -> list[float]:
    value = parse(text)
    try:
        num = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        num = 0.0
    finite = math.isfinite(num)
    return [len(text), len(text.split()), text.count("\n"), len(re.findall(r"\d+(?:\.\d+)?", text)),
            math.log10(abs(num) + 1) if finite else 0.0, float(finite and num == int(num)) if finite else 0.0,
            float(value is not None), float(num < 0) if finite else 0.0]


def c2_rows(d: dict):
    x, y, g = [], [], []
    for pid, k in one_right_pairs(d):
        first_ok = d["correct"][(pid, 1)]
        for order in ("ab", "ba"):
            a, b = (d["samples"][(pid, 1)], d["samples"][(pid, k)]) if order == "ab" else \
                   (d["samples"][(pid, k)], d["samples"][(pid, 1)])
            fa, fb = surface(a), surface(b)
            x.append(fa + fb + [vb - va for va, vb in zip(fa, fb)])
            y.append(int((order == "ab") != first_ok))  # 1 = right solution shown as B
            g.append(pid)
    return np.array(x), np.array(y), np.array(g)


def c2_surface(svamp: dict, gsm: dict) -> dict:
    xs, ys, gs = c2_rows(svamp)
    xg, yg, gg = c2_rows(gsm)
    out = {}
    for c in (0.01, 1.0):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
        model.fit(xs, ys)
        cross = float((model.predict(xg) == yg).mean())
        cv = []
        for tr, te in GroupKFold(5).split(xg, yg, gg):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(xg[tr], yg[tr])
            cv.append(float((m.predict(xg[te]) == yg[te]).mean()))
        out[f"C={c}"] = {"svamp_to_gsm8k": cross, "within_gsm8k_cv": float(np.mean(cv))}
    out["features"] = ("per solution: chars, words, newlines (steps), number count, log10|answer|, "
                       "integer answer, answer parsed, negative answer; for A, B, and B-A")
    return out


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if not str(k).startswith("_")}
    return obj


def main() -> None:
    gsm, svamp = load("gsm8k"), load("svamp")
    report = {"schema_version": "paper_cpu_checks_v1", "post_hoc": True}
    report["A1_contamination_audit"] = a1_audit()
    report["A2_split_integrity"] = a2_splits()
    report["A3_parse_rates"] = {"gsm8k": a3_parse(gsm), "svamp": a3_parse(svamp)}
    tables = {"gsm8k": judge_tables(gsm), "svamp": judge_tables(svamp)}
    report["B1_B2_B4"] = {name: {"one_right_pairs": t["_pairs"], "problems": t["_problems"],
                                 **{j: strip(v) for j, v in t.items() if not j.startswith("_")}}
                          for name, t in tables.items()}
    report["B3_paired_gsm8k"] = b3_paired(tables["gsm8k"])
    report["B5_F3"] = {"gsm8k": b5_f3(gsm), "svamp": b5_f3(svamp)}
    report["C2_surface_baseline"] = c2_surface(svamp, gsm)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in report.items() if k != "A1_contamination_audit"}, indent=1, ensure_ascii=False)[:9000])
    print(json.dumps({k: v for k, v in report["A1_contamination_audit"].items()}, indent=1)[:2500])


if __name__ == "__main__":
    main()
