"""Build Phase 13 direction-B DPO pairs: constrained verdicts, order-swapped.

Source: Phase 10 training-pool judgments (as in Phases 11-12). Only one-right
pairs are used, because a constrained verdict cannot name a third answer.

  chosen    the first verifier-correct judgment + "Verdict: Solution <right>"
  rejected  (a) a judgment that sided with the wrong solution
                + "Verdict: Solution <wrong>"
            (b) a judgment that invented a third answer, left without a valid
                verdict line
Every pair is also emitted with Solution A and B exchanged (prompt, judgment
labels, and verdict). The prompt is the constrained prompt from
constrained.py. Run with Python 3.10 + SymPy 1.14.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.core.schema import Problem  # noqa: E402
from src.data.verifiers.math import MathVerifier  # noqa: E402
from phase9.scripts.answers import parse, same  # noqa: E402
from phase11.scripts.build_dpo import MIRROR, POOL, checked, lf_sha256, read_jsonl  # noqa: E402
from phase12.scripts.build_dpo import swap_labels  # noqa: E402
from phase13.scripts.constrained import constrained_prompt, verdict_line  # noqa: E402

OUT = ROOT / "outputs/phase13_v1/dpo_b"
REPORT = ROOT / "phase13/data/dpo_b_v1_report.json"


def order(key: str) -> str:
    return hashlib.sha256(f"phase13_dpo_b_v1|{key}".encode()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite: {OUT}")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use Python 3.10 with SymPy 1.14.0 (matches the GPU verifier)")
    pool = {row["id"]: row for row in read_jsonl(POOL)}
    samples = checked("train_samples")
    candidates = checked("judge_candidates")
    verifier = MathVerifier()
    built, stats = [], Counter()
    for pair in read_jsonl(MIRROR / "train_pairs.jsonl"):
        kind = pair["kind"]
        if kind not in ("A_right", "B_right"):
            continue
        row = pool[pair["problem_id"]]
        gold = Problem(id=row["id"], domain="math", question=row["question"],
                       reference_answer=row["reference_answer"])
        a, b = samples[(row["id"], pair["a"])], samples[(row["id"], pair["b"])]
        texts = [candidates[(pair["pair_id"], j)] for j in range(1, 5)]
        ok = [bool(verifier.verify(gold, t).passed) for t in texts]
        if not any(ok) or all(ok):
            stats["skipped_no_contrast"] += 1
            continue
        right, wrong = ("A", "B") if kind == "A_right" else ("B", "A")
        pa, pb = parse(a), parse(b)
        wrong_value = pb if kind == "A_right" else pa
        chosen = texts[ok.index(True)].rstrip() + verdict_line(right)
        incorrect = [t for t, good in zip(texts, ok) if not good]
        sided = [t for t in incorrect if same(parse(t), wrong_value)]
        invented = [t for t in incorrect if not same(parse(t), pa) and not same(parse(t), pb)]
        rejected = []
        if sided:
            rejected.append(("sided_with_wrong_solution", sided[0].rstrip() + verdict_line(wrong)))
        if invented:
            rejected.append(("invented_third_answer", invented[0].rstrip()))
        if not rejected:
            stats["skipped_no_usable_rejection"] += 1
            continue
        problem = Problem(id=row["id"], domain="math", question=row["question"])
        prompts = {"original": constrained_prompt(problem, a, b), "swapped": constrained_prompt(problem, b, a)}
        for rtype, text in rejected:
            stats[f"{kind}:{rtype}"] += 1
            key = f"{pair['pair_id']}|{rtype}"
            built.append({"key": key, "problem_id": row["id"], "kind": kind, "rejected_type": rtype,
                          "order": "original", "prompt": prompts["original"],
                          "chosen": chosen, "rejected": text})
            built.append({"key": key + "|swapped", "problem_id": row["id"],
                          "kind": "B_right" if kind == "A_right" else "A_right", "rejected_type": rtype,
                          "order": "swapped", "prompt": prompts["swapped"],
                          "chosen": swap_labels(chosen), "rejected": swap_labels(text)})
    built.sort(key=lambda p: order(p["key"]))
    validation = {pid for pid in {p["problem_id"] for p in built} if int(order(pid)[:8], 16) % 10 == 0}
    OUT.mkdir(parents=True)
    report = {"schema_version": "phase13_dpo_b_data_v1",
              "source": "Phase 10 training-pool pairs and judge candidates (verified mirror)",
              "verifier": f"Python {sys.version.split()[0]}, SymPy",
              "construction": "constrained verdict line + order swap; one-right pairs only",
              "stats": dict(sorted(stats.items())), "pairs": len(built)}
    for split in ("train", "validation"):
        rows = [p for p in built if (p["problem_id"] in validation) == (split == "validation")]
        path = OUT / f"{split}.jsonl"
        path.write_text("".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n" for p in rows),
                        encoding="utf-8", newline="\n")
        report[split] = {"rows": len(rows), "sources": len({p["problem_id"] for p in rows}),
                         "kinds": dict(Counter(p["kind"] for p in rows)),
                         "orders": dict(Counter(p["order"] for p in rows)),
                         "rejected_types": dict(Counter(p["rejected_type"] for p in rows)),
                         "sha256": lf_sha256(path)}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
