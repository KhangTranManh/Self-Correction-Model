"""Build Phase 12 order-swapped DPO pairs from Phase 10 training-pool judgments.

Extends Phase 11's construction in two ways:

1. Order swap. Every preference pair is also emitted with Solution A and B
   exchanged; standalone "A" and "B" tokens in the chosen and rejected
   judgments are exchanged as well, so labels keep pointing at the same
   solution. This makes A-right and B-right exactly symmetric.
2. Invented-answer negatives. When a pair has an incorrect judgment whose
   final answer matches neither solution, it is added as a second rejected
   example for the same chosen judgment.

Only Phase 10 training-pool sources are read; no holdout of any phase.
Run with Python 3.10 + SymPy 1.14 (matches the GPU verifier).
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
sys.path.insert(0, str(ROOT))
from src.core.schema import Problem  # noqa: E402
from src.data.verifiers.math import MathVerifier  # noqa: E402
from phase9.scripts.answers import parse, same  # noqa: E402
from phase9.scripts.collect_checkpoint import judge_prompt  # noqa: E402
from phase11.scripts.build_dpo import MIRROR, POOL, checked, lf_sha256, read_jsonl  # noqa: E402

OUT = ROOT / "outputs/phase12_v1/dpo"
REPORT = ROOT / "phase12/data/dpo_v1_report.json"
BOTH_WRONG_SHARE = 0.25
_AB = re.compile(r"\b([AB])\b")


def swap_labels(text: str) -> str:
    return _AB.sub(lambda m: "B" if m.group(1) == "A" else "A", text)


def order(key: str) -> str:
    return hashlib.sha256(f"phase12_dpo_v1|{key}".encode()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite: {OUT}")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use Python 3.10 with SymPy 1.14.0 (matches the GPU verifier)")
    pool = {row["id"]: row for row in read_jsonl(POOL)}
    samples = checked("train_samples")
    candidates = checked("judge_candidates")
    verifier = MathVerifier()
    base_pairs, stats = [], Counter()
    for pair in read_jsonl(MIRROR / "train_pairs.jsonl"):
        kind = pair["kind"]
        if kind not in ("A_right", "B_right", "both_wrong"):
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
        chosen = texts[ok.index(True)]
        pa, pb = parse(a), parse(b)
        wrong_side = None if kind == "both_wrong" else (pb if kind == "A_right" else pa)
        incorrect = [t for t, good in zip(texts, ok) if not good]
        sided = [t for t in incorrect if wrong_side is not None and same(parse(t), wrong_side)]
        invented = [t for t in incorrect if not same(parse(t), pa) and not same(parse(t), pb)]
        rejected = []
        if sided:
            rejected.append(("sided_with_wrong_solution", sided[0]))
        if invented:
            rejected.append(("invented_third_answer", invented[0]))
        if not rejected:
            rejected.append(("other_incorrect", incorrect[0]))
        prompt = judge_prompt(Problem(id=row["id"], domain="math", question=row["question"]), a, b)
        swapped_prompt = judge_prompt(Problem(id=row["id"], domain="math", question=row["question"]), b, a)
        for rtype, text in rejected:
            stats[f"{kind}:{rtype}"] += 1
            key = f"{pair['pair_id']}|{rtype}"
            base_pairs.append({"key": key, "problem_id": row["id"], "kind": kind, "rejected_type": rtype,
                               "order": "original", "prompt": prompt, "chosen": chosen, "rejected": text})
            swapped_kind = {"A_right": "B_right", "B_right": "A_right"}.get(kind, kind)
            base_pairs.append({"key": key + "|swapped", "problem_id": row["id"], "kind": swapped_kind,
                               "rejected_type": rtype, "order": "swapped", "prompt": swapped_prompt,
                               "chosen": swap_labels(chosen), "rejected": swap_labels(text)})
    one_right = [p for p in base_pairs if p["kind"] != "both_wrong"]
    both_wrong = sorted((p for p in base_pairs if p["kind"] == "both_wrong"), key=lambda p: order(p["key"]))
    selected = one_right + both_wrong[:int(len(one_right) * BOTH_WRONG_SHARE)]
    selected.sort(key=lambda p: order(p["key"]))
    validation = {pid for pid in {p["problem_id"] for p in selected} if int(order(pid)[:8], 16) % 10 == 0}
    OUT.mkdir(parents=True)
    report = {"schema_version": "phase12_dpo_data_v1",
              "source": "Phase 10 training-pool pairs and judge candidates (verified mirror)",
              "verifier": f"Python {sys.version.split()[0]}, SymPy",
              "construction": "Phase 11 contrast + order swap (A/B tokens exchanged) + invented-answer negatives",
              "stats": dict(sorted(stats.items())), "selected": len(selected)}
    for split in ("train", "validation"):
        rows = [p for p in selected if (p["problem_id"] in validation) == (split == "validation")]
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
