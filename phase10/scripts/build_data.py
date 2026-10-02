"""Build Phase 10 judge training data from the model's own verified outputs.

--pairs  Pair the first sample (A, initial prompt) with each blind sample (B)
         whose final answer differs, on training sources only. Label each side
         with the deterministic verifier. At most two pairs per source.
--sft    From four sampled judge outputs per pair, keep the first whose final
         answer the verifier accepts (rejection sampling). Balance A-right and
         B-right pairs exactly; allow both-wrong pairs up to a quarter of that
         count. Split 90/10 by source for training and validation loss.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1"))
from src.core.schema import Problem  # noqa: E402
from src.data.verifiers.math import MathVerifier  # noqa: E402
sys.path.insert(0, str(ROOT))

from phase9.scripts.answers import parse, same  # noqa: E402
from phase9.scripts.collect_checkpoint import judge_prompt  # noqa: E402
from phase10.scripts.generate import OUT, PAIRS, completed, read_jsonl, rows  # noqa: E402

SFT = OUT / "sft"
MAX_PAIRS_PER_SOURCE = 2
MAX_EXAMPLES = 1200


def order(key: str) -> str:
    return hashlib.sha256(f"phase10_sft_v1|{key}".encode()).hexdigest()


def build_pairs() -> None:
    samples = completed("train_samples")
    verifier = MathVerifier()
    pairs, kinds = [], Counter()
    for row in rows("train_pool"):
        problem = Problem(id=row["id"], domain="math", question=row["question"],
                          reference_answer=row["reference_answer"])
        a_text = samples[(row["id"], 1)]
        a_ok = bool(verifier.verify(problem, a_text).passed)
        taken = 0
        for b in (2, 3, 4):
            b_text = samples[(row["id"], b)]
            if taken == MAX_PAIRS_PER_SOURCE or same(parse(a_text), parse(b_text)):
                continue
            b_ok = bool(verifier.verify(problem, b_text).passed)
            kind = "A_right" if a_ok and not b_ok else "B_right" if b_ok and not a_ok else \
                "both_wrong" if not (a_ok or b_ok) else "both_right"
            kinds[kind] += 1
            pairs.append({"pair_id": f"{row['id']}|{b}", "problem_id": row["id"], "a": 1, "b": b,
                          "kind": kind})
            taken += 1
    PAIRS.write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in pairs),
                     encoding="utf-8", newline="\n")
    print(json.dumps({"pairs": len(pairs), "kinds": dict(kinds)}))


def build_sft() -> None:
    samples = completed("train_samples")
    candidates = completed("judge_candidates")
    by_id = {row["id"]: row for row in rows("train_pool")}
    verifier = MathVerifier()
    accepted = {"A_right": [], "B_right": [], "both_wrong": []}
    tried = Counter()
    for pair in read_jsonl(PAIRS):
        if pair["kind"] not in accepted:
            continue
        tried[pair["kind"]] += 1
        row = by_id[pair["problem_id"]]
        problem = Problem(id=row["id"], domain="math", question=row["question"],
                          reference_answer=row["reference_answer"])
        for j in range(1, 5):
            text = candidates[(pair["pair_id"], j)]
            if verifier.verify(problem, text).passed:
                prompt = judge_prompt(Problem(id=row["id"], domain="math", question=row["question"]),
                                      samples[(row["id"], pair["a"])], samples[(row["id"], pair["b"])])
                accepted[pair["kind"]].append({"pair_id": pair["pair_id"], "problem_id": row["id"],
                                               "kind": pair["kind"], "candidate": j,
                                               "messages": [{"role": "user", "content": prompt},
                                                            {"role": "assistant", "content": text}]})
                break
    for kind in accepted:
        accepted[kind].sort(key=lambda ex: order(ex["pair_id"]))
    n = min(len(accepted["A_right"]), len(accepted["B_right"]), MAX_EXAMPLES * 4 // 9)
    both = min(len(accepted["both_wrong"]), n // 4)
    chosen = accepted["A_right"][:n] + accepted["B_right"][:n] + accepted["both_wrong"][:both]
    chosen.sort(key=lambda ex: order(ex["pair_id"]))
    val_sources = {pid for pid in {ex["problem_id"] for ex in chosen}
                   if int(order(pid)[:8], 16) % 10 == 0}
    SFT.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "phase10_sft_v1", "pairs_tried": dict(tried),
              "accepted": {k: len(v) for k, v in accepted.items()},
              "selected": {"A_right": n, "B_right": n, "both_wrong": both}}
    for split in ("train", "validation"):
        subset = [ex for ex in chosen if (ex["problem_id"] in val_sources) == (split == "validation")]
        path = SFT / f"{split}.jsonl"
        path.write_text("".join(json.dumps(ex, ensure_ascii=False, sort_keys=True) + "\n" for ex in subset),
                        encoding="utf-8", newline="\n")
        report[f"{split}_rows"] = len(subset)
        report[f"{split}_kinds"] = dict(Counter(ex["kind"] for ex in subset))
        report[f"{split}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (SFT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", action="store_true")
    parser.add_argument("--sft", action="store_true")
    args = parser.parse_args()
    if args.pairs:
        build_pairs()
    if args.sft:
        build_sft()


if __name__ == "__main__":
    main()
