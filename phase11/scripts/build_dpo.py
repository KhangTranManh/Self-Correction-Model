"""Build Phase 11 DPO preference pairs from Phase 10 training-pool judgments.

For each Phase 10 training pair (same problem, same Solution A and B), the
four sampled judgments are labeled by the verifier. A preference pair needs
at least one correct and one incorrect judgment of that same prompt:

  chosen    the first verifier-correct judgment
  rejected  for one-right pairs, preferably a judgment whose final answer
            sides with the wrong solution; otherwise the first incorrect one

A-right and B-right pairs are balanced exactly; both-wrong pairs are capped
at a quarter of that count. Split 90/10 by source. Only Phase 10 training-pool
sources are used; no holdout source of any phase is read.
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
from phase9.scripts.collect_checkpoint import judge_prompt  # noqa: E402

MIRROR = ROOT / "outputs/phase10_remote_v100/outputs/phase10_v1"
POOL = ROOT / "phase10/data/sources_v1/train_pool.jsonl"
OUT = ROOT / "outputs/phase11_v1/dpo"
REPORT = ROOT / "phase11/data/dpo_v1_report.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def checked(stage: str) -> dict[tuple, str]:
    summary = json.loads((MIRROR / stage / "summary.json").read_text(encoding="utf-8"))
    if lf_sha256(MIRROR / stage / "answers.jsonl") != summary["answers_sha256"]:
        raise RuntimeError(f"Phase 10 {stage} output changed")
    return {tuple(r["task"]): r["output"] for r in read_jsonl(MIRROR / stage / "answers.jsonl")}


def order(key: str) -> str:
    return hashlib.sha256(f"phase11_dpo_v1|{key}".encode()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite: {OUT}")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use Python 3.10 with SymPy 1.14.0 (matches the GPU verifier)")
    pool = {row["id"]: row for row in read_jsonl(POOL)}
    samples = checked("train_samples")
    candidates = checked("judge_candidates")
    verifier = MathVerifier()
    built = {"A_right": [], "B_right": [], "both_wrong": []}
    stats = Counter()
    for pair in read_jsonl(MIRROR / "train_pairs.jsonl"):
        kind = pair["kind"]
        if kind not in built:
            continue
        row = pool[pair["problem_id"]]
        problem = Problem(id=row["id"], domain="math", question=row["question"],
                          reference_answer=row["reference_answer"])
        a, b = samples[(row["id"], pair["a"])], samples[(row["id"], pair["b"])]
        texts = [candidates[(pair["pair_id"], j)] for j in range(1, 5)]
        ok = [bool(verifier.verify(problem, t).passed) for t in texts]
        stats[f"{kind}_pairs"] += 1
        if not any(ok) or all(ok):
            stats[f"{kind}_{'all_wrong' if not any(ok) else 'all_right'}"] += 1
            continue
        chosen = texts[ok.index(True)]
        wrong_side = parse(b if kind == "A_right" else a) if kind != "both_wrong" else None
        incorrect = [t for t, good in zip(texts, ok) if not good]
        sided_wrong = [t for t in incorrect if wrong_side is not None and same(parse(t), wrong_side)]
        rejected = (sided_wrong or incorrect)[0]
        built[kind].append({
            "pair_id": pair["pair_id"], "problem_id": row["id"], "kind": kind,
            "rejected_type": "sided_with_wrong_solution" if sided_wrong else "other_incorrect",
            "prompt": judge_prompt(Problem(id=row["id"], domain="math", question=row["question"]), a, b),
            "chosen": chosen, "rejected": rejected})
    for kind in built:
        built[kind].sort(key=lambda ex: order(ex["pair_id"]))
    n = min(len(built["A_right"]), len(built["B_right"]))
    both = min(len(built["both_wrong"]), n // 4)
    chosen_rows = built["A_right"][:n] + built["B_right"][:n] + built["both_wrong"][:both]
    chosen_rows.sort(key=lambda ex: order(ex["pair_id"]))
    validation = {pid for pid in {ex["problem_id"] for ex in chosen_rows} if int(order(pid)[:8], 16) % 10 == 0}
    OUT.mkdir(parents=True)
    report = {"schema_version": "phase11_dpo_data_v1",
              "source": "Phase 10 training-pool pairs and judge candidates (verified mirror)",
              "verifier": f"Python {sys.version.split()[0]}, SymPy",
              "pair_stats": dict(sorted(stats.items())),
              "usable": {k: len(v) for k, v in built.items()},
              "selected": {"A_right": n, "B_right": n, "both_wrong": both}}
    for split in ("train", "validation"):
        rows = [ex for ex in chosen_rows if (ex["problem_id"] in validation) == (split == "validation")]
        path = OUT / f"{split}.jsonl"
        path.write_text("".join(json.dumps(ex, ensure_ascii=False, sort_keys=True) + "\n" for ex in rows),
                        encoding="utf-8", newline="\n")
        report[split] = {"rows": len(rows), "kinds": dict(Counter(ex["kind"] for ex in rows)),
                         "rejected_types": dict(Counter(ex["rejected_type"] for ex in rows)),
                         "sha256": lf_sha256(path)}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
