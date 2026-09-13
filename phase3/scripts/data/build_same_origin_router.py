"""Fresh-verify stochastic generations and build same-origin router data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase3.lib.behavior import ProvenanceResolver, VerificationEngine

REVIEW = "Review your previous answer carefully and decide whether it should be kept or revised."
BAD_CODE = re.compile(r"syntax|timeout|exit_code|nameerror|typeerror|indentation|no code|empty", re.I)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)


def symbols(text: str) -> set[str]:
    return set(re.findall(r"(?:def|class)\s+([A-Za-z_]\w*)", text))


def stable(value: str) -> str:
    return hashlib.sha256(f"20260907|{value}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-pairs", type=int, default=100)
    args = parser.parse_args()
    manifest = {row["task_id"]: row for row in read_jsonl(Path(args.manifest).resolve())}
    candidates = []
    for path in args.candidate:
        candidates.extend(read_jsonl(Path(path).resolve()))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[row["task_id"]].append(row)
    resolver = ProvenanceResolver()
    verifier = VerificationEngine()
    accepted = []
    audit = []
    for task_id, task in manifest.items():
        source = resolver.resolve(task["problem_ref"], task["source_id"])
        correct = task["existing_verified_correct_answer"].strip()
        correct_check = verifier.verify(source, correct)
        options = []
        for candidate in grouped.get(task_id, []):
            wrong = candidate["raw_answer"].strip()
            ratio = min(len(correct), len(wrong)) / max(1, max(len(correct), len(wrong)))
            similarity = SequenceMatcher(None, correct, wrong).ratio()
            check = verifier.verify(source, wrong) if wrong else {"passed": False, "detail": "empty"}
            reasons = []
            if not correct_check["passed"]: reasons.append("existing_correct_failed_fresh_verifier")
            if check["passed"]: reasons.append("candidate_passed_verifier")
            if not wrong or len(wrong) < 30: reasons.append("empty_or_too_short")
            if wrong == correct: reasons.append("identical")
            if ratio < 0.50: reasons.append("length_ratio_below_0.5")
            if similarity < (0.35 if task["domain"] == "math" else 0.50): reasons.append("similarity_too_low")
            if task["domain"] == "code":
                if BAD_CODE.search(str(check["detail"])): reasons.append("non_semantic_code_failure")
                if symbols(correct) and not (symbols(correct) & symbols(wrong)): reasons.append("interface_mismatch")
            record = {
                "candidate_id": candidate["candidate_id"], "task_id": task_id,
                "source_id": task["source_id"], "dataset": task["dataset"], "domain": task["domain"],
                "model_origin": candidate["model_origin"], "length_ratio": ratio, "similarity": similarity,
                "fresh_verifier_passed": bool(check["passed"]), "fresh_verifier_detail": check["detail"],
                "accepted": not reasons, "rejection_reasons": reasons,
            }
            audit.append(record)
            if not reasons:
                options.append((similarity + ratio, candidate, record, correct_check))
        if options:
            _, candidate, record, correct_check = max(options, key=lambda item: (item[0], stable(item[1]["candidate_id"])))
            accepted.append({
                "pair_id": f"same_origin::{task['source_id']}", "source_id": task["source_id"],
                "problem_ref": task["problem_ref"], "dataset": task["dataset"], "domain": task["domain"],
                "model_origin": task["model_origin_to_sample"], "same_model_origin": True,
                "correct_answer": correct, "wrong_answer": candidate["raw_answer"].strip(),
                "correct_verifier_detail": correct_check["detail"], "wrong_verifier_detail": record["fresh_verifier_detail"],
                "similarity": record["similarity"], "length_ratio": record["length_ratio"],
                "candidate_id": candidate["candidate_id"],
            })
    accepted.sort(key=lambda row: (-row["similarity"], -row["length_ratio"], stable(row["pair_id"])))
    accepted = accepted[: args.max_pairs]
    # Group-safe 80/20 split; each pair contributes both labels and identical review wording.
    ordered = sorted(accepted, key=lambda row: stable(row["pair_id"]))
    dev_count = max(1, round(len(ordered) * 0.2)) if ordered else 0
    dev_ids = {row["pair_id"] for row in ordered[:dev_count]}
    train_rows, dev_rows = [], []
    for pair in accepted:
        task = manifest[f"same_origin_wrong::{pair['source_id']}"]
        for label, answer in (("KEEP", pair["correct_answer"]), ("REVISE", pair["wrong_answer"])):
            target = f"<decision>{label}</decision>"
            row = {
                "construction_id": f"same_origin_router::{pair['source_id']}::{label}",
                "pair_id": pair["pair_id"], "source_id": pair["source_id"], "dataset": pair["dataset"],
                "domain": pair["domain"], "label": label, "decision": label,
                "model_origin": pair["model_origin"], "same_model_origin": True,
                "messages": [task["problem_message"], {"role": "assistant", "content": answer}, {"role": "user", "content": REVIEW}, {"role": "assistant", "content": target}],
            }
            (dev_rows if pair["pair_id"] in dev_ids else train_rows).append(row)
    output = Path(args.output_dir).resolve()
    write_jsonl(output / "verified_pairs.jsonl", accepted)
    write_jsonl(output / "candidate_audit.jsonl", audit)
    write_jsonl(output / "train.jsonl", train_rows)
    write_jsonl(output / "dev.jsonl", dev_rows)
    summary = {
        "schema_version": "phase3_same_origin_router_v1", "candidate_rows": len(candidates),
        "accepted_pairs": len(accepted), "train_rows": len(train_rows), "dev_rows": len(dev_rows),
        "labels": dict(Counter(row["label"] for row in train_rows + dev_rows)),
        "datasets": dict(Counter(row["dataset"] for row in accepted)),
        "domains": dict(Counter(row["domain"] for row in accepted)),
        "fresh_correct_pass": len(accepted), "fresh_wrong_fail": len(accepted),
        "same_model_origin_pairs": len(accepted), "synthetic_wrong_answers": 0,
        "train_dev_source_overlap": sorted({r["source_id"] for r in train_rows} & {r["source_id"] for r in dev_rows}),
        "training_started": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
