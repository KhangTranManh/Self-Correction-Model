"""Prepare fresh GSM8K/HumanEval sources without opening confirmation data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from phase4.scripts.data.prepare_math_expansion import ANSWER_RE, CALC_RE, existing_problem_hashes, stable_rank, text_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to replace existing diagnostic sources")
    historical_blocked = existing_problem_hashes()
    blocked = set(historical_blocked)
    for relative in ("phase4/data/pilot_v1/train_problems.jsonl", "phase4/data/pilot_v1/dev_problems.jsonl",
                     "phase4/data/expansion_v1/train_problems.jsonl"):
        with (ROOT / relative).open(encoding="utf-8") as handle:
            for line in handle:
                blocked.add(text_hash(json.loads(line)["question"]))
    eligible = []
    for index, source in enumerate(load_dataset("openai/gsm8k", "main", split="train")):
        raw = str(source["answer"])
        match = ANSWER_RE.search(raw)
        if match and text_hash(source["question"]) not in historical_blocked:
            eligible.append((index, source, match.group(1).replace(",", "")))
    eligible.sort(key=lambda x: stable_rank(x[0]))
    # Reconstruct the original allocation order after the historical exclusion.
    # The first 800 positions include sealed candidates and are skipped by index,
    # without reading their file or their model outcomes.
    rows = []
    for index, source, answer in eligible[2800:3000]:
        rows.append({"id": f"phase4_fresh_gsm8k_{index:04d}", "domain": "math",
                     "question": source["question"], "reference_answer": answer,
                     "calc_steps": [[a.strip(), b.strip()] for a,b in CALC_RE.findall(source["answer"])],
                     "dataset": "gsm8k", "source_split": "train", "source_index": index})
    if len(rows) != 200:
        raise ValueError("Insufficient untouched math sources")
    for source in load_dataset("openai/openai_humaneval", split="test"):
        if text_hash(source["prompt"]) in blocked:
            continue
        rows.append({"id": "phase4_fresh_" + source["task_id"].replace("/", "_"),
                     "domain": "code", "question": source["prompt"], "entry_point": source["entry_point"],
                     "tests": [source["test"], f"check({source['entry_point']})"],
                     "dataset": "humaneval", "source_split": "test", "source_id": source["task_id"]})
    if any(text_hash(row["question"]) in blocked for row in rows):
        raise ValueError("Diagnostic source overlap detected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    (args.output_dir / "candidate_problems.jsonl").write_text(payload, encoding="utf-8", newline="\n")
    report = {"rows": len(rows), "math": 200, "code": len(rows)-200,
              "sha256": hashlib.sha256(payload.encode()).hexdigest(),
              "confirmation_candidates_read": False, "purpose": "fresh development diagnostic; not confirmation"}
    (args.output_dir / "source_report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
