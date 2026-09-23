"""Store model-hidden reference arithmetic steps for frozen Phase 5 candidates."""

import argparse
import hashlib
import json
from pathlib import Path
import re


STEP = re.compile(r"<<([^=<>]+)=([^<>]+)>>")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("phase5/data/raw/gsm8k_train.jsonl"))
    parser.add_argument("--candidates", type=Path,
                        default=Path("phase5/data/candidates_v1/candidate_problems.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("phase5/data/candidates_v1/reference_steps.jsonl"))
    args = parser.parse_args()
    raw = read_jsonl(args.raw)
    candidates = read_jsonl(args.candidates)
    rows = []
    for candidate in candidates:
        source = raw[candidate["dataset_index"]]
        if source["question"] != candidate["question"]:
            raise ValueError(f"Question/index mismatch: {candidate['id']}")
        steps = [[expression.strip(), declared.strip()]
                 for expression, declared in STEP.findall(source["answer"])]
        if len(steps) != candidate["reference_step_count"]:
            raise ValueError(f"Step-count mismatch: {candidate['id']}")
        rows.append({"id": candidate["id"], "reference_steps": steps})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                   for row in rows), encoding="utf-8", newline="\n")
    report = {"rows": len(rows), "raw_sha256": sha256(args.raw),
              "candidates_sha256": sha256(args.candidates),
              "reference_steps_sha256": sha256(args.output),
              "model_visible": False}
    report_path = args.output.with_name("reference_steps_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
