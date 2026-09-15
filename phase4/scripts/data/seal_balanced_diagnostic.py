"""Fix a balanced diagnostic from verified natural selected-policy initials."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1"))
from phase4.scripts.collect_warmstart import read_jsonl, verify, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--initials", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-bucket", type=int, default=20)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to replace fixed diagnostic")
    problems = {x["id"]:x for x in read_jsonl(args.problems)}
    buckets = defaultdict(list)
    for row in read_jsonl(args.initials):
        passed = verify(row, row["initial_output"])
        if passed != bool(row["initial_correct"]):
            raise ValueError(f"Initial label mismatch: {row['problem_id']}")
        buckets[(row["domain"],passed)].append(row)
    counts = {f"{domain}_{label}":len(buckets[(domain,label)]) for domain in ('math','code') for label in (True,False)}
    if min(counts.values()) < args.per_bucket:
        raise ValueError(f"Insufficient natural coverage for fixed {args.per_bucket}/bucket: {counts}")
    selected=[]
    for key in sorted(buckets):
        values=sorted(buckets[key],key=lambda x: hashlib.sha256(f"20260915:fresh:{x['problem_id']}".encode()).hexdigest())
        selected.extend(values[:args.per_bucket])
    selected.sort(key=lambda x:x['problem_id'])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    hashes={
        'problems':write_jsonl(args.output_dir/'problems.jsonl',[problems[x['problem_id']] for x in selected]),
        'initials':write_jsonl(args.output_dir/'initials.jsonl',selected),
    }
    report={'rows':len(selected),'per_bucket':args.per_bucket,'candidate_bucket_counts':counts,
            'sha256':hashes,'model_outputs_used_for_selection':'natural selected-V2 initial correctness only',
            'purpose':'fixed fresh development diagnostic; not scientific confirmation',
            'confirmation_candidates_read':False}
    (args.output_dir/'manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
