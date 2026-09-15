"""Evaluate policies on fixed initials and report per-seed transition metrics."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase4.scripts.collect_warmstart import read_jsonl


def metrics(rows):
    counts=Counter(x['decision'] for x in rows)
    correct=[x for x in rows if x['initial_correct']]
    wrong=[x for x in rows if not x['initial_correct']]
    keep=sum(x['decision']=='KEEP' for x in correct)/len(correct) if correct else None
    revise=sum(x['decision']=='REVISE' and x['contract_valid'] for x in wrong)/len(wrong) if wrong else None
    return {
        'rows':len(rows),'initial_accuracy':sum(x['initial_correct'] for x in rows)/len(rows),
        'final_accuracy':sum(x['final_correct'] for x in rows)/len(rows),
        'wrong_to_correct':sum(not x['initial_correct'] and x['final_correct'] for x in rows),
        'correct_to_wrong':sum(x['initial_correct'] and not x['final_correct'] for x in rows),
        'contract_valid_rate':sum(x['contract_valid'] for x in rows)/len(rows),
        'decision_counts':dict(counts),'keep_recall':keep,'revise_recall':revise,
        'balanced_decision_accuracy':(keep+revise)/2 if keep is not None and revise is not None else None,
        'domains':{d:{'rows':len(v),'initial_accuracy':sum(x['initial_correct'] for x in v)/len(v),
                      'final_accuracy':sum(x['final_correct'] for x in v)/len(v)}
                   for d in sorted({x['domain'] for x in rows}) if (v:=[x for x in rows if x['domain']==d])},
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',action='append',required=True,help='NAME=PROBLEMS_PATH,INITIALS_PATH')
    parser.add_argument('--policy',action='append',required=True,help='LABEL=SERVING_MODEL')
    parser.add_argument('--seeds',type=int,nargs='+',default=[20260914,20260915,20260916])
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'phase4/configs/transition_rl_v1.yaml')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    protocol={'datasets':args.dataset,'policies':args.policy,'seeds':args.seeds,
              'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),'input_sha256':{}}
    for dataset in args.dataset:
        _, paths=dataset.split('=',1)
        for path in paths.split(','):
            protocol['input_sha256'][path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest=args.output_dir/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:
        raise ValueError('Existing evaluation has a different protocol')
    manifest.write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    summary={}
    for dataset in args.dataset:
        name, paths=dataset.split('=',1)
        problems,initials=paths.split(',')
        initial_by_id={x['problem_id']:x for x in read_jsonl(Path(initials))}
        expected_ids={x['id'] for x in read_jsonl(Path(problems))}
        summary[name]={}
        for policy in args.policy:
            label,model=policy.split('=',1)
            per_seed={}; combined=[]
            for seed in args.seeds:
                path=args.output_dir/f'{name}_{label}_{seed}.jsonl'
                if not path.exists():
                    subprocess.run([sys.executable,str(ROOT/'phase4/scripts/collect_rollouts.py'),
                        '--problems',problems,'--initial-rollouts',initials,'--output',str(path),
                        '--cycle-id',f'{name}-{label}-{seed}','--policy-checkpoint',model,
                        '--serving-model',model,'--seed',str(seed),'--config',str(args.config),
                        '--concurrency','8'],check=True)
                rows=read_jsonl(path)
                if len(rows)!=len(expected_ids) or {x['problem_id'] for x in rows}!=expected_ids:
                    raise ValueError(f'Incomplete or mismatched evaluation output: {path}')
                if any(x['initial_output']!=initial_by_id[x['problem_id']]['initial_output'] for x in rows):
                    raise ValueError(f'Initial answers changed in evaluation: {path}')
                per_seed[str(seed)]=metrics(rows); combined.extend(rows)
                print(f'EVALUATED {name} {label} {seed} accuracy={per_seed[str(seed)]["final_accuracy"]:.3f}',flush=True)
            summary[name][label]={'per_seed':per_seed,'aggregate':metrics(combined)}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
