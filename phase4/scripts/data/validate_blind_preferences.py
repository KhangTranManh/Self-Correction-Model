"""Freshly verify real-output pairs against their collection audit and prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'phase1'))
from phase4.lib.transition import parse_review
from phase4.scripts.collect_warmstart import read_jsonl,verify


def validate_pairs(train_path,dev_path,audit_path,initials_path,config_path):
    attempts=read_jsonl(audit_path)
    audit={(x['problem_id'],x['candidate_index']):x for x in attempts}
    if len(audit)!=len(attempts): raise ValueError('Duplicate audit generations')
    initials={x['problem_id']:x for x in read_jsonl(initials_path)}
    manifest=json.loads((audit_path.parent/'collection_manifest.json').read_text())
    if manifest['initials_sha256']!=hashlib.sha256(initials_path.read_bytes()).hexdigest() or manifest['config_sha256']!=hashlib.sha256(config_path.read_bytes()).hexdigest():
        raise ValueError('Collection input provenance changed')
    rollout=yaml.safe_load(config_path.read_text(encoding='utf-8'))['rollout']
    seen=set(); counts={}
    for split,path in (('train',train_path),('dev',dev_path)):
        rows=read_jsonl(path)
        counts[split]={'KEEP':0,'REVISE':0}
        for pair in rows:
            key=pair['source_id']
            if pair.get('model_visible_verifier_hints') is not False:
                raise ValueError('Pair is not labeled as blind generation')
            if key in seen: raise ValueError('Duplicate source or train/dev overlap')
            seen.add(key); source=initials[key]
            correct=verify(source,source['initial_output'])
            expected=[{'role':'system','content':str(rollout['review_system_prompt'])},
                      {'role':'user','content':source['task_prompt']},
                      {'role':'assistant','content':source['initial_output']},
                      {'role':'user','content':str(rollout['neutral_prompt'])}]
            if pair['prompt']!=expected: raise ValueError(f'Non-deployment prompt: {key}')
            outcomes={}
            for side in ('chosen','rejected'):
                record=audit[(key,pair[f'{side}_candidate_index'])]
                output=pair[side][0]['content']
                if pair[side]!=[{'role':'assistant','content':output}]:
                    raise ValueError('Pair side must contain exactly one assistant response')
                if record['split']!=split or record['output']!=output or record['model_visible_verifier_hints'] is not False:
                    raise ValueError(f'Pair does not match its blind audit: {key}')
                action=parse_review(output)
                if not action.valid: raise ValueError('Pair contains invalid contract')
                passed=correct if action.decision=='KEEP' else verify(source,action.revised_answer or '')
                if passed!=record['final_correct'] or correct!=record['initial_correct']:
                    raise ValueError(f'Fresh verifier disagrees with audit: {key}')
                outcomes[side]=(action.decision,passed)
            if correct:
                if outcomes['chosen']!=('KEEP',True) or outcomes['rejected']!=('REVISE',False):
                    raise ValueError('KEEP pair lacks a real harmful revision')
                label='KEEP'
            else:
                if outcomes['chosen']!=('REVISE',True) or outcomes['rejected'][1]:
                    raise ValueError('REVISE pair lacks a verified fix and real failure')
                label='REVISE'
            if pair['decision']!=label: raise ValueError('Pair label mismatch')
            counts[split][label]+=1
        if counts[split]['KEEP']!=counts[split]['REVISE']: raise ValueError('Action imbalance')
    return {'all_passed':True,'fresh_verification':True,'counts':counts,'sha256':{
        'train':hashlib.sha256(train_path.read_bytes()).hexdigest(),
        'dev':hashlib.sha256(dev_path.read_bytes()).hexdigest(),
        'audit':hashlib.sha256(audit_path.read_bytes()).hexdigest()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('train','dev','audit','initials','config','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    report=validate_pairs(args.train,args.dev,args.audit,args.initials,args.config)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
