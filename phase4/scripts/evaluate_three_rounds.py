"""Resumable, paired three-round autonomous review diagnostic."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import random
import sys

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phase4.lib.rollout import as_problem, completion, read_jsonl, sha256, verify
from phase4.lib.transition import parse_review
from src.core.prompts import build_prompt


def next_answer(current, review):
    action = parse_review(review)
    return (action.revised_answer if action.valid and action.decision == 'REVISE'
            else current), action


def summarize(rows, seeds):
    result = {'sources': len(rows) // len(seeds), 'seeds': {}, 'rounds': {}}
    for seed in seeds:
        group = [r for r in rows if r['seed'] == seed]
        result['seeds'][str(seed)] = metrics(group)
    result['rounds'] = metrics(rows)
    buckets = {}
    for r in rows:
        buckets.setdefault((r['domain'], r['initial_correct']), {}).setdefault(r['problem_id'], []).append(r)
    differences = []
    for bucket in buckets.values():
        differences.append([sum(int(r['rounds'][2]['correct']) - int(r['rounds'][0]['correct'])
                                for r in source) / len(source) for source in bucket.values()])
    rng = random.Random(20260915)
    samples = sorted(sum(sum(rng.choices(b, k=len(b))) for b in differences)
                     / result['sources'] for _ in range(10000))
    result['round3_minus_round1_accuracy_95pct_source_bootstrap'] = [samples[249], samples[9749]]
    m = result['rounds']
    result['preliminary_benefit'] = (
        m['3']['accuracy'] > m['1']['accuracy']
        and m['3']['initial_to_final_harms'] <= m['1']['initial_to_final_harms']
        and m['3']['domain_accuracy']['code'] >= m['1']['domain_accuracy']['code']
        and m['3']['contract_validity'] >= m['1']['contract_validity'])
    result['confirmation_claim'] = False
    return result


def metrics(rows):
    output = {'initial_accuracy': sum(r['initial_correct'] for r in rows) / len(rows)}
    for i in range(3):
        def before(r):
            return r['initial_correct'] if i == 0 else r['rounds'][i-1]['correct']
        output[str(i+1)] = {
            'accuracy': sum(r['rounds'][i]['correct'] for r in rows) / len(rows),
            'contract_validity': sum(r['rounds'][i]['contract_valid'] for r in rows) / len(rows),
            'consecutive_fixes': sum(not before(r) and r['rounds'][i]['correct'] for r in rows),
            'consecutive_harms': sum(before(r) and not r['rounds'][i]['correct'] for r in rows),
            'initial_to_final_fixes': sum(not r['initial_correct'] and r['rounds'][i]['correct'] for r in rows),
            'initial_to_final_harms': sum(r['initial_correct'] and not r['rounds'][i]['correct'] for r in rows),
            'domain_accuracy': {d: sum(r['rounds'][i]['correct'] for r in rows if r['domain'] == d)
                                / sum(r['domain'] == d for r in rows) for d in ('math', 'code')},
        }
    return output


async def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))['rollout']
    problems = read_jsonl(args.problems)
    initials = {r['problem_id']: r for r in read_jsonl(args.initials)}
    if len(problems) != 80 or len({r['id'] for r in problems}) != 80:
        raise ValueError('Expected fixed 80-source diagnostic')
    protocol = {'problems_sha256': sha256(args.problems), 'initials_sha256': sha256(args.initials),
                'config_sha256': sha256(args.config), 'model': args.model,
                'seeds': args.seeds, 'rounds': 3, 'invalid_policy': 'retain_current_answer',
                'seed_formula': 'seed + 1000000 + source_index*10 + round_index',
                'verifier_feedback': False}
    protocol_path = args.output_dir / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Existing protocol mismatch')
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    audit = args.output_dir / 'trajectories.jsonl'
    done_rows = read_jsonl(audit) if audit.exists() else []
    done = {(r['seed'], r['problem_id']) for r in done_rows}
    if len(done) != len(done_rows):
        raise ValueError('Duplicate audit trajectories')
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip('/')+'/',
                                timeout=httpx.Timeout(600, connect=30)) as client:
        async def trajectory(row, index, seed):
            if (seed, row['id']) in done:
                return
            problem = as_problem(row)
            current = str(initials[problem.id]['initial_output'])
            initial = await asyncio.to_thread(verify, problem, current)
            if initial['passed'] != initials[problem.id]['initial_correct']:
                raise ValueError('Initial verifier mismatch: '+problem.id)
            record = {'problem_id': problem.id, 'domain': problem.domain, 'seed': seed,
                      'initial_output': current, 'initial_correct': initial['passed'], 'rounds': []}
            for round_index in range(3):
                request_seed = seed + 1000000 + index*10 + round_index
                review = await completion(client, semaphore, model=args.model,
                    messages=[{'role':'system','content':cfg['review_system_prompt']},
                              {'role':'user','content':build_prompt(problem)},
                              {'role':'assistant','content':current},
                              {'role':'user','content':cfg['neutral_prompt']}],
                    temperature=float(cfg['review_temperature']),
                    max_tokens=int(cfg['review_max_tokens']), seed=request_seed)
                current, action = next_answer(current, review)
                outcome = await asyncio.to_thread(verify, problem, current)
                record['rounds'].append({'round':round_index+1, 'request_seed':request_seed,
                    'review_output':review, 'answer':current, 'decision':action.decision,
                    'contract_valid':action.valid, 'correct':outcome['passed'],
                    'verifier_detail':outcome['detail']})
            with audit.open('a', encoding='utf-8', newline='\n') as handle:
                handle.write(json.dumps(record, ensure_ascii=False)+'\n')
                handle.flush()
                os.fsync(handle.fileno())
            done_rows.append(record)
            print(f'SAVED {len(done_rows)}/240 trajectories', flush=True)
        # Bound whole trajectories as well as HTTP requests.
        queue = asyncio.Queue()
        for seed in args.seeds:
            for index, row in enumerate(problems):
                queue.put_nowait((row,index,seed))
        async def worker():
            while not queue.empty():
                row,index,seed = queue.get_nowait()
                await trajectory(row,index,seed)
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    if len(done_rows) != 240:
        raise ValueError('Incomplete audit')
    result = summarize(done_rows, args.seeds)
    result['trajectories_sha256'] = sha256(audit)
    (args.output_dir/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problems', type=Path, default=Path('phase4/data/fresh_diagnostic_v1/fixed/problems.jsonl'))
    parser.add_argument('--initials', type=Path, default=Path('phase4/data/fresh_diagnostic_v1/fixed/initials.jsonl'))
    parser.add_argument('--config', type=Path, default=Path('phase4/configs/cycle000_rollout.yaml'))
    parser.add_argument('--output-dir', type=Path, default=Path('phase4/runs/three_round_v1'))
    parser.add_argument('--model', default='phase4-actor')
    parser.add_argument('--base-url', default='http://127.0.0.1:8999/v1')
    parser.add_argument('--seeds', type=int, nargs=3, default=[20260914,20260915,20260916])
    parser.add_argument('--concurrency', type=int, default=4)
    asyncio.run(run(parser.parse_args()))
