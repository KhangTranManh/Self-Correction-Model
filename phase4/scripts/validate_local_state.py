"""Audit the closed Phase 4 archive without loading models or confirmation data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from phase4.lib.rollout import read_jsonl, sha256
from phase4.scripts.evaluate_three_rounds import next_answer, summarize


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(hash_adapters: bool = False):
    registry = yaml.safe_load((ROOT / 'phase4/configs/experiments.yaml').read_text(encoding='utf-8'))
    require(registry['status'] == 'closed', 'Phase 4 must be closed')
    require(registry['promotion'] == 'none', 'Unexpected model promotion')
    require(not registry['further_experiments_authorized'], 'Unexpected experiment authorization')
    require(not registry['confirmation_opened'] and not registry['confirmation_completed'],
            'Confirmation disposition changed')
    experiments = registry['experiments']
    for report in registry['reports']:
        require((ROOT / report).is_file(), f'Missing closure report: {report}')
    adapters = 0
    for name, experiment in experiments.items():
        summary = read_json(ROOT / experiment['summary_path'])
        if 'final_accuracy' in experiment:
            require(summary['final_accuracy'] == experiment['final_accuracy'],
                    f'{name}: registered accuracy differs from evidence')
        if 'rerun_summary_path' in experiment:
            rerun = read_json(ROOT / experiment['rerun_summary_path'])
            require(rerun['final_accuracy'] == experiment['rerun_final_accuracy'],
                    f'{name}: rerun accuracy mismatch')
        if 'adapter_path' in experiment:
            path = ROOT / experiment['adapter_path']
            require(path.is_file() and path.with_name('adapter_config.json').is_file(),
                    f'{name}: missing final adapter')
            if hash_adapters:
                require(sha256(path) == experiment['adapter_sha256'], f'{name}: adapter hash mismatch')
            adapters += 1

    fixed = ROOT / 'phase4/data/fresh_diagnostic_v1/fixed'
    manifest = read_json(fixed / 'manifest.json')
    for kind in ('problems', 'initials'):
        require(sha256(fixed / f'{kind}.jsonl') == manifest['sha256'][kind],
                f'Fixed diagnostic {kind} hash mismatch')
    problems = read_jsonl(fixed / 'problems.jsonl')
    initial_rows = read_jsonl(fixed / 'initials.jsonl')
    initials = {r['problem_id']: r for r in initial_rows}
    ids = {r['id'] for r in problems}
    require(len(problems) == len(ids) == len(initial_rows) == len(initials) == 80,
            'Fixed diagnostic must have 80 unique sources')
    require(ids == set(initials), 'Diagnostic source/initial ID mismatch')
    buckets = Counter((r['domain'], initials[r['id']]['initial_correct']) for r in problems)
    require(buckets == Counter({(d, c): 20 for d in ('math', 'code') for c in (True, False)}),
            'Diagnostic domain/correctness balance mismatch')

    blind_dir = ROOT / 'phase4/data/blind_preferences_v2'
    blind = read_json(blind_dir / 'summary.json')
    collection = read_json(blind_dir / 'collection_manifest.json')
    blind_rows = read_jsonl(blind_dir / 'blind_attempts.jsonl')
    require(len(blind_rows) == blind['attempts'] == experiments['blind_preference_v2']['attempts'] == 16064,
            'Blind collection incomplete')
    keys = {(r['problem_id'], r['candidate_index']) for r in blind_rows}
    require(len(keys) == len(blind_rows), 'Duplicate blind candidates')
    sources = Counter(r['problem_id'] for r in blind_rows)
    require(len(sources) == blind['sources'] == 1004 and set(sources.values()) == {collection['candidates']},
            'Blind per-source candidate coverage mismatch')
    require(dict(Counter(r['decision'] for r in blind_rows)) == blind['decision_counts'],
            'Blind decision counts mismatch')
    fixes = sum(not r['initial_correct'] and r['final_correct'] and r['contract_valid']
                for r in blind_rows)
    harms = sum(r['initial_correct'] and not r['final_correct'] and r['contract_valid']
                for r in blind_rows)
    require(fixes == blind['verified_fixes'] and harms == blind['harmful_revisions'],
            'Blind transition counts mismatch')
    require(all(not r['model_visible_verifier_hints'] for r in blind_rows), 'Blind audit contains hinted rows')
    require(not blind['training_gate_passed'], 'Unexpected blind training authorization')
    splits = {}
    for split in ('train', 'dev'):
        rows = read_jsonl(blind_dir / f'{split}.jsonl')
        require(len(rows) == blind[f'{split}_rows'], f'Blind {split} size mismatch')
        require(sha256(blind_dir / f'{split}.jsonl') == blind['sha256'][split],
                f'Blind {split} hash mismatch')
        counts = Counter(r['decision'] for r in rows)
        require(counts['KEEP'] == counts['REVISE'], f'Blind {split} action imbalance')
        splits[split] = {r['source_id'] for r in rows}
        require(len(splits[split]) == len(rows), f'Blind {split} repeated source')
    require(not splits['train'] & splits['dev'], 'Blind train/dev overlap')
    prepared = yaml.safe_load((ROOT / 'phase4/configs/preference_dpo_v2.yaml').read_text(encoding='utf-8'))
    require(prepared['experiment']['status'] == 'not_run_insufficient_blind_coverage',
            'Prepared DPO V2 status inconsistent with failed gate')

    run_dir = ROOT / 'phase4/runs/three_round_v1'
    protocol = read_json(run_dir / 'protocol.json')
    require(protocol['problems_sha256'] == manifest['sha256']['problems'] and
            protocol['initials_sha256'] == manifest['sha256']['initials'], 'Three-round inputs changed')
    require(protocol['config_sha256'] == sha256(ROOT / 'phase4/configs/cycle000_rollout.yaml'),
            'Frozen three-round settings changed')
    require(protocol['seeds'] == [20260914, 20260915, 20260916] and not protocol['verifier_feedback'],
            'Three-round protocol changed')
    rows = read_jsonl(run_dir / 'trajectories.jsonl')
    expected = {(seed, problem_id) for seed in protocol['seeds'] for problem_id in ids}
    require(len(rows) == 240 and {(r['seed'], r['problem_id']) for r in rows} == expected,
            'Three-round trajectories missing or duplicated')
    for row in rows:
        initial = initials[row['problem_id']]
        current = initial['initial_output']
        require(current == row['initial_output'] and initial['initial_correct'] == row['initial_correct'],
                'Three-round initial answer mismatch')
        require(len(row['rounds']) == 3, 'Incomplete trajectory')
        for index, step in enumerate(row['rounds'], start=1):
            current, action = next_answer(current, step['review_output'])
            require(step['round'] == index and current == step['answer'] and
                    action.valid == step['contract_valid'] and action.decision == step['decision'],
                    'Three-round state transition mismatch')
    summary = read_json(run_dir / 'summary.json')
    computed = summarize(rows, protocol['seeds'])
    require(all(summary[k] == v for k, v in computed.items()), 'Three-round metrics mismatch')
    require(sha256(run_dir / 'trajectories.jsonl') == summary['trajectories_sha256'] ==
            experiments['three_round_v1']['trajectories_sha256'], 'Three-round evidence hash mismatch')
    require(not summary['preliminary_benefit'] and not summary['confirmation_claim'],
            'Unexpected three-round benefit claim')
    return {'all_passed': True, 'phase_status': 'closed', 'promotion': 'none',
            'final_adapters_checked': adapters, 'adapter_hashes_checked': hash_adapters,
            'blind_attempts': len(blind_rows), 'three_round_trajectories': len(rows),
            'confirmation_data_read': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hash-adapters', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.hash_adapters), indent=2))
    except (ValueError, KeyError, OSError) as error:
        print(f'PHASE4_AUDIT_FAILED: {error}', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
