#!/usr/bin/env python3
"""Summarize completed dual Verifier/Solver OPSD runs; enforce paired cohorts.

Treat problem_id as statistical unit. A finite verifier audit is not a
correctness proof. Bootstrap intervals are conditional on training seeds.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np


def _lines(path):
    if not path.is_file():return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def load_one(path:Path):
    plan=json.loads((path/'plan.json').read_text())
    completion=json.loads((path/'completion.json').read_text())
    if completion['status']!='completed' or plan['status']!='executing':
        raise ValueError('incomplete experiment')
    rounds=completion['rounds']
    evaluation=[];metrics=[]
    for r in range(rounds+1):
        e=json.loads((path/f'round_{r}'/'evaluation.json').read_text())
        values={str(x['problem_id']):float(x['all_pass']) for x in e['rows']}
        if len(values)!=len(e['rows']) or not values:
            raise ValueError('evaluation must have one row per distinct problem_id')
        evaluation.append(values)
        if r==rounds:continue
        receipt=json.loads((path/f'round_{r}'/'round_receipt.json').read_text())
        verifier=[x for x in _lines(path/f'round_{r}'/'verifier_proposals.jsonl')
                  if 'reserved_audit_reporting_only' in x]
        solver=_lines(path/f'round_{r}'/'solver_updates.jsonl')
        audited=[x['reserved_audit_reporting_only'] for x in verifier if x['selected_for_solver']]
        gates=[x['gate_audit'] for x in verifier if x['selected_for_solver']]
        v_steps=[x['verifier_opsd_update'] for x in _lines(path/f'round_{r}'/'verifier_proposals.jsonl')
                 if 'verifier_opsd_update' in x]
        s_steps=[x.get('update',{}) for x in solver if 'update' in x]
        metrics.append({'round':r,'accepted_verifier_tasks':receipt['accepted_verifiers'],
                        'solver_candidate_count':receipt['solver_candidate_count'],
                        'verifier_hidden_audit_n':len(audited),
                        'verifier_hidden_balanced_accuracy':float(np.mean([x['balanced_accuracy'] for x in audited])) if audited else None,
                        'verifier_hidden_false_accept_rate':float(np.mean([x['false_accept_rate'] for x in audited])) if audited else None,
                        'verifier_hidden_false_reject_rate':float(np.mean([x['false_reject_rate'] for x in audited])) if audited else None,
                        'verifier_gate_balanced_accuracy':float(np.mean([x['balanced_accuracy'] for x in gates])) if gates else None,
                        'verifier_opsd_steps':sum(bool(x.get('step')) for x in v_steps),
                        'solver_opsd_steps':sum(bool(x.get('step')) for x in s_steps),
                        'solver_fixed_feedback_fraction':sum(x.get('source')=='fixed_trusted_fallback' for x in solver)/len(solver) if solver else None,
                        'elapsed_seconds':receipt['elapsed_seconds']})
    for a in evaluation[1:]:
        if set(a)!=set(evaluation[0]):raise ValueError('evaluation problem drift across rounds')
    return {'path':str(path),'arm':plan['arm'],'seed':plan['args']['seed'],
            'init_adapter_sha256':plan['init_adapter_sha256'],
            'dataset_sha256':plan['dataset_sha256'],
            'model_revision':plan['model_revision'],
            'train_distinct_problem_ids':plan['train_distinct_problem_ids'],
            'evaluation':evaluation,'metrics':metrics,
            'round_pass_at_1':[float(np.mean(list(x.values()))) for x in evaluation]}


def paired_bootstrap(a:dict[str,float],b:dict[str,float],*,seed:int=1701,n_boot:int=4000):
    if set(a)!=set(b):raise ValueError('paired evaluations require identical problem IDs')
    pids=sorted(a)
    d=np.array([a[p]-b[p] for p in pids],dtype=float)
    if len(d)<2:raise ValueError('at least two distinct problems required')
    rng=np.random.default_rng(seed)
    means=np.mean(d[rng.integers(0,len(d),size=(n_boot,len(d)))],axis=1)
    return {'n_problems':len(d),'difference':float(d.mean()),
            'bootstrap_95':list(map(float,np.quantile(means,[.025,.975])))}


def summarize(runs):
    if not runs:raise ValueError('at least one run required')
    rows=[load_one(Path(x)) for x in runs]
    if len({(r['arm'],r['seed']) for r in rows})!=len(rows):
        raise ValueError('duplicate (arm,seed)')
    key=['dataset_sha256','model_revision','train_distinct_problem_ids']
    for r in rows[1:]:
        if any(r[k]!=rows[0][k] for k in key):
            raise ValueError('cannot compare different manifests/models/train problems')
    byseed=defaultdict(dict)
    for r in rows:byseed[r['seed']][r['arm']]=r
    comparisons=[]
    for seed,arms in byseed.items():
        if len({x['init_adapter_sha256'] for x in arms.values()})>1:
            raise ValueError('same-seed arms require identical initial adapters')
        for left,right in [('dual','solver_fixed'),('dual','solver_static_verifier'),
                           ('dual','solver_regenerated_verifier')]:
            if left in arms and right in arms:
                a=arms[left]['evaluation'];b=arms[right]['evaluation']
                comparisons.append({'seed':seed,'contrast':left+'-minus-'+right,
                    'round_zero':paired_bootstrap(a[0],b[0],seed=seed),
                    'final':paired_bootstrap(a[-1],b[-1],seed=seed)})
    return {'schema':'dual-recursive-opsd-summary-v1',
            'disclaimer':'Finite trusted anchors are not formal proof; local RunBugRun matcher equivalence is unqualified. No heldout results were used for training selection.',
            'runs':[{'path':r['path'],'arm':r['arm'],'seed':r['seed'],
                     'round_pass_at_1':r['round_pass_at_1'],'metrics':r['metrics']} for r in rows],
            'contrasts':comparisons}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    out=summarize(args.runs)
    if args.output.exists():raise FileExistsError('refuse to overwrite summary')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(out,f,indent=2,ensure_ascii=False)
    print(json.dumps({'runs':len(out['runs']),'contrasts':len(out['contrasts']),
                      'output':str(args.output)},indent=2))


if __name__=='__main__':main()
