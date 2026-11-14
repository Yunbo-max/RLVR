#!/usr/bin/env python3
"""Paired problem-level summaries of completed Joint/Mix five-round runs.

Never equate multiple training seeds with independent new problem IDs. The
cluster bootstrap samples problem IDs, with seed-specific contrasts retained.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

CONTRASTS=(('joint_pair','mix_pair'),('joint_search','random_search'),
           ('joint_search','joint_pair'),('joint_pair','no_update'))


def _read(run:Path):
    plan=json.loads((run/'plan.json').read_text())
    complete=json.loads((run/'completion.json').read_text())
    if plan.get('schema')!='eesd-joint-mix-plan-v1' or plan.get('experiment')!='train' or complete.get('status')!='completed':
        raise ValueError(f'{run}: not a completed real five-round run')
    arm=plan['arm'];seed=int(plan['seed']); rnd=int(plan['rounds'])
    if arm not in ('joint_pair','mix_pair','joint_search','random_search','no_update') or rnd!=5:
        raise ValueError('unexpected arm or round number')
    report=json.loads((run/f'round_{rnd}'/'evaluation.json').read_text())
    rows={}
    for r in report['rows']:
        key=str(r['problem_id'])
        if key in rows:raise ValueError('duplicate evaluation problem ID')
        rows[key]=int(bool(r['all_pass']))
    if not rows:raise ValueError('no final evaluation rows')
    return plan,arm,seed,rows


def paired_interval(deltas:np.ndarray, *, draws:int=4000, seed:int=731):
    d=np.asarray(deltas,dtype=float)
    if d.ndim!=1 or len(d)<2 or not np.isfinite(d).all():
        raise ValueError('need >=2 independent paired problem differences')
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(draws):
        index=rng.integers(len(d),size=len(d))
        samples.append(float(d[index].mean()))
    return [float(x) for x in np.quantile(samples,[.025,.975])]


def aggregate(run_folders:list[Path], *, draws:int=4000):
    seen={}; baseline=None
    for path in run_folders:
        plan,arm,seed,rows=_read(path)
        key=(seed,arm)
        if key in seen:raise ValueError(f'duplicate seed/arm {key}')
        identities=(plan['dataset_manifest_sha256'],plan['model_revision'],
                    plan['train_distinct_problems'],plan['final_distinct_problems'],
                    plan['rounds'])
        if baseline is None:baseline=identities
        if identities!=baseline:raise ValueError('data/model/protocol do not match')
        seen[key]=(plan,rows)
    by_seed=defaultdict(dict)
    for (seed,arm),(plan,rows) in seen.items():by_seed[seed][arm]=(plan,rows)
    contrasts=[]
    for left,right in CONTRASTS:
        seeds=sorted(s for s in by_seed if left in by_seed[s] and right in by_seed[s])
        if not seeds:continue
        arrays=[];one_seed=[];keys=None
        for s in seeds:
            lp,l=by_seed[s][left];rp,r=by_seed[s][right]
            if lp['initial_adapter_sha256']!=rp['initial_adapter_sha256']:
                raise ValueError('pair was trained from different initial adapters')
            if set(l)!=set(r):raise ValueError('paired evaluation IDs differ')
            if keys is None:keys=sorted(l)
            if sorted(l)!=keys:raise ValueError('evaluation IDs differ between seeds')
            d=np.array([l[k]-r[k] for k in keys],dtype=float)
            arrays.append(d)
            one_seed.append({'seed':s,'delta_pp':float(100*d.mean()),
                             'fixes':int((d==1).sum()),'regressions':int((d==-1).sum()),
                             'left_pass':int(sum(l.values())),'right_pass':int(sum(r.values()))})
        mean_by_problem=np.mean(np.stack(arrays),axis=0)
        low,high=paired_interval(mean_by_problem,draws=draws,seed=100+len(seeds))
        contrasts.append({'left':left,'right':right,'seeds':seeds,
            'problem_ids':len(keys),'delta_pp':float(100*mean_by_problem.mean()),
            'cluster_bootstrap_95pp':[100*low,100*high],
            'individual_seed_results':one_seed,
            'note':'Clustered by problem_id; CI conditional on observed training seeds.'})
    return {'schema':'eesd-joint-mix-paired-summary-v1','status':'completed',
            'input_runs':len(seen),'seed_count':len(by_seed),'contrasts':contrasts,
            'method_note':'Intervals do not account for additional seeds or prior model selection.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap-draws',type=int,default=4000)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.bootstrap_draws<100:raise ValueError('bootstrap draws must be at least 100')
    value=aggregate(args.runs,draws=args.bootstrap_draws)
    with args.output.open('x') as f:json.dump(value,f,indent=2,sort_keys=True)
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
