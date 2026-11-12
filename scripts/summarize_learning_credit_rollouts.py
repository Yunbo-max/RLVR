#!/usr/bin/env python3
"""Paired G={8,16,32} scale results, separate from the seven-arm main study."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

G_VALUES=(8,16,32)
ARMS=('shapley','direct_max')


def summarize(root:Path,*,draws:int=2000)->dict:
    if draws<100:raise ValueError('need >=100 bootstrap resamples')
    result={'schema':'eesd-learning-credit-rollout-scale-v1',
            'feedback_tests':4,'exact_coalitions':16,
            'rollout_counts':list(G_VALUES),'by_g':{}}
    initial=None;dataset=None
    rng=np.random.default_rng(314159)
    for g in G_VALUES:
        by={}
        rows_by_arm={}
        for arm in ARMS:
            exp=root/f'g{g}-{arm}'
            plan=json.loads((exp/'plan.json').read_text())
            if plan.get('arm')!=arm or plan.get('rollouts')!=g or plan.get('rounds')!=5:
                raise ValueError('scale arm/protocol mismatch')
            identity=(plan['dataset_manifest_sha256'],plan['initial_adapter_sha256'])
            if dataset is None:dataset=identity
            if dataset!=identity:raise ValueError('not the same dataset/initial adapter')
            evals={}
            for k in (0,5):
                data=json.loads((exp/f'round_{k}'/'evaluation.json').read_text())
                pairs={r['source_id']:r for r in data['rows']}
                if len(pairs)!=len(data['rows']):raise ValueError('duplicate test source')
                if k==0:
                    ref=sorted(pairs)
                    hashes=[pairs[i]['code_sha256'] for i in ref]
                    if initial is None:initial=hashes
                    elif initial!=hashes:raise ValueError('starting policy differs')
                elif sorted(pairs)!=ref:raise ValueError('source population differs')
                evals[k]=np.array([int(pairs[i]['all_pass']) for i in ref])
            by[arm]={'round0':float(evals[0].mean()),'round5':float(evals[5].mean()),
                     'step_count':sum(json.loads((exp/f'round_{r}'/'summary.json').read_text())['optimizer_steps'] for r in range(5))}
            rows_by_arm[arm]=evals[5]
        diff=rows_by_arm['shapley']-rows_by_arm['direct_max']
        draws_idx=rng.integers(0,len(diff),size=(draws,len(diff)))
        ci=np.quantile(diff[draws_idx].mean(axis=1),[.025,.975])
        result['by_g'][str(g)]={'arms':by,'shapley_minus_direct':float(diff.mean()),
           'paired_ci95':[float(ci[0]),float(ci[1])],
           'n_sources':len(diff)}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('refuse overwrite')
    result=summarize(a.root)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))
