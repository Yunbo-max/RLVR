#!/usr/bin/env python3
"""Source-paired RunBugRun recursive results. Never accesses training feedback."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

ARMS=('no_update','sdpo','loo','shapley')


def summarize(root:Path,*,bootstrap_draws:int=2000,seed:int=314159)->dict:
    if bootstrap_draws<1:raise ValueError('positive bootstrap_draws required')
    data={}
    identity=None
    ids=None
    for arm in ARMS:
        plan=json.loads((root/arm/'plan.json').read_text())
        check={k:plan[k] for k in ('model_id','model_revision','data_manifest_sha256','train_sources','eval_sources','rollouts')}
        if identity is not None and identity!=check:
            raise ValueError('arms do not use the identical frozen experiment')
        identity=check
        data[arm]={}
        for rnd in range(4):
            report=json.loads((root/arm/f'round_{rnd}'/'evaluation.json').read_text())
            byid={row['source_id']:row for row in report['rows']}
            if len(byid)!=len(report['rows']) or len(byid)!=check['eval_sources']:
                raise ValueError('source coverage mismatch')
            if ids is None:ids=sorted(byid)
            if sorted(byid)!=ids:raise ValueError('evaluation source identities differ')
            data[arm][rnd]=np.array([int(byid[k]['all_pass']) for k in ids],dtype=int)
    base=data['no_update'][0]
    for arm in ARMS:
        if not np.array_equal(base,data[arm][0]):
            raise ValueError('initial model evaluation differs; invalid matched baseline')
    rng=np.random.default_rng(seed)
    result={'schema':'eesd-recursive-paired-results-v1','identity':identity,
            'sources':len(ids),'bootstrap_draws':bootstrap_draws,
            'status':'paired_evaluation_not_official_benchmark_submission','arms':{}}
    for arm in ARMS:
        by=[]
        for rnd in range(4):
            end=data[arm][rnd]
            diff=end-data['sdpo'][rnd]
            draw=rng.integers(0,len(base),size=(bootstrap_draws,len(base)))
            ci=np.quantile(diff[draw].mean(axis=1),[.025,.975])
            by.append({'round':rnd,'pass_at_1':float(end.mean()),
                       'delta_vs_sdpo':float(diff.mean()),
                       'paired_ci95_vs_sdpo':[float(ci[0]),float(ci[1])],
                       'fixes_vs_round0':int(((base==0)&(end==1)).sum()),
                       'regressions_vs_round0':int(((base==1)&(end==0)).sum())})
        result['arms'][arm]=by
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('refuse to overwrite summary')
    report=summarize(args.root)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({arm:rows[-1] for arm,rows in report['arms'].items()},indent=2))
