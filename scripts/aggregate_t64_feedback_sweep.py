#!/usr/bin/env python3
"""Compare T=4/8/16/32/64 with fixed F64 execution and exact K4 Shapley.

Reject changes to data, initial LoRA, final test sources, model and RNG.
Only the prefix of the F64 bank is exposed to the selector; C8 and A8
never change. Evaluating multiple T values on the same final holdout is a
planned analysis, not license to tune T post-hoc then claim a new test result.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

T_VALUES=(4,8,16,32,64)


def _vector(folder:Path):
    r=json.loads((folder/'round_5'/'evaluation.json').read_text())
    rows={z['source_id']:z for z in r['rows']}
    if len(rows)!=len(r['rows']):raise ValueError('duplicate eval source')
    order=sorted(rows)
    return order, np.array([int(rows[x]['all_pass']) for x in order]),\
        [rows[x]['problem_id'] for x in order]


def aggregate(root:Path,*,draws:int=3000)->dict:
    if draws<100:raise ValueError('need >=100 paired draws')
    rng=np.random.default_rng(271828)
    results={}
    initial=None;identity=None;ids=None;problem_ids=None
    t4_shapley=None
    for t in T_VALUES:
        arm=root/f't{t}'
        plan=json.loads((arm/'shapley'/'plan.json').read_text())
        direct=json.loads((arm/'direct_max'/'plan.json').read_text())
        if plan.get('exposed_feedback')!=t or direct.get('exposed_feedback')!=t:
            raise ValueError('incorrect disclosed T test count')
        keys=('model','revision','seed','rounds','rollouts',
              'train_bug_instances','eval_distinct_problem_ids',
              'dataset_manifest_sha256','initial_adapter_sha256',
              'executed_public_feedback_tests_per_candidate',
              'group_shapley_players','exact_group_coalitions',
              'feedback_group_chars','panel_mode','calibration_tests',
              'reserved_audit_tests')
        x={k:plan.get(k) for k in keys}
        y={k:direct.get(k) for k in keys}
        if x!=y:raise ValueError('within-T Shapley vs direct mismatched')
        if identity is None:identity=x
        elif identity!=x:raise ValueError('feedback-level sweep changed a non-T variable')
        for which in ('shapley','direct_max'):
            r0=json.loads((arm/which/'round_0'/'evaluation.json').read_text())
            records={r['source_id']:r for r in r0['rows']}
            hashes={id:row['code_sha256'] for id,row in records.items()}
            if initial is None:initial=hashes
            elif hashes!=initial:raise ValueError('initial model outputs differ across T')
        source_a,shapley,pids=_vector(arm/'shapley')
        source_b,direct,pids_b=_vector(arm/'direct_max')
        if source_a!=source_b or pids!=pids_b:raise ValueError('T-level evaluation mismatch')
        if ids is None:
            ids=source_a;problem_ids=pids
            if len(set(pids))!=len(pids):raise ValueError('problem-level heldout duplicates')
        elif ids!=source_a or problem_ids!=pids:
            raise ValueError('different final problem IDs across T')
        if t==4:t4_shapley=shapley
        n=len(shapley)
        resample=rng.integers(0,n,size=(draws,n))
        def diff(a,b):
            d=a-b
            ci=np.quantile(d[resample].mean(axis=1),[.025,.975])
            return {'mean_pp':float(100*d.mean()),
                    'problem_paired_ci95_pp':[float(100*ci[0]),float(100*ci[1])],
                    'source_fixes':int(((a==1)&(b==0)).sum()),
                    'source_regressions':int(((a==0)&(b==1)).sum())}
        def cost(label):
            summaries=[json.loads((arm/label/f'round_{r}'/'summary.json').read_text())
                       for r in range(5)]
            return {field:sum(s[field] for s in summaries) for field in
              ('teacher_subset_scoring_calls','public_test_executions',
               'calibration_test_executions','optimizer_steps','training_wallclock_seconds')}
        results[str(t)]={
            'shapley_pass_at_1':float(shapley.mean()),
            'direct_pass_at_1':float(direct.mean()),
            'shapley_minus_direct':diff(shapley,direct),
            'shapley_minus_T4':diff(shapley,t4_shapley),
            'shapley_compute':cost('shapley'),
            'direct_compute':cost('direct_max'),
        }
    return {'schema':'eesd-nested-T64-budget-sweep-v1',
            'identity':identity,'levels':list(T_VALUES),
            'n_distinct_eval_problems':len(ids),'by_feedback_count':results,
            'limitations':['Group-level attribution: four players, NOT 64 individual-test Shapley',
                           'Source-cluster bootstrap is not independent training-seed uncertainty',
                           'T sweep reads the same final test; do not use it for unregistered hyperparameter tuning']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('do not overwrite feedback sweep')
    result=aggregate(a.root)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({t:{k:v for k,v in results.items() if k not in ('shapley_compute','direct_compute')}
      for t,results in result['by_feedback_count'].items()},indent=2))
