#!/usr/bin/env python3
"""Five-round paired RunBugRun evaluation for nested T4..T64 group Shapley.

The PRIMARY statistical unit is a *distinct problem_id*. Repeated buggy
submissions to the same programming problem must not be treated as
independent problems. Requires EXACT initial adapter identity and enforces
the same executed-64/public-group4 evidence budget in Shapley vs DirectMax.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ARMS=('no_update','sdpo','audit_gate','direct_max','shapley',
      'random_subset','matched_random_updates')


def _paired(aa:np.ndarray,bb:np.ndarray,draws:np.ndarray)->dict:
    delta=aa-bb
    bounds=np.quantile(delta[draws].mean(axis=1),[.025,.975])
    wins=int(((aa==1)&(bb==0)).sum());losses=int(((aa==0)&(bb==1)).sum())
    mismatches=wins+losses
    prob=(min(1.0,2*sum(math.comb(mismatches,x)
                   for x in range(min(wins,losses)+1))/2**mismatches)
          if mismatches else 1.)
    return {'delta':float(delta.mean()),'paired_cluster_ci95':bounds.tolist(),
            'fixes':wins,'regressions':losses,'mcnemar_exact_two_sided_p':prob}


def summarize(root:Path,*,arms:tuple[str,...]=ARMS,
              bootstrap:int=2000,seed:int=314159)->dict:
    if bootstrap<100:raise ValueError('at least 100 bootstrap draws')
    if 'shapley' not in arms or 'direct_max' not in arms:
        raise ValueError('Shapley and direct-J must both be present')
    plans={}
    reports={}
    identity=None
    baselines=None
    group_ids=None
    for arm in arms:
        plan=json.loads((root/arm/'plan.json').read_text())
        if plan.get('schema')!='eesd-rich64-five-v1' or plan.get('arm')!=arm:
            raise ValueError('wrong grouped-rich-feedback experiment')
        keys=('model','revision','seed','rounds','rollouts','train_bug_instances',
              'eval_distinct_problem_ids','dataset_manifest_sha256',
              'initial_adapter_sha256','executed_public_feedback_tests_per_candidate',
              'exposed_feedback','group_shapley_players','exact_group_coalitions',
              'calibration_tests','reserved_audit_tests','panel_mode',
              'feedback_group_chars','max_prompt_tokens','max_new_tokens',
              'temperature','learning_rate','beta')
        ident={k:plan.get(k) for k in keys}
        if identity is not None and identity!=ident:
            raise ValueError('different model/test/teacher/group/seed budget across arms')
        identity=ident
        if ident['executed_public_feedback_tests_per_candidate']!=64:
            raise ValueError('not equal 64 executed public tests')
        if ident['exact_group_coalitions']!=16 or ident['rounds']!=5:
            raise ValueError('expected four evidence groups and five rounds')
        plans[arm]=plan
        reports[arm]={}
        for rnd in range(6):
            data=json.loads((root/arm/f'round_{rnd}'/'evaluation.json').read_text())
            if data.get('schema')!='eesd-behavioral-fresh-eval-v1' or data.get('round')!=rnd:
                raise ValueError('wrong round evaluation')
            rows={r['source_id']:r for r in data['rows']}
            if len(rows)!=len(data['rows']):
                raise ValueError('repeated final source ID')
            if group_ids is None:
                group_ids=sorted(rows)
            if sorted(rows)!=group_ids:
                raise ValueError('not the same final heldout source IDs')
            assert group_ids is not None
            if baselines is None:
                baselines=[rows[k]['code_sha256'] for k in group_ids]
            if rnd==0 and [rows[k]['code_sha256'] for k in group_ids]!=baselines:
                raise ValueError('initial greedy programs differ across arms')
            y=np.array([int(rows[k]['all_pass']) for k in group_ids])
            frac=np.array([rows[k]['passes']/rows[k]['total_tests'] for k in group_ids])
            problemids=[rows[k]['problem_id'] for k in group_ids]
            if len(set(problemids))!=len(problemids):
                raise ValueError('final sample has repeated problem IDs; use problem-cluster aggregation instead')
            reports[arm][rnd]=(y,frac,problemids)
        if arm=='no_update':
            frozen=reports[arm][0][0]
            for rnd in range(1,6):
                if not np.array_equal(reports[arm][rnd][0],frozen):
                    raise ValueError('frozen model inexplicably changed')
    rng=np.random.default_rng(seed);n=len(group_ids)
    draws=rng.integers(0,n,size=(bootstrap,n))
    output={'schema':'eesd-rich64-five-paired-v1','status':'computed_only_after_real_model_recepts',
            'identity':identity,'eval_distinct_problems':n,'arms':{},'rounds':6}
    base=reports['no_update'][0][0] if 'no_update' in arms else None
    for arm in arms:
        rounds=[]
        for rnd in range(6):
            y,frac,_=reports[arm][rnd]
            record={'round':rnd,'pass_at_1':float(y.mean()),
                    'mean_test_pass_fraction':float(frac.mean()),
                    'versus_shapley':_paired(y,reports['shapley'][rnd][0],draws),
                    'versus_direct_max':_paired(y,reports['direct_max'][rnd][0],draws)}
            if base is not None:record['versus_frozen']=_paired(y,base,draws)
            if rnd<5 and arm!='no_update':
                record['train']=json.loads((root/arm/f'round_{rnd}'/'summary.json').read_text())
            rounds.append(record)
        output['arms'][arm]=rounds
    output['main_claim_test']={
        'shapley_vs_direct_max_round5':
            _paired(reports['shapley'][5][0],reports['direct_max'][5][0],draws),
        'interpretation':'Shapley can only beat direct-J by INDEPENDENT transfer, not in-sample J',
    }
    if 'no_update' in arms:
        output['main_claim_test']['shapley_vs_frozen_round5']=_paired(
            reports['shapley'][5][0],reports['no_update'][0][0],draws)
    for arm in ('shapley','direct_max'):
        output['main_claim_test'][arm+'_cost']={
          'teacher_subset_scoring_calls':sum(output['arms'][arm][k]['train']['teacher_subset_scoring_calls'] for k in range(5)),
          'public_test_executions':sum(output['arms'][arm][k]['train']['public_test_executions'] for k in range(5)),
          'calibration_test_executions':sum(output['arms'][arm][k]['train']['calibration_test_executions'] for k in range(5)),
          'optimizer_steps':sum(output['arms'][arm][k]['train']['optimizer_steps'] for k in range(5)),
          'training_wallclock_seconds':sum(output['arms'][arm][k]['train']['training_wallclock_seconds'] for k in range(5)),
        }
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arms',nargs='+',default=ARMS)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('result summary already exists')
    report=summarize(a.root,arms=tuple(a.arms))
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report['main_claim_test'],indent=2))
