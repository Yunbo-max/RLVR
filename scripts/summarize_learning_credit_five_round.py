#!/usr/bin/env python3
"""Source-paired five-round holdout evaluation, mechanism ablation and regret.

No train/model decisions use these final evaluation results. Require identical
data, model and initial LoRA adapter across all seven arms; reject silently
missing / overwritten / mismatched runs. Bootstrap over held-out *problem IDs*.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np

ARMS=('no_update','sdpo','audit_gate','direct_max','shapley',
      'random_subset','matched_random_updates')


def _paired(a:np.ndarray,b:np.ndarray,draws:np.ndarray)->dict:
    if a.shape!=b.shape or a.ndim!=1:
        raise ValueError('matched 1D source vectors required')
    d=a-b
    ci=np.quantile(d[draws].mean(axis=1),[.025,.975])
    gains=int(((a==1)&(b==0)).sum())
    harms=int(((a==0)&(b==1)).sum())
    n=gains+harms
    p=min(1.0,2*sum(math.comb(n,k) for k in range(min(gains,harms)+1))/(2**n)) if n else 1.
    return {'difference':float(d.mean()),'paired_ci95':[float(ci[0]),float(ci[1])],
            'wins':gains,'losses':harms,'exact_mcnemar_two_sided_p':float(p)}


def summarize(root:Path,*,bootstrap:int=3000,seed:int=314159)->dict:
    if bootstrap<100:raise ValueError('>=100 paired bootstrap draws required')
    plans={}
    evals={}
    source_ids=None
    initial_codes=None
    identity=None
    for arm in ARMS:
        plan=json.loads((root/arm/'plan.json').read_text())
        if plan.get('schema')!='eesd-learning-credit-five-v1' or plan.get('arm')!=arm:
            raise ValueError(f'missing or wrong experiment plan for {arm}')
        id_keys=('model','revision','seed','rollouts','rounds','train_sources',
                 'eval_sources','dataset_manifest_sha256','initial_adapter_sha256',
                 'feedback_tests','calibration_budget_per_rollout','isolation',
                 'max_new_tokens','max_prompt_tokens','temperature','beta',
                 'learning_rate')
        current={k:plan.get(k) for k in id_keys}
        if identity is None:identity=current
        if identity!=current:raise ValueError(f'unmatched experimental budget: {arm}')
        if plan.get('rounds')!=5 or plan.get('feedback_tests')!=4:
            raise ValueError('require five recursive updates and four feedback tests')
        plans[arm]=plan
        evals[arm]={}
        for rnd in range(6):
            path=root/arm/f'round_{rnd}'/'evaluation.json'
            report=json.loads(path.read_text())
            if report.get('round')!=rnd or report.get('schema')!='eesd-behavioral-fresh-eval-v1':
                raise ValueError('invalid evaluation receipt')
            rows={r['source_id']:r for r in report['rows']}
            if len(rows)!=len(report['rows']) or len(rows)!=identity['eval_sources']:
                raise ValueError('duplicate or missing final heldout source')
            if source_ids is None:source_ids=sorted(rows)
            if sorted(rows)!=source_ids:raise ValueError('unmatched final problem IDs')
            code_hashes=[rows[k]['code_sha256'] for k in source_ids]
            if rnd==0:
                if initial_codes is None:initial_codes=code_hashes
                elif initial_codes!=code_hashes:
                    raise ValueError('initial generated code differs across arms')
            success=np.array([int(rows[k]['all_pass']) for k in source_ids],dtype=np.int8)
            fractional=np.array([rows[k]['passes']/rows[k]['total_tests']
                                 for k in source_ids],dtype=float)
            evals[arm][rnd]={'success':success,'fractional':fractional,'code_hashes':code_hashes}
    for rnd in range(6):
        if evals['no_update'][rnd]['code_hashes']!=initial_codes:
            raise ValueError('frozen reference unexpectedly changed across rounds')

    rng=np.random.default_rng(seed)
    source_count=len(source_ids)
    draws=rng.integers(0,source_count,size=(bootstrap,source_count))
    base=evals['no_update'][0]['success']
    result={'schema':'eesd-five-round-paired-report-v1','status':'computed_not_independent_model_reproduction',
            'identity':identity,'source_count':source_count,'arms':{},
            'notes':['Source-paired uncertainty, not independent training-seed uncertainty',
                     'A +2 problem advantage alone cannot establish Shapley superiority',
                     'Official SDPO must be reproduced separately']}
    for arm in ARMS:
        per_round=[]
        for rnd in range(6):
            y=evals[arm][rnd]['success'];frac=evals[arm][rnd]['fractional']
            score={'round':rnd,'all_tests_pass_at_1':float(y.mean()),
                   'mean_test_pass_fraction':float(frac.mean()),
                   'versus_frozen':_paired(y,base,draws),
                   'versus_sdpo':_paired(y,evals['sdpo'][rnd]['success'],draws),
                   'versus_direct_max':_paired(y,evals['direct_max'][rnd]['success'],draws),
                   'versus_shapley':_paired(y,evals['shapley'][rnd]['success'],draws)}
            if rnd<5 and arm!='no_update':
                summary=json.loads((root/arm/f'round_{rnd}'/'summary.json').read_text())
                score['training']=summary
            per_round.append(score)
        result['arms'][arm]=per_round
    # Diagnostic: Did Shapley achieve more held-out transfer than direct J(S)
    # (despite never having larger in-sample J(S))? Report all five rounds.
    result['primary_comparisons']={
        'shapley_vs_direct':result['arms']['shapley'][-1]['versus_direct_max'],
        'shapley_vs_sdpo':result['arms']['shapley'][-1]['versus_sdpo'],
        'shapley_vs_frozen':result['arms']['shapley'][-1]['versus_frozen'],
        'shapley_vs_random_subset':_paired(evals['shapley'][5]['success'],
                  evals['random_subset'][5]['success'],draws),
        'shapley_vs_matched_updates':_paired(evals['shapley'][5]['success'],
                  evals['matched_random_updates'][5]['success'],draws),
    }
    # Round 0 has byte-identical starting weights and per-source rollout RNG.
    # This is the only round in which direct vs Shapley selection can be
    # compared causally on exactly the SAME candidate pool.
    first_round={}
    for arm in ('shapley','direct_max'):
        path=root/arm/'round_0'/'training.jsonl'
        if not path.is_file():raise FileNotFoundError(path)
        items=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        byid={v['source_id']:v for v in items}
        if len(byid)!=len(items):raise ValueError('duplicate training source receipts')
        first_round[arm]=byid
    if set(first_round['shapley'])!=set(first_round['direct_max']):
        raise ValueError('direct/Shapley round-0 source population differs')
    diff_masks=0;comparable=0;direct_regret=[];identical=0
    for src in sorted(first_round['shapley']):
        sp=first_round['shapley'][src];dr=first_round['direct_max'][src]
        ca=sp.get('candidate_code_sha256');cb=dr.get('candidate_code_sha256')
        if ca is not None or cb is not None:
            if ca!=cb:raise ValueError('round-0 code candidates not identical')
            identical+=1
        if sp.get('selected_mask',0)!=dr.get('selected_mask',0):
            diff_masks+=1
        js=sp.get('j_calibration');jd=dr.get('j_calibration')
        if js is not None and jd is not None:
            comparable+=1
            if float(jd)+1e-9<float(js):
                raise ValueError('direct J(S) cannot be lower than Shapley on matched training data')
            direct_regret.append(float(jd)-float(js))
    result['round0_direct_vs_shapley_selection']={
        'sources':len(first_round['shapley']),
        'matched_candidate_pools':identical,
        'subset_choice_disagreement_rate':diff_masks/len(first_round['shapley']),
        'both_nonempty_comparable':comparable,
        'mean_direct_minus_shapley_training_J':float(np.mean(direct_regret)) if direct_regret else None,
        'interpretation':'direct wins or ties in-sample J; Shapley must win only on heldout transfer'}
    for random_arm in ('random_subset','matched_random_updates'):
        requested=[result['arms'][random_arm][r]['training']['scheduled_optimizer_steps'] for r in range(5)]
        realized=[result['arms'][random_arm][r]['training']['optimizer_steps'] for r in range(5)]
        result['primary_comparisons'][random_arm+'_update_matching']={
            'scheduled_per_round':requested,'realized_per_round':realized,
            'exactly_matched':requested==realized,
            'interpretation':'if false, this is schedule-matched but NOT actual update-count matched'}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=3000)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('refuse overwriting paired results')
    report=summarize(args.root,bootstrap=args.bootstrap)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report['primary_comparisons'],indent=2))
