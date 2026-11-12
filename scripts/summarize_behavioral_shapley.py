#!/usr/bin/env python3
"""Source-matched unseen-problem summary of all behavioral Shapley controls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

ARMS=('no_update','sdpo','audit_gate','shapley_select')


def summarize(root:Path, *, seed:int=314159, bootstraps:int=2000)->dict:
    if bootstraps<100:raise ValueError('at least 100 bootstrap draws')
    measures={}; ids=None; protocols=None; initial_codes=None
    for arm in ARMS:
        plan=json.loads((root/arm/'plan.json').read_text())
        if plan['arm']!=arm or plan['isolation']!='bubblewrap_mandatory_no_soft_fallback':
            raise ValueError('unmatched arm or unsafe execution provenance')
        proto={k:plan.get(k) for k in ('model','revision','train_sources','eval_sources','rollouts','feedback_tests','calibration_budget_per_rollout','rounds','dataset_manifest_sha256','seed','temperature','max_new_tokens')}
        if protocols is None: protocols=proto
        elif protocols!=proto:raise ValueError('protocol mismatch between arms')
        measures[arm]={}
        for r in range(4):
            data=json.loads((root/arm/f'round_{r}'/'evaluation.json').read_text())
            if data.get('schema')!='eesd-behavioral-fresh-eval-v1' or data.get('round')!=r:
                raise ValueError('not a sealed behavioral evaluation report')
            rows={row['source_id']:row for row in data['rows']}
            if len(rows)!=len(data['rows']) or len(rows)!=protocols['eval_sources']:
                raise ValueError('duplicate or missing evaluation source')
            if ids is None:ids=sorted(rows)
            elif ids!=sorted(rows):raise ValueError('matched task identities differ')
            measure=np.asarray([int(rows[s]['all_pass']) for s in ids])
            measure_f=np.asarray([rows[s]['passes']/rows[s]['total_tests'] for s in ids])
            codes=[rows[s]['code_sha256'] for s in ids]
            if r==0:
                if initial_codes is None:initial_codes=codes
                elif codes!=initial_codes:raise ValueError('arms do not share an identical initial checkpoint')
            measures[arm][r]=(measure,measure_f)
    rng=np.random.default_rng(seed)
    n=len(ids)
    base=measures['no_update'][0][0]
    output={'schema':'eesd-behavioral-shapley-paired-v1','status':'completed_metrics_not_an_accuracy_claim',
            'protocol':protocols,'eval_sources':n,'arms':{},'baseline':'sdpo'}
    for arm in ARMS:
        rounds=[]
        for r in range(4):
            y,y_fraction=measures[arm][r]
            ref,ref_fraction=measures['sdpo'][r]
            sampled=rng.integers(0,n,size=(bootstraps,n))
            delta=(y-ref)[sampled].mean(axis=1)
            frac_delta=(y_fraction-ref_fraction)[sampled].mean(axis=1)
            rounds.append({'round':r,'pass_at_1':float(y.mean()),
                'test_pass_fraction':float(y_fraction.mean()),
                'pass_at_1_delta_vs_sdpo':float((y-ref).mean()),
                'paired_bootstrap_ci95_vs_sdpo':np.quantile(delta,[.025,.975]).tolist(),
                'test_pass_fraction_delta_vs_sdpo':float((y_fraction-ref_fraction).mean()),
                'test_pass_fraction_ci95_vs_sdpo':np.quantile(frac_delta,[.025,.975]).tolist(),
                'fixes_vs_initial':int(((base==0)&(y==1)).sum()),
                'regressions_vs_initial':int(((base==1)&(y==0)).sum())})
        output['arms'][arm]=rounds
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    opts=ap.parse_args()
    if opts.output.exists():raise FileExistsError('refuse to overwrite experiment summary')
    report=summarize(opts.root)
    opts.output.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:v[-1] for k,v in report['arms'].items()},indent=2))
