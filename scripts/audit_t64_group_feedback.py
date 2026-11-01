#!/usr/bin/env python3
"""Post-training ONLY audit of heldout 8 input/output probes for grouped feedback.

This recomputes feedback-behavior alignment using saved 16 group-subset
teacher scores and independently executed candidates. It never touches
optimizer weights or influences which evidence is chosen. A safer bwrap
isolation backend is mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from eesd.behavioral_shapley import audit_value_game
from eesd.strict_execution import audit_vector


def inspect(record:dict,reserved:list[dict],*,execute=audit_vector)->dict|None:
    if 'candidate_sources' not in record or 'candidate_shift_means' not in record:
        return None
    codes=record['candidate_sources']
    shifts=np.asarray(record['candidate_shift_means'],dtype=np.float64)
    if len(codes)<2 or shifts.shape!=(len(codes),16) or not np.isfinite(shifts).all():
        raise ValueError('bad 16 group-coalition score provenance')
    results=np.asarray([execute(code,reserved) for code in codes],dtype=float)
    if results.shape!=(len(codes),len(reserved)):
        raise ValueError('heldout calibration dimension mismatch')
    game=audit_value_game([x[:,None] for x in shifts],results)
    mask=record.get('selected_mask',0)
    if type(mask) is not int or not 0<=mask<16:
        raise ValueError('invalid selected evidence-group mask')
    return {'source_id':record['source_id'],'round':record['round'],
        'selected_mask':mask,'step':bool(record['step']),
        'evidence_groups':4,'exposed_public_tests':record['exposed_feedback_test_count'],
        'reserved_score_selected':float(game[mask].mean()),
        'reserved_score_all_groups':float(game[-1].mean()),
        'selected_minus_all_groups':float((game[mask]-game[-1]).mean()),
        'reserved_behavioral_variance':float(results.var(axis=0).mean())}


def audit(data_dir:Path,run_dir:Path)->dict:
    manifest=json.loads((data_dir/'manifest.json').read_text())
    if manifest.get('schema')!='eesd-rich64-frozen-v1':
        raise ValueError('not formal T64 frozen data')
    plan=json.loads((run_dir/'plan.json').read_text())
    if (plan.get('schema')!='eesd-rich64-five-v1'
        or plan['dataset_manifest_sha256']!=hashlib.sha256(
            (data_dir/'manifest.json').read_bytes()).hexdigest()):
        raise ValueError('mismatched experiment/data identities')
    lines=(data_dir/'train_audit_do_not_train.jsonl').read_text().splitlines()
    protected=[json.loads(x) for x in lines if x.strip()]
    byid={x['source_id']:x for x in protected}
    if len(byid)!=len(protected):
        raise ValueError('repeated reserved train source identity')
    results=[]
    for k in range(5):
        file=run_dir/f'round_{k}'/'training.jsonl'
        if not file.exists():raise FileNotFoundError(file)
        for line in file.read_text().splitlines():
            if not line.strip():continue
            r=json.loads(line)
            row=byid.get(r['source_id'])
            if row is None:raise ValueError('unrecognized source in group audit')
            outcome=inspect(r,row['audit_tests'])
            if outcome is not None:results.append(outcome)
    accepted=[r for r in results if r['step']]
    return {'schema':'eesd-grouped-T64-posthoc-audit-v1',
            'method_scope':'group-level Shapley, not individual test Shapley',
            'status':'no_audit_probes_used_in_training',
            'arm':plan['arm'],'exposed_feedback':plan['exposed_feedback'],
            'tested_source_rounds':len(results),
            'realized_optimizer_updates':len(accepted),
            'accepted_selected_minus_all_mean':float(np.mean([
              v['selected_minus_all_groups'] for v in accepted])) if accepted else None,
            'accepted_positive_alignment_fraction':float(np.mean([
              v['reserved_score_selected']>0 for v in accepted])) if accepted else None,
            'records':results}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--experiment-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('refuse to overwrite audit receipt')
    result=audit(a.data_dir,a.experiment_dir)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
