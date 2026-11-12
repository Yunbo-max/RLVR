#!/usr/bin/env python3
"""After ALL five updates, audit feedback choices against reserved train probes.

Never use these labels for feedback selection, LoRA training, or hyperparameter
tuning on the same run. A sandbox infrastructure failure aborts the audit.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from eesd.behavioral_shapley import audit_value_game
from eesd.strict_execution import audit_vector


def inspect_record(record:dict, reserved_tests:list[dict], execute=audit_vector)->dict|None:
    if 'candidate_sources' not in record:
        return None
    codes=record['candidate_sources']
    values=np.asarray(record.get('candidate_shift_means'),dtype=np.float64)
    if (len(codes)<2 or values.ndim!=2 or values.shape!=(len(codes),16)
        or not np.isfinite(values).all()):
        raise ValueError('required complete 16-coalition provenance')
    outcomes=np.asarray([execute(c,reserved_tests) for c in codes],dtype=np.float64)
    if outcomes.shape!=(len(codes),len(reserved_tests)):
        raise ValueError('reserved test dimension mismatch')
    game=audit_value_game([v[:,None] for v in values],outcomes)
    mask=record.get('selected_mask',0)
    if not isinstance(mask,int) or not 0<=mask<16:
        raise ValueError('invalid chosen subset')
    return {'source_id':record['source_id'],'round':record['round'],
        'selected_mask':mask,'training_choice_rule':record.get('choice_rule'),
        'actual_optimizer_step':bool(record['step']),
        'reserved_audit_score_selected':float(game[mask].mean()),
        'reserved_audit_score_full':float(game[-1].mean()),
        'reserved_selected_minus_full':float((game[mask]-game[-1]).mean()),
        'reserved_audit_behavior_contrast':float(outcomes.var(axis=0).mean())}


def audit(data_dir:Path, arm_dir:Path)->dict:
    manifest=json.loads((data_dir/'manifest.json').read_text())
    if manifest.get('learning_credit_schema')!='eesd-five-round-frozen-data-v1':
        raise ValueError('incorrect dataset protocol')
    plan=json.loads((arm_dir/'plan.json').read_text())
    if plan.get('schema')!='eesd-learning-credit-five-v1' or plan.get('rounds')!=5:
        raise ValueError('wrong five-round trained policy')
    if plan.get('dataset_manifest_sha256') != __import__('hashlib').sha256((data_dir/'manifest.json').read_bytes()).hexdigest():
        raise ValueError('data/experiment identity mismatch')
    protected=[json.loads(x) for x in (data_dir/'train_audit_do_not_train.jsonl').read_text().splitlines() if x.strip()]
    audit_by_id={x['source_id']:x for x in protected}
    if len(audit_by_id)!=len(protected):raise ValueError('repeated reserved audit source')
    result=[]
    for round_number in range(5):
        rows_path=arm_dir/f'round_{round_number}'/'training.jsonl'
        if not rows_path.is_file():raise FileNotFoundError(rows_path)
        for line in rows_path.read_text().splitlines():
            if not line.strip():continue
            row=json.loads(line)
            reserved=audit_by_id.get(row['source_id'])
            if reserved is None:raise ValueError('audit source missing')
            outcome=inspect_record(row,reserved['audit_tests'])
            if outcome is not None:result.append(outcome)
    accepted=[x for x in result if x['actual_optimizer_step']]
    return {'schema':'eesd-five-round-reserved-probe-audit-v1',
            'status':'post-training-only-no-gradient',
            'arm':plan['arm'],'rounds':5,'source_rounds_with_candidates':len(result),
            'actual_updates_with_candidates':len(accepted),
            'mean_selected_minus_full':float(np.mean([x['reserved_selected_minus_full'] for x in accepted])) if accepted else None,
            'positive_selected_alignment_fraction':float(np.mean([x['reserved_audit_score_selected']>0 for x in accepted])) if accepted else None,
            'records':result}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--experiment-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('refuse to overwrite reserved audit')
    report=audit(a.data_dir,a.experiment_dir)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
