#!/usr/bin/env python3
"""Post-training, entirely independent audit of selected Shapley feedback.

This diagnostic NEVER updates or loads model weights and NEVER runs as part of
feedback selection. Uses reserved train audits only AFTER candidate generation
and training are complete. Requires proper bubblewrap sandbox; no soft fallback.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from eesd.behavioral_shapley import audit_value_game
from eesd.strict_execution import audit_vector


def audit_selection(record:dict, tests:list[dict], execute=audit_vector)->dict|None:
    if 'candidate_sources' not in record or 'candidate_shift_means' not in record:
        return None
    codes=record['candidate_sources']; shifts=record['candidate_shift_means']
    if len(codes)<2 or len(codes)!=len(shifts):raise ValueError('candidate provenance mismatch')
    arr=np.asarray(shifts,dtype=np.float64)
    if arr.ndim!=2 or arr.shape[1]!=16 or not np.isfinite(arr).all():
        raise ValueError('require exact 16 feedback subset scores')
    outcomes=np.asarray([execute(code,tests) for code in codes],dtype=np.float64)
    # Each candidate's logP contrast across coalitions is its stored mean shift.
    value=audit_value_game([v[:,None] for v in arr],outcomes)
    chosen=int(record.get('selected_mask',0))
    if not 0<=chosen<16:raise ValueError('invalid selected subset')
    return {'source_id':record['source_id'],'round':int(record['round']) if 'round' in record else None,
            'selected_mask':chosen,'selected_probe_alignment':float(value[chosen].mean()),
            'full_probe_alignment':float(value[15].mean()),
            'selected_minus_full':float((value[chosen]-value[15]).mean()),
            'calibration_selection_reason':record.get('selection',{}).get('reason'),
            'heldout_probe_count':len(tests),'heldout_variance':float(outcomes.var(axis=0).mean())}


def run(data_dir:Path,experiment_dir:Path)->dict:
    manifest=json.loads((data_dir/'manifest.json').read_text())
    if manifest.get('behavioral_schema')!='eesd-runbugrun-behavioral-v1':
        raise ValueError('requires the independently split 4+4+4 data manifest')
    rows=[json.loads(line) for line in (data_dir/'train_audit_do_not_train.jsonl').read_text().splitlines()]
    by_source={r['source_id']:r for r in rows}
    if len(by_source)!=len(rows):raise ValueError('duplicate reserved-audit sources')
    entries=[]
    for r in range(3):
        records_path=experiment_dir/f'round_{r}'/'training.jsonl'
        if not records_path.is_file():raise FileNotFoundError(records_path)
        for line in records_path.read_text().splitlines():
            record=json.loads(line)
            row=by_source.get(record['source_id'])
            if row is None:raise ValueError('unknown reserved-audit source')
            value=audit_selection({**record,'round':r},row['audit_tests'])
            if value is not None:entries.append(value)
    accepted=[e for e in entries if e['selected_mask']]
    result={'schema':'eesd-post-training-holdout-feedback-audit-v1',
        'status':'diagnostic_only_never_used_in_training',
        'sources_audited':len(entries),'accepted_sources':len(accepted),
        'accept_fraction':len(accepted)/len(entries) if entries else None,
        'mean_selected_minus_full_on_heldout':float(np.mean([e['selected_minus_full'] for e in accepted])) if accepted else None,
        'fraction_accepted_with_positive_heldout_alignment':float(np.mean([e['selected_probe_alignment']>0 for e in accepted])) if accepted else None,
        'by_source':entries}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--experiment-dir',type=Path,required=True,
            help='Shapley arm output folder containing round_0..2/training.jsonl')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('never overwrite audit receipt')
    output=run(args.data_dir,args.experiment_dir)
    args.output.write_text(json.dumps(output,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='by_source'},indent=2))
