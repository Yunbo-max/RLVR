#!/usr/bin/env python3
"""Seal random control schedules using ONLY grouped-Shapley TRAINING receipts.

Supports both five-round repeated small task panels and 8K bug-instance
five-round rotating panels. Never reads final evaluation outcomes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from eesd.learning_credit_selection import sample_matched_update_sources


def make(root:Path)->dict:
    plan=json.loads((root/'plan.json').read_text())
    if (plan.get('schema')!='eesd-rich64-five-v1' or plan.get('arm')!='shapley'
        or plan.get('rounds')!=5):
        raise ValueError('must be a completed five-round grouped-Shapley training run')
    allrounds=[];population=None
    for rnd in range(5):
        path=root/f'round_{rnd}'/'training.jsonl'
        if not path.is_file():raise FileNotFoundError(path)
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        id_map={r['source_id']:r for r in rows}
        if len(id_map)!=len(rows) or not rows:
            raise ValueError('duplicate or empty Shapley training source rows')
        if population is None:population=sorted(id_map)
        if sorted(id_map)!=population:raise ValueError('population changed across rounds')
        masks={}
        panel=[]
        for sid in population:
            r=id_map[sid]
            if r.get('skip')!='not_in_round_panel':
                panel.append(sid)
            if r.get('step') is True:
                mask=int(r.get('selected_mask',0))
                if not 0<mask<16:
                    raise ValueError('actual Shapley step has invalid group subset')
                masks[sid]=mask
            else:masks[sid]=0
        count=sum(x>0 for x in masks.values())
        if count>len(panel):
            raise ValueError('more optimizer steps than active panel candidates')
        selected=sample_matched_update_sources(panel,count,
              seed=plan['seed'],round_index=rnd)
        allrounds.append({
            'round':rnd,'accepted_masks':masks,'panel_active_ids':sorted(panel),
            'shapley_actual_optimizer_steps':count,
            'random_update_sources':selected,
            'training_receipt_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return {'schema':'eesd-rich64-schedule-v1','seed':plan['seed'],
        'rounds':5,'exposed_feedback':plan['exposed_feedback'],
        'panel_mode':plan['panel_mode'],
        'dataset_manifest_sha256':plan['dataset_manifest_sha256'],
        'initial_adapter_sha256':plan['initial_adapter_sha256'],
        'source_ids':population,'training_rounds':allrounds,
        'note':'TRAINING-only receipt, no heldout score used to schedule random controls'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shapley-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError('schedule already exists')
    data=make(a.shapley_run)
    a.output.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'round_counts':[x['shapley_actual_optimizer_steps']
       for x in data['training_rounds']]},indent=2))
