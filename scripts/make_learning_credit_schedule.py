#!/usr/bin/env python3
"""Freeze Shapley acceptance schedule for matched-update/subset ablations.

Reads TRAINING receipts ONLY, never final or meta performance. The resulting
schedule must be fixed BEFORE running any control. It is deliberately NOT an
independent randomized controlled trial: closed-loop policies diverge in later
rounds. The schedule isolates source/update count, not identical trajectories.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from eesd.learning_credit_selection import sample_matched_update_sources


def _sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_schedule(run_root:Path, *, seed:int=1701,rounds:int=5)->dict:
    plan=json.loads((run_root/'plan.json').read_text())
    if plan.get('schema')!='eesd-learning-credit-five-v1' or plan.get('arm')!='shapley':
        raise ValueError('schedule must originate from the Shapley five-round arm')
    if plan.get('rounds')!=rounds or plan.get('seed')!=seed:
        raise ValueError('schedule and training seed/rounds differ')
    all_rounds=[];population=None
    for k in range(rounds):
        p=run_root/f'round_{k}'/'training.jsonl'
        if not p.is_file():raise FileNotFoundError(p)
        rows=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        ids=[r.get('source_id') for r in rows]
        if not ids or len(ids)!=len(set(ids)) or any(not isinstance(x,str) for x in ids):
            raise ValueError('duplicate, missing or empty source ID')
        if population is None:population=sorted(ids)
        if sorted(ids)!=population:raise ValueError('training population changes over rounds')
        accepted={}
        for r in rows:
            done=r.get('step') is True
            mask=int(r.get('selected_mask',0))
            if not 0<=mask<=15:raise ValueError('illegal selected feedback mask')
            if done and mask==0:raise ValueError('Shapley marked step with zero feedback')
            accepted[r['source_id']]=mask if done else 0
        count=sum(bool(x) for x in accepted.values())
        sampled=sample_matched_update_sources(population,count,
                          seed=seed,round_index=k)
        all_rounds.append({'round':k,'accepted_masks':accepted,
            'shapley_actual_optimizer_steps':count,
            'random_update_sources':sampled,
            'training_receipt_sha256':_sha(p)})
    return {'schema':'eesd-learning-credit-schedule-v1','seed':seed,
            'rounds':rounds,'train_sources':len(population),
            'data_manifest_sha256':plan['dataset_manifest_sha256'],
            'initial_adapter_sha256':plan['initial_adapter_sha256'],
            'source_ids':population,'training_rounds':all_rounds,
            'origin':'only Shapley training decisions, not final/meta evaluation'}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shapley-run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=1701)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('refuse schedule overwrite')
    data=make_schedule(a.shapley_run,seed=a.seed)
    a.output.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'schedule':str(a.output),'update_counts':
        [x['shapley_actual_optimizer_steps'] for x in data['training_rounds']]},indent=2))
