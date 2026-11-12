#!/usr/bin/env python3
"""Freeze RunBugRun five-round learning-credit data, with completely disjoint tasks.

Train (official split 0): 4 public feedback tests + 4 training calibration tests
+ 4 reserved *post-training* audit tests. Meta (separate problem IDs from
split 0): never used for selectors or training, reserved for intervention
evaluation of REAL parameter changes. Final (official split 2): never
accessed by any selection/diagnostic trainer.

No fixed/reference corrected code is queried or exported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from prepare_recursive_runbugrun import _select


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def prepare(db_path: Path, root: Path, *, train_n: int=64, meta_n: int=24,
            final_n: int=128, seed: int=1701) -> dict:
    if min(train_n,meta_n,final_n)<1 or root.exists() or not db_path.is_file():
        raise ValueError('positive counts, existing database, fresh output required')
    with sqlite3.connect(f'file:{db_path.resolve().as_posix()}?mode=ro',uri=True) as db:
        # _select is deterministic; splitting AFTER selecting in one call
        # prevents train/meta overlap even if there are multiple buggy files.
        development=_select(db,0,train_n+meta_n,seed,12)
        final=_select(db,2,final_n,seed+7301,12,
             exclude_problem_ids={r['problem_id'] for r in development})
    train=development[:train_n]
    meta=development[train_n:]
    train_public=[];calibration=[];reserved=[];meta_data=[];final_data=[]
    for r in train:
        x={k:r[k] for k in ('source_id','problem_id','task','buggy_code')}
        train_public.append({**x,'public_tests':r['tests'][:4]})
        calibration.append({'source_id':r['source_id'],'problem_id':r['problem_id'],
                            'calibration_tests':r['tests'][4:8]})
        reserved.append({'source_id':r['source_id'],'problem_id':r['problem_id'],
                         'audit_tests':r['tests'][8:12]})
    for r in meta:
        x={k:r[k] for k in ('source_id','problem_id','task','buggy_code')}
        meta_data.append({**x,'evaluation_tests':r['tests']})
    for r in final:
        x={k:r[k] for k in ('source_id','problem_id','task','buggy_code')}
        final_data.append({**x,'evaluation_tests':r['tests']})
    populations=[
        {r['problem_id'] for r in a} for a in (train_public,meta_data,final_data)]
    if any(populations[i]&populations[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError('train, meta and final problem IDs overlap')
    root.mkdir(parents=True,exist_ok=False)
    output={
        'train_public.jsonl':train_public,
        'train_calibration.jsonl':calibration,
        'train_audit_do_not_train.jsonl':reserved,
        'meta_probes_do_not_train.jsonl':meta_data,
        'eval_private.jsonl':final_data,
    }
    for fname,items in output.items():
        with (root/fname).open('x') as file:
            for row in items:file.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n')
    manifest={
        'schema':'eesd-runbugrun-public-private-v1',
        'behavioral_schema':'eesd-runbugrun-behavioral-v1',
        'learning_credit_schema':'eesd-five-round-frozen-data-v1',
        'dataset':'RunBugRun official Python SQLite',
        'db_sha256':_sha(db_path),
        'train_sources':train_n,'meta_sources':meta_n,'eval_sources':final_n,
        'public_tests':4,'calibration_tests':4,'audit_tests':4,
        'split_policy':'train=0, meta=train=0/disjoint problem ID, final=test=2',
        'never_read_fixed_code':True,'no_gold_code':True,
        'seed':seed,
        'file_sha256':{p:_sha(root/p) for p in output},
    }
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-sources',type=int,default=64)
    p.add_argument('--meta-sources',type=int,default=24)
    p.add_argument('--eval-sources',type=int,default=128)
    p.add_argument('--seed',type=int,default=1701)
    a=p.parse_args()
    print(json.dumps(prepare(a.db,a.output,train_n=a.train_sources,
          meta_n=a.meta_sources,final_n=a.eval_sources,seed=a.seed),indent=2))
