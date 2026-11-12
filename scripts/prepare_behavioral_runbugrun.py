#!/usr/bin/env python3
"""Freeze 4 feedback + 4 training calibration + 4 held-out audit tests.

RunBugRun official SQLite, Python only. No corrected reference code queried.
Training may use calibration tests to weight/choose feedback; the audit file is
NEVER opened by model training, only used for a test-ID-disjointness check.
Unseen evaluation problems are a completely separate database split.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from prepare_recursive_runbugrun import _select


def _sha_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def extract(db_file:Path, output:Path, *,train_sources:int=64,eval_sources:int=64,seed:int=1701)->dict:
    if train_sources<1 or eval_sources<1 or output.exists() or not db_file.is_file():
        raise ValueError('bad source counts or input/output paths')
    with sqlite3.connect(f'file:{db_file.resolve().as_posix()}?mode=ro',uri=True) as db:
        train=_select(db,0,train_sources,seed,12)
        evaluation=_select(db,2,eval_sources,seed+7301,12,
                exclude_problem_ids={r['problem_id'] for r in train})
    train_public=[];calibration=[];audit=[];eval_rows=[]
    for r in train:
        identity={k:r[k] for k in ('source_id','problem_id','task','buggy_code')}
        train_public.append({**identity,'public_tests':r['tests'][:4]})
        calibration.append({'source_id':r['source_id'],'problem_id':r['problem_id'],
                            'calibration_tests':r['tests'][4:8]})
        audit.append({'source_id':r['source_id'],'problem_id':r['problem_id'],
                      'audit_tests':r['tests'][8:12]})
    for r in evaluation:
        identity={k:r[k] for k in ('source_id','problem_id','task','buggy_code')}
        eval_rows.append({**identity,'evaluation_tests':r['tests']})
    output.mkdir(parents=True,exist_ok=False)
    for filename,rows in (
        ('train_public.jsonl',train_public),
        ('train_calibration.jsonl',calibration),
        ('train_audit_do_not_train.jsonl',audit),
        ('eval_private.jsonl',eval_rows)):
        with (output/filename).open('x') as f:
            for row in rows:f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n')
    checksums={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.jsonl')}
    manifest={'schema':'eesd-runbugrun-public-private-v1',
              'behavioral_schema':'eesd-runbugrun-behavioral-v1',
              'dataset':'RunBugRun official SQLite',
              'train_sources':train_sources,'eval_sources':eval_sources,
              'public_tests':4,'calibration_tests':4,'audit_tests':4,
              'seed':seed,'file_sha256':checksums,'never_read_fixed_code':True,
              'split':'train=0, test=2, Python language=5',
              'db_sha256':_sha_file(db_file),
              'split_policy':'problem_id disjoint; 4 feedback + 4 calibration + 4 reserved audit',
              'uses_gold_code':False}
    (output/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-sources',type=int,default=64)
    p.add_argument('--eval-sources',type=int,default=64)
    p.add_argument('--seed',type=int,default=1701)
    args=p.parse_args()
    print(json.dumps(extract(args.db,args.output,train_sources=args.train_sources,
                 eval_sources=args.eval_sources,seed=args.seed),indent=2))
