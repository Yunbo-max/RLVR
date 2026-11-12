#!/usr/bin/env python3
"""Extract a leakage-safe Python-only RunBugRun slice from OFFICIAL SQLite.

Uses original buggy code and testcase input/output. Never reads fixed_code.
Train/public and evaluation source problem IDs are disjoint, enforced by code.
Only the four public train tests go into the training JSONL; no held-out tests
are exposed to the trainer. The raw DB must be acquired independently.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sqlite3
from pathlib import Path


def _select(db, split: int, n: int, seed: int, minimum_tests: int,
            exclude_problem_ids: set[str] | None = None) -> list[dict]:
    rows=[]; seen=set()
    # SQLite official schema: 5=python, 0=train, 2=test, active=1.
    query='''SELECT b.id, b.problem_id, b.buggy_code, p.text
        FROM bugs b JOIN problems p ON p.problem_id=b.problem_id
        WHERE b.language=5 AND b.split=? AND b.active=1
        ORDER BY b.problem_id, b.id'''
    for bug_id,pid,buggy,task in db.execute(query,(split,)):
        if pid in seen or pid in (exclude_problem_ids or set()) or not isinstance(buggy,str) or not buggy.strip():
            continue
        seen.add(pid)
        raw=db.execute('''SELECT id, input, output FROM tests
            WHERE problem_id=? AND active=1 ORDER BY test_id, id''',(pid,)).fetchall()
        if len(raw)<minimum_tests or not isinstance(task,str) or not task.strip():
            continue
        tests=[{'id':str(tid),'input':str(inp),'expected':str(out)} for tid,inp,out in raw]
        if len({t['id'] for t in tests}) != len(tests):
            raise ValueError('duplicate test identity')
        rng=random.Random(f'{seed}:{pid}')
        rng.shuffle(tests)
        rows.append({'source_id':f'runbugrun/{pid}/{bug_id}',
                     'problem_id':str(pid),'task':task,'buggy_code':buggy,
                     'tests':tests})
    rng=random.Random(seed+split)
    rng.shuffle(rows)
    if len(rows)<n:
        raise ValueError(f'only {len(rows)} eligible sources, require {n}')
    return rows[:n]


def extract(db_path: str|Path, output: str|Path, *, train_sources: int,
            eval_sources: int, seed: int, public_tests: int=4, audit_tests: int=4) -> dict:
    path=Path(db_path); dest=Path(output)
    if not path.is_file() or dest.exists():
        raise ValueError('SQLite DB missing or output directory already exists')
    if train_sources<1 or eval_sources<1 or public_tests!=4 or audit_tests<1:
        raise ValueError('require positive source counts, four public and audit tests')
    uri=f'file:{path.resolve().as_posix()}?mode=ro'
    with sqlite3.connect(uri, uri=True) as db:
        train=_select(db,0,train_sources,seed,public_tests+audit_tests)
        test=_select(db,2,eval_sources,seed+7301,public_tests+audit_tests,
                     exclude_problem_ids={r['problem_id'] for r in train})
    train_public=[]; train_audit=[]; eval_rows=[]
    for r in train:
        prefix={k:r[k] for k in ('source_id','problem_id','task','buggy_code')}
        train_public.append({**prefix,'public_tests':r['tests'][:public_tests]})
        train_audit.append({'source_id':r['source_id'],'problem_id':r['problem_id'],
                            'audit_tests':r['tests'][public_tests:public_tests+audit_tests]})
    for r in test:
        prefix={k:r[k] for k in ('source_id','problem_id','task','buggy_code')}
        eval_rows.append({**prefix,'evaluation_tests':r['tests']})
    dest.mkdir(parents=True,exist_ok=False)
    for name,rows in [('train_public.jsonl',train_public),('train_audit_do_not_train.jsonl',train_audit),
                      ('eval_private.jsonl',eval_rows)]:
        with (dest/name).open('x') as out:
            for row in rows:out.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('*.jsonl')}
    manifest={'schema':'eesd-runbugrun-public-private-v1','dataset':'RunBugRun official SQLite',
              'db_path':str(path.resolve()),'db_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'train_sources':train_sources,'eval_sources':eval_sources,'public_tests':public_tests,
              'audit_tests':audit_tests,'seed':seed,'file_sha256':hashes,
              'never_read_fixed_code':True,'split':'train=0, test=2, Python language=5',
              'selection':'unique problem_id per split, fail on train/test overlap'}
    (dest/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-sources',type=int,default=64)
    parser.add_argument('--eval-sources',type=int,default=64)
    parser.add_argument('--seed',type=int,default=1701)
    args=parser.parse_args()
    print(json.dumps(extract(args.db,args.output,train_sources=args.train_sources,
                     eval_sources=args.eval_sources,seed=args.seed),indent=2))
