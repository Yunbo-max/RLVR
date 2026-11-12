#!/usr/bin/env python3
"""Prepare a leakage-audited T=4 EXACT-Shapley RunBugRun formal cohort.

Each selected train problem has >=24 distinct test inputs, with no conflicting
expected outputs. Freeze ONE bank of 8 potential public feedback tests and
independent 8 calibration + 8 untouched audit inputs. The T=4 model runner
uses ONLY first 4 of the 8 public feedback tests. Thus subsequent T=8
robustness can reuse the same source population without resampling problems.

This script samples AT MOST ONE buggy Python code per problem_id. There is no
fiction of multiple same-problem bugs being independent. If a requested number
of eligible distinct problem IDs is unavailable, FAIL and report counts.
Training never sees gold fixed_code, test outcomes, or meta evaluation labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
from pathlib import Path

SCHEMA='eesd-five-round-frozen-data-v1'
MASTER_TESTS=8
CALIBRATION_TESTS=8
AUDIT_TESTS=8
MIN_UNIQUE_INPUTS=MASTER_TESTS+CALIBRATION_TESTS+AUDIT_TESTS


def _sha_bytes(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def _choose_bug_per_problem(db:sqlite3.Connection,split:int,seed:int,
                            *,max_bug_chars:int=16000)->dict[str,dict]:
    """Deterministic per-problem reservoir based on SHA of bug ID."""
    out={}
    q='''SELECT b.id,b.problem_id,b.buggy_code,p.text
         FROM bugs b JOIN problems p ON p.problem_id=b.problem_id
         WHERE b.active=1 AND b.language=5 AND b.split=?
         ORDER BY b.problem_id,b.id'''
    for bug_id,pid,code,problem_text in db.execute(q,(split,)):
        if not isinstance(pid,str) or not isinstance(code,str) or not isinstance(problem_text,str):
            continue
        if not code.strip() or len(code)>max_bug_chars or not problem_text.strip():
            continue
        rank=_sha_bytes(f'{seed}:{split}:{pid}:{bug_id}'.encode())
        old=out.get(pid)
        if old is None or rank<old['rank']:
            out[pid]={'source_id':f'runbugrun/{pid}/{bug_id}',
                     'problem_id':pid,'task':problem_text,'buggy_code':code,
                     'rank':rank}
    return out


def _unique_input_tests(db:sqlite3.Connection,pid:str,seed:int)->list[dict] | None:
    """Return active tests with unique input, or None if oracle conflicts."""
    examples={}
    sql='''SELECT id,input,output FROM tests WHERE problem_id=? AND active=1
           ORDER BY test_id,id'''
    for tid,inp,out in db.execute(sql,(pid,)):
        if inp is None or out is None:
            continue
        inp=str(inp);out=str(out)
        prev=examples.get(inp)
        if prev is not None:
            if prev['expected']!=out:
                return None   # inconsistent oracle; do not cherry-pick
            continue
        examples[inp]={'id':str(tid),'input':inp,'expected':out}
    rows=list(examples.values())
    if len(rows)<MIN_UNIQUE_INPUTS:return None
    rng=random.Random(int(_sha_bytes(f'{seed}:{pid}:testbank'.encode())[:16],16))
    rng.shuffle(rows)
    return rows


def freeze(db_path:Path,output:Path,*,train_n:int,meta_n:int,final_n:int,
           seed:int=1701)->dict:
    if min(train_n,meta_n,final_n)<=0 or not db_path.is_file() or output.exists():
        raise ValueError('positive counts, restored DB and new output directory required')
    uri=f'file:{db_path.resolve().as_posix()}?mode=ro'
    db=sqlite3.connect(uri,uri=True)
    try:
        db.execute('PRAGMA query_only=ON')
        def take(split:int,needed:int,exclude:set[str]):
            candidates=_choose_bug_per_problem(db,split,seed)
            selected=[]
            for pid in sorted(candidates,key=lambda p:_sha_bytes(f'{seed}:{split}:{p}:sample'.encode())):
                if pid in exclude:continue
                tests=_unique_input_tests(db,pid,seed)
                if tests is None:continue
                row={k:v for k,v in candidates[pid].items() if k!='rank'}
                selected.append((row,tests))
                if len(selected)==needed:break
            if len(selected)!=needed:
                raise ValueError(f'RunBugRun Python split {split}: need {needed} '
                                 f'eligible UNIQUE problem IDs with >=24 different '
                                 f'valid inputs, available {len(selected)}; '
                                 'reduce requested sample size only BEFORE freezing')
            return selected
        development=take(0,train_n+meta_n,set())
        excluded={r[0]['problem_id'] for r in development}
        final=take(2,final_n,excluded)
    finally:
        db.close()

    train=development[:train_n]
    meta=development[train_n:]
    all_ids=[{x['problem_id'] for x,_ in pop} for pop in (train,meta,final)]
    if any(all_ids[i]&all_ids[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError('overlapping problem IDs across populations')

    records={name:[] for name in (
        'train_public.jsonl','train_feedback_master_8_do_not_train.jsonl',
        'train_calibration.jsonl','train_audit_do_not_train.jsonl',
        'meta_probes_do_not_train.jsonl','eval_private.jsonl')}
    for row,tests in train:
        master=tests[:8];cal=tests[8:16];audit=tests[16:24]
        if len({t['input'] for t in master+cal+audit})!=24:
            raise ValueError('test input collision across evidence splits')
        records['train_public.jsonl'].append({**row,'public_tests':master[:4]})
        records['train_feedback_master_8_do_not_train.jsonl'].append({
            'source_id':row['source_id'],'problem_id':row['problem_id'],
            'master_feedback_tests':master})
        records['train_calibration.jsonl'].append({
            'source_id':row['source_id'],'problem_id':row['problem_id'],
            'calibration_tests':cal})
        records['train_audit_do_not_train.jsonl'].append({
            'source_id':row['source_id'],'problem_id':row['problem_id'],
            'audit_tests':audit})
    for row,tests in meta:
        records['meta_probes_do_not_train.jsonl'].append({
            **row,'evaluation_tests':tests})
    for row,tests in final:
        records['eval_private.jsonl'].append({**row,'evaluation_tests':tests})

    output.mkdir(parents=True,exist_ok=False)
    for filename,entries in records.items():
        with (output/filename).open('x') as file:
            for record in entries:
                file.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+'\n')
    manifest={
        'schema':'eesd-runbugrun-public-private-v1',
        'behavioral_schema':'eesd-runbugrun-behavioral-v1',
        'learning_credit_schema':SCHEMA,
        'formal_protocol':'eesd-nested-T4-vs-T8-20261101',
        'dataset':'RunBugRun official SQLite Python',
        'db_sha256':_file_sha(db_path),'seed':seed,
        'train_sources':train_n,'meta_sources':meta_n,'eval_sources':final_n,
        'public_tests':4,'master_feedback_tests':8,
        'calibration_tests':8,'audit_tests':8,
        'unique_test_inputs_required_per_problem':24,
        'one_bug_per_problem':True,
        'fixed_code_never_queried':True,
        'never_read_fixed_code':True,
        'split_policy':'train/meta official split 0, final official split 2, unique disjoint problem_id',
        'file_sha256':{name:_file_sha(output/name) for name in records}}
    (output/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-sources',type=int,default=256)
    parser.add_argument('--meta-sources',type=int,default=64)
    parser.add_argument('--eval-sources',type=int,default=128)
    parser.add_argument('--seed',type=int,default=1701)
    a=parser.parse_args()
    print(json.dumps(freeze(a.db,a.output,train_n=a.train_sources,
                      meta_n=a.meta_sources,final_n=a.eval_sources,seed=a.seed),indent=2))
