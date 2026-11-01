#!/usr/bin/env python3
"""Feasibility audit for a *publication-scale* RunBugRun Python experiment.

Question: how many genuinely distinct input tests exist per problem, and can
we fairly compare 4 vs 8 public feedback tests while reserving calibration
and audit tests? Reads the OFFICIAL SQLite in read-only mode. Never selects or
reads fixed_code. Does not execute unknown code and requires no GPU.

Important: "different test IDs" and "different input/output pairs" need NOT
mean independent evidence. We count unique INPUTS as the conservative test
eligibility criterion, and explicitly report inconsistent expected outputs
for identical inputs. Behavioral independence must still be checked from
per-candidate PASS/FAIL response vectors in a later pilot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

THRESHOLDS=(4,8,12,16,20,24,32,40,48)
SPLITS={0:"train",1:"validation",2:"test"}


def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def quantiles(v:list[int])->dict:
    if not v:return {'n':0,'min':None,'p10':None,'median':None,'p90':None,'max':None}
    a=sorted(v);n=len(a)
    def at(p:float)->int:return a[min(n-1,math.ceil(p*n)-1)]
    return {'n':n,'min':a[0],'p10':at(.1),'median':statistics.median(a),
            'p90':at(.9),'max':a[-1]}


def audit(db_path:Path,*,min_tokens:int|None=None,max_tokens:int|None=None)->dict:
    if not db_path.is_file():raise FileNotFoundError(db_path)
    if min_tokens is not None and min_tokens<0:raise ValueError('negative min_tokens')
    if max_tokens is not None and max_tokens<1:raise ValueError('invalid max_tokens')
    if min_tokens is not None and max_tokens is not None and min_tokens>max_tokens:
        raise ValueError('invalid token range')

    con=sqlite3.connect(f'file:{db_path.resolve().as_posix()}?mode=ro',uri=True)
    try:
        con.execute('PRAGMA query_only=ON')
        active_raw=dict(con.execute(
            'SELECT problem_id,COUNT(*) FROM tests WHERE active=1 GROUP BY problem_id'))
        # Unique input, not just test IDs or input-output row identities.
        unique_inputs=dict(con.execute(
            'SELECT problem_id,COUNT(*) FROM ('
            ' SELECT problem_id,input FROM tests WHERE active=1'
            ' GROUP BY problem_id,input'
            ') GROUP BY problem_id'))
        unique_pairs=dict(con.execute(
            'SELECT problem_id,COUNT(*) FROM ('
            ' SELECT problem_id,input,output FROM tests WHERE active=1'
            ' GROUP BY problem_id,input,output'
            ') GROUP BY problem_id'))
        inconsistent=dict(con.execute(
            'SELECT problem_id,COUNT(*) FROM ('
            ' SELECT problem_id,input FROM tests WHERE active=1'
            ' GROUP BY problem_id,input HAVING COUNT(DISTINCT output)>1'
            ') GROUP BY problem_id'))
        token_condition=''
        args=[]
        if min_tokens is not None:
            token_condition+=' AND b.token_count>=?';args.append(min_tokens)
        if max_tokens is not None:
            token_condition+=' AND b.token_count<=?';args.append(max_tokens)
        query=('SELECT b.split,b.problem_id,COUNT(*) AS bugs,'
               'MIN(b.token_count),MAX(b.token_count)'
               ' FROM bugs b WHERE b.active=1 AND b.language=5'
               ' AND b.split IN (0,1,2)'+token_condition+
               ' GROUP BY b.split,b.problem_id')
        rows=list(con.execute(query,args))
    finally:
        con.close()

    bysplit={v:[] for v in SPLITS}
    for split,pid,bugs,low_tok,high_tok in rows:
        if pid not in unique_inputs:continue
        bysplit[split].append({'problem':pid,'bugs':int(bugs),
            'unique_inputs':int(unique_inputs[pid]),
            'unique_pairs':int(unique_pairs[pid]),
            'active_tests':int(active_raw[pid]),
            'inconsistent_expected_inputs':int(inconsistent.get(pid,0)),
            'min_source_tokens':low_tok,'max_source_tokens':high_tok})
    stats={}
    for split,name in SPLITS.items():
        records=bysplit[split]
        pool=[x for x in records if x['inconsistent_expected_inputs']==0]
        eligible={}
        for threshold in THRESHOLDS:
            included=[x for x in pool if x['unique_inputs']>=threshold]
            eligible[str(threshold)]={'distinct_problem_ids':len(included),
                'total_bug_instances':sum(x['bugs'] for x in included)}
        stats[name]={
            'python_distinct_problem_ids_with_tests':len(records),
            'python_bug_instances_with_tests':sum(x['bugs'] for x in records),
            'problems_with_conflicting_expected_output_for_same_input':
                sum(x['inconsistent_expected_inputs']>0 for x in records),
            'unique_input_tests_per_problem':quantiles([x['unique_inputs'] for x in records]),
            'active_test_ids_per_problem':quantiles([x['active_tests'] for x in records]),
            'duplicate_test_rows_per_problem':quantiles(
                [x['active_tests']-x['unique_pairs'] for x in records]),
            'eligible_without_conflicting_inputs':eligible,
        }
    ps={i:set(x['problem'] for x in bysplit[i]) for i in SPLITS}
    return {
        'schema':'eesd-runbugrun-feasibility-audit-v1',
        'dataset':'RunBugRun official SQLite, Python language=5',
        'db_sha256':digest(db_path),
        'model_input_token_filter':{'min':min_tokens,'max':max_tokens},
        'split_stats':stats,
        'problem_id_overlap_raw_dataset':{
            'train_test':len(ps[0]&ps[2]),
            'train_validation':len(ps[0]&ps[1]),
            'validation_test':len(ps[1]&ps[2]),
        },
        'interpretation':{
            'T4_requires_12_unique_test_inputs':'4 feedback + 4 calibration + 4 reserved',
            'T4_with_better_calibration_requires_20':'4 feedback + 8 calibration + 8 reserved',
            'T8_requires_24_unique_test_inputs':'8 feedback + 8 calibration + 8 reserved',
            'nested_T4_vs_T8_pool':'Use the exact SAME >=24-unique-input eligible problem population, split once, hold C8/A8 fixed',
            'statistical_unit':'problem_id, not independent bugs sampled from the same problem',
            'execution_feedback_diversity':'not measurable from SQLite test rows alone; run a fixed-model pilot',
            'safety':'No buggy/fixed code was executed, fixed_code was never queried',
        },
    }


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--min-source-tokens',type=int)
    parser.add_argument('--max-source-tokens',type=int)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError('refuse to overwrite feasibility receipt')
    report=audit(a.db,min_tokens=a.min_source_tokens,max_tokens=a.max_source_tokens)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report['split_stats'].items()},indent=2))
