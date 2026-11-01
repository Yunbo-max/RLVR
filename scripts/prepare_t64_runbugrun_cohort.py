#!/usr/bin/env python3
"""Freeze an 8K-BUG INSTANCE RunBugRun Python cohort with 64 public tests.

RunBugRun derives from CodeNet (~4053 source PROBLEMS overall): 8000 bug
instances are possible in principle; 8000 independent problem IDs are NOT.
Require a census of distinct input tests before accepting the requested
sample size. Select at most max_per_problem bugs per problem and compute
future confidence intervals CLUSTERED by problem_id.

F64 public execution inputs + C8 training calibration + A8 never-used audit
are disjoint by INPUT STRING, not merely by test ID. One fixed F64 bank
allows nested T4/8/16/32/64 ablations with exactly the SAME bug instances.
No fixed_code column is queried, and this script executes no generated code.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sqlite3


def sha_string(s:str)->str:
    return hashlib.sha256(s.encode()).hexdigest()


def file_sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as file:
        for buf in iter(lambda:file.read(1048576),b''):
            h.update(buf)
    return h.hexdigest()


def get_unique_input_bank(db:sqlite3.Connection,pid:str,*,seed:int,
                          feedback:int=64,calibration:int=8,audit:int=8,
                          max_io_chars:int=8192) -> list[dict] | None:
    """Exclude inconsistent or oversized test oracle records."""
    rows={}
    for tid,inp,out,origin in db.execute(
        'SELECT id,input,output,origin FROM tests '
        'WHERE problem_id=? AND active=1 ORDER BY test_id,id',(pid,)):
        if not isinstance(inp,str) or not isinstance(out,str):
            continue
        if len(inp)>max_io_chars or len(out)>max_io_chars:
            continue
        item=rows.get(inp)
        if item is not None:
            if item['expected']!=out:return None
            continue
        rows[inp]={'id':str(tid),'input':inp,'expected':out,
                   'origin':int(origin or 0)}
    if len(rows)<feedback+calibration+audit:
        return None
    bank=list(rows.values())
    rng=random.Random(int(sha_string(f'{seed}:{pid}:frozen-testbank')[:16],16))
    rng.shuffle(bank)
    return bank


def _top_bugs_by_problem(db:sqlite3.Connection,split:int,*,seed:int,
                         cap:int,max_code_chars:int=16000,
                         max_task_chars:int=18000)->dict[str,list[dict]]:
    if cap<1:raise ValueError('positive maximum instances/problem required')
    import heapq
    heaps=defaultdict(list)
    query='''SELECT b.id,b.problem_id,b.buggy_code,p.text
           FROM bugs b JOIN problems p ON p.problem_id=b.problem_id
           WHERE b.active=1 AND b.language=5 AND b.split=?
           ORDER BY b.problem_id,b.id'''
    for bugid,pid,code,task in db.execute(query,(split,)):
        if not isinstance(pid,str) or not isinstance(code,str) or not isinstance(task,str):
            continue
        if not code.strip() or not task.strip() or len(code)>max_code_chars or len(task)>max_task_chars:
            continue
        # Prefer lowest stable hash ranks within each problem. Keep only cap.
        rank=int(sha_string(f'{seed}:{pid}:{bugid}')[:16],16)
        item={'source_id':f'runbugrun/{pid}/{bugid}',
              'problem_id':pid,'task':task,'buggy_code':code}
        heap=heaps[pid]
        if len(heap)<cap:
            heapq.heappush(heap,(-rank,str(bugid),item))
        elif rank<-heap[0][0]:
            heapq.heapreplace(heap,(-rank,str(bugid),item))
    result={}
    for pid,heap in heaps.items():
        items=sorted(heap,key=lambda v:(-v[0],v[1]))
        result[pid]=[x[-1] for x in items]
    return result


def select_balanced(bugs:dict[str,list[dict]],banks:dict[str,list[dict]],
                    target:int,*,seed:int,max_per_problem:int)->list[tuple[dict,list[dict]]]:
    """Round-robin across distinct problem IDs, cap submissions/problem."""
    if target<=0:raise ValueError('positive target')
    problems=sorted(set(bugs)&set(banks),key=lambda pid:sha_string(f'{seed}:problem:{pid}'))
    chosen=[]
    for rank in range(max_per_problem):
        for pid in problems:
            if rank<len(bugs[pid]):
                chosen.append((bugs[pid][rank],banks[pid]))
                if len(chosen)==target:
                    return chosen
    raise ValueError(f'insufficient eligible BUG INSTANCES: requested {target}, '
                     f'available {len(chosen)} from {len(problems)} problems '
                     f'at maximum {max_per_problem} instances/problem; '
                     'do not silently lower test count or invent test inputs')


def freeze(db_path:Path,root:Path,*,train_bugs:int=8000,
           meta_problems:int=64,eval_problems:int=256,
           max_per_problem:int=16,seed:int=1701,
           public_tests:int=64,calibration_tests:int=8,
           audit_tests:int=8)->dict:
    if (min(train_bugs,meta_problems,eval_problems,max_per_problem,
            public_tests,calibration_tests,audit_tests)<1
        or public_tests!=64 or not db_path.is_file() or root.exists()):
        raise ValueError('fresh output, official SQLite, T64 and positive counts required')
    con=sqlite3.connect(f'file:{db_path.resolve().as_posix()}?mode=ro',uri=True)
    try:
        con.execute('PRAGMA query_only=ON')
        train_bugs_per_problem=_top_bugs_by_problem(con,0,seed=seed,cap=max_per_problem)
        test_bugs_per_problem=_top_bugs_by_problem(con,2,seed=seed,cap=1)
        def find_banks(pid_collection,exclude=None):
            out={}
            for pid in pid_collection:
                if pid in (exclude or set()):continue
                bank=get_unique_input_bank(con,pid,seed=seed,feedback=64,
                         calibration=calibration_tests,audit=audit_tests)
                if bank is not None:out[pid]=bank
            return out
        training_banks=find_banks(train_bugs_per_problem)
        # Meta is deliberately reserved from training, chosen first as
        # independent *problem IDs* from the train split.
        pids=sorted(training_banks,key=lambda pid:sha_string(f'{seed}:meta:{pid}'))
        if len(pids)<meta_problems+1:
            raise ValueError('too few eligible independent training problems to reserve meta')
        meta_ids=set(pids[:meta_problems])
        selected_train=select_balanced(train_bugs_per_problem,
                {pid:bank for pid,bank in training_banks.items() if pid not in meta_ids},
                train_bugs,seed=seed,max_per_problem=max_per_problem)
        eval_banks=find_banks(test_bugs_per_problem,exclude=set(training_banks))
        eval_ids=sorted(eval_banks,key=lambda pid:sha_string(f'{seed}:eval:{pid}'))
        if len(eval_ids)<eval_problems:
            raise ValueError(f'need {eval_problems} distinct final problem IDs '
                             f'with >= {public_tests+calibration_tests+audit_tests} '
                             f'unique input tests, available only {len(eval_ids)}')
        selected_meta=[(train_bugs_per_problem[pid][0],training_banks[pid])
                       for pid in pids[:meta_problems]]
        selected_eval=[(test_bugs_per_problem[pid][0],eval_banks[pid])
                       for pid in eval_ids[:eval_problems]]
    finally:
        con.close()

    train_pids={row['problem_id'] for row,_ in selected_train}
    meta_pids={row['problem_id'] for row,_ in selected_meta}
    eval_pids={row['problem_id'] for row,_ in selected_eval}
    if train_pids&meta_pids or train_pids&eval_pids or meta_pids&eval_pids:
        raise ValueError('problem_id contamination')

    output={k:[] for k in ('train_public_master64.jsonl',
                          'train_calibration.jsonl','train_audit_do_not_train.jsonl',
                          'meta_do_not_train.jsonl','eval_private.jsonl')}
    for row,bank in selected_train:
        public=bank[:64]
        cal=bank[64:64+calibration_tests]
        reserved=bank[64+calibration_tests:64+calibration_tests+audit_tests]
        if len({t['input'] for t in public+cal+reserved})!=64+calibration_tests+audit_tests:
            raise ValueError('duplicated input across feedback/calibration/audit')
        output['train_public_master64.jsonl'].append({**row,'public_tests':public})
        output['train_calibration.jsonl'].append({'source_id':row['source_id'],
            'problem_id':row['problem_id'],'calibration_tests':cal})
        output['train_audit_do_not_train.jsonl'].append({'source_id':row['source_id'],
            'problem_id':row['problem_id'],'audit_tests':reserved})
    for row,bank in selected_meta:
        output['meta_do_not_train.jsonl'].append({**row,
                     'evaluation_tests':bank[:64+calibration_tests+audit_tests]})
    for row,bank in selected_eval:
        output['eval_private.jsonl'].append({**row,
                     'evaluation_tests':bank[:64+calibration_tests+audit_tests]})

    root.mkdir(parents=True,exist_ok=False)
    for filename,records in output.items():
        with (root/filename).open('x') as file:
            for record in records:
                file.write(json.dumps(record,sort_keys=True,ensure_ascii=False)+'\n')
    counts={pid:0 for pid in train_pids}
    for row,_ in selected_train:counts[row['problem_id']]+=1
    manifest={
        'schema':'eesd-rich64-frozen-v1',
        'dataset':'RunBugRun official SQLite Python only',
        'seed':seed,'db_sha256':file_sha(db_path),
        'train_bug_instances':len(selected_train),
        'train_distinct_problem_ids':len(train_pids),
        'meta_distinct_problem_ids':len(meta_pids),
        'eval_distinct_problem_ids':len(eval_pids),
        'min_public_feedback_tests':64,
        'calibration_tests':calibration_tests,'reserved_audit_tests':audit_tests,
        'unique_input_tests_per_train_problem_required':
                 64+calibration_tests+audit_tests,
        'max_bug_instances_per_problem':max(counts.values()),
        'train_problem_instance_histogram':dict(sorted(Counter(counts.values()).items())) if False else None,
        'all_arms_execute_64_public_tests_but_expose_only_nested_T':True,
        'gold_fixed_code_never_queried':True,
        'final_evaluation_protocol':'fixed first 80 (or 64+C+A) active test inputs per distinct problem, not all official tests',
        'file_sha256':{p:file_sha(root/p) for p in output},
    }
    del manifest['train_problem_instance_histogram']
    (root/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-bugs',type=int,default=8000)
    parser.add_argument('--meta-problems',type=int,default=64)
    parser.add_argument('--eval-problems',type=int,default=256)
    parser.add_argument('--max-per-problem',type=int,default=16)
    parser.add_argument('--seed',type=int,default=1701)
    a=parser.parse_args()
    print(json.dumps(freeze(a.db,a.output,train_bugs=a.train_bugs,
          meta_problems=a.meta_problems,eval_problems=a.eval_problems,
          max_per_problem=a.max_per_problem,seed=a.seed),indent=2))
