import importlib.util
import json
import sqlite3
import sys
import types
from pathlib import Path
import pytest

stub=types.ModuleType('prepare_recursive_runbugrun')
def select(db,split,n,seed,minimum_tests,exclude_problem_ids=None):
    rows=[]
    for i,pid,code,text in db.execute(
        'SELECT b.id,b.problem_id,b.buggy_code,p.text FROM bugs b JOIN problems p ON p.problem_id=b.problem_id WHERE b.split=? ORDER BY b.id',(split,)):
        tests=[{'id':str(j),'input':str(j),'expected':str(j*3)} for j in range(13)]
        if pid in (exclude_problem_ids or set()) or len(tests)<minimum_tests:continue
        rows.append({'source_id':f'src/{pid}/{i}','problem_id':pid,'task':text,
                     'buggy_code':code,'tests':tests})
    if len(rows)<n:raise ValueError('need more sources')
    return rows[:n]
stub._select=select
prior=sys.modules.get('prepare_recursive_runbugrun')
sys.modules['prepare_recursive_runbugrun']=stub
p=Path(__file__).resolve().parents[1]/'scripts'/'prepare_learning_credit_data.py'
spec=importlib.util.spec_from_file_location('prepare_learning_credit_data',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if prior is None:sys.modules.pop('prepare_recursive_runbugrun',None)
else:sys.modules['prepare_recursive_runbugrun']=prior


def test_data_split_separates_training_calibration_audit_meta_and_final(tmp_path):
    db_path=tmp_path/'rbr.sqlite'
    with sqlite3.connect(db_path) as db:
        db.executescript('CREATE TABLE problems(problem_id TEXT,text TEXT);'
          'CREATE TABLE bugs(id INTEGER,problem_id TEXT,buggy_code TEXT,fixed_code TEXT,split INTEGER);')
        for n,(pid,split) in enumerate((('t1',0),('t2',0),('t3',0),('t4',0),('e1',2),('e2',2))):
            db.execute('INSERT INTO bugs VALUES(?,?,?,?,?)',(n,pid,'print(0)','SECRET GOLD REPAIR',split))
            db.execute('INSERT INTO problems VALUES(?,?)',(pid,'task'))
    root=tmp_path/'frozen'
    manifest=m.prepare(db_path,root,train_n=2,meta_n=2,final_n=2)
    assert manifest['learning_credit_schema']=='eesd-five-round-frozen-data-v1'
    def read(name):
        return [json.loads(v) for v in (root/name).read_text().splitlines()]
    train=read('train_public.jsonl')
    cal=read('train_calibration.jsonl')
    held=read('train_audit_do_not_train.jsonl')
    meta=read('meta_probes_do_not_train.jsonl')
    final=read('eval_private.jsonl')
    assert len(train)==len(cal)==len(held)==len(meta)==len(final)==2
    assert not set(r['problem_id'] for r in train)&set(r['problem_id'] for r in meta)
    assert not set(r['problem_id'] for r in meta)&set(r['problem_id'] for r in final)
    for row,c,h in zip(train,cal,held):
        p={x['id'] for x in row['public_tests']}
        a={x['id'] for x in c['calibration_tests']}
        b={x['id'] for x in h['audit_tests']}
        assert len(p)==len(a)==len(b)==4
        assert not p&a and not p&b and not a&b
    assert 'SECRET GOLD REPAIR' not in ''.join(
        p.read_text() for p in root.iterdir() if p.suffix in ('.jsonl','.json'))
    with pytest.raises(ValueError):m.prepare(db_path,root,train_n=2,meta_n=2,final_n=2)
