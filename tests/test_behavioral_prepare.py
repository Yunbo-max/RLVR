import importlib.util
import json
import sqlite3
import sys
import types
from pathlib import Path
import pytest

# Stub official _select with the same schema and split semantics. The existing
# prepare_recursive_runbugrun tests separately exercise the real SQL extractor.
stub=types.ModuleType('prepare_recursive_runbugrun')
def fake_select(db,split,n,seed,minimum_tests,exclude_problem_ids=None):
    rows=[]
    for bug_id,pid,buggy,task in db.execute('SELECT b.id,b.problem_id,b.buggy_code,p.text FROM bugs b JOIN problems p ON p.problem_id=b.problem_id WHERE b.split=?',(split,)):
        tests=[{'id':str(j),'input':str(j),'expected':str(j)} for j in range(13)]
        if pid in (exclude_problem_ids or set()) or len(tests)<minimum_tests:continue
        rows.append({'source_id':f'src/{pid}/{bug_id}','problem_id':pid,
                     'task':task,'buggy_code':buggy,'tests':tests})
    if len(rows)<n:raise ValueError('not enough samples')
    return rows[:n]
stub._select=fake_select
previous=sys.modules.get('prepare_recursive_runbugrun')
sys.modules['prepare_recursive_runbugrun']=stub
P=Path(__file__).resolve().parents[1]/'scripts'/'prepare_behavioral_runbugrun.py'
s=importlib.util.spec_from_file_location('prepare_behavioral_runbugrun',P)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
if previous is None:sys.modules.pop('prepare_recursive_runbugrun',None)
else:sys.modules['prepare_recursive_runbugrun']=previous


def test_behavioral_dataset_4_4_4_and_no_fixed_code(tmp_path):
    db_file=tmp_path/'db.sqlite';out=tmp_path/'export'
    with sqlite3.connect(db_file) as db:
        db.executescript('''CREATE TABLE problems(problem_id TEXT,text TEXT);
         CREATE TABLE bugs(id INTEGER,problem_id TEXT,buggy_code TEXT,fixed_code TEXT,split INTEGER);''')
        for i,(p,split) in enumerate([('p1',0),('p2',0),('e1',2),('e2',2)]):
            db.execute('INSERT INTO problems VALUES(?,?)',(p,'task'))
            db.execute('INSERT INTO bugs VALUES(?,?,?,?,?)',(i,p,'print(0)','SENSITIVE GOLD CODE',split))
    manifest=m.extract(db_file,out,train_sources=2,eval_sources=2)
    train=[json.loads(x) for x in (out/'train_public.jsonl').read_text().splitlines()]
    cal=[json.loads(x) for x in (out/'train_calibration.jsonl').read_text().splitlines()]
    held=[json.loads(x) for x in (out/'train_audit_do_not_train.jsonl').read_text().splitlines()]
    assert manifest['behavioral_schema']=='eesd-runbugrun-behavioral-v1'
    for a,b,c in zip(train,cal,held):
        A={z['id'] for z in a['public_tests']};B={z['id'] for z in b['calibration_tests']};C={z['id'] for z in c['audit_tests']}
        assert len(A)==len(B)==len(C)==4 and not A&B and not A&C and not B&C
    assert 'SENSITIVE GOLD' not in ''.join(p.read_text() for p in out.glob('*.json*'))
    assert len((out/'eval_private.jsonl').read_text().splitlines())==2
    with pytest.raises(ValueError):m.extract(db_file,out,train_sources=2,eval_sources=2)
