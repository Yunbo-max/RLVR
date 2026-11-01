import importlib.util
import json
import sqlite3
from pathlib import Path
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'prepare_formal_runbugrun_t4.py'
s=importlib.util.spec_from_file_location('formal_rbr_t4',SCRIPT)
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)


def build_db(path):
    with sqlite3.connect(path) as db:
        db.executescript(
          'CREATE TABLE problems(problem_id TEXT,text TEXT);'
          'CREATE TABLE bugs(id INTEGER,problem_id TEXT,buggy_code TEXT,'
          ' fixed_code TEXT,language INTEGER,split INTEGER,active INTEGER);'
          'CREATE TABLE tests(id INTEGER,problem_id TEXT,test_id INTEGER,'
          ' input TEXT,output TEXT,active INTEGER);')
        names=[('p1',0,25),('p2',0,24),('p3',0,24),
               ('p_bad',0,25),('q1',2,24),('q2',2,30)]
        for i,(pid,split,num) in enumerate(names):
            db.execute('INSERT INTO problems VALUES(?,?)',(pid,'repair task'))
            db.execute('INSERT INTO bugs VALUES(?,?,?,?,?,?,?)',
                       (i,pid,'print(0)','SECRET GOLD FIX',5,split,1))
            for j in range(num):
                db.execute('INSERT INTO tests VALUES(?,?,?,?,?,1)',
                           (i*1000+j,pid,j,f'input_{j}',f'output_{j}'))
        db.execute('INSERT INTO tests VALUES(?,?,?,?,?,1)',
                   (90001,'p1',90001,'input_0','output_0'))
        db.execute('INSERT INTO tests VALUES(?,?,?,?,?,1)',
                   (90002,'p_bad',90002,'input_0','CONFLICT'))


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_formal_nested_input_test_bank_and_problem_identity(tmp_path):
    db=tmp_path/'rbr.db';build_db(db)
    root=tmp_path/'frozen'
    manifest=m.freeze(db,root,train_n=2,meta_n=1,final_n=2,seed=1701)
    assert manifest['formal_protocol']=='eesd-nested-T4-vs-T8-20261101'
    assert manifest['calibration_tests']==manifest['audit_tests']==8
    train=read(root/'train_public.jsonl')
    feedback8=read(root/'train_feedback_master_8_do_not_train.jsonl')
    cal=read(root/'train_calibration.jsonl')
    audit=read(root/'train_audit_do_not_train.jsonl')
    meta=read(root/'meta_probes_do_not_train.jsonl')
    final=read(root/'eval_private.jsonl')
    assert len(train)==len(feedback8)==len(cal)==len(audit)==2
    assert len(meta)==1 and len(final)==2
    assert not ({x['problem_id'] for x in train}&
                {x['problem_id'] for x in meta+final})
    assert 'p_bad' not in {x['problem_id'] for x in train+meta}
    for t,f,c,a in zip(train,feedback8,cal,audit):
        x=t['public_tests'];master=f['master_feedback_tests']
        assert master[:4]==x and len(master)==len(c['calibration_tests'])==len(a['audit_tests'])==8
        inp=[t['input'] for group in (master,c['calibration_tests'],a['audit_tests']) for t in group]
        assert len(inp)==len(set(inp))==24
    assert 'SECRET GOLD' not in ''.join(
        file.read_text() for file in root.iterdir() if file.suffix=='.jsonl')
    with pytest.raises(ValueError):m.freeze(db,root,train_n=2,meta_n=1,final_n=2)


def test_formal_source_loader_accepts_eight_calibration_tests(tmp_path):
    import importlib.util
    import sys
    db=tmp_path/'rbr.db';build_db(db)
    root=tmp_path/'frozen'
    m.freeze(db,root,train_n=2,meta_n=1,final_n=2)
    runner=Path(__file__).resolve().parents[1]/'scripts'/'run_learning_credit_five_round.py'
    spec=importlib.util.spec_from_file_location('formal_five_round',runner)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train,ev,by_id,man=module.load_matched_data(root,2,2)
    assert len(train)==2 and len(ev)==2
    assert man['calibration_tests']==8
    assert len(by_id[train[0]['source_id']]['calibration_tests'])==8
