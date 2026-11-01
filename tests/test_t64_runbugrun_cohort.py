import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

path=Path(__file__).resolve().parents[1]/'scripts'/'prepare_t64_runbugrun_cohort.py'
spec=importlib.util.spec_from_file_location('t64_cohort',path)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def build_db(path):
    with sqlite3.connect(path) as db:
        db.executescript(
            'CREATE TABLE problems(problem_id TEXT,text TEXT);'
            'CREATE TABLE bugs(id INTEGER,problem_id TEXT,buggy_code TEXT,'
            'fixed_code TEXT,language INTEGER,split INTEGER,active INTEGER);'
            'CREATE TABLE tests(id INTEGER,problem_id TEXT,test_id INTEGER,'
            'input TEXT,output TEXT,origin INTEGER,active INTEGER);'
        )
        for i,(pid,split,n) in enumerate([
           ('train1',0,88),('train2',0,81),('train3',0,80),
           ('bad_test',0,80),('eval1',2,81),('eval2',2,82)]):
            db.execute('INSERT INTO problems VALUES(?,?)',(pid,'repair this program'))
            for j in range(3):
                db.execute('INSERT INTO bugs VALUES(?,?,?,?,?,?,?)',
                    (i*100+j,pid,'print(0)','GOLD_CORRECTION_SECRET',5,split,1))
            for j in range(n):
                db.execute('INSERT INTO tests VALUES(?,?,?,?,?,?,1)',
                    (i*1000+j,pid,j,f'input{j}',f'expected{j}',1))
        db.execute('INSERT INTO tests VALUES(?,?,?,?,?,?,1)',
            (99001,'train1',9901,'input0','expected0',1))
        db.execute('INSERT INTO tests VALUES(?,?,?,?,?,?,1)',
            (99002,'bad_test',9902,'input0','different expected',1))
        db.commit()


def read(file):
    return [json.loads(s) for s in file.read_text().splitlines()]


def test_nested_64_public_plus_8cal_8audit_and_clustered_train(tmp_path):
    db=tmp_path/'runbugrun.db';build_db(db)
    out=tmp_path/'cohort'
    manifest=mod.freeze(db,out,train_bugs=4,meta_problems=1,
                        eval_problems=2,max_per_problem=3,seed=1701)
    assert manifest['train_bug_instances']==4
    assert manifest['train_distinct_problem_ids']>=1
    assert manifest['meta_distinct_problem_ids']==1
    assert manifest['eval_distinct_problem_ids']==2
    train=read(out/'train_public_master64.jsonl')
    cal=read(out/'train_calibration.jsonl')
    audit=read(out/'train_audit_do_not_train.jsonl')
    meta=read(out/'meta_do_not_train.jsonl')
    fin=read(out/'eval_private.jsonl')
    assert len(train)==len(cal)==len(audit)==4
    assert len(meta)==1 and len(fin)==2
    assert not ({r['problem_id'] for r in train}&{r['problem_id'] for r in meta+fin})
    for row,c,a in zip(train,cal,audit):
        public=row['public_tests']; calibration=c['calibration_tests']; reserved=a['audit_tests']
        assert len(public)==64 and len(calibration)==len(reserved)==8
        inputs=[x['input'] for x in public+calibration+reserved]
        assert len(inputs)==len(set(inputs))==80
        assert row['public_tests'][:4]==public[:4]
    alltext=''.join(p.read_text() for p in out.iterdir())
    assert 'GOLD_CORRECTION_SECRET' not in alltext
    assert 'bad_test' not in {x['problem_id'] for x in train+meta}


def test_insufficient_unique_80_input_pool_refused(tmp_path):
    db=tmp_path/'runbugrun.db';build_db(db)
    with pytest.raises(ValueError,match='insufficient eligible BUG INSTANCES'):
        mod.freeze(db,tmp_path/'nope',train_bugs=100,meta_problems=1,
                    eval_problems=2,max_per_problem=3)


def test_input_conflict_eliminated_before_shapley(tmp_path):
    db=tmp_path/'runbugrun.db';build_db(db)
    with sqlite3.connect(f'file:{db}?mode=ro',uri=True) as con:
        assert mod.get_unique_input_bank(con,'bad_test',seed=2) is None
        assert len(mod.get_unique_input_bank(con,'train3',seed=2))>=80


def test_one_bug_per_problem_budget_is_not_eight_thousand_independent_tasks(tmp_path):
    db=tmp_path/'runbugrun.db';build_db(db)
    with pytest.raises(ValueError,match='insufficient eligible BUG INSTANCES'):
        mod.freeze(db,tmp_path/'too_much',train_bugs=4,
                   meta_problems=1,eval_problems=2,max_per_problem=1)
