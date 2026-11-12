import json
import sqlite3
from pathlib import Path
import pytest
import importlib.util

P=Path(__file__).resolve().parents[1]/'scripts'/'prepare_recursive_runbugrun.py'
s=importlib.util.spec_from_file_location('prepare_recursive_runbugrun',P)
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)


def _db(path):
    with sqlite3.connect(path) as db:
        db.executescript('''CREATE TABLE bugs(id INTEGER,problem_id TEXT,buggy_code TEXT,
            fixed_code TEXT, language INTEGER, split INTEGER, active INTEGER);
            CREATE TABLE problems(problem_id TEXT,text TEXT);
            CREATE TABLE tests(id INTEGER,problem_id TEXT, test_id INTEGER,
                input TEXT,output TEXT,active INTEGER);''')
        for n,(pid,split) in enumerate([('p0',0),('p1',0),('p2',2),('p3',2)]):
            db.execute('INSERT INTO bugs VALUES(?,?,?,?,?,?,?)',
                       (n,pid,'print(0)','SECRET GOLD NEVER EXPORT',5,split,1))
            db.execute('INSERT INTO problems VALUES(?,?)',(pid,'description'))
            for j in range(10):
                db.execute('INSERT INTO tests VALUES(?,?,?,?,?,1)',
                           (100*n+j,pid,j,str(j),str(j)))
        db.commit()


def test_extract_does_not_read_gold_and_has_disjoint_sources(tmp_path):
    p=tmp_path/'data.db';_db(p)
    manifest=m.extract(p,tmp_path/'export',train_sources=2,eval_sources=2,seed=1701)
    root=tmp_path/'export'
    train=[json.loads(x) for x in (root/'train_public.jsonl').read_text().splitlines()]
    test=[json.loads(x) for x in (root/'eval_private.jsonl').read_text().splitlines()]
    assert len(train)==len(test)==2 and manifest['never_read_fixed_code']
    assert not ({r['problem_id'] for r in train}&{r['problem_id'] for r in test})
    assert all(len(r['public_tests'])==4 for r in train)
    assert all(len(r['evaluation_tests'])==10 for r in test)
    assert 'SECRET' not in '\n'.join((root/x).read_text() for x in ('train_public.jsonl','eval_private.jsonl'))
    with pytest.raises(ValueError):m.extract(p,root,train_sources=2,eval_sources=2,seed=1701)


def test_fail_when_test_split_unavailable(tmp_path):
    p=tmp_path/'data.db';_db(p)
    with pytest.raises(ValueError,match='eligible sources'):
        m.extract(p,tmp_path/'out',train_sources=2,eval_sources=3,seed=1701)
