import importlib.util
import sqlite3
from pathlib import Path
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'audit_runbugrun_feedback_feasibility.py'
spec=importlib.util.spec_from_file_location('rbr_feedback_feasibility',SCRIPT)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def _make_db(p):
    with sqlite3.connect(p) as db:
        db.executescript(
            'CREATE TABLE bugs(id INTEGER,problem_id TEXT,language INTEGER,'
            ' split INTEGER,active INTEGER,token_count INTEGER,fixed_code TEXT);'
            'CREATE TABLE tests(id INTEGER,problem_id TEXT,input TEXT,'
            ' output TEXT,active INTEGER);')
        for i,(pid,split,num) in enumerate(
                [('trA',0,26),('trB',0,12),('teA',2,24),
                 ('teB',2,22),('vaA',1,20)]):
            for b in range(3):
                db.execute('INSERT INTO bugs VALUES(?,?,?,?,?,?,?)',
                           (10*i+b,pid,5,split,1,50,'NEVER_READ_GOLD'))
            for j in range(num):
                db.execute('INSERT INTO tests VALUES(?,?,?,?,1)',
                           (100*i+j,pid,f'in{j}',f'out{j}'))
        # Repeated identical test ID different and same I/O: not independent.
        db.execute('INSERT INTO tests VALUES(?,?,?,?,1)',(9999,'trA','in0','out0'))
        # Conflicting expected outputs for identical input must be flagged.
        db.execute('INSERT INTO tests VALUES(?,?,?,?,1)',(9998,'teB','in0','wrong'))
        # Inactive/other-language examples do not qualify.
        db.execute('INSERT INTO tests VALUES(?,?,?,?,0)',(9997,'trA','inactive','x'))
        db.execute('INSERT INTO bugs VALUES(?,?,?,?,?,?,?)',
                   (19999,'trA',1,0,1,100,'NEVER_READ_GOLD'))


def test_formal_eligibility_is_based_on_distinct_inputs_and_reports_conflicts(tmp_path):
    db=tmp_path/'rbr.db';_make_db(db)
    report=m.audit(db)
    train=report['split_stats']['train']
    test=report['split_stats']['test']
    assert train['python_distinct_problem_ids_with_tests']==2
    assert train['python_bug_instances_with_tests']==6
    assert train['eligible_without_conflicting_inputs']['24']['distinct_problem_ids']==1
    assert train['eligible_without_conflicting_inputs']['12']['distinct_problem_ids']==2
    assert train['unique_input_tests_per_problem']['max']==26
    assert train['duplicate_test_rows_per_problem']['max']==1
    assert test['problems_with_conflicting_expected_output_for_same_input']==1
    assert test['eligible_without_conflicting_inputs']['24']['distinct_problem_ids']==1
    assert 'db_sha256' in report


def test_source_token_filter_does_not_look_at_gold(tmp_path):
    db=tmp_path/'rbr.db';_make_db(db)
    report=m.audit(db,min_tokens=51,max_tokens=200)
    assert report['split_stats']['train']['python_distinct_problem_ids_with_tests']==0
    with pytest.raises(ValueError):m.audit(db,min_tokens=90,max_tokens=40)
    with pytest.raises(FileNotFoundError):m.audit(tmp_path/'missing.db')


def test_quantile_empty_population_is_explicit():
    assert m.quantiles([])['median'] is None
    assert m.quantiles([2,4,8])['median']==4
    assert m.quantiles([1])['p90']==1
