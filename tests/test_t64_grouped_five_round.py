import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest

PATH=Path(__file__).resolve().parents[1]/'scripts'/'run_t64_grouped_five_round.py'
spec=importlib.util.spec_from_file_location('t64_runner',PATH)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def _frozen(root,n=5):
    root.mkdir()
    public=[];cal=[];audit=[];final=[]
    for i in range(n):
        source=f'train_{i}'
        tests=[{'id':f'p{i}-{j}','input':f'input{j}','expected':f'out{j}'} for j in range(64)]
        public.append({'source_id':source,'problem_id':f'p{i}',
                       'task':'repair this','buggy_code':'print(1)',
                       'public_tests':tests})
        cal.append({'source_id':source,'problem_id':f'p{i}',
                    'calibration_tests':[{'id':f'c{i}-{j}',
                      'input':f'cal{j}','expected':f'out{j}'} for j in range(8)]})
        audit.append({'source_id':source,'problem_id':f'p{i}',
                    'audit_tests':[{'id':f'a{i}-{j}',
                      'input':f'audit{j}','expected':f'out{j}'} for j in range(8)]})
    for i in range(2):
        final.append({'source_id':f'eval_{i}','problem_id':f'final_{i}',
            'task':'repair','buggy_code':'print(1)',
            'evaluation_tests':[{'id':str(j),'input':str(j),'expected':str(j)}
                                for j in range(10)]})
    files={'train_public_master64.jsonl':public,
         'train_calibration.jsonl':cal,
         'train_audit_do_not_train.jsonl':audit,
         'eval_private.jsonl':final,
         'meta_do_not_train.jsonl':[]}
    for filename,rows in files.items():
        (root/filename).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    manifest={'schema':'eesd-rich64-frozen-v1',
      'train_bug_instances':n,'train_distinct_problem_ids':n,
      'calibration_tests':8,'reserved_audit_tests':8,
      'file_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                     for name in files}}
    (root/'manifest.json').write_text(json.dumps(manifest))
    return root


def test_loader_allows_64_public_and_8_calibration_but_no_final_overlap(tmp_path):
    root=_frozen(tmp_path/'bank')
    train,eval_rows,cal,manifest=m._load_data(root,5,2)
    assert len(train)==5 and len(eval_rows)==2
    assert len(train[0]['public_tests'])==64
    assert len(cal[train[0]['source_id']]['calibration_tests'])==8
    assert manifest['schema']=='eesd-rich64-frozen-v1'


def test_same_round_panels_repeat_or_rotate(tmp_path):
    train,_,_,_=m._load_data(_frozen(tmp_path/'bank',n=11),11,2)
    repeat=[m._panel(train,k,seed=1701,mode='repeat') for k in range(5)]
    rotated=[m._panel(train,k,seed=1701,mode='rotate') for k in range(5)]
    assert all(len(p)==11 for p in repeat)
    assert sum(map(len,rotated))==11
    assert len(set.union(*rotated))==11
    assert all(not a&b for i,a in enumerate(rotated) for b in rotated[i+1:])


@pytest.mark.parametrize('exposed',[4,8,16,32,64])
def test_dry_run_tracks_64_real_executions_and_four_group_players(
    tmp_path,monkeypatch,capsys,exposed):
    root=_frozen(tmp_path/'bank')
    init=tmp_path/'initial';init.mkdir()
    (init/'adapter_config.json').write_text('{"peft_type":"LORA"}')
    monkeypatch.setattr(sys,'argv',[
         'run_t64_grouped_five_round.py','--data-dir',str(root),
         '--output',str(tmp_path/'do_not_create'),
         '--init-adapter',str(init),'--arm','shapley',
         '--train-limit','5','--eval-limit','2',
         '--rollouts','8','--exposed-feedback',str(exposed),
         '--dry-run'])
    m.main()
    plan=json.loads(capsys.readouterr().out)
    assert plan['executed_public_feedback_tests_per_candidate']==64
    assert plan['exposed_feedback']==exposed
    assert plan['group_shapley_players']==4
    assert plan['exact_group_coalitions']==16
    assert plan['rounds']==5
    assert plan['panel_sizes']==[5]*5
    assert not (tmp_path/'do_not_create').exists()


def test_full_public_test_outcomes_not_exposed_to_t4_grouping(monkeypatch):
    # If T=4 only, group assignment sees only first four test outcomes.
    tests=[{'id':str(j),'input':str(j),'expected':str(j)} for j in range(64)]
    transcript=[
      [{'id':str(j),'outcome':'PASS','input':str(j),'expected':str(j),
        'actual':str(j),'stderr':''} for j in range(64)]
      for _ in range(8)]
    four=m.form_groups(tests[:4],[v[:4] for v in transcript],groups=4)
    assert sum(len(group) for group in four.members)==4
    assert all(j<4 for group in four.members for j in group)
