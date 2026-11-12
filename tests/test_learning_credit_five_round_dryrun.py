import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

p=Path(__file__).resolve().parents[1]/'scripts'/'run_learning_credit_five_round.py'
s=importlib.util.spec_from_file_location('five_round_smoke',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def test_five_round_cli_dry_run_no_gpu_and_no_final_label_selection(monkeypatch,tmp_path,capsys):
    data=tmp_path/'frozen';data.mkdir()
    train={'source_id':'train-1','problem_id':'p1','task':'compute sum',
           'buggy_code':'print(0)','public_tests':[
             {'id':f'f{i}','input':'','expected':'0'} for i in range(4)]}
    cal={'source_id':'train-1','problem_id':'p1','calibration_tests':[
             {'id':f'c{i}','input':'','expected':'0'} for i in range(4)]}
    held={'source_id':'train-1','problem_id':'p1','audit_tests':[
             {'id':f'h{i}','input':'','expected':'0'} for i in range(4)]}
    meta={'source_id':'meta-1','problem_id':'p2','task':'compute',
           'buggy_code':'print(0)',
           'evaluation_tests':[{'id':'m0','input':'','expected':'0'}]}
    ev={'source_id':'final-1','problem_id':'p3','task':'compute',
           'buggy_code':'print(0)',
           'evaluation_tests':[{'id':'e0','input':'','expected':'0'}]}
    output={'train_public.jsonl':train,
            'train_calibration.jsonl':cal,
            'train_audit_do_not_train.jsonl':held,
            'meta_probes_do_not_train.jsonl':meta,
            'eval_private.jsonl':ev}
    for fname,row in output.items():
        (data/fname).write_text(json.dumps(row)+'\n')
    manifest={'schema':'eesd-runbugrun-public-private-v1',
         'behavioral_schema':'eesd-runbugrun-behavioral-v1',
         'learning_credit_schema':'eesd-five-round-frozen-data-v1',
         'file_sha256':{name:hashlib.sha256((data/name).read_bytes()).hexdigest() for name in output},
         'public_tests':4,'calibration_tests':4,'audit_tests':4,
         'meta_sources':1,'never_read_fixed_code':True}
    (data/'manifest.json').write_text(json.dumps(manifest))
    init=tmp_path/'init';init.mkdir()
    (init/'adapter_config.json').write_text('{"peft_type":"LORA"}')
    monkeypatch.setattr(sys,'argv',['run_learning_credit_five_round.py',
      '--data-dir',str(data),'--output',str(tmp_path/'unused'), '--arm','shapley',
      '--init-adapter',str(init),'--rollouts','8','--rounds','5',
      '--train-limit','1','--eval-limit','1','--dry-run'])
    m.main()
    plan=json.loads(capsys.readouterr().out)
    assert plan['rounds']==5
    assert plan['feedback_tests']==4
    assert plan['scoring_cost']['exact_coalitions']==16
    assert plan['scoring_cost']['total_scoring_calls']==5*8*1*16
    assert plan['meta_sources']==1
    assert plan['initial_adapter_sha256']
    assert not (tmp_path/'unused').exists()
