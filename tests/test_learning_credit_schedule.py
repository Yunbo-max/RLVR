import importlib.util
import json
from pathlib import Path
import pytest

p=Path(__file__).resolve().parents[1]/'scripts'/'make_learning_credit_schedule.py'
spec=importlib.util.spec_from_file_location('learning_credit_schedule',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def _fake_run(root):
    root.mkdir(parents=True,exist_ok=True)
    plan={'schema':'eesd-learning-credit-five-v1','arm':'shapley',
          'rounds':5,'seed':1701,'dataset_manifest_sha256':'datasum',
          'initial_adapter_sha256':'adaptersum'}
    (root/'plan.json').write_text(json.dumps(plan))
    sources=['s0','s1','s2']
    for r in range(5):
        rd=root/f'round_{r}';rd.mkdir()
        data=[{'source_id':source,'step':i<2 if r%2==0 else i==0,
               'selected_mask':(1<<i) if (i<2 if r%2==0 else i==0) else 0}
             for i,source in enumerate(sources)]
        (rd/'training.jsonl').write_text('\n'.join(json.dumps(x) for x in data)+'\n')


def test_training_only_schedule_has_actual_step_count(tmp_path):
    _fake_run(tmp_path)
    x=m.make_schedule(tmp_path)
    assert len(x['training_rounds'])==5
    assert [r['shapley_actual_optimizer_steps'] for r in x['training_rounds']]==[2,1,2,1,2]
    assert all(len(r['random_update_sources'])==r['shapley_actual_optimizer_steps'] for r in x['training_rounds'])
    assert all(set(r['accepted_masks'])==set(x['source_ids']) for r in x['training_rounds'])


def test_rejects_unsafe_false_training_provenance(tmp_path):
    _fake_run(tmp_path)
    p=tmp_path/'round_0/training.jsonl'
    data=[json.loads(x) for x in p.read_text().splitlines()]
    data[0]['step']=True;data[0]['selected_mask']=0
    p.write_text('\n'.join(json.dumps(x) for x in data))
    with pytest.raises(ValueError,match='zero feedback'):
        m.make_schedule(tmp_path)


def test_seeds_must_match_frozen_experiment(tmp_path):
    _fake_run(tmp_path)
    with pytest.raises(ValueError):m.make_schedule(tmp_path,seed=1702)
