import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

P=Path(__file__).resolve().parents[1]/'scripts'/'run_learning_credit_five_round.py'
spec=importlib.util.spec_from_file_location('run_learning_credit_five_round',P)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def _schedule(path):
    d={'schema':'eesd-learning-credit-schedule-v1','seed':1701,'rounds':5,
      'data_manifest_sha256':'frozen','initial_adapter_sha256':'initial',
      'source_ids':['s1','s2'],
      'training_rounds':[{'round':r,'accepted_masks':{'s1':1,'s2':0},
                    'shapley_actual_optimizer_steps':1,
                    'random_update_sources':['s2']} for r in range(5)]}
    path.write_text(json.dumps(d))
    return d


def test_random_controls_require_frozen_matching_schedule(tmp_path):
    path=tmp_path/'frozen.json';_schedule(path)
    args=SimpleNamespace(arm='random_subset',seed=1701,rounds=5)
    assert len(m._load_schedule(path,args,'frozen','initial',['s1','s2'])['training_rounds'])==5
    with pytest.raises(ValueError,match='mismatches'):
        m._load_schedule(path,args,'frozen','different',['s1','s2'])
    with pytest.raises(ValueError,match='mismatches'):
        m._load_schedule(path,args,'frozen','initial',['s1','unlisted'])


def test_nonrandom_arms_cannot_consume_schedule(tmp_path):
    path=tmp_path/'frozen.json';_schedule(path)
    args=SimpleNamespace(arm='shapley',seed=1701,rounds=5)
    assert m._load_schedule(None,args,'frozen','initial',['s1','s2']) is None
    with pytest.raises(ValueError,match='only for the two frozen random'):
        m._load_schedule(path,args,'frozen','initial',['s1','s2'])


def test_scheduler_skips_reference_rejected_sources_without_executing(monkeypatch):
    pytest.importorskip('torch')
    def not_called(*a,**kw):
        raise AssertionError('must not execute unselected training sample')
    monkeypatch.setattr(m.legacy,'_sample_rollouts',not_called)
    args=SimpleNamespace(arm='random_subset')
    ref={'accepted_masks':{'s1':0},'random_update_sources':[]}
    rec=m._train_source(None,None,None,None,{'source_id':'s1'},None,args,0,ref)
    assert rec['step'] is False and rec['skip']=='reference_abstained'


def test_count_matched_random_control_skips_unselected_without_execution(monkeypatch):
    pytest.importorskip('torch')
    monkeypatch.setattr(m.legacy,'_sample_rollouts',lambda *args: (_ for _ in ()).throw(
             AssertionError('unscheduled candidate evaluated')))
    args=SimpleNamespace(arm='matched_random_updates')
    rec=m._train_source(None,None,None,None,{'source_id':'s1'},None,args,0,
                     {'accepted_masks':{'s1':1},'random_update_sources':['s2']})
    assert rec['step'] is False
