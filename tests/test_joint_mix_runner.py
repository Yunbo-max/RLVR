"""CPU integration contract for the new runner without HF or a GPU.

These fakes test information-flow and tensor plumbing, NOT model quality.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def _load(monkeypatch):
    fake=types.ModuleType('run_t64_grouped_five_round')
    monkeypatch.setitem(sys.modules,'run_t64_grouped_five_round',fake)
    strict=types.ModuleType('eesd.strict_execution')
    strict.run_test=lambda *a,**k:None
    strict.audit_vector=lambda *a,**k:None
    monkeypatch.setitem(sys.modules,'eesd.strict_execution',strict)
    spec=importlib.util.spec_from_file_location('run_joint_mix',ROOT/'scripts/run_joint_mix.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(monkeypatch,module):
    tokens=[([1,2], 'good'),([1,2], 'bad')]
    feedback=[[{'id':str(j),'outcome':'WRONG_OUTPUT'} for j in range(64)] for _ in tokens]
    monkeypatch.setattr(module,'_candidates',lambda *args: tokens)
    monkeypatch.setattr(module,'_feedback',lambda *args: feedback)
    # Every singleton is uninformative, but the exact joint pair is useful.
    pair=module.deterministic_pair(64,seed=1701,round_index=0,source_id='source')
    pair_mask=sum(1<<j for j in pair)
    def scoring(_teacher,_tk,_row,_tokens,_feedback,mask,_args,full=False):
        if full:raise AssertionError('unrequested full-vocabulary pass')
        idx=0 if _feedback is feedback[0] else 1
        delta=(.2 if idx==0 else -.2) if mask==pair_mask else 0.
        return np.array([-.7+delta,-.7+delta]),None
    monkeypatch.setattr(module,'_teacher_score',scoring)
    monkeypatch.setattr(module,'audit_vector',lambda code,tests:
                        [1 if code=='good' else 0 for _ in tests])
    return pair


def test_frozen_uses_exactly_four_teacher_contexts_and_never_chooses_from_audit(monkeypatch):
    module=_load(monkeypatch)
    pair=_fixture(monkeypatch,module)
    row={'source_id':'source','problem_id':'problem','public_tests':[{'id':str(j)} for j in range(64)]}
    args=types.SimpleNamespace(seed=1701,mix_grid=5,uncertainty=.5,probe_tokens=0)
    a=module.frozen_source(None,None,row,[1]*8,[1]*8,args)
    assert a['status']=='completed'
    assert a['feedback_indices']==list(pair)
    assert a['teacher_scoring_conditions']==8
    assert a['a8_joint_minus_mix']>0
    assert a['alpha_q0_qa_qb'][0]==1.


def test_full_distribution_arithmetic_mixing_normalizes(monkeypatch):
    module=_load(monkeypatch)
    import torch
    probs=[torch.tensor([[.2,.8]]),torch.tensor([[.9,.1]]),torch.tensor([[.4,.6]])]
    mixed=module._torch_mix([p.log() for p in probs],(.2,.5,.3)).exp()
    expected=.2*probs[0]+.5*probs[1]+.3*probs[2]
    assert torch.allclose(mixed,expected)
    assert torch.allclose(mixed.sum(-1),torch.ones(1))
    additive=module._additive_logq(*(p.log() for p in probs))
    assert torch.allclose(additive.exp().sum(-1),torch.ones(1),atol=1e-6)
