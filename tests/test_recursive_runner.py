import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
torch=pytest.importorskip("torch")

FILE=Path(__file__).resolve().parents[1]/'scripts'/'run_recursive_shapley_sdpo.py'
spec=importlib.util.spec_from_file_location('run_recursive_shapley_sdpo',FILE)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class TinyTokenizer:
    pad_token_id=0
    eos_token_id=0
    def apply_chat_template(self,messages,*,tokenize,add_generation_prompt):
        return ' '.join(item['content'] for item in messages)
    def __call__(self,text,*,add_special_tokens):
        return {'input_ids':[2+(ord(c)%17) for c in text]}


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=torch.nn.Embedding(24,6)
        self.proj=torch.nn.Linear(6,24)
        self.device=torch.device('cpu')
    def forward(self,input_ids,**kwargs):
        return SimpleNamespace(logits=self.proj(self.emb(input_ids)))


def _row():
    return {'source_id':'r1','problem_id':'p1','task':'implement sum',
            'buggy_code':'print(0)',
            'public_tests':[{'id':str(i),'input':str(i),'expected':str(i)} for i in range(4)]}


def _feedback():
    return [{'id':str(i),'input':str(i),'expected':str(i),
             'actual':'0','stderr':'','outcome':'WRONG_OUTPUT'} for i in range(4)]


def test_teacher_scores_use_exact_response_positions():
    torch.manual_seed(6)
    model=ToyModel();tokenizer=TinyTokenizer()
    row=_row();response=[4,5,6]
    scores,logp=m._logprobs_for_fixed_response(model,tokenizer,row,None,response,
                                       return_distribution=True)
    np.testing.assert_allclose(scores,logp.gather(1,torch.tensor(response)[:,None]).squeeze(1).numpy())
    assert len(scores)==3 and logp.shape==(3,24)


def test_sdpo_and_shapley_both_produce_dense_token_targets():
    torch.manual_seed(7)
    model=ToyModel();tokenizer=TinyTokenizer();row=_row();feedback=_feedback()
    old,full,w,info=m._teacher_for_rollout(model,tokenizer,row,[4,5],feedback,'sdpo')
    assert len(w)==2 and np.all(w==1)
    old,full,w,info=m._teacher_for_rollout(model,tokenizer,row,[4,5],feedback,'shapley')
    assert len(w)==2 and info['negative_shapley_fraction'] is not None
    assert old.shape==full.shape==(2,24)


def test_optimizer_gradient_exists_on_distillation_with_frozen_teacher():
    torch.manual_seed(8)
    student=ToyModel();teacher=ToyModel()
    tokenizer=TinyTokenizer();row=_row();ids=[7,8]
    old,fb,w,_=m._teacher_for_rollout(teacher,tokenizer,row,ids,_feedback(),'sdpo')
    context=m.prompt_ids(tokenizer,row)
    full=torch.tensor([context+ids])
    loss,metrics=m._distill_one(student,old,fb,np.ones(2),full,len(context),beta=.02)
    loss.backward()
    assert np.isfinite(metrics['distillation_kl'])
    assert student.proj.weight.grad is not None
    assert not any(p.grad is not None for p in teacher.parameters())


def test_response_token_terminates_on_eos_without_dropping_first_token():
    assert m.response_token_ids([4,5,0,0],0,0)==[4,5,0]
    assert m.response_token_ids([4,5,7,7],0,0)==[4,5,7,7]


def test_prompt_never_contains_gold_code():
    row=_row();text=str(m.prompt_messages(row,_feedback()))
    assert 'buggy_code' not in text and 'fixed_code' not in text
    assert 'WRONG_OUTPUT' in text
