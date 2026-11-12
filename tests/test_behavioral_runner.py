import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

# The real existing runner is imported in production. Stub just its adapter
# interface so we can test this new layer without downloading Qwen weights.
stub=types.ModuleType('run_recursive_shapley_sdpo')
stub.MODEL_REVISION='c03e6d358207e414f1eca0bb1891e29f1db0e242'
stub.prompt_ids=lambda tok,row,fb=None:[2,3,4]
stub._logprobs_for_fixed_response=lambda model,tok,row,fb,tokens,return_distribution: (
    np.array([.25 if fb else .10]*len(tokens)),None)
stub._extract_python=lambda raw:raw
previous=sys.modules.get('run_recursive_shapley_sdpo')
sys.modules['run_recursive_shapley_sdpo']=stub
P=Path(__file__).resolve().parents[1]/'scripts'/'run_behavioral_shapley.py'
spec=importlib.util.spec_from_file_location('run_behavioral_shapley',P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if previous is None: sys.modules.pop('run_recursive_shapley_sdpo',None)
else: sys.modules['run_recursive_shapley_sdpo']=previous


def test_clean_code_tokenization_drops_entire_markdown_prefix():
    class FakeTokenizer:
        def __call__(self,text,add_special_tokens):
            assert not text.startswith(chr(96)*3)
            return {'input_ids':list(range(len(text)))}
    assert len(m._normalized_ids(FakeTokenizer(),'print(1)',100))>0
    assert m._normalized_ids(FakeTokenizer(),' ',100)==[]
    assert m._normalized_ids(FakeTokenizer(),'print(1)',1)==[]


def test_subset_scoring_exact_vs_cheap(monkeypatch):
    calls=[]
    def fake_score(model,tok,row,fb,tokens,return_distribution):
        calls.append(fb)
        return np.array([len(fb) if fb else 0.]*len(tokens)),None
    monkeypatch.setattr(m.legacy,'_logprobs_for_fixed_response',fake_score)
    feedback=[{'id':str(j),'outcome':'PASS'} for j in range(4)]
    values=m.score_candidate(None,None,{'task':'a'},[1,2],feedback,
          arm='shapley_select',max_prompt_tokens=100)
    assert len(calls)==16
    assert values[0,0]==0 and values[15,0]==4
    calls.clear()
    values=m.score_candidate(None,None,{'task':'a'},[1,2],feedback,
          arm='audit_gate',max_prompt_tokens=100)
    assert len(calls)==2 and values[15,0]==4


def test_teacher_context_never_includes_audit_expected(monkeypatch):
    captured=[]
    def fake_score(model,tok,row,fb,tokens,return_distribution):
        captured.append(fb)
        return np.array([0.]*len(tokens)),None
    monkeypatch.setattr(m.legacy,'_logprobs_for_fixed_response',fake_score)
    feedback=[{'id':str(j),'expected':'PUBLIC'} for j in range(4)]
    m.score_candidate(None,None,{'task':'a'},[1,2],feedback,arm='shapley_select',max_prompt_tokens=100)
    assert all('AUDIT_SECRET' not in str(fb) for fb in captured)


def test_soft_sandbox_fallback_not_used(monkeypatch):
    from eesd import strict_execution
    monkeypatch.setattr(strict_execution.shutil,'which',lambda _:None)
    with pytest.raises(RuntimeError,match='refuse untrusted code'):
        strict_execution.run_test('print(1)',{'input':'','expected':'1'})
    assert strict_execution.audit_vector('',[{'id':'x','input':'','expected':'a'}])==[0]


def test_weak_token_credit_reduces_gradient_not_renormalized():
    torch=pytest.importorskip('torch')
    import torch.nn.functional as F
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.e=torch.nn.Embedding(20,6);self.o=torch.nn.Linear(6,20)
        def forward(self,input_ids,use_cache=False):
            return SimpleNamespace(logits=self.o(self.e(input_ids)))
    student=TinyModel()
    context=[2,3]; response=[4,5]
    ids=torch.tensor([context+response])
    out=student(ids).logits[0,1:-1]
    altered=out.detach().clone(); altered[:,0] += 1.0
    logp=F.log_softmax(altered,dim=-1)
    old=F.log_softmax(out.detach(),dim=-1)
    loss0,stat0=m.distill_token_kl(student,old,logp,[0,0],ids,len(context),beta=.02)
    assert loss0==0
    student.zero_grad(set_to_none=True)
    loss1,stat1=m.distill_token_kl(student,old,logp,[.01,.01],ids,len(context),beta=.02)
    loss1.backward()
    grad1=sum(float(p.grad.norm()) for p in student.parameters() if p.grad is not None)
    assert grad1>0 and stat1['mean_weight']==pytest.approx(.01)
    student.zero_grad(set_to_none=True)
    loss2,_=m.distill_token_kl(student,old,logp,[1,1],ids,len(context),beta=.02)
    loss2.backward()
    grad2=sum(float(p.grad.norm()) for p in student.parameters() if p.grad is not None)
    assert grad2/grad1==pytest.approx(100,rel=.001)


def test_training_calibration_and_reserved_audit_are_disjoint(monkeypatch,tmp_path):
    train=[{'source_id':'s1','problem_id':'p1','task':'t','buggy_code':'print(0)',
            'public_tests':[{'id':str(j)} for j in range(4)]}]
    cal=[{'source_id':'s1','problem_id':'p1','calibration_tests':[{'id':str(j)} for j in range(4,8)]}]
    held=[{'source_id':'s1','problem_id':'p1','audit_tests':[{'id':str(j)} for j in range(8,12)]}]
    manifest={'behavioral_schema':'eesd-runbugrun-behavioral-v1','calibration_tests':4,'audit_tests':4}
    monkeypatch.setattr(m.legacy,'read_dataset',lambda *a:(train,[{'source_id':'e1'}],manifest),raising=False)
    monkeypatch.setattr(m.legacy,'read_records',lambda path,keys:cal if path.name=='train_calibration.jsonl' else held,raising=False)
    _,_,by_id,_=m.load_matched_data(tmp_path,1,1)
    assert by_id['s1']['calibration_tests'][0]['id']=='4'
    held[0]['audit_tests'][0]['id']='4'
    with pytest.raises(ValueError,match='contamination'):
        m.load_matched_data(tmp_path,1,1)
