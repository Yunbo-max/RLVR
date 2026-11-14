from pathlib import Path
import sys
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eesd.dual_verifier_loss import reverse_kl_from_logits


def test_exact_reverse_kl_matches_privileged():
    target=torch.tensor([[.1,.9],[.3,.7]],dtype=torch.float32)
    plain=torch.tensor([[.8,.2],[.8,.2]],dtype=torch.float32).log()
    student=target.log().detach().clone().requires_grad_(True)
    loss,info=reverse_kl_from_logits(student,plain,target.log(),beta=0.)
    assert float(loss.detach())==pytest.approx(0.,abs=1e-6)
    assert info['sample_tokens']==2
    loss.backward()
    assert torch.allclose(student.grad,torch.zeros_like(student),atol=1e-6)


def test_kl_target_shifts_gradient():
    target=torch.tensor([[.2,.8]],dtype=torch.float32).log()
    base=torch.tensor([[.8,.2]],dtype=torch.float32).log()
    logits=torch.tensor([[0.,0.]],requires_grad=True)
    loss,info=reverse_kl_from_logits(logits,base,target,beta=.02)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[0,1]<0  # SGD raises probability of second token
    assert info['teacher_kl']>0


def test_wrong_shapes_fail():
    with pytest.raises(ValueError):
        reverse_kl_from_logits(torch.zeros((2,3)),torch.zeros((2,4)),torch.zeros((2,3)))
