"""Exact full-vocabulary reverse KL for both recursive OPSD roles.

The privileged teacher distributions are detached. Unlike SFT, this matches
all vocabulary probabilities at each ON-POLICY sampled response prefix.
"""
from __future__ import annotations


def reverse_kl_from_logits(student_logits, plain_teacher_logq, privileged_teacher_logq,
                           *, beta:float=0.02):
    import torch
    import torch.nn.functional as F
    if not 0 <= beta < float('inf'):
        raise ValueError('invalid KL anchor coefficient')
    if (student_logits.ndim!=2 or student_logits.shape!=plain_teacher_logq.shape or
        student_logits.shape!=privileged_teacher_logq.shape or not len(student_logits)):
        raise ValueError('teacher/student [tokens,vocab] shape mismatch')
    if not torch.isfinite(student_logits).all():
        raise ValueError('nonfinite student logits')
    if not torch.isfinite(plain_teacher_logq).all() or not torch.isfinite(privileged_teacher_logq).all():
        raise ValueError('nonfinite teacher log-probabilities')
    logp=F.log_softmax(student_logits.float(),dim=-1)
    p=logp.exp()
    priv=privileged_teacher_logq.detach().to(logp.device).float()
    plain=plain_teacher_logq.detach().to(logp.device).float()
    token_kl=(p*(logp-priv)).sum(dim=-1)
    anchor=(p*(logp-plain)).sum(dim=-1)
    loss=(token_kl+beta*anchor).mean()
    return loss,{'teacher_kl':float(token_kl.mean().detach().cpu()),
                 'anchor_kl':float(anchor.mean().detach().cpu()),
                 'sample_tokens':len(student_logits)}
