"""Run these in the independent sdpo-official environment (actual upstream code)."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec('verl') is None,
                                reason='requires pinned official SDPO environment')


def test_actual_upstream_reverse_kl_tail_loss_and_gradient():
    import torch
    from verl.trainer.ppo.core_algos import compute_self_distillation_loss
    cfg = SimpleNamespace(full_logit_distillation=True, distillation_topk=2,
                          distillation_add_tail=True, alpha=1., is_clip=2.)
    z = torch.tensor([[[1., 2., -1., .5]]], requires_grad=True)
    student = z.log_softmax(-1)
    teacher = torch.tensor([[[.8, -.2, 1., .3]]]).log_softmax(-1)
    idx = student.topk(2, dim=-1).indices
    s, t = student.gather(-1, idx), teacher.gather(-1, idx)
    chosen = student[:, :, 0]
    loss, _ = compute_self_distillation_loss(
        chosen, teacher[:, :, 0], torch.ones(1, 1), cfg,
        old_log_probs=chosen.detach(), student_topk_log_probs=s, teacher_topk_log_probs=t)
    sp = torch.cat([s.exp(), 1-s.exp().sum(-1, keepdim=True)], -1)
    tp = torch.cat([t.exp(), 1-t.exp().sum(-1, keepdim=True)], -1)
    expected = (sp * (sp.log()-tp.log())).sum()
    torch.testing.assert_close(loss, expected)
    g1 = torch.autograd.grad(loss, z, retain_graph=True)[0]
    g2 = torch.autograd.grad(expected, z)[0]
    torch.testing.assert_close(g1, g2)
    assert torch.isfinite(g1).all() and g1.abs().sum() > 0


def test_actual_upstream_solution_selection_excludes_self_and_other_problem():
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    dummy = SimpleNamespace(_remove_thinking_trace=RayPPOTrainer._remove_thinking_trace)
    uids = ['a', 'a', 'b']
    successes = {'a': [0], 'b': [2]}
    texts = ['<think>reason</think>answer A', 'bad', 'answer B']
    f = RayPPOTrainer._get_solution
    assert f(dummy, 0, successes, uids, texts, True, True) is None
    assert f(dummy, 1, successes, uids, texts, True, True) == 'answer A'
    assert f(dummy, 2, successes, uids, texts, True, True) is None


def test_actual_upstream_masks_examples_without_demonstration_or_feedback():
    import torch
    from verl.trainer.ppo.core_algos import compute_self_distillation_loss
    cfg = SimpleNamespace(full_logit_distillation=True, distillation_topk=None,
                          distillation_add_tail=True, alpha=1., is_clip=2.)
    z = torch.randn(2, 3, 5, requires_grad=True)
    s, t = z.log_softmax(-1), torch.randn(2, 3, 5).log_softmax(-1)
    loss, _ = compute_self_distillation_loss(
        s[:, :, 0], t[:, :, 0], torch.ones(2, 3), cfg,
        old_log_probs=s[:, :, 0].detach(), student_all_log_probs=s,
        teacher_all_log_probs=t, self_distillation_mask=torch.tensor([1., 0.]))
    loss.backward()
    assert z.grad[0].abs().sum() > 0
    assert z.grad[1].abs().sum() == 0


def test_actual_upstream_ema_updates_only_after_called_and_matches_rate():
    import torch
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from omegaconf import OmegaConf
    teacher, student = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    for p in teacher.parameters():
        p.data.zero_()
    for p in student.parameters():
        p.data.fill_(2.)
    dummy = SimpleNamespace(teacher_module=teacher, actor_module=student,
                            config=OmegaConf.create({'policy_loss': {'loss_mode': 'sdpo'},
                                'self_distillation': {'teacher_regularization':'ema', 'teacher_update_rate':.01}}))
    DataParallelPPOActor._update_teacher(dummy)
    for p in teacher.parameters():
        torch.testing.assert_close(p, torch.full_like(p, .02))
