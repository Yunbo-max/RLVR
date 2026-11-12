import importlib.util
from pathlib import Path
import pytest

PATH = Path(__file__).resolve().parents[1]/'scripts/run_sdpo_official.py'
spec = importlib.util.spec_from_file_location('official_launcher', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_matched_profile_does_not_silently_use_historical_loss_or_private_feedback(tmp_path):
    options = m.build_overrides(data_dir=tmp_path/'data', output=tmp_path/'run',
                                model_path=tmp_path/'model', init_adapter=tmp_path/'adapter',
                                gpus=1, profile='rich64-pilot', root=PATH.parents[1])
    opts = dict(x.lstrip('+').split('=', 1) for x in options)
    assert opts['actor_rollout_ref.actor.policy_loss.loss_mode'] == 'sdpo'
    assert opts['actor_rollout_ref.actor.self_distillation.alpha'] == '1.0'
    assert opts['actor_rollout_ref.actor.self_distillation.distillation_topk'] == '20'
    assert opts['actor_rollout_ref.actor.self_distillation.distillation_add_tail'] == 'true'
    assert opts['actor_rollout_ref.actor.self_distillation.teacher_regularization'] == 'ema'
    assert opts['actor_rollout_ref.actor.self_distillation.teacher_update_rate'] == '0.01'
    assert opts['actor_rollout_ref.actor.use_kl_loss'] == 'false'
    assert opts['actor_rollout_ref.rollout.val_kwargs.do_sample'] == 'false'
    assert opts['actor_rollout_ref.rollout.val_kwargs.n'] == '1'
    assert opts['trainer.resume_mode'] == 'disable'
    assert opts['data.filter_overlong_prompts'] == 'false'


def test_upstream_hyperparameters_are_a_separate_profile(tmp_path):
    args = dict(data_dir=tmp_path, output=tmp_path/'out', model_path=tmp_path,
                init_adapter=tmp_path, gpus=1, root=PATH.parents[1])
    opts = dict(x.lstrip('+').split('=', 1) for x in m.build_overrides(
        **args, profile='upstream-hparams'))
    assert opts['actor_rollout_ref.actor.optim.lr'] == '1e-06'
    assert opts['data.train_batch_size'] == '32'
    assert opts['actor_rollout_ref.rollout.n'] == '8'


def test_mechanism_stage_can_preserve_eight_rollouts(tmp_path):
    opts = dict(x.lstrip('+').split('=', 1) for x in m.build_overrides(
        data_dir=tmp_path, output=tmp_path/'out', model_path=tmp_path,
        init_adapter=tmp_path, gpus=1, profile='rich64-pilot', root=PATH.parents[1],
        train_count=512, rollouts=8))
    assert opts['actor_rollout_ref.rollout.n'] == '8'
    assert opts['trainer.test_freq'] == '512'


@pytest.mark.skipif(importlib.util.find_spec('verl') is None, reason='requires official environment')
def test_overrides_match_actual_upstream_model_dataclass(tmp_path):
    import verl
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from verl.workers.config.model import HFModelConfig
    with initialize_config_dir(config_dir=str(Path(verl.__file__).parent/'trainer/config'), version_base=None):
        cfg = compose(config_name='sdpo', overrides=m.build_overrides(
            data_dir=tmp_path, output=tmp_path/'out', model_path=tmp_path,
            init_adapter=tmp_path, gpus=1, profile='rich64-pilot', root=PATH.parents[1]))
    model = OmegaConf.merge(OmegaConf.structured(HFModelConfig), cfg.actor_rollout_ref.model)
    assert model.lora_rank == 16
