#!/usr/bin/env python3
"""Plan, check, or run the pinned authors' SDPO trainer on frozen Rich64 data.

This is an algorithm baseline adaptation, not a reproduction of LCB paper scores.
No training happens unless the explicit `run` subcommand is used.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from eesd.sdpo_official import (MODEL_ID, MODEL_REVISION, sha256,
                                validate_upstream, verify_export)


def build_overrides(*, data_dir: Path, output: Path, model_path: Path,
                    init_adapter: Path, gpus: int, profile: str, root: Path,
                    train_count: int = 64, seed: int = 1701,
                    rollouts: int | None = None) -> list[str]:
    if profile not in ('rich64-pilot', 'upstream-hparams'):
        raise ValueError('unknown profile')
    pilot = profile == 'rich64-pilot'
    if rollouts is not None and rollouts not in (4, 8, 16, 32):
        raise ValueError('rollouts must be 4, 8, 16 or 32')
    batch, response = (1, 256) if pilot else (32, 8192)
    if train_count < batch or train_count % batch:
        raise ValueError('train size must be a positive multiple of batch size; no dropped rows')
    opts = {
        'vars.dir': str(root), 'vars.task': 'runbugrun_rich64',
        'vars.log_dir': str(output), 'vars.ckpt_dir': str(output/'checkpoints'),
        'data.train_files': [str(data_dir/'train.parquet')],
        'data.val_files': [str(data_dir/'test.parquet')],
        'data.train_batch_size': batch, 'data.val_batch_size': 1,
        'data.max_prompt_length': 8192 if pilot else 2048,
        'data.max_response_length': response,
        'data.filter_overlong_prompts': False, 'data.truncation': 'error',
        'data.trust_remote_code': False, 'data.seed': seed,
        'data.dataloader_num_workers': 0,
        'max_model_len': 10496 if pilot else 18944,
        'actor_rollout_ref.model.path': str(model_path),
        'actor_rollout_ref.model.trust_remote_code': False,
        'actor_rollout_ref.model.lora_rank': 16,
        'actor_rollout_ref.model.lora_alpha': 32,
        'actor_rollout_ref.model.lora_adapter_path': str(init_adapter),
        # PeftModel.from_pretrained reads target modules from the sealed adapter.
        # Upstream HFModelConfig accepts a string here, not a list.
        'actor_rollout_ref.model.use_remove_padding': False,
        'actor_rollout_ref.model.use_fused_kernels': False,
        '+actor_rollout_ref.model.override_config.attn_implementation': 'sdpa',
        'actor_rollout_ref.actor.policy_loss.loss_mode': 'sdpo',
        'actor_rollout_ref.actor.self_distillation.full_logit_distillation': True,
        'actor_rollout_ref.actor.self_distillation.alpha': 1.0,
        'actor_rollout_ref.actor.self_distillation.distillation_topk': 20,
        'actor_rollout_ref.actor.self_distillation.distillation_add_tail': True,
        'actor_rollout_ref.actor.self_distillation.teacher_regularization': 'ema',
        'actor_rollout_ref.actor.self_distillation.teacher_update_rate': .01,
        'actor_rollout_ref.actor.self_distillation.is_clip': 2.0,
        'actor_rollout_ref.actor.self_distillation.success_reward_threshold': .5,
        'actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success': True,
        'actor_rollout_ref.actor.self_distillation.include_environment_feedback': True,
        'actor_rollout_ref.actor.self_distillation.environment_feedback_only_without_solution': True,
        'actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration': True,
        'actor_rollout_ref.actor.use_kl_loss': False,
        'actor_rollout_ref.actor.ppo_mini_batch_size': 1,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.actor.optim.lr': 1e-5 if pilot else 1e-6,
        'actor_rollout_ref.actor.optim.weight_decay': 0. if pilot else .01,
        'actor_rollout_ref.actor.optim.lr_warmup_steps': 0,
        'actor_rollout_ref.actor.fsdp_config.model_dtype': 'bf16',
        'actor_rollout_ref.actor.fsdp_config.seed': seed,
        'actor_rollout_ref.actor.use_torch_compile': False,
        'actor_rollout_ref.ref.fsdp_config.model_dtype': 'bf16',
        'actor_rollout_ref.ref.fsdp_config.seed': seed,
        'actor_rollout_ref.ref.use_torch_compile': False,
        'actor_rollout_ref.rollout.name': 'vllm',
        'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
        'actor_rollout_ref.rollout.n': rollouts if rollouts is not None else (4 if pilot else 8),
        'actor_rollout_ref.rollout.temperature': .7 if pilot else 1.,
        'actor_rollout_ref.rollout.top_p': .95 if pilot else 1.,
        'actor_rollout_ref.rollout.calculate_log_probs': True,
        '+actor_rollout_ref.rollout.engine_kwargs.vllm.seed': seed,
        'actor_rollout_ref.rollout.enforce_eager': True,
        'actor_rollout_ref.rollout.gpu_memory_utilization': .3,
        'actor_rollout_ref.rollout.max_num_seqs': 8,
        'actor_rollout_ref.rollout.agent.num_workers': 1,
        'actor_rollout_ref.rollout.val_kwargs.do_sample': False,
        'actor_rollout_ref.rollout.val_kwargs.n': 1,
        'actor_rollout_ref.rollout.val_kwargs.temperature': 0.,
        'actor_rollout_ref.rollout.val_kwargs.top_p': 1.,
        'algorithm.adv_estimator': 'grpo',  # Upstream SDPO disables the critic this way.
        'algorithm.use_kl_in_reward': False,
        'algorithm.rollout_correction.rollout_is': 'token',
        'algorithm.rollout_correction.rollout_is_threshold': 2.,
        'custom_reward_function.path': str(root/'src/eesd/sdpo_official.py'),
        'custom_reward_function.name': 'compute_score',
        'reward_model.use_reward_loop': False,
        'trainer.project_name': 'RLVR-rich64',
        'trainer.group_name': 'sdpo_official_verl_rich64',
        'trainer.experiment_name': output.name,
        'trainer.logger': ['console'], 'trainer.nnodes': 1,
        'trainer.n_gpus_per_node': gpus,
        'trainer.total_epochs': 5 if pilot else 30,
        'trainer.test_freq': train_count//batch,
        'trainer.save_freq': train_count//batch,
        'trainer.max_actor_ckpt_to_keep': 5 if pilot else 30,
        'trainer.val_before_train': True,
        'trainer.resume_mode': 'disable',
        'trainer.default_local_dir': str(output/'checkpoints'),
        'trainer.validation_data_dir': str(output/'validation'),
        'ray_kwargs.ray_init.num_cpus': 8,
    }
    def value(v):
        if isinstance(v, str):
            # Hydra's quotes must survive parsing; subprocess receives argument arrays.
            return json.dumps(v) if any(c in v for c in ' []{},=:') else v
        return json.dumps(v)
    return [f'{k}={value(v)}' for k, v in opts.items()]


def validate_adapter(path: Path, seed: int) -> dict:
    from create_learning_credit_init import digest_tree
    receipt = json.loads((path.parent/(path.name+'.receipt.json')).read_text())
    if (receipt.get('schema') != 'eesd-shared-initial-lora-v1'
        or receipt.get('seed') != seed or receipt.get('model_id') != MODEL_ID
        or receipt.get('revision') != MODEL_REVISION
        or receipt.get('sha256') != digest_tree(path)):
        raise ValueError('shared initialization identity/hash mismatch')
    config = json.loads((path/'adapter_config.json').read_text())
    if config['r'] != 16 or config['lora_alpha'] != 32 or config['lora_dropout'] != 0:
        raise ValueError('shared adapter must match Rich64 rank16/alpha32/dropout0')
    if set(config['target_modules']) != {'q_proj', 'k_proj', 'v_proj', 'o_proj',
                                         'gate_proj', 'up_proj', 'down_proj'}:
        raise ValueError('shared adapter target modules differ from Rich64')
    return receipt


def check_inputs(args) -> tuple[dict, list[str], dict]:
    upstream = validate_upstream(args.upstream)
    data = verify_export(args.data_dir)
    adapter = validate_adapter(args.init_adapter, args.seed)
    from huggingface_hub import snapshot_download
    model = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    index = json.loads((model/'model.safetensors.index.json').read_text())
    if any(not (model/s).is_file() for s in set(index['weight_map'].values())):
        raise ValueError('incomplete pinned model snapshot')
    overrides = build_overrides(data_dir=args.data_dir, output=args.output, model_path=model,
        init_adapter=args.init_adapter, gpus=args.gpus, profile=args.profile, root=ROOT,
        train_count=data['train_bugs'], seed=args.seed, rollouts=args.rollouts)
    env = dict(os.environ, SDPO_UPSTREAM_ROOT=str(args.upstream),
               HF_HUB_OFFLINE='1', WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false',
               VLLM_ATTENTION_BACKEND='TRITON_ATTN', VLLM_NO_USAGE_STATS='1',
               DO_NOT_TRACK='1',
               TASK='runbugrun_rich64', EXPERIMENT=args.output.name,
               PYTHONPATH=os.pathsep.join([str(args.upstream), str(ROOT/'src')]))
    command = [sys.executable, '-m', 'verl.trainer.main_ppo', '--config-name', 'sdpo', *overrides]
    receipt = {'schema': 'sdpo-official-rich64-run-v1', 'status': 'planned_not_trained',
        'method': 'sdpo_official_verl_rich64', 'profile': args.profile,
        'upstream': upstream, 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
        'data': data, 'shared_initial_adapter': adapter, 'command': command,
        'runtime_environment': {k: env.get(k) for k in (
            'CUDA_VISIBLE_DEVICES', 'SDPO_UPSTREAM_ROOT', 'HF_HUB_OFFLINE',
            'WANDB_MODE', 'TOKENIZERS_PARALLELISM', 'VLLM_ATTENTION_BACKEND',
            'VLLM_NO_USAGE_STATS', 'DO_NOT_TRACK', 'PYTHONPATH')},
        'bridge_sha256': sha256(ROOT/'src/eesd/sdpo_official.py'),
        'launcher_sha256': sha256(Path(__file__)),
        'packages': {n: importlib.metadata.version(n) for n in
                     ('torch', 'vllm', 'transformers', 'peft', 'ray', 'numpy', 'verl')},
        'deviations': [
            'RunBugRun repair and Qwen2.5-Coder7B replace paper LCBv6 and Qwen3-8B.',
            'LoRA rank16/shared initial adapter replace paper full-parameter training.',
            'BF16 base weights replace historical Rich64 NF4; do not claim precision parity.',
            'Greedy single-sample evaluation replaces paper stochastic n4 evaluation.',
            'Mandatory bubblewrap execution; original generated tokens still used for loss.',
            'Pilot profile changes batch/rollout/lr/length/epochs; upstream-hparams preserves selected published knobs only.',
            'WSL CUDA runtime/vLLM pins, eager inference, Triton rollout attention and SDPA actor replace GH200 cluster runtime.',
            'Online per-batch EMA SDPO updates are not Rich64 frozen-per-round update scheduling.',
        ]}
    return receipt, command, env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'check', 'run'])
    p.add_argument('--upstream', type=Path, default=Path('/root/ai-work/SDPO-official'))
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--init-adapter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile', choices=['rich64-pilot', 'upstream-hparams'], default='rich64-pilot')
    p.add_argument('--seed', type=int, default=1701)
    p.add_argument('--rollouts', type=int, choices=[4, 8, 16, 32],
                   help='explicit stage override; otherwise use the selected profile default')
    p.add_argument('--gpus', type=int, choices=[1, 2], default=1)
    a = p.parse_args()
    for field in ('upstream', 'data_dir', 'init_adapter', 'output'):
        setattr(a, field, getattr(a, field).resolve())
    receipt, command, env = check_inputs(a)
    if a.action == 'plan':
        print(json.dumps(receipt, indent=2))
        return
    # Resolve the real upstream Hydra config, rather than validating our own copy.
    resolved = subprocess.run([*command, '--cfg', 'job', '--resolve'], env=env,
                              cwd=a.upstream, text=True, capture_output=True)
    if resolved.returncode:
        raise RuntimeError('upstream configuration failed:\n'+resolved.stderr)
    from eesd.strict_execution import run_tests_batched
    probe = run_tests_batched('print(input())', [{'id': 'probe', 'input': 'ok', 'expected': 'ok'}])
    if probe[0].status != 'PASS':
        raise RuntimeError('bubblewrap execution probe failed')
    # Fail on overlong prompts before allocating GPU workers. Never silently change cohort.
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    model_path = next(x.split('=', 1)[1] for x in command if x.startswith('actor_rollout_ref.model.path='))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    limit = 8192 if a.profile == 'rich64-pilot' else 2048
    for name in ('train.parquet', 'test.parquet'):
        rows = pq.read_table(a.data_dir/name).to_pylist()
        sizes = [len(tokenizer.apply_chat_template(r['prompt'], add_generation_prompt=True,
                                                   enable_thinking=False)) for r in rows]
        if max(sizes) > limit:
            raise ValueError(f'{name}: prompt exceeds {limit}; explicit protocol revision required')
        receipt[name+'_max_prompt_tokens'] = max(sizes)
    receipt['status'] = 'preflight_passed_not_trained'
    if a.action == 'check':
        print(json.dumps(receipt, indent=2))
        return
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'resolved_config.yaml').write_text(resolved.stdout)
    lock = '\n'.join(sorted(f'{d.metadata["Name"]}=={d.version}'
                            for d in importlib.metadata.distributions()))+'\n'
    (a.output/'environment.freeze.txt').write_text(lock)
    receipt['status'] = 'running'
    report = a.output/'run_manifest.json'
    report.write_text(json.dumps(receipt, indent=2)+'\n')
    with (a.output/'training.log').open('w') as log:
        result = subprocess.run(command, cwd=a.upstream, env=env, stdout=log, stderr=subprocess.STDOUT)
    receipt['status'] = 'trainer_exited_zero_review_metrics' if result.returncode == 0 else 'failed'
    receipt['returncode'] = result.returncode
    report.write_text(json.dumps(receipt, indent=2)+'\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
