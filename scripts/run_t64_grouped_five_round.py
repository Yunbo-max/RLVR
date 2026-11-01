#!/usr/bin/env python3
"""Five-round T=4..64 rich-execution, K=4 exact *GROUP* Shapley runner.

Executes 64 public tests per sampled candidate for EVERY arm, but exposes
only first T=4/8/16/32/64 to the evidence selector. T64 outcomes are
compressed into K=4 deterministic behavioral groups, and each group is
a Shapley player (2**4=16 coalition prompts, for every T).

This is GROUP Shapley, not individual-test Shapley. The 64-individual-test
Monte Carlo estimator lives in eesd.feedback_group_credit as a SECONDARY
experiment. No real H100 benefit is assumed. Requires working bwrap.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_learning_credit_five_round as previous

from eesd.feedback_group_credit import VALID_T,form_groups,compact_candidate_groups
from eesd.learning_credit_selection import (
    selection_from_arm,sample_matched_update_sources)
from eesd.strict_execution import run_test,audit_vector

ARMS=('no_update','sdpo','audit_gate','direct_max','shapley',
      'random_subset','matched_random_updates')
K_GROUPS=4


def _sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def _load_data(folder:Path,ntrain:int,neval:int):
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest.get('schema')!='eesd-rich64-frozen-v1':
        raise ValueError('use strict nested 64-execution RunBugRun manifest')
    for filename,digest in manifest['file_sha256'].items():
        if _sha(folder/filename)!=digest:raise ValueError(f'dataset SHA mismatch: {filename}')
    read=previous.legacy.read_records
    train=read(folder/'train_public_master64.jsonl',
        {'source_id','problem_id','task','buggy_code','public_tests'})
    cal=read(folder/'train_calibration.jsonl',
        {'source_id','problem_id','calibration_tests'})
    evaluation=read(folder/'eval_private.jsonl',
        {'source_id','problem_id','task','buggy_code','evaluation_tests'})
    if min(ntrain,neval)<1 or ntrain>len(train) or neval>len(evaluation):
        raise ValueError('requested formal cohort exceeds frozen file')
    train=train[:ntrain];evaluation=evaluation[:neval]
    bycal={r['source_id']:r for r in cal}
    for r in train:
        if len(r['public_tests'])!=64:raise ValueError('not 64 public tests')
        z=bycal.get(r['source_id'])
        if z is None or z['problem_id']!=r['problem_id']:
            raise ValueError('training calibration task mismatch')
        if len(z['calibration_tests'])!=manifest['calibration_tests']:
            raise ValueError('training calibration count mismatch')
        public={t['input'] for t in r['public_tests']}
        calibration={t['input'] for t in z['calibration_tests']}
        if len(public)!=64 or len(calibration)!=len(z['calibration_tests']) or public&calibration:
            raise ValueError('public/calibration input contamination')
    if ({r['problem_id'] for r in train}&
        {r['problem_id'] for r in evaluation}):
        raise ValueError('train/final heldout problem overlap')
    return train,evaluation,bycal,manifest


def _panel(train:list[dict],round_id:int,*,seed:int,mode:str)->set[str]:
    if mode=='repeat':return {r['source_id'] for r in train}
    if mode!='rotate':raise ValueError('unknown training panel mode')
    order=sorted((r['source_id'] for r in train),
       key=lambda s:hashlib.sha256(f'{seed}:{s}:panel'.encode()).hexdigest())
    return {sid for i,sid in enumerate(order) if i%5==round_id}


def _safe_feedback(code:str,tests:list[dict])->list[dict]:
    """Faithful compact execution transcript, no reference fixed code."""
    rows=[]
    for test in tests:
        v=run_test(code,test)
        rows.append({'id':str(test['id']),
            'input':test['input'][:512],
            'expected':test['expected'][:512],
            'actual':v.actual[:512], 'stderr':v.stderr[:512],
            'outcome':v.status})
    return rows


def _train_one(teacher,student,optimizer,tokenizer,row,calibration_tests,
               args,round_idx:int,schedule_round:dict|None=None):
    import torch
    sid=row['source_id']
    reference_mask=None
    random_sources=None
    if schedule_round is not None:
        reference_mask=schedule_round['accepted_masks'].get(sid)
        random_sources=set(schedule_round['random_update_sources'])
        if args.arm=='random_subset' and not reference_mask:
            return {'source_id':sid,'round':round_idx,'step':False,
                    'selected_mask':0,'skip':'not_on_reference_shapley_schedule'}
        if args.arm=='matched_random_updates' and sid not in random_sources:
            return {'source_id':sid,'round':round_idx,'step':False,
                    'selected_mask':0,'skip':'not_on_random_count_schedule'}

    sampled=previous.legacy._sample_rollouts(teacher,tokenizer,row,args,round_idx)
    clean=[]
    for _,code in sampled:
        tokens=previous._normalized_ids(tokenizer,code,args.max_new_tokens)
        if tokens and len(code)<=16000:
            clean.append((tokens,code))
    if len(clean)<2:
        return {'source_id':sid,'round':round_idx,'step':False,'selected_mask':0,
                'skip':'too_few_executable_candidate_programs'}

    # All arms execute EXACTLY 64 public tests on every evaluated candidate.
    public=row['public_tests']
    transcript=[_safe_feedback(code,public) for _,code in clean]
    cal=np.array([audit_vector(code,calibration_tests)
                  for _,code in clean],dtype=np.float64)
    # Crucially: do NOT read the hidden 64-T test results into the group builder.
    exposed_public=public[:args.exposed_feedback]
    exposed_transcripts=[v[:args.exposed_feedback] for v in transcript]
    grouping=form_groups(exposed_public,exposed_transcripts,
           groups=K_GROUPS,seed=args.seed,source_id=sid)
    feedback=[compact_candidate_groups(grouping,v,per_group_chars=args.group_char_budget)
              for v in exposed_transcripts]
    # With four *groups*, group Shapley and DirectMax each score EXACTLY
    # sixteen coalition prompts independent of whether T is 4 or 64.
    scoring_arm='shapley_select' if args.arm in (
           'shapley','direct_max','random_subset') else 'audit_gate'
    scores=[previous.score_candidate(teacher,tokenizer,row,tokens,fb,
               arm=scoring_arm,max_prompt_tokens=args.max_prompt_tokens)
            for (tokens,_),fb in zip(clean,feedback)]
    choice=selection_from_arm(args.arm,scores,cal,
            round_index=round_idx,source_id=sid,seed=args.seed,
            reference_mask=reference_mask,reference_update_sources=random_sources,
            uncertainty=args.audit_uncertainty,subset_cost=args.subset_cost,
            min_gain=args.min_gain)
    mask=choice.mask
    if args.arm in ('sdpo','matched_random_updates'):
        weights=[np.ones(len(tok),dtype=np.float32) for tok,_ in clean]
    else:
        weights=previous.directional_token_weights(scores,cal,mask,
                                                    shift_scale=args.shift_scale)

    base={
        'source_id':sid,'round':round_idx,'selected_mask':mask,
        'choice_rule':choice.rule,'j_calibration':choice.j_calibration,
        'shapley_phi':list(choice.phi) if choice.phi is not None else None,
        'public_test_executions':64*len(clean),
        'calibration_test_executions':len(calibration_tests)*len(clean),
        'teacher_subset_scoring_calls':len(clean)*(16 if scoring_arm=='shapley_select' else 2),
        'executed_public_test_count':64,
        'exposed_feedback_test_count':args.exposed_feedback,
        'evidence_group_count':K_GROUPS,
        'distinct_outcome_signatures':grouping.distinct_outcome_signatures,
        'calibration_behavior_variance':float(cal.var(axis=0).mean()),
        'calibration_pass_fraction':float(cal.mean()),
        'candidate_code_sha256':[hashlib.sha256(c.encode()).hexdigest() for _,c in clean],
        'candidate_sources':[c for _,c in clean],
        'candidate_shift_means':[(v.mean(axis=1)-v[0].mean()).tolist() for v in scores],
        'candidate_calibration_pass_fraction':cal.mean(axis=1).tolist(),
        'group_test_indices':[list(x) for x in grouping.members],
    }
    if mask==0 or sum(float(w.sum()) for w in weights)<=1e-9:
        return {**base,'step':False,'skip':'abstained_or_zero_token_credit'}
    optimizer.zero_grad(set_to_none=True)
    n=0;stats=[]
    for (tokens,_),fb,w in zip(clean,feedback,weights):
        if not np.any(w>0):continue
        tq,oldq=previous.masked_teacher(teacher,tokenizer,row,tokens,fb,mask)
        context=previous.legacy.prompt_ids(tokenizer,row)
        ids=torch.tensor([context+tokens],dtype=torch.long,device=student.device)
        loss,info=previous.distill_token_kl(student,oldq,tq,w,ids,
                                          len(context),beta=args.beta)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite group distillation loss')
        (loss/len(clean)).backward()
        n+=1;stats.append(info)
        del loss,ids,tq,oldq
    if n==0:return {**base,'step':False,'skip':'no_distilled_programs'}
    norm=torch.nn.utils.clip_grad_norm_(
        [p for p in student.parameters() if p.requires_grad],1.)
    if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient')
    if float(norm.detach().cpu())<=1e-10:
        return {**base,'step':False,'skip':'zero_gradient'}
    optimizer.step()
    return {**base,'step':True,'grad_norm':float(norm.detach().cpu()),
            'distilled_candidates':n,
            'mean_weight':float(np.mean([s['mean_weight'] for s in stats]))}


def _save(path:Path,payload:dict):
    with path.open('x') as out:
        json.dump(payload,out,indent=2,sort_keys=True,ensure_ascii=False)


def _check_schedule(args,manifest_sha:str,adapter_sha:str,source_ids:list[str])->dict|None:
    if args.arm not in ('random_subset','matched_random_updates'):
        if args.schedule is not None:raise ValueError('only random controls accept schedule')
        return None
    if args.schedule is None or not args.schedule.is_file():
        raise ValueError('random controls require a frozen Shapley training schedule')
    schedule=json.loads(args.schedule.read_text())
    if (schedule.get('schema')!='eesd-rich64-schedule-v1'
        or schedule.get('seed')!=args.seed or schedule.get('rounds')!=5
        or schedule.get('dataset_manifest_sha256')!=manifest_sha
        or schedule.get('initial_adapter_sha256')!=adapter_sha
        or schedule.get('exposed_feedback')!=args.exposed_feedback
        or schedule.get('panel_mode')!=args.panel_mode
        or schedule.get('source_ids')!=sorted(source_ids)):
        raise ValueError('training-only schedule mismatch')
    if len(schedule['training_rounds'])!=5:
        raise ValueError('missing reference rounds')
    for k,rd in enumerate(schedule['training_rounds']):
        if rd.get('round')!=k or set(rd.get('accepted_masks',{}))!=set(source_ids):
            raise ValueError('source IDs/round mismatch in schedule')
        chosen=rd.get('random_update_sources',[])
        active=rd.get('panel_active_ids',[])
        if (len(chosen)!=rd.get('shapley_actual_optimizer_steps')
            or len(set(chosen))!=len(chosen)
            or not set(chosen)<=set(active)):
            raise ValueError('random schedule does not match panel/update budget')
    return schedule


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--init-adapter',type=Path,required=True)
    p.add_argument('--schedule',type=Path)
    p.add_argument('--model-config',type=Path,default=ROOT/'rlvr/models/qwen25_7b.yaml')
    p.add_argument('--seed',type=int,default=1701)
    p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--exposed-feedback',type=int,choices=VALID_T,default=64)
    p.add_argument('--group-char-budget',type=int,default=440)
    p.add_argument('--panel-mode',choices=('repeat','rotate'),default='repeat')
    p.add_argument('--rollouts',type=int,default=8)
    p.add_argument('--train-limit',type=int,default=256)
    p.add_argument('--eval-limit',type=int,default=128)
    p.add_argument('--generate-batch',type=int,default=4)
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--max-prompt-tokens',type=int,default=8192)
    p.add_argument('--temperature',type=float,default=.7)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--beta',type=float,default=.02)
    p.add_argument('--audit-uncertainty',type=float,default=.5)
    p.add_argument('--subset-cost',type=float,default=.001)
    p.add_argument('--min-gain',type=float,default=.001)
    p.add_argument('--shift-scale',type=float,default=.15)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    previous.legacy._require_model_config(args.model_config)
    if (args.rounds!=5 or args.rollouts<2 or args.group_char_budget<200
        or args.group_char_budget>2048 or not args.init_adapter.is_dir()
        or not (args.init_adapter/'adapter_config.json').exists()):
        raise ValueError('five updates, >=2 rollouts, bounded group evidence and fixed LoRA adapter required')
    train,ev,bycal,manifest=_load_data(args.data_dir,args.train_limit,args.eval_limit)
    manifest_sha=_sha(args.data_dir/'manifest.json')
    adapter_sha=previous._digest_adapter(args.init_adapter)
    schedule=_check_schedule(args,manifest_sha,adapter_sha,
                              [r['source_id'] for r in train])
    planned_panels=[_panel(train,k,seed=args.seed,mode=args.panel_mode)
                    for k in range(5)]
    plan={'schema':'eesd-rich64-five-v1',
          'status':'prospective_not_executed',
          'arm':args.arm,'model':previous.legacy.MODEL_ID,
          'revision':previous.legacy.MODEL_REVISION,
          'seed':args.seed,'rounds':args.rounds,'rollouts':args.rollouts,
          'train_bug_instances':len(train),
          'train_distinct_problem_ids':len({r['problem_id'] for r in train}),
          'eval_distinct_problem_ids':len(ev),
          'executed_public_feedback_tests_per_candidate':64,
          'exposed_feedback':args.exposed_feedback,
          'group_shapley_players':K_GROUPS,
          'exact_group_coalitions':16,
          'calibration_tests':manifest['calibration_tests'],
          'reserved_audit_tests':manifest['reserved_audit_tests'],
          'panel_mode':args.panel_mode,'panel_sizes':[len(p) for p in planned_panels],
          'dataset_manifest_sha256':manifest_sha,
          'initial_adapter_sha256':adapter_sha,
          'init_adapter':str(args.init_adapter),
          'feedback_group_chars':args.group_char_budget,
          'max_new_tokens':args.max_new_tokens,
          'max_prompt_tokens':args.max_prompt_tokens,
          'temperature':args.temperature,'learning_rate':args.learning_rate,
          'beta':args.beta,
          'schedule_sha256':_sha(args.schedule) if args.schedule else None,
          'code_execution_sandbox':'strict_bubblewrap_no_network',
          'method_scope':'GROUP Shapley over 4 evidence aggregates, NOT test-level Shapley'}
    if args.dry_run:
        print(json.dumps(plan,indent=2,sort_keys=True));return
    if args.output.exists():raise FileExistsError('experiment output already exists')
    preflight=run_test('print(47)',{'input':'','expected':'47'})
    if preflight.status!='PASS':raise RuntimeError('bubblewrap isolated positive control failed')
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('real BF16 CUDA GPU required')
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    _,tokenizer,attach=previous.legacy._models(args)
    args.output.mkdir(parents=True,exist_ok=False)
    plan['status']='executing';_save(args.output/'plan.json',plan)
    prev_adapter=args.init_adapter
    for rnd in range(6):
        folder=args.output/f'round_{rnd}';folder.mkdir(exist_ok=False)
        teacher=attach(prev_adapter,trainable=False)
        evaluation=previous._evaluate(teacher,tokenizer,ev,args,rnd)
        _save(folder/'evaluation.json',evaluation)
        if rnd==5 or args.arm=='no_update':
            del teacher;gc.collect();torch.cuda.empty_cache()
            continue
        student=attach(prev_adapter,trainable=True)
        student.train()
        opt=torch.optim.AdamW([x for x in student.parameters() if x.requires_grad],
                  lr=args.learning_rate,weight_decay=0.)
        torch.cuda.reset_peak_memory_stats()
        begun=time.perf_counter()
        panel=planned_panels[rnd]
        panel_rows=[r for r in train if r['source_id'] in panel]
        ref=schedule['training_rounds'][rnd] if schedule else None
        target=None
        if args.arm=='matched_random_updates':
            target=ref['shapley_actual_optimizer_steps']
            pref=set(ref['random_update_sources'])
            ranked=lambda r:hashlib.sha256(f'{args.seed}:{rnd}:{r["source_id"]}:backup'.encode()).hexdigest()
            panel_rows=([r for r in panel_rows if r['source_id'] in pref]
                +sorted([r for r in panel_rows if r['source_id'] not in pref],key=ranked))
        receipt={}
        actual_steps=0
        with (folder/'training.jsonl').open('x') as file:
            for row in panel_rows:
                active_ref=ref
                if args.arm=='matched_random_updates':
                    if actual_steps>=target:
                        record={'source_id':row['source_id'],'round':rnd,'step':False,
                               'selected_mask':0,'skip':'step_budget_reached'}
                        receipt[row['source_id']]=record
                        continue
                    active_ref=dict(ref,random_update_sources=[row['source_id']])
                record=_train_one(teacher,student,opt,tokenizer,row,
                      bycal[row['source_id']]['calibration_tests'],
                      args,rnd,active_ref)
                actual_steps+=int(record.get('step',False))
                receipt[row['source_id']]=record
            for row in train:
                if row['source_id'] not in receipt:
                    receipt[row['source_id']]={'source_id':row['source_id'],
                       'round':rnd,'step':False,'selected_mask':0,
                       'skip':'not_in_round_panel'}
                file.write(json.dumps(receipt[row['source_id']],ensure_ascii=False,sort_keys=True)+'\n')
        if target is not None and target!=actual_steps:
            raise RuntimeError('matched random-control actual LoRA optimizer steps differ')
        updated=folder/'adapter';student.save_pretrained(updated)
        sample=list(receipt.values())
        _save(folder/'summary.json',{
           'round':rnd,'panel_size':len(panel),
           'sources':len(sample),'optimizer_steps':actual_steps,
           'shapley_accepted_steps_target':target,
           'selected_masks':[r.get('selected_mask',0) for r in sample],
           'teacher_subset_scoring_calls':sum(r.get('teacher_subset_scoring_calls',0) for r in sample),
           'public_test_executions':sum(r.get('public_test_executions',0) for r in sample),
           'calibration_test_executions':sum(r.get('calibration_test_executions',0) for r in sample),
           'group_discrimination_count':sum(r.get('distinct_outcome_signatures',0)>1 for r in sample),
           'choice_reasons':dict(Counter(r.get('choice_rule',r.get('skip','unknown')) for r in sample)),
           'training_wallclock_seconds':time.perf_counter()-begun,
           'peak_gpu_bytes':int(torch.cuda.max_memory_allocated()),
        })
        del teacher,student,opt;gc.collect();torch.cuda.empty_cache()
        prev_adapter=updated
    print(json.dumps({'status':'completed','output':str(args.output),'arm':args.arm}))


if __name__=='__main__':main()
