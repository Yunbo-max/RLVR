#!/usr/bin/env python3
"""Run frozen Joint/Mix mechanism study or five-round on-policy study.

Use separate wrappers run_joint_mix_experiment1.py and
run_joint_mix_experiment2.py. Existing Rich64 and CI are never modified.
Real model, datasets and mandatory bubblewrap are required; no fake outputs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
import run_t64_grouped_five_round as existing
from eesd.joint_mix import (QueryCache,behavior_value,deterministic_pair,
    fit_mixture,mask_indices,mix_sampled_logprobs,select_budgeted_joint,token_weights)
from eesd.strict_execution import run_test,audit_vector

TRAIN_ARMS=('joint_pair','mix_pair','joint_search','random_search','no_update')


def _save(path:Path,payload:dict):
    with path.open('x',encoding='utf-8') as out:
        json.dump(payload,out,ensure_ascii=False,indent=2,sort_keys=True)
        out.write('\n')


def _hash(path:Path)->str:
    return existing._sha(path)


def _load_data(args, *, include_audit:bool):
    manifest=json.loads((args.data_dir/'manifest.json').read_text())
    train,final,cal,m=existing._load_data(
        args.data_dir,int(manifest['train_bug_instances']),min(args.eval_limit,int(manifest['eval_distinct_problem_ids'])))
    # One bug instance per problem for primary analyses; no pseudo-replication.
    unique=[];seen=set()
    for r in train:
        if r['problem_id'] not in seen:
            seen.add(r['problem_id']);unique.append(r)
        if len(unique)==args.train_limit:break
    if len(unique)!=args.train_limit:
        raise ValueError('requested independent training problems exceed eligible cohort')
    heldout={}
    if include_audit:
        records=existing.previous.legacy.read_records(
            args.data_dir/'train_audit_do_not_train.jsonl',
            {'source_id','problem_id','audit_tests'})
        heldout={r['source_id']:r for r in records}
        for row in unique:
            a=heldout.get(row['source_id'])
            if a is None or a['problem_id']!=row['problem_id']:
                raise ValueError('reserved audit source/prob identity mismatch')
            probes=cal[row['source_id']]['calibration_tests']
            ids=[str(z['id']) for z in row['public_tests']+probes+a['audit_tests']]
            inputs=[str(z['input']) for z in row['public_tests']+probes+a['audit_tests']]
            if len(set(ids))!=len(ids) or len(set(inputs))!=len(inputs):
                raise ValueError('feedback/calibration/reserved audit contamination')
    return unique,final,cal,heldout,m


def _candidates(teacher,tokenizer,row,args,rnd):
    sampled=existing.previous.legacy._sample_rollouts(teacher,tokenizer,row,args,rnd)
    out=[]
    for _,code in sampled:
        ids=existing.previous._normalized_ids(tokenizer,code,args.max_new_tokens)
        if ids and len(code)<=16000:
            out.append((ids,code))
    return out


def _feedback(candidates,row):
    # Same F64 execution protocol as original Rich64; 64 is data, not 4 groups.
    return [existing._safe_feedback(code,row['public_tests']) for _,code in candidates]


def _teacher_score(teacher,tokenizer,row,tokens,feedback,mask,args,full=False):
    subset=[feedback[j] for j in mask_indices(mask,len(feedback))]
    ctx=subset if subset else None
    if len(existing.previous.legacy.prompt_ids(tokenizer,row,ctx))>args.max_prompt_tokens:
        raise ValueError('joint/mix teacher prompt exceeds fixed context budget')
    return existing.previous.legacy._logprobs_for_fixed_response(
        teacher,tokenizer,row,ctx,tokens,return_distribution=full)


def _cache(teacher,tokenizer,row,candidates,feedback,args,budget):
    def score(i,mask):
        return _teacher_score(teacher,tokenizer,row,candidates[i][0],feedback[i],
                              mask,args)[0]
    return QueryCache(len(candidates),len(feedback[0]),score,budget)


def _pair_masks(t:int,seed:int,rnd:int,sid:str):
    a,b=deterministic_pair(t,seed=seed,round_index=rnd,source_id=sid)
    return (a,b),(1<<a),(1<<b),(1<<a)|(1<<b)


def _mix_choice(cache,cal,ma,mb,args):
    base=cache.get(0);first=cache.get(ma);second=cache.get(mb)
    alpha,score=fit_mixture(base,first,second,cal,
            grid_denominator=args.mix_grid,uncertainty=args.uncertainty)
    mixed=mix_sampled_logprobs(base,first,second,alpha)
    return alpha,score,mixed


def _full_dist(teacher,tokenizer,row,tokens,feedback,mask,args):
    # Full [L,V] log distribution is materialized ONLY for an accepted update
    # or a bounded diagnostic; it is never cached for all 64 singletons.
    _,out=_teacher_score(teacher,tokenizer,row,tokens,feedback,mask,args,full=True)
    if out is None:
        raise RuntimeError('teacher did not return full vocabulary distribution')
    return out


def _torch_mix(logq,alpha):
    import torch
    if len(logq)!=3 or len(alpha)!=3:
        raise ValueError('expected q0, qA, qB with one coefficient each')
    if not all(x.shape==logq[0].shape for x in logq):
        raise ValueError('teacher vocab-token mismatch')
    weighted=[x+float(np.log(a)) for x,a in zip(logq,alpha) if a>0]
    if not weighted:
        raise ValueError('empty teacher mixture')
    mixed=torch.logsumexp(torch.stack(weighted),dim=0) if len(weighted)>1 else weighted[0]
    return mixed


def _additive_logq(q0,qa,qb):
    import torch
    raw=qa+qb-q0
    return raw-torch.logsumexp(raw,dim=-1,keepdim=True)


def _frozen_diagnostics(teacher,tokenizer,row,candidates,feedback,alpha,ma,mb,mab,args):
    """Diagnostic subset of tokens, full-vocab comparison, no labels used.

    Reports Joint-vs-convex-mixture and Joint-vs-log-additive evidence fusion.
    These are not final accuracy scores and do not fit mixture weights.
    """
    import torch
    report=[]
    for i,(tokens,_) in enumerate(candidates[:args.probe_candidates]):
        with torch.no_grad():
            q0=_full_dist(teacher,tokenizer,row,tokens,feedback[i],0,args)
            qa=_full_dist(teacher,tokenizer,row,tokens,feedback[i],ma,args)
            qb=_full_dist(teacher,tokenizer,row,tokens,feedback[i],mb,args)
            qj=_full_dist(teacher,tokenizer,row,tokens,feedback[i],mab,args)
            end=min(len(tokens),args.probe_tokens)
            qm=_torch_mix((q0[:end],qa[:end],qb[:end]),alpha)
            qadd=_additive_logq(q0[:end],qa[:end],qb[:end])
            logj=qj[:end]
            pj=logj.exp()
            klm=(pj*(logj-qm)).sum(-1).mean()
            kla=(pj*(logj-qadd)).sum(-1).mean()
            report.append({'candidate_index':i,'probed_tokens':end,
                           'kl_joint_to_calibrated_mix':float(klm.cpu()),
                           'kl_joint_to_logadd':float(kla.cpu())})
            del q0,qa,qb,qj,qm,qadd
    return report


def frozen_source(teacher,tokenizer,row,cal_tests,audit_tests,args):
    candidates=_candidates(teacher,tokenizer,row,args,0)
    if len(candidates)<2:
        return {'source_id':row['source_id'],'problem_id':row['problem_id'],
                'status':'too_few_valid_candidates','candidate_count':len(candidates)}
    feedback=_feedback(candidates,row)
    cal=np.asarray([audit_vector(code,cal_tests) for _,code in candidates],dtype=float)
    (ia,ib),ma,mb,mab=_pair_masks(len(feedback[0]),args.seed,0,row['source_id'])
    cache=_cache(teacher,tokenizer,row,candidates,feedback,args,budget=4)
    alpha,mix_cal,mix=_mix_choice(cache,cal,ma,mb,args)
    joint=cache.get(mab);base=cache.get(0)
    joint_cal=behavior_value(joint,base,cal,uncertainty=args.uncertainty)
    # The reserved A8 audit is read and executed AFTER the pair, alpha and
    # all calibration-driven choices are fixed. It cannot enter any selector.
    audit=np.asarray([audit_vector(code,audit_tests) for _,code in candidates],dtype=float)
    joint_audit=behavior_value(joint,base,audit,uncertainty=args.uncertainty)
    mix_audit=behavior_value(mix,base,audit,uncertainty=args.uncertainty)
    probe=_frozen_diagnostics(teacher,tokenizer,row,candidates,feedback,
                        alpha,ma,mb,mab,args) if args.probe_tokens>0 else []
    return {'source_id':row['source_id'],'problem_id':row['problem_id'],
            'status':'completed','candidate_count':len(candidates),
            'candidate_code_sha256':[hashlib.sha256(c.encode()).hexdigest() for _,c in candidates],
            'feedback_indices':[ia,ib],'feedback_test_ids':[row['public_tests'][ia]['id'],row['public_tests'][ib]['id']],
            'alpha_q0_qa_qb':list(alpha),'c8_joint_J':joint_cal.criterion,
            'c8_mix_J':mix_cal.criterion,'a8_joint_value':joint_audit.mean,
            'a8_mix_value':mix_audit.mean,'a8_joint_minus_mix':joint_audit.mean-mix_audit.mean,
            'a8_joint_probe_values':list(joint_audit.by_probe),
            'a8_mix_probe_values':list(mix_audit.by_probe),
            'full_vocab_probe':probe,'unique_teacher_subsets':list(cache.entries),
            'teacher_scoring_conditions':cache.forward_conditions,
            'public_execution_count':len(candidates)*len(feedback[0]),
            'calibration_execution_count':len(candidates)*len(cal_tests),
            'reserved_audit_execution_count':len(candidates)*len(audit_tests)}


def _train_source(teacher,student,opt,tokenizer,row,cal_tests,args,rnd):
    import torch
    candidates=_candidates(teacher,tokenizer,row,args,rnd)
    sid=row['source_id']
    if len(candidates)<2:
        return {'source_id':sid,'problem_id':row['problem_id'],
                'step':False,'skip':'too_few_valid_candidates'}
    feedback=_feedback(candidates,row)
    cal=np.asarray([audit_vector(code,cal_tests) for _,code in candidates],dtype=float)
    (ia,ib),ma,mb,mab=_pair_masks(len(feedback[0]),args.seed,rnd,sid)
    budget=(args.query_budget if args.arm in ('joint_search','random_search') else 4)
    cache=_cache(teacher,tokenizer,row,candidates,feedback,args,budget)
    base=cache.get(0);alpha=None
    if args.arm=='mix_pair':
        alpha,value,target=_mix_choice(cache,cal,ma,mb,args)
        selected=tuple(j for j,a in ((ia,alpha[1]),(ib,alpha[2])) if a>0)
        mask=sum(1<<j for j in selected)
        # Keep same correction-relative acceptance standard as Joint.
        accept=value.criterion>args.min_gain
        query_record=None
    elif args.arm=='joint_pair':
        target=cache.get(mab);mask=mab
        value=behavior_value(target,base,cal,uncertainty=args.uncertainty)
        accept=value.criterion>args.min_gain
        query_record=None
    else:
        search=select_budgeted_joint(cache,cal,seed=args.seed,round_index=rnd,
               source_id=sid,policy='adaptive' if args.arm=='joint_search' else 'random',
               max_items=args.max_feedback_items,uncertainty=args.uncertainty,
               subset_cost=args.subset_cost,min_gain=args.min_gain)
        mask=search.selected_mask
        target=cache.get(mask)
        value=behavior_value(target,base,cal,uncertainty=args.uncertainty,
                 cost=args.subset_cost*mask.bit_count())
        accept=not search.abstained
        query_record={'queried_masks':list(search.queried_masks),
                      'queried_criterion':[[int(m),float(x)] for m,x in search.scored]}
    weights=token_weights(target,base,cal,scale=args.shift_scale) if accept else []
    diag={'source_id':sid,'problem_id':row['problem_id'],'round':rnd,'arm':args.arm,
          'feedback_indices':list(mask_indices(mask,len(feedback[0]))),
          'selected_mask':mask,'teacher_criterion':value.criterion,
          'alpha_q0_qa_qb':list(alpha) if alpha is not None else None,
          'candidate_count':len(candidates),'calibration_contrast':float(cal.var(axis=0).mean()),
          'mean_weight':float(np.mean([w.mean() for w in weights])) if weights else 0.,
          'teacher_query_subsets':len(cache.entries),
          'teacher_scoring_conditions':cache.forward_conditions,
          'public_execution_count':len(candidates)*len(feedback[0]),
          'calibration_execution_count':len(candidates)*len(cal_tests),
          'candidate_code_sha256':[hashlib.sha256(c.encode()).hexdigest() for _,c in candidates],
          'search_diagnostics':query_record}
    if not accept or not any(np.any(w>0) for w in weights):
        return {**diag,'step':False,'skip':'no_positive_calibrated_signal'}
    opt.zero_grad(set_to_none=True)
    used=0
    for i,((tokens,_),w) in enumerate(zip(candidates,weights)):
        if not np.any(w>0):continue
        if args.arm=='mix_pair':
            q0=_full_dist(teacher,tokenizer,row,tokens,feedback[i],0,args)
            qa=_full_dist(teacher,tokenizer,row,tokens,feedback[i],ma,args)
            qb=_full_dist(teacher,tokenizer,row,tokens,feedback[i],mb,args)
            logq=_torch_mix((q0,qa,qb),alpha)
            del qa,qb
        else:
            logq=_full_dist(teacher,tokenizer,row,tokens,feedback[i],mask,args)
            q0=_full_dist(teacher,tokenizer,row,tokens,feedback[i],0,args)
        ctx=existing.previous.legacy.prompt_ids(tokenizer,row)
        ids=torch.tensor([ctx+tokens],dtype=torch.long,device=student.device)
        loss,_=existing.previous.distill_token_kl(student,q0,logq,w,ids,len(ctx),beta=args.beta)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite joint/mix distillation loss')
        (loss/len(candidates)).backward();used+=1
        del q0,logq,loss,ids
    if not used:return {**diag,'step':False,'skip':'zero_effective_token_weight'}
    norm=torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],1.)
    if not torch.isfinite(norm):raise RuntimeError('nonfinite LoRA gradient')
    if float(norm.detach().cpu())<=1e-10:
        return {**diag,'step':False,'skip':'zero_parameter_gradient'}
    opt.step()
    return {**diag,'step':True,'grad_norm':float(norm.detach().cpu()),'distilled_candidates':used}


def run_frozen(args,teacher,tokenizer,rows,bycal,heldout,manifest):
    records=[]
    for row in rows:
        sid=row['source_id']
        records.append(frozen_source(teacher,tokenizer,row,
                       bycal[sid]['calibration_tests'],heldout[sid]['audit_tests'],args))
        with (args.output/'rows.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(records[-1],ensure_ascii=False,sort_keys=True)+'\n')
    good=[r for r in records if r['status']=='completed']
    # Descriptive only; final independent evaluation is experiment 2.
    differences=np.asarray([x['a8_joint_minus_mix'] for x in good],dtype=float)
    if len(differences)>=2:
        rng=np.random.default_rng(args.seed+10114)
        boot=[float(differences[rng.integers(len(differences),size=len(differences))].mean())
              for _ in range(4000)]
        interval=[float(x) for x in np.quantile(boot,[.025,.975])]
    else:
        interval=None
    summary={'schema':'eesd-joint-mix-frozen-v1','status':'completed',
             'total_sources':len(records),'completed_sources':len(good),
             'a8_mean_joint_minus_mix':float(differences.mean()) if good else None,
             'a8_problem_bootstrap_95':interval,
             'total_teacher_scoring_conditions':sum(x.get('teacher_scoring_conditions',0) for x in records),
             'notes':'A8 is reserved for reporting; never used to fit alpha or choose pairs. Bootstrap conditional on sampled teacher and cohort.'}
    _save(args.output/'summary.json',summary)
    return summary


def run_train(args,tokenizer,attach,rows,final,bycal,manifest):
    import torch
    prev=args.init_adapter
    for rnd in range(args.rounds+1):
        folder=args.output/f'round_{rnd}';folder.mkdir(exist_ok=False)
        teacher=attach(prev,trainable=False)
        evaluation=existing.previous._evaluate(teacher,tokenizer,final,args,rnd)
        _save(folder/'evaluation.json',evaluation)
        if rnd==args.rounds or args.arm=='no_update':
            del teacher;gc.collect();torch.cuda.empty_cache();continue
        student=attach(prev,trainable=True)
        student.train()
        opt=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                              lr=args.learning_rate,weight_decay=0.)
        begun=time.perf_counter();records=[]
        panel=existing._panel(rows,rnd,seed=args.seed,mode=args.panel_mode)
        with (folder/'training.jsonl').open('x',encoding='utf-8') as stream:
            for row in rows:
                if row['source_id'] not in panel:
                    record={'source_id':row['source_id'],'problem_id':row['problem_id'],
                            'round':rnd,'step':False,'skip':'not_in_panel'}
                else:
                    record=_train_source(teacher,student,opt,tokenizer,row,
                         bycal[row['source_id']]['calibration_tests'],args,rnd)
                stream.write(json.dumps(record,sort_keys=True,ensure_ascii=False)+'\n')
                stream.flush();records.append(record)
        updated=folder/'adapter';student.save_pretrained(updated)
        _save(folder/'summary.json',{'round':rnd,'arm':args.arm,
               'optimizer_steps':sum(int(x.get('step',False)) for x in records),
               'teacher_scoring_conditions':sum(x.get('teacher_scoring_conditions',0) for x in records),
               'public_test_executions':sum(x.get('public_execution_count',0) for x in records),
               'calibration_executions':sum(x.get('calibration_execution_count',0) for x in records),
               'wallclock_seconds':time.perf_counter()-begun,
               'status':'completed'})
        del teacher,student,opt;gc.collect();torch.cuda.empty_cache()
        prev=updated
    return {'schema':'eesd-joint-mix-train-v1','status':'completed',
            'arm':args.arm,'final_round':args.rounds,
            'final_evaluation':str(args.output/f'round_{args.rounds}'/'evaluation.json')}


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=('frozen','train'),required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=TRAIN_ARMS,default='joint_pair')
    p.add_argument('--init-adapter',type=Path)
    p.add_argument('--model-config',type=Path,default=ROOT/'configs/models/qwen25_7b.yaml')
    p.add_argument('--seed',type=int,default=1701)
    p.add_argument('--train-limit',type=int,default=64)
    p.add_argument('--eval-limit',type=int,default=64)
    p.add_argument('--rollouts',type=int,default=8)
    p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--panel-mode',choices=('repeat','rotate'),default='repeat')
    p.add_argument('--generate-batch',type=int,default=4)
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--max-prompt-tokens',type=int,default=8192)
    p.add_argument('--temperature',type=float,default=.7)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--beta',type=float,default=.02)
    p.add_argument('--shift-scale',type=float,default=.15)
    p.add_argument('--uncertainty',type=float,default=.5)
    p.add_argument('--min-gain',type=float,default=.001)
    p.add_argument('--subset-cost',type=float,default=.001)
    p.add_argument('--mix-grid',type=int,default=20)
    p.add_argument('--query-budget',type=int,default=16)
    p.add_argument('--max-feedback-items',type=int,default=8)
    p.add_argument('--probe-tokens',type=int,default=16)
    p.add_argument('--probe-candidates',type=int,default=2)
    p.add_argument('--dry-run',action='store_true')
    return p


def main(argv=None):
    p=build_parser();args=p.parse_args(argv)
    if args.init_adapter is None:
        p.error('both experiments require --init-adapter with the shared frozen LoRA initialization')
    if args.experiment=='frozen' and args.arm!='joint_pair':
        p.error('experiment 1 is a matched Joint vs Mix comparison; omit --arm')
    if (args.train_limit<1 or args.eval_limit<1 or args.rollouts<2
        or args.rounds!=5 or args.generate_batch<1 or args.beta<0
        or args.max_new_tokens<1 or args.probe_candidates<1
        or not 1<=args.mix_grid<=100 or args.query_budget<3
        or args.max_feedback_items>64 or args.max_feedback_items<2):
        p.error('invalid frozen study dimensions')
    existing.previous.legacy._require_model_config(args.model_config)
    if not (args.init_adapter/'adapter_config.json').is_file():
        p.error('missing shared immutable LoRA adapter')
    rows,final,bycal,heldout,manifest=_load_data(args,include_audit=args.experiment=='frozen')
    manifest_sha=_hash(args.data_dir/'manifest.json')
    receipt={'schema':'eesd-joint-mix-plan-v1','experiment':args.experiment,
             'arm':args.arm if args.experiment=='train' else 'joint_vs_calibrated_mix',
             'status':'planned_not_executed','model_id':existing.previous.legacy.MODEL_ID,
             'model_revision':existing.previous.legacy.MODEL_REVISION,
             'dataset_manifest_sha256':manifest_sha,
             'train_distinct_problems':len(rows),'final_distinct_problems':len(final),
             'seed':args.seed,'rounds':args.rounds,'rollouts':args.rollouts,
             'feedback_pool':len(rows[0]['public_tests']),
             'selection_rule':'hashed_pair' if args.experiment=='frozen' or args.arm in ('joint_pair','mix_pair')
                else 'budgeted_true_joint_coalitions',
             'max_joint_teacher_queries':args.query_budget,
             'group_count':None,'calibration_tests':manifest['calibration_tests'],
             'audit_used_for_selection':False,
             'evaluation_used_for_selection':False,
             'scorer_note':'Existing strict-bwrap scorer; native RunBugRun output-equivalence audit must be reconciled before paper claims.',
             'initial_adapter_sha256':existing.previous._digest_adapter(args.init_adapter)
                if args.init_adapter else None,
             'args':{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}}
    if args.dry_run:
        print(json.dumps(receipt,indent=2,sort_keys=True));return
    if args.output.exists():raise FileExistsError('refuse to overwrite experiment results')
    smoke=run_test('print(47)',{'input':'','expected':'47'})
    if smoke.status!='PASS':raise RuntimeError('strict bubblewrap sandbox unavailable')
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('real BF16 CUDA GPU required')
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    _,tokenizer,attach=existing.previous.legacy._models(args)
    args.output.mkdir(parents=True,exist_ok=False)
    receipt['status']='executing';_save(args.output/'plan.json',receipt)
    if args.experiment=='frozen':
        teacher=attach(args.init_adapter,trainable=False)
        result=run_frozen(args,teacher,tokenizer,rows,bycal,heldout,manifest)
        del teacher
    else:
        result=run_train(args,tokenizer,attach,rows,final,bycal,manifest)
    _save(args.output/'completion.json',result)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
