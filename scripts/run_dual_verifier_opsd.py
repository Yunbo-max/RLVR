#!/usr/bin/env python3
"""Dual Recursive OPSD: executable-verifier synthesis, anchored reflection,
and solver distillation on the same current-round code-policy rollouts.

Research implementation, not scientific validation. Real training requires a
locally cached pinned Qwen model/LoRA, a CUDA BF16 GPU, and mandatory bwrap.
Every role has a separate LoRA adapter; protected data never enters prompts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
import gc
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
import run_joint_mix as previous  # reuse pinned RunBugRun cohort, eval and model setup
from eesd.dual_verifier import (Audit,VerdictCase,make_cases,split_tests,execute_verifier,
    audit_verdicts,admit,verifier_training_mistakes,choose_candidate,verifier_source_sanity,
    safe_feedback_for_solver,oracle_verdict_feedback)
from eesd.strict_execution import run_test,run_tests_batched
from eesd.dual_verifier_loss import reverse_kl_from_logits

ARMS=('dual','solver_fixed','solver_oracle_verdict','solver_static_verifier','solver_regenerated_verifier','verifier_only')


def _save_json(path:Path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(payload,f,sort_keys=True,ensure_ascii=False,indent=2)
        f.write('\n')


def _sha_file(path:Path)->str:
    h=sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def _json_line(f,payload):
    f.write(json.dumps(payload,ensure_ascii=False,sort_keys=True)+'\n')
    f.flush()


def _hash_seed(*parts)->int:
    return int(sha256(':'.join(map(str,parts)).encode()).hexdigest()[:8],16)


def _prompt_ids(tokenizer,messages)->list[int]:
    chat=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    return list(tokenizer(chat,add_special_tokens=False)['input_ids'])


def verifier_messages(row:dict, *, mistakes:list[dict]|None=None,
                      memory:list[dict]|None=None)->list[dict]:
    """Student prompt lacks current reference labels; privileged teacher gets mistakes."""
    task=row['task']
    content=(
        'Design an executable, general-purpose Python verifier for the programming problem. '
        'Return a complete Python script only. Define EXACTLY this function:\n'
        'def verify(input_text: str, candidate_output: str) -> bool:\n'
        'It must determine whether candidate_output is a valid answer to the task on '
        'input_text. Do not depend on pre-recorded test answers, hidden files, tools, '
        'network access, or additional arguments. Avoid hard-coding examples. '
        'Valid alternative outputs should be accepted. '
        'Only use safe Python standard-library modules.\n\n'
        'Problem:\n'+str(task)
    )
    if memory:
        content+='\n\nPreviously found TRAINING mistakes (not held-out):\n'+json.dumps(memory[:4],ensure_ascii=False)
    if mistakes is not None:
        content+='\n\nTrusted evaluator discovered these mistakes in earlier generated verifier code. '\
                 'Use them to revise the VERIFICATION LOGIC, not the task semantics:\n'+json.dumps(mistakes[:8],ensure_ascii=False)
    return [
        {'role':'system','content':'You are an expert in writing reliable executable Python verification functions.'},
        {'role':'user','content':content},
    ]


def sample_verifiers(model,tokenizer,row,args,rnd:int,*,memory=None)->list[tuple[list[int],str]]:
    import torch
    ids=_prompt_ids(tokenizer,verifier_messages(row,memory=memory))
    if len(ids)>args.max_prompt_tokens:raise ValueError('verifier prompt over token limit')
    inp=torch.tensor([ids],dtype=torch.long,device=model.device)
    torch.manual_seed(_hash_seed(args.seed,rnd,row['source_id'],'verifier'))
    if torch.cuda.is_available():torch.cuda.manual_seed_all(_hash_seed(args.seed,rnd,row['source_id'],'verifier'))
    collected=[]
    for off in range(0,args.verifier_proposals,args.generate_batch):
        count=min(args.generate_batch,args.verifier_proposals-off)
        with torch.no_grad():
            out=model.generate(inp,attention_mask=torch.ones_like(inp),do_sample=True,
                temperature=args.temperature,top_p=.95,max_new_tokens=args.max_verifier_tokens,
                num_return_sequences=count,pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
        for seq in out:
            raw=previous.existing.previous.legacy.response_token_ids(
                seq[len(ids):].tolist(),tokenizer.eos_token_id,tokenizer.pad_token_id)
            source=previous.existing.previous.legacy._extract_python(
                tokenizer.decode(raw,skip_special_tokens=True))
            collected.append((raw,source))
    return collected


def _logq(model, context_ids:list[int], response_ids:list[int]):
    """Full teacher distribution for ONE on-policy response. No corpus cache."""
    import torch
    import torch.nn.functional as F
    ids=torch.tensor([context_ids+response_ids],dtype=torch.long,device=model.device)
    with torch.no_grad():
        logits=model(input_ids=ids,use_cache=False).logits[0,len(context_ids)-1:-1].float()
        dist=F.log_softmax(logits,dim=-1).detach()
    return dist


def _distill_response(teacher,student,tokenizer,plain_messages,privileged_messages,
                      response_ids,opt,*,beta:float,max_prompt_tokens:int)->dict:
    """Actual token-level reverse KL, not an SFT imitation or scalar RL reward."""
    import torch
    plain=_prompt_ids(tokenizer,plain_messages)
    private=_prompt_ids(tokenizer,privileged_messages)
    if max(len(plain),len(private))>max_prompt_tokens:
        return {'step':False,'skip':'teacher_or_student_context_over_budget'}
    if not response_ids:
        return {'step':False,'skip':'empty_response'}
    q0=_logq(teacher,plain,response_ids)
    qp=_logq(teacher,private,response_ids)
    full=torch.tensor([plain+response_ids],dtype=torch.long,device=student.device)
    opt.zero_grad(set_to_none=True)
    logits=student(input_ids=full,use_cache=False).logits[0,len(plain)-1:-1].float()
    loss,stats=reverse_kl_from_logits(logits,q0,qp,beta=beta)
    if not torch.isfinite(loss):raise RuntimeError('nonfinite verifier/solver OPSD loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],1.)
    if not torch.isfinite(norm):raise RuntimeError('nonfinite OPSD gradient')
    if float(norm.detach().cpu())<=1e-10:
        return {'step':False,'skip':'zero_gradient','grad_norm':float(norm.detach().cpu())}
    opt.step()
    return {'step':True,'grad_norm':float(norm.detach().cpu()),'loss':float(loss.detach().cpu()),
            'token_count':len(response_ids),**stats}


def _safe_own_feedback(code:str,tests:list[dict]):
    return previous.existing._safe_feedback(code,tests)


def _sample_solver(teacher,tokenizer,row,args,rnd):
    # Source tokens are the ACTUAL on-policy model continuation; execute the
    # extracted Python separately. No re-tokenizing canonical source as rollout.
    return previous.existing.previous.legacy._sample_rollouts(teacher,tokenizer,row,args,rnd)


def _actual_negative_outputs(codes:list[tuple[list[int],str]], tests:list[dict]):
    """One mandatory-bwrap execution batch per rollout; return both transcripts
    and genuine WRONG_OUTPUT negatives keyed by public test ID."""
    negatives=defaultdict(list)
    transcripts=[]
    for _,code in codes:
        out=_safe_own_feedback(code,tests)
        transcripts.append(out)
        for entry in out:
            if entry['outcome']=='WRONG_OUTPUT' and entry['actual'].strip()!=entry['expected'].strip():
                negatives[str(entry['id'])].append(entry['actual'])
    return dict(negatives),transcripts


def _task_verifier_rows(teacher,tokenizer,row,solver_samples,args,rnd:int,*,memory=None):
    a,b,c=split_tests(row['public_tests'],seed=args.seed,source_id=row['source_id'],
                     train=args.verifier_train_tests,gate=args.verifier_gate_tests,
                     solver=args.solver_feedback_tests)
    # Generated alternatives can include actual wrong on-policy outputs.
    negatives,_=_actual_negative_outputs(solver_samples,a+b)
    train_cases=make_cases(a,seed=args.seed,negatives=negatives,tag='train:')
    gate_cases=make_cases(b,seed=args.seed,negatives=negatives,tag='gate:')
    proposals=sample_verifiers(teacher,tokenizer,row,args,rnd,memory=memory)
    records=[]; scoring=[]
    for i,(tokens,source) in enumerate(proposals):
        okay,reason=verifier_source_sanity(source)
        pred_train=execute_verifier(source,train_cases) if okay else [None]*len(train_cases)
        pred_gate=execute_verifier(source,gate_cases) if okay else [None]*len(gate_cases)
        train_audit=audit_verdicts(train_cases,pred_train)
        gate_audit=audit_verdicts(gate_cases,pred_gate)
        corrections=verifier_training_mistakes(source,okay,reason,train_cases,pred_train,
                                               limit=args.error_examples)
        r={'source_id':row['source_id'],'problem_id':row['problem_id'],
           'proposal':i,'verifier_code_sha256':sha256(source.encode()).hexdigest(),
           'verifier_code':source,'response_ids':tokens,'static_valid':okay,'static_reason':reason,
           'train_audit':asdict(train_audit),'gate_audit':asdict(gate_audit),
           'train_mistakes':corrections}
        records.append(r);scoring.append((source,gate_audit))
    selected=choose_candidate(scoring,max_fpr=args.max_false_accept_rate,
        max_fnr=args.max_false_reject_rate,min_per_class=args.min_per_class) if records else None
    # Truth for solver is not revised based on selected verifier guesses.
    return records,selected,c


def _trusted_feedback_as_teacher(code,tests):
    records=_safe_own_feedback(code,tests)
    return [{k:v for k,v in rec.items() if k!='stderr' or v} for rec in records]


def _verifier_feedback_as_teacher(code,tests,source):
    return safe_feedback_for_solver(source,code,tests)


def _torch_models(args):
    torch,tokenizer,attach=previous.existing.previous.legacy._models(args)
    return torch,tokenizer,attach


def _unload(*models):
    import torch
    for model in models:del model
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def _evaluate_solver(attach,adapter,tokenizer,eval_rows,args,rnd:int):
    model=attach(adapter,trainable=False)
    try:
        return previous.existing.previous._evaluate(model,tokenizer,eval_rows,args,rnd)
    finally:
        del model;gc.collect()
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()


def _load(args):
    # Historical cohort has a fully hashed manifest, source-ID integrity and
    # problem-ID disjointness.  We additionally forbid using the reserved
    # verifier audit data to gate or train candidate verifiers.
    rows,final,bycal,audit,manifest=previous._load_data(args,include_audit=True)
    if len(rows)!=args.train_limit:
        raise AssertionError('unexpected training cohort size')
    for row in rows:
        public_inputs={str(x['input']) for x in row['public_tests']}
        reserved=audit[row['source_id']]['audit_tests']
        if public_inputs&{str(x['input']) for x in reserved}:
            raise ValueError('reserved audit/public overlap')
        if (args.verifier_train_tests+args.verifier_gate_tests+args.solver_feedback_tests
            >len(row['public_tests'])):
            raise ValueError('not enough public tests for verifier experiment')
    return rows,final,bycal,audit,manifest


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--init-adapter',type=Path,required=True,
                   help='Shared initial LoRA snapshot; roles use distinct updated adapters')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--model-config',type=Path,default=ROOT/'configs/models/qwen25_7b.yaml')
    p.add_argument('--seed',type=int,default=1701)
    p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--train-limit',type=int,default=32)
    p.add_argument('--eval-limit',type=int,default=32)
    p.add_argument('--rollouts',type=int,default=4,help='Solver on-policy programs per problem')
    p.add_argument('--verifier-proposals',type=int,default=2)
    p.add_argument('--verifier-train-tests',type=int,default=8)
    p.add_argument('--verifier-gate-tests',type=int,default=8)
    p.add_argument('--solver-feedback-tests',type=int,default=8)
    p.add_argument('--max-false-accept-rate',type=float,default=0.125)
    p.add_argument('--max-false-reject-rate',type=float,default=0.25)
    p.add_argument('--min-per-class',type=int,default=4)
    p.add_argument('--error-examples',type=int,default=6)
    p.add_argument('--max-verifier-updates-per-problem',type=int,default=1)
    p.add_argument('--memory-limit',type=int,default=0)
    p.add_argument('--missing-verifier',choices=('skip','fixed'),default='skip',
                   help='Generated-verifier arms: skip when no verifier is admitted; fixed is explicit oracle-backed ablation')
    p.add_argument('--max-verifier-tokens',type=int,default=256)
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--max-prompt-tokens',type=int,default=8192)
    p.add_argument('--generate-batch',type=int,default=2)
    p.add_argument('--temperature',type=float,default=.7)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--beta',type=float,default=.02)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true',help='Resume only after complete, sealed rounds')
    return p


def _check_saved_adapter_integrity(receipt:dict,digest_fn)->None:
    """Resume must verify both adapter trees, not merely path existence."""
    for role in ('solver','verifier'):
        path=Path(receipt[role+'_adapter'])
        if not path.is_dir():
            raise FileNotFoundError('missing '+role+' round adapter')
        if receipt.get(role+'_adapter_sha256') != digest_fn(path):
            raise ValueError(role+' round adapter hash mismatch')


def _validate(args):
    if (args.rounds<1 or args.train_limit<1 or args.eval_limit<1 or
        args.rollouts<2 or args.verifier_proposals<1 or
        min(args.verifier_train_tests,args.verifier_gate_tests,args.solver_feedback_tests)<1 or
        args.min_per_class<1 or args.max_verifier_updates_per_problem<1 or
        args.max_new_tokens<1 or args.max_verifier_tokens<1 or
        args.generate_batch<1 or not 0<=args.max_false_accept_rate<=1 or
        not 0<=args.max_false_reject_rate<=1 or args.error_examples<1 or
        not 0<=args.memory_limit<=16 or args.beta<0 or
        not 0<args.learning_rate<1 or not 0<args.temperature<=2 or
        args.max_prompt_tokens<1):
        raise ValueError('invalid dual recursive experiment configuration')
    previous.existing.previous.legacy._require_model_config(args.model_config)
    if not (args.init_adapter/'adapter_config.json').is_file():
        raise FileNotFoundError('missing shared LoRA adapter; run create_learning_credit_init.py')


def main(argv=None):
    args=build_parser().parse_args(argv);_validate(args)
    rows,final,bycal,audit,manifest=_load(args)
    receipt={'schema':'dual-recursive-verifier-opsd-v1',
      'status':'planned_not_executed','arm':args.arm,
      'dataset_sha256':_sha_file(args.data_dir/'manifest.json'),
      'init_adapter_sha256':previous.existing.previous._digest_adapter(args.init_adapter),
      'model_revision':previous.existing.previous.legacy.MODEL_REVISION,
      'train_distinct_problem_ids':[x['problem_id'] for x in rows],
      'eval_distinct_problem_ids':[x['problem_id'] for x in final],
      'training_rounds':args.rounds,'gold_source_code_used':False,
      'verifier_audit_used_for_selection':False,
      'verifier_programs_sandboxed':True,
      'scorer_limit':'pinned local stdout comparator; official RunBugRun equivalence unqualified',
      'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('resume','dry_run')}}
    if args.dry_run:
        print(json.dumps(receipt,ensure_ascii=False,indent=2));return
    probe=run_test('print(47)',{'input':'','expected':'47'})
    if probe.status!='PASS':raise RuntimeError('bubblewrap verification failed')
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('requires real BF16 CUDA GPU')
    # Existing model setup loads a frozen pinned checkpoint and LoRA.
    _,tokenizer,attach=_torch_models(args)
    if args.output.exists() and not args.resume:
        raise FileExistsError('existing output: pass --resume or change path')
    if args.resume:
        old=json.loads((args.output/'plan.json').read_text())
        expected={**receipt,'status':'executing'}
        if old!=expected:raise ValueError('resume plan mismatch')
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        _save_json(args.output/'plan.json',{**receipt,'status':'executing'})
    verifier_adapter=args.init_adapter
    solver_adapter=args.init_adapter
    frozen_verifiers={}
    memories={}
    start_round=0
    if args.resume:
        for rnd in range(args.rounds):
            p=args.output/f'round_{rnd}'/'round_receipt.json'
            if not p.exists():break
            data=json.loads(p.read_text())
            if data.get('status')!='completed':raise ValueError('non-complete round receipt')
            solver_adapter=Path(data['solver_adapter'])
            verifier_adapter=Path(data['verifier_adapter'])
            _check_saved_adapter_integrity(data,previous.existing.previous._digest_adapter)
            frozen_verifiers=data.get('frozen_verifiers',{})
            memories=data.get('training_memories',{})
            start_round=rnd+1
        if start_round==args.rounds and (args.output/'completion.json').exists():
            print((args.output/'completion.json').read_text());return
        if (args.output/f'round_{start_round}').exists():
            raise FileExistsError('partial round exists; manually inspect before resume')
    for rnd in range(start_round,args.rounds):
        start=time.perf_counter()
        folder=args.output/f'round_{rnd}';folder.mkdir(exist_ok=False)
        # PHASE 1: sample SOLVER programs from exactly current solver policy.
        model_s=attach(solver_adapter,trainable=False)
        try:
            evaluation=previous.existing.previous._evaluate(model_s,tokenizer,final,args,rnd)
            _save_json(folder/'evaluation.json',evaluation)
            solver_rollouts={r['source_id']:_sample_solver(model_s,tokenizer,r,args,rnd) for r in rows}
        finally:
            del model_s;gc.collect();torch.cuda.empty_cache()
        # PHASE 2: verifier proposes programs, audit misjudgments, distill its own
        # ON-POLICY continuation tokens from the privileged mistake-conditioned
        # self-teacher. Independent reserved audit is read only AFTER selection.
        selected_verifiers={};solver_tests={};train_proposals={};new_memories={}
        if args.arm not in ('solver_fixed','solver_oracle_verdict'):
            model_v=attach(verifier_adapter,trainable=False)
            update_v=args.arm in ('dual','verifier_only')
            student_v=attach(verifier_adapter,trainable=True) if update_v else None
            if update_v:student_v.train()
            opt_v=torch.optim.AdamW([p for p in student_v.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=0.) if update_v else None
            vf_path=folder/'verifier_proposals.jsonl'
            with vf_path.open('x',encoding='utf-8') as fp:
                for row in rows:
                    sid=row['source_id']
                    _,_,ft=split_tests(row['public_tests'],seed=args.seed,source_id=sid,
                      train=args.verifier_train_tests,gate=args.verifier_gate_tests,
                      solver=args.solver_feedback_tests)
                    solver_tests[sid]=ft
                    if args.arm=='solver_static_verifier' and rnd>0:
                        selected_verifiers[sid]=frozen_verifiers.get(sid)
                        _json_line(fp,{'source_id':sid,'status':'frozen_verifier_reused',
                             'selected_code_sha256':sha256((selected_verifiers[sid] or '').encode()).hexdigest()})
                        continue
                    rec,chosen,ft=_task_verifier_rows(model_v,tokenizer,row,
                                solver_rollouts[sid],args,rnd,memory=memories.get(sid))
                    solver_tests[sid]=ft
                    new_memories[sid]=(rec[chosen]['train_mistakes'][:args.memory_limit] if chosen is not None else
                             (rec[0]['train_mistakes'][:args.memory_limit] if rec else []))
                    selected_verifiers[sid]=rec[chosen]['verifier_code'] if chosen is not None else None
                    # Independent verifier audit is strictly reporting-only.
                    independent=make_cases(audit[sid]['audit_tests'],seed=args.seed,
                                           tag='reserved_audit:')
                    audits=[]
                    for item in rec:
                        audited=audit_verdicts(independent,execute_verifier(item['verifier_code'],independent))
                        audits.append(asdict(audited))
                    v_updates=0
                    for i,item in enumerate(rec):
                        record={**item,'selected_for_solver':i==chosen,
                                'reserved_audit_reporting_only':audits[i]}
                        _json_line(fp,record)
                        if update_v and v_updates<args.max_verifier_updates_per_problem and item['train_mistakes']:
                            plain=verifier_messages(row,memory=memories.get(sid))
                            privileged=verifier_messages(row,memory=memories.get(sid),
                                                  mistakes=item['train_mistakes'])
                            result=_distill_response(model_v,student_v,tokenizer,plain,
                                privileged,item['response_ids'],opt_v,beta=args.beta,
                                max_prompt_tokens=args.max_prompt_tokens)
                            _json_line(fp,{'source_id':sid,'proposal':i,'verifier_opsd_update':result})
                            if result.get('step'):v_updates+=1
            if update_v:
                next_v=folder/'verifier_adapter'
                student_v.save_pretrained(next_v)
            else:next_v=verifier_adapter
            del model_v,student_v,opt_v;gc.collect();torch.cuda.empty_cache()
        else:
            next_v=verifier_adapter
            for row in rows:
                sid=row['source_id']
                _,_,ft=split_tests(row['public_tests'],seed=args.seed,source_id=sid,
                         train=args.verifier_train_tests,gate=args.verifier_gate_tests,
                         solver=args.solver_feedback_tests)
                solver_tests[sid]=ft
        # PHASE 3: student solver distils current frozen solver as a privileged
        # feedback-conditioned teacher; verifier accepts are finite-sample only.
        update_s=args.arm!='verifier_only'
        if update_s:
            model_s=attach(solver_adapter,trainable=False)
            student_s=attach(solver_adapter,trainable=True)
            student_s.train()
            opt_s=torch.optim.AdamW([p for p in student_s.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=0.)
            with (folder/'solver_updates.jsonl').open('x',encoding='utf-8') as f:
                for row in rows:
                    sid=row['source_id'];source=selected_verifiers.get(sid)
                    for i,(tokens,code) in enumerate(solver_rollouts[sid]):
                        if not tokens or len(code)>16000 or not code.strip():
                            _json_line(f,{'source_id':sid,'rollout':i,'step':False,'skip':'invalid_code'});continue
                        tests=solver_tests[sid]
                        if args.arm=='solver_fixed':
                            feedback=_trusted_feedback_as_teacher(code,tests)
                            disagreement={}
                            feedback_source='fixed_full_oracle'
                        elif args.arm=='solver_oracle_verdict':
                            feedback=oracle_verdict_feedback(code,tests)
                            disagreement={}
                            feedback_source='oracle_verdict_information_matched'
                        elif source:
                            feedback,disagreement=_verifier_feedback_as_teacher(code,tests,source)
                            feedback_source='generated_audit_gated_verifier'
                            if all(x['verifier_verdict']=='ABSTAIN' for x in feedback):
                                _json_line(f,{'source_id':sid,'problem_id':row['problem_id'],
                                     'rollout':i,'step':False,'source':feedback_source,
                                     'skip':'all_generated_verdicts_abstained'})
                                continue
                        else:
                            if args.missing_verifier=='skip':
                                _json_line(f,{'source_id':sid,'problem_id':row['problem_id'],
                                     'rollout':i,'step':False,'source':'no_admitted_verifier',
                                     'skip':'no_admitted_verifier'})
                                continue
                            feedback=_trusted_feedback_as_teacher(code,tests)
                            disagreement={}
                            feedback_source='fixed_trusted_fallback_opt_in'
                        base=previous.existing.previous.legacy.prompt_messages(row)
                        priv=previous.existing.previous.legacy.prompt_messages(row,feedback)
                        result=_distill_response(model_s,student_s,tokenizer,base,priv,
                              tokens,opt_s,beta=args.beta,max_prompt_tokens=args.max_prompt_tokens)
                        _json_line(f,{'source_id':sid,'problem_id':row['problem_id'],
                           'rollout':i,'source':feedback_source,
                           'generated_code_sha256':sha256(code.encode()).hexdigest(),
                           'feedback_diagnostics':disagreement,'update':result})
            next_s=folder/'solver_adapter'
            student_s.save_pretrained(next_s)
            del model_s,student_s,opt_s;gc.collect();torch.cuda.empty_cache()
        else:next_s=solver_adapter
        if args.arm=='solver_static_verifier' and rnd==0:
            frozen_verifiers={k:v for k,v in selected_verifiers.items() if v}
        # If verifier role updated, next-round prompts include only TRAINING
        # mistakes, never gate or reserved-audit cases.
        if args.arm in ('dual','verifier_only'):
            memories.update(new_memories)
        result={'schema':'dual-opsd-round-v1','status':'completed','round':rnd,
           'arm':args.arm,'solver_adapter':str(next_s.resolve()),
           'verifier_adapter':str(next_v.resolve()),
           'frozen_verifiers':frozen_verifiers,
           'training_memories':memories,
           'solver_adapter_sha256':previous.existing.previous._digest_adapter(next_s),
           'verifier_adapter_sha256':previous.existing.previous._digest_adapter(next_v),
           'accepted_verifiers':sum(bool(v) for v in selected_verifiers.values()),
           'train_tasks':len(rows),'solver_candidate_count':sum(len(x) for x in solver_rollouts.values()),
           'elapsed_seconds':time.perf_counter()-start}
        _save_json(folder/'round_receipt.json',result)
        solver_adapter=next_s;verifier_adapter=next_v
    final_eval=_evaluate_solver(attach,solver_adapter,tokenizer,final,args,args.rounds)
    (args.output/f'round_{args.rounds}').mkdir(exist_ok=False)
    _save_json(args.output/f'round_{args.rounds}'/'evaluation.json',final_eval)
    completed={'schema':'dual-recursive-verifier-opsd-v1','status':'completed',
       'arm':args.arm,'rounds':args.rounds,'final_solver_adapter':str(solver_adapter),
       'final_verifier_adapter':str(verifier_adapter),
       'final_evaluation':str(args.output/f'round_{args.rounds}'/'evaluation.json')}
    _save_json(args.output/'completion.json',completed)
    print(json.dumps(completed,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
