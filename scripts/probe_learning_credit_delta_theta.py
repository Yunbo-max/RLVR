#!/usr/bin/env python3
"""Real LoRA parameter-update intervention: feedback -> delta(theta) -> meta correctness.

For each round-k *training-source* anchor, freeze the same pi_k checkpoint,
re-use exactly the SAME fixed code candidates and public feedback, and
reinitialize a separate student from pi_k for each feedback selector.
Perform ONE controlled LoRA optimizer update; evaluate changed STUDENT on a
fixed set of disjoint meta programming problems. Never use these meta results
to choose tests or adjust any hyperparameter within this run.

This is an OPTIONAL (GPU expensive) causal mechanism diagnostic, not the
final benchmark and not a guarantee of full training generalization.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_learning_credit_five_round as run
from eesd.learning_credit_selection import selection_from_arm
from eesd.strict_execution import feedback_for_tests, audit_vector

ARMS=('shapley','direct_max','random_subset','full_feedback')
def _json(path:Path,value)->None:
    with path.open('x') as fp:json.dump(value,fp,sort_keys=True,indent=2)


def _read_anchors(shapley_run:Path, round_number:int, train:list[dict], limit:int):
    if not 0<=round_number<=4 or limit<1:raise ValueError('invalid round/anchor limit')
    receipt=shapley_run/f'round_{round_number}'/'training.jsonl'
    if not receipt.is_file():raise FileNotFoundError(receipt)
    byid={r['source_id']:r for r in train}
    selected=[]
    for line in receipt.read_text().splitlines():
        record=json.loads(line)
        if record.get('step') is True:
            if record['source_id'] not in byid:raise ValueError('unknown trained source')
            if 'candidate_sources' not in record:raise ValueError('no frozen candidate provenance')
            selected.append((byid[record['source_id']],record))
    if not selected:raise ValueError('no Shapley-accepted training anchors in this round')
    # The training-selected anchors are chosen without ever looking at meta
    # performance. Stable order allows exact rerun checks.
    return selected[:limit]


def _cached_candidates(tokenizer,record,maximum:int)->list[tuple[list[int],str]]:
    out=[]
    for code in record['candidate_sources']:
        tokens=run._normalized_ids(tokenizer,code,maximum)
        if tokens:out.append((tokens,code))
    if len(out)<2:raise ValueError('not enough immutable candidate codes')
    return out


def _bundle(teacher,tokenizer,row,record,calibration_tests,args):
    candidates=_cached_candidates(tokenizer,record,args.max_new_tokens)
    fb=[feedback_for_tests(code,row['public_tests']) for _,code in candidates]
    cal=np.asarray([audit_vector(code,calibration_tests) for _,code in candidates])
    # Score same 16 subsets ONCE. All intervention arms reuse these arrays,
    # the underlying fixed code pool and the same frozen Qwen.
    scores=[run.score_candidate(teacher,tokenizer,row,tok,feedback,
         arm='shapley_select',max_prompt_tokens=args.max_prompt_tokens)
         for (tok,_),feedback in zip(candidates,fb)]
    return candidates,fb,cal,scores


def _one_step(student,teacher,tokenizer,row,candidates,feedback,calibration,
              scores,choice,args):
    import torch
    if choice.mask==0:return {'updated':False,'gradient_norm':0.,'reason':'selector_abstained'}
    if choice.rule=='full_feedback':
        weights=[np.ones(len(tok),dtype=np.float32) for tok,_ in candidates]
    else:
        weights=run.directional_token_weights(scores,calibration,choice.mask,
                                               shift_scale=args.shift_scale)
    if sum(float(w.sum()) for w in weights)<=1e-10:
        return {'updated':False,'gradient_norm':0.,'reason':'no_aligned_token'}
    optim=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
               lr=args.learning_rate,weight_decay=0.)
    optim.zero_grad(set_to_none=True)
    count=0
    for (tok,_),fb,w in zip(candidates,feedback,weights):
        if not np.any(w>0):continue
        tq,oldq=run.masked_teacher(teacher,tokenizer,row,tok,fb,choice.mask)
        ctx=run.legacy.prompt_ids(tokenizer,row)
        ids=torch.tensor([ctx+tok],dtype=torch.long,device=student.device)
        loss,_=run.distill_token_kl(student,oldq,tq,w,ids,len(ctx),beta=args.beta)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite controlled intervention loss')
        (loss/len(candidates)).backward()
        count+=1
        del loss,ids,tq,oldq
    if not count:
        return {'updated':False,'gradient_norm':0.,'reason':'all_candidates_zero_weight'}
    norm=torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],1.)
    if not torch.isfinite(norm):raise RuntimeError('nonfinite intervention gradient')
    if float(norm.detach().cpu()) <= 1e-10:
        return {'updated':False,'gradient_norm':float(norm.detach().cpu()),
                'reason':'zero_parameter_gradient'}
    optim.step()
    return {'updated':True,'gradient_norm':float(norm.detach().cpu()),
            'updated_candidates':count,'reason':'one_real_lora_step'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--shapley-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--round',type=int,default=0,choices=range(5))
    p.add_argument('--anchors',type=int,default=4)
    p.add_argument('--meta-limit',type=int,default=12)
    p.add_argument('--seed',type=int,default=1701)
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--max-prompt-tokens',type=int,default=8192)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--beta',type=float,default=.02)
    p.add_argument('--shift-scale',type=float,default=.15)
    p.add_argument('--audit-uncertainty',type=float,default=.5)
    p.add_argument('--subset-cost',type=float,default=.001)
    p.add_argument('--min-gain',type=float,default=.001)
    p.add_argument('--model-config',type=Path,default=ROOT/'configs/models/qwen25_7b.yaml')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    train,_,cal,manifest=run.load_matched_data(a.data_dir,manifest_count(a.data_dir,'train_sources'),1)
    meta=run.legacy.read_records(a.data_dir/'meta_probes_do_not_train.jsonl',
             {'source_id','problem_id','task','buggy_code','evaluation_tests'})
    if manifest.get('learning_credit_schema')!='eesd-five-round-frozen-data-v1':
        raise ValueError('wrong frozen meta protocol')
    if not 0<a.meta_limit<=len(meta):raise ValueError('meta population smaller than requested')
    if ({z['problem_id'] for z in meta} & {z['problem_id'] for z in train}):
        raise ValueError('intervention meta examples contaminate training')
    anchors=_read_anchors(a.shapley_run,a.round,train,a.anchors)
    plan=json.loads((a.shapley_run/'plan.json').read_text())
    if (plan.get('schema')!='eesd-learning-credit-five-v1'
        or plan.get('arm')!='shapley' or plan.get('seed')!=a.seed
        or plan.get('dataset_manifest_sha256')!=run.legacy.digest(a.data_dir/'manifest.json')):
        raise ValueError('wrong Shapley source experiment')
    snapshot=(a.shapley_run / f'round_{a.round-1}' / 'adapter'
              if a.round>0 else Path(plan['init_adapter']))
    if run._digest_adapter(snapshot)!=plan['initial_adapter_sha256'] and a.round==0:
        raise ValueError('baseline adapter checksum mismatch')
    jobs={'schema':'eesd-intervention-probe-v1','status':'dry_run',
          'round':a.round,'anchors':[row['source_id'] for row,_ in anchors],
          'meta_sources':[row['source_id'] for row in meta[:a.meta_limit]],
          'arms':ARMS,'checkpoint_path':str(snapshot),
          'candidate_source':'frozen Shapley training receipt; no sampling difference'}
    if a.dry_run:
        print(json.dumps(jobs,indent=2));return
    if a.output.exists():raise FileExistsError('refuse overwrite of intervention result')
    smoke=run.run_test('print(47)',{'input':'','expected':'47'})
    if smoke.status!='PASS':raise RuntimeError('secure sandbox unavailable')
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('requires real H100 BF16 GPU')
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    _,tokenizer,attach=run.legacy._models(a)
    a.output.mkdir(parents=True,exist_ok=False)
    jobs['status']='executing';_json(a.output/'plan.json',jobs)
    records=[]
    teacher=attach(snapshot,trainable=False)
    for row,reference in anchors:
        candidate,feedback,calibration,scores=_bundle(
            teacher,tokenizer,row,reference,cal[row['source_id']]['calibration_tests'],a)
        before=run._evaluate(teacher,tokenizer,meta[:a.meta_limit],a,a.round)
        before_bits={z['source_id']:int(z['all_pass']) for z in before['rows']}
        choices={
          'shapley':selection_from_arm('shapley',scores,calibration,
             round_index=a.round,source_id=row['source_id'],seed=a.seed),
          'direct_max':selection_from_arm('direct_max',scores,calibration,
             round_index=a.round,source_id=row['source_id'],seed=a.seed),
        }
        if choices['shapley'].mask != int(reference['selected_mask']):
            raise RuntimeError(
                "Shapley mask changed when replaying a fixed candidate/teacher "
                "snapshot; refuse incomparable parameter interventions")
        choices['random_subset']=selection_from_arm(
             'random_subset',scores,calibration,round_index=a.round,
             source_id=row['source_id'],seed=a.seed,
             reference_mask=choices['shapley'].mask)
        choices['full_feedback']=type(choices['shapley'])(
             mask=15,rule='full_feedback',j_calibration=None,
             phi=None,abstained=False,audit_contrast=None)
        for name,choice in choices.items():
            student=attach(snapshot,trainable=True)
            student.train()
            updated=_one_step(student,teacher,tokenizer,row,candidate,feedback,
                              calibration,scores,choice,a)
            after=run._evaluate(student,tokenizer,meta[:a.meta_limit],a,a.round)
            after_bits={z['source_id']:int(z['all_pass']) for z in after['rows']}
            if sorted(before_bits)!=sorted(after_bits):raise ValueError('meta source mismatch')
            gains=sum(not before_bits[s] and after_bits[s] for s in before_bits)
            regressions=sum(before_bits[s] and not after_bits[s] for s in before_bits)
            item={'source_id':row['source_id'],'round':a.round,
                'method':name,'selected_mask':choice.mask,
                'J_calibration':choice.j_calibration,
                'reference_shapley_mask':reference['selected_mask'],
                'update':updated,'meta_delta_all_tests_pass_at_1':
                  after['pass_at_1']-before['pass_at_1'],
                'meta_delta_mean_test_fraction':
                  after['mean_test_pass_fraction']-before['mean_test_pass_fraction'],
                'meta_fixes':gains,'meta_regressions':regressions,
                'meta_pass_at_1_before':before['pass_at_1'],
                'meta_pass_at_1_after':after['pass_at_1']}
            records.append(item)
            del student;gc.collect();torch.cuda.empty_cache()
    del teacher;gc.collect();torch.cuda.empty_cache()
    _json(a.output/'interventions.json',{'schema':'eesd-intervention-results-v1',
           'state':'completed','round':a.round,'meta_selected_by_training':False,
           'records':records})
    print(json.dumps({'output':str(a.output),'experiments':len(records),
                      'real_updates':sum(x['update']['updated'] for x in records)},indent=2))


def manifest_count(data_dir:Path,key:str)->int:
    data=json.loads((data_dir/'manifest.json').read_text())
    if data.get('learning_credit_schema')!='eesd-five-round-frozen-data-v1':
        raise ValueError('wrong meta-only data')
    return int(data[key])


if __name__=='__main__':main()
