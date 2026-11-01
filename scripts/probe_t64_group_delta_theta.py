#!/usr/bin/env python3
"""Real one-step Delta(theta) intervention on T64 -> four GROUP evidence inputs.

For accepted Shapley training anchors, freeze exactly one pre-update Qwen
checkpoint, re-use the SAME canonical code candidates and F64/C8 outcomes,
then compare one REAL LoRA optimizer step under Shapley groups, Direct-J,
random same-cardinality groups or all four evidence groups.

Generate fresh repairs on disjoint Meta problems and evaluate heldout tests.
This does NOT prove causality on the entire dataset; anchor selection is
conditioned on Shapley acceptance. Never choose feedback based on meta scores.
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
import run_t64_grouped_five_round as grouped
import probe_learning_credit_delta_theta as probe
from eesd.learning_credit_selection import selection_from_arm
from eesd.strict_execution import run_test,audit_vector

ARMS=('shapley','direct_max','random_subset','full_feedback')


def _frozen_bundle(teacher,tokenizer,row,record,calibration,args):
    candidates=probe._cached_candidates(tokenizer,record,args.max_new_tokens)
    transcript=[grouped._safe_feedback(code,row['public_tests'])
                for _,code in candidates]
    observed=[x[:args.exposed_feedback] for x in transcript]
    grouping=grouped.form_groups(row['public_tests'][:args.exposed_feedback],
          observed,groups=4,seed=args.seed,source_id=row['source_id'])
    fb=[grouped.compact_candidate_groups(grouping,v,
            per_group_chars=args.group_char_budget) for v in observed]
    cal=np.asarray([audit_vector(code,calibration) for _,code in candidates],dtype=float)
    scores=[grouped.previous.score_candidate(teacher,tokenizer,row,tokens,feedback,
        arm='shapley_select',max_prompt_tokens=args.max_prompt_tokens)
        for (tokens,_),feedback in zip(candidates,fb)]
    current_hashes=[hashlib.sha256(code.encode()).hexdigest() for _,code in candidates]
    if current_hashes!=record.get('candidate_code_sha256'):
        raise ValueError('replayed code candidates do not match frozen training receipt')
    return candidates,fb,cal,scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--shapley-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--round',type=int,choices=(0,1,2,3,4),default=0)
    p.add_argument('--anchors',type=int,default=16)
    p.add_argument('--meta-limit',type=int,default=64)
    p.add_argument('--seed',type=int,default=1701)
    p.add_argument('--model-config',type=Path,default=ROOT/'rlvr/models/qwen25_7b.yaml')
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--max-prompt-tokens',type=int,default=8192)
    p.add_argument('--group-char-budget',type=int,default=440)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--beta',type=float,default=.02)
    p.add_argument('--shift-scale',type=float,default=.15)
    p.add_argument('--audit-uncertainty',type=float,default=.5)
    p.add_argument('--subset-cost',type=float,default=.001)
    p.add_argument('--min-gain',type=float,default=.001)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    grouped.previous.legacy._require_model_config(args.model_config)
    plan=json.loads((args.shapley_run/'plan.json').read_text())
    if (plan.get('schema')!='eesd-rich64-five-v1' or plan.get('arm')!='shapley'
        or plan.get('rounds')!=5
        or plan.get('seed')!=args.seed
        or plan.get('dataset_manifest_sha256')!=grouped._sha(args.data_dir/'manifest.json')):
        raise ValueError('invalid seed/anchor model/dataset provenance')
    args.exposed_feedback=plan['exposed_feedback']
    args.group_char_budget=plan['feedback_group_chars']
    train,_,cal,manifest=grouped._load_data(
         args.data_dir,plan['train_bug_instances'],1)
    meta=grouped.previous.legacy.read_records(
          args.data_dir/'meta_do_not_train.jsonl',
          {'source_id','problem_id','task','buggy_code','evaluation_tests'})
    if not 0<args.meta_limit<=len(meta) or args.anchors<1:
        raise ValueError('insufficient meta sources or anchor count')
    if {v['problem_id'] for v in train}&{v['problem_id'] for v in meta}:
        raise ValueError('meta sources contaminated by training')
    anchors=probe._read_anchors(args.shapley_run,args.round,train,args.anchors)
    snapshot=(args.shapley_run/f'round_{args.round-1}'/'adapter'
              if args.round else Path(plan['init_adapter']))
    if not snapshot.is_dir():raise ValueError('pre-update Qwen checkpoint missing')
    metadata={'schema':'eesd-rich64-real-parameter-intervention-v1',
        'status':'dry_run','round':args.round,
        'shapley_preupdate_checkpoint':str(snapshot),
        'exposed_feedback_tests':args.exposed_feedback,
        'exact_evidence_group_players':4,
        'meta_source_ids':[x['source_id'] for x in meta[:args.meta_limit]],
        'accepted_training_anchors':[x['source_id'] for x,_ in anchors],
        'methods':list(ARMS),'selected_by_meta_score':False}
    if args.dry_run:
        print(json.dumps(metadata,indent=2));return
    if args.output.exists():raise FileExistsError('refuse to overwrite intervention')
    smoke=run_test('print(47)',{'input':'','expected':'47'})
    if smoke.status!='PASS':raise RuntimeError('strict bwrap isolation unavailable')
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('real BF16 GPU required')
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    _,tokenizer,attach=grouped.previous.legacy._models(args)
    args.output.mkdir(parents=True,exist_ok=False)
    metadata['status']='executing'
    (args.output/'plan.json').write_text(json.dumps(metadata,indent=2)+'\n')
    teacher=attach(snapshot,trainable=False)
    all_results=[]
    for row,receipt in anchors:
        candidates,feedback,calibration,scores=_frozen_bundle(teacher,tokenizer,row,
            receipt,cal[row['source_id']]['calibration_tests'],args)
        pre=grouped.previous._evaluate(teacher,tokenizer,meta[:args.meta_limit],
                                      args,args.round)
        preflags={x['source_id']:bool(x['all_pass']) for x in pre['rows']}
        choices={
            x:selection_from_arm(x,scores,calibration,
               seed=args.seed,round_index=args.round,source_id=row['source_id'],
               uncertainty=args.audit_uncertainty,subset_cost=args.subset_cost,
               min_gain=args.min_gain) for x in ('shapley','direct_max')}
        if choices['shapley'].mask!=int(receipt['selected_mask']):
            raise ValueError('replayed feedback group does not reproduce Shapley selection')
        choices['random_subset']=selection_from_arm(
            'random_subset',scores,calibration,seed=args.seed,
            round_index=args.round,source_id=row['source_id'],
            reference_mask=choices['shapley'].mask)
        choices['full_feedback']=type(choices['shapley'])(
            mask=15,rule='full_feedback',j_calibration=None,
            phi=None,abstained=False,audit_contrast=None)
        for method,choice in choices.items():
            student=attach(snapshot,trainable=True)
            student.train()
            update=probe._one_step(student,teacher,tokenizer,row,candidates,
                        feedback,calibration,scores,choice,args)
            post=grouped.previous._evaluate(student,tokenizer,
                                      meta[:args.meta_limit],args,args.round)
            postflags={x['source_id']:bool(x['all_pass']) for x in post['rows']}
            if set(preflags)!=set(postflags):
                raise ValueError('intervention changed meta evaluation population')
            result={'source_id':row['source_id'],'round':args.round,
                    'selector':method,'selected_group_mask':choice.mask,
                    'actual_update':update,'meta_pass_at_1_before':pre['pass_at_1'],
                    'meta_pass_at_1_after':post['pass_at_1'],
                    'meta_pass_at_1_delta':post['pass_at_1']-pre['pass_at_1'],
                    'meta_test_pass_delta':post['mean_test_pass_fraction']-pre['mean_test_pass_fraction'],
                    'meta_fixes':sum(not preflags[k] and postflags[k] for k in preflags),
                    'meta_regressions':sum(preflags[k] and not postflags[k] for k in preflags)}
            all_results.append(result)
            del student;gc.collect();torch.cuda.empty_cache()
    del teacher;gc.collect();torch.cuda.empty_cache()
    (args.output/'interventions.json').write_text(json.dumps({
        'schema':'eesd-rich64-intervention-results-v1','status':'completed',
        'measure':'actual delta theta -> fresh disjoint meta problem correctness',
        'records':all_results},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'experiments':len(all_results),
         'real_updates':sum(x['actual_update']['updated'] for x in all_results)},indent=2))


if __name__=='__main__':main()
