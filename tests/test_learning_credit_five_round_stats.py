import importlib.util
import json
from pathlib import Path
import pytest

p=Path(__file__).resolve().parents[1]/'scripts'/'summarize_learning_credit_five_round.py'
spec=importlib.util.spec_from_file_location('summarize_learning_credit_five_round',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def _tree(root):
    names=['a','b','c','d']
    for arm in m.ARMS:
        folder=root/arm;folder.mkdir(parents=True)
        plan={'schema':'eesd-learning-credit-five-v1','arm':arm,'rounds':5,
           'model':'Qwen','revision':'abcdef','seed':1701,'rollouts':8,
           'train_sources':5,'eval_sources':len(names),
           'feedback_tests':4,'calibration_budget_per_rollout':4,
           'dataset_manifest_sha256':'dataset','initial_adapter_sha256':'initial',
           'isolation':'bubblewrap_mandatory_no_soft_fallback',
           'max_new_tokens':256,'max_prompt_tokens':8192,
           'temperature':.7,'beta':.02,'learning_rate':1e-5}
        (folder/'plan.json').write_text(json.dumps(plan))
        for rnd in range(6):
            rd=folder/f'round_{rnd}';rd.mkdir()
            successes=[1,0,0,1] if rnd==0 or arm=='no_update' else (
                [1,1,0,1] if arm=='shapley' else [1,0,0,1] if arm=='direct_max'
                else [0,0,0,1] if arm=='sdpo' else [1,0,0,1])
            data={'schema':'eesd-behavioral-fresh-eval-v1','round':rnd,
                 'rows':[{'source_id':name,'all_pass':bool(successes[i]),
                  'passes':int(successes[i]),'total_tests':1,
                  'code_sha256':f'initial{i}' if rnd==0 or arm=='no_update'
                   else f'{arm}-{rnd}-{i}'} for i,name in enumerate(names)]}
            (rd/'evaluation.json').write_text(json.dumps(data))
            if rnd<5 and arm!='no_update':
                d={'optimizer_steps':2,'scheduled_optimizer_steps':2,
                   'selected_masks':[1,1,0,0],'abstention_rate':.5}
                (rd/'summary.json').write_text(json.dumps(d))
                if arm in ('shapley','direct_max'):
                    receipts=[{'source_id':name,'step':i==1,
                       'selected_mask':1 if i==1 else 0,
                       'j_calibration':.10 if i==1 and arm=='shapley'
                           else .15 if i==1 else None,
                       'candidate_code_sha256':['c1','c2']}
                       for i,name in enumerate(names)]
                    (rd/'training.jsonl').write_text(
                        '\n'.join(json.dumps(v) for v in receipts)+'\n')


def test_paired_final_and_realized_matching(tmp_path):
    _tree(tmp_path)
    x=m.summarize(tmp_path,bootstrap=200)
    assert x['primary_comparisons']['shapley_vs_direct']['difference']==pytest.approx(.25)
    assert x['primary_comparisons']['shapley_vs_direct']['exact_mcnemar_two_sided_p']==1.
    assert x['primary_comparisons']['random_subset_update_matching']['exactly_matched']
    assert len(x['arms']['shapley'])==6


def test_refuse_initial_checkpoint_mismatch(tmp_path):
    _tree(tmp_path)
    p=tmp_path/'direct_max/round_0/evaluation.json'
    r=json.loads(p.read_text());r['rows'][0]['code_sha256']='unexpected'
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='initial generated code'):
        m.summarize(tmp_path,bootstrap=200)


def test_detect_mismatched_gpu_protocol(tmp_path):
    _tree(tmp_path)
    p=tmp_path/'direct_max/plan.json'
    r=json.loads(p.read_text());r['rollouts']=32;p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='unmatched experimental budget'):
        m.summarize(tmp_path,bootstrap=200)


def test_mcnemar_two_sided_small_effect_not_claimed_significant():
    import numpy as np
    a=np.array([1,1,0,0]);b=np.array([0,0,0,0])
    draws=np.tile(np.arange(4),(200,1))
    x=m._paired(a,b,draws)
    assert x['wins']==2 and x['losses']==0
    assert x['exact_mcnemar_two_sided_p']==pytest.approx(.5)
