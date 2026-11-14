from pathlib import Path
import importlib.util
import json
import sys
import pytest

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dual_summary',root/'scripts'/'summarize_dual_verifier_opsd.py')
summary=importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_paired_bootstrap_and_identity():
    x=summary.paired_bootstrap({'a':1.,'b':0.,'c':1.},
                               {'a':0.,'b':0.,'c':0.},seed=15,n_boot=200)
    assert x['n_problems']==3
    assert abs(x['difference']-2/3)<1e-10
    assert x['bootstrap_95'][0]<=x['difference']<=x['bootstrap_95'][1]
    with pytest.raises(ValueError):
        summary.paired_bootstrap({'a':1.,'b':1.},{'a':0.,'c':0.})


def _fake_run(tmp_path,arm,seed,initial='same',flip=False):
    p=tmp_path/f'{arm}-{seed}';p.mkdir()
    plan={'status':'executing','arm':arm,'args':{'seed':seed},
          'init_adapter_sha256':initial,'dataset_sha256':'dataset',
          'model_revision':'rev','train_distinct_problem_ids':['a','b']}
    (p/'plan.json').write_text(json.dumps(plan))
    (p/'completion.json').write_text(json.dumps({'status':'completed','rounds':1}))
    for r in range(2):
        d=p/f'round_{r}';d.mkdir()
        values=[{'problem_id':'x','all_pass':r==1 and (arm=='dual')},
                {'problem_id':'y','all_pass':True}]
        (d/'evaluation.json').write_text(json.dumps({'rows':values}))
    (p/'round_0'/'round_receipt.json').write_text(json.dumps({
        'accepted_verifiers':1,'solver_candidate_count':2,'elapsed_seconds':.1}))
    return p


def test_real_run_file_contract_and_contrast(tmp_path):
    a=_fake_run(tmp_path,'dual',1701)
    b=_fake_run(tmp_path,'solver_fixed',1701)
    r=summary.summarize([a,b])
    assert len(r['contrasts'])==1
    assert r['contrasts'][0]['final']['difference']==.5


def test_reject_different_initial_adapters(tmp_path):
    a=_fake_run(tmp_path,'dual',1701,initial='a')
    b=_fake_run(tmp_path,'solver_fixed',1701,initial='b')
    with pytest.raises(ValueError):summary.summarize([a,b])
