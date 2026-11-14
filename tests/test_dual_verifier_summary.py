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
        'accepted_verifiers':1,'train_tasks':2,'solver_candidate_count':2,'elapsed_seconds':.1}))
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


def test_summary_counts_explicit_skip_and_never_mislabels_oracle_sources(tmp_path):
    a=_fake_run(tmp_path,'dual',1701)
    file=a/'round_0'/'solver_updates.jsonl'
    file.write_text('\n'.join(map(json.dumps,[
        {'source':'no_admitted_verifier','skip':'no_admitted_verifier'},
        {'source':'generated_audit_gated_verifier','update':{'step':True}},
        {'source':'fixed_trusted_fallback_opt_in','update':{'step':True}}]))+'\n')
    metrics=summary.load_one(a)['metrics'][0]
    assert metrics['solver_fixed_feedback_fraction']==1/3
    assert metrics['missing_verifier_skip_fraction']==1/3
    assert metrics['generated_verifier_rollout_fraction']==1/3


def test_unselected_verifier_quality_still_counts(tmp_path):
    a=_fake_run(tmp_path,'dual',1701)
    p=a/'round_0'/'verifier_proposals.jsonl'
    p.write_text('\n'.join(map(json.dumps,[
        {'static_valid':True,'selected_for_solver':True,
         'gate_audit':{'balanced_accuracy':1.0},
         'reserved_audit_reporting_only':{'balanced_accuracy':.9,'coverage':1.0,'false_accept_rate':0,'false_reject_rate':.2}},
        {'static_valid':False,'selected_for_solver':False,
         'gate_audit':{'balanced_accuracy':0.0},
         'reserved_audit_reporting_only':{'balanced_accuracy':0.0,'coverage':0.0,'false_accept_rate':0,'false_reject_rate':0}}]))+'\n')
    m=summary.load_one(a)['metrics'][0]
    assert m['verifier_acceptance_rate']==.5
    assert m['all_proposal_source_valid_fraction']==.5
    assert m['all_proposal_hidden_balanced_accuracy']==.45
    assert m['verifier_hidden_balanced_accuracy']==.9
    assert m['all_proposal_hidden_coverage']==.5
