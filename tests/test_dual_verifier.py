"""CPU tests: no untrusted generated source runs outside a mock bubblewrap."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eesd.dual_verifier import (
    VerdictCase,audit_verdicts,admit,verifier_source_sanity,make_cases,split_tests,
    execute_verifier,feedback_mistakes,verifier_training_mistakes,choose_candidate,safe_feedback_for_solver,oracle_verdict_feedback,
)


@dataclass
class Outcome:
    status:str
    actual:str
    expected:str=''


def test_balanced_accuracy_and_collapse_gates():
    cases=[VerdictCase(str(i),'x','o',i%2==0,'fixture') for i in range(8)]
    pass_all=audit_verdicts(cases,[True]*8)
    assert pass_all.false_accept_rate==1 and pass_all.false_reject_rate==0
    assert not admit(pass_all,min_per_class=4)
    reject_all=audit_verdicts(cases,[False]*8)
    assert reject_all.false_reject_rate==1
    assert not admit(reject_all,min_per_class=4)
    perfect=audit_verdicts(cases,[c.label for c in cases])
    assert perfect.balanced_accuracy==1 and admit(perfect,min_per_class=4)
    missing=audit_verdicts(cases,[None]*8)
    assert missing.coverage==0 and not admit(missing,min_per_class=4)


def test_ast_policy_rejects_obvious_unsafe_operations():
    assert verifier_source_sanity('def verify(input_text, candidate_output):\n    return True')[0]
    assert not verifier_source_sanity('import os\ndef verify(a,b):\n    return True')[0]
    assert not verifier_source_sanity('def verify(a,b):\n    return eval(a)')[0]
    assert not verifier_source_sanity('def verify(a,b):\n    open("/etc/passwd")\n    return True')[0]
    assert not verifier_source_sanity('def verify(a,b,c):\n    return True')[0]
    assert not verifier_source_sanity('print(123)\ndef verify(a,b):\n    return True')[0]


def test_split_and_cases_are_disjoint_deterministic():
    ts=[{'id':f't{i}','input':f'{i}','expected':str(i*2)} for i in range(24)]
    a,b,c=split_tests(ts,seed=1701,source_id='task-1',train=8,gate=8,solver=8)
    aa,bb,cc=split_tests(ts,seed=1701,source_id='task-1',train=8,gate=8,solver=8)
    assert (a,b,c)==(aa,bb,cc)
    assert len({x['input'] for x in a+b+c})==24
    cases=make_cases(a,seed=1,tag='anchor:')
    assert len(cases)==16 and sum(x.label for x in cases)==8
    assert all(candidates.candidate_output.strip()!=positive.candidate_output.strip()
               for positive,candidates in zip(cases[::2],cases[1::2]))
    assert all(positive.origin=='trusted_expected' for positive in cases[::2])


def test_verifier_batch_receives_no_labels_and_follows_bwrap_api():
    cases=[VerdictCase('one','4','8',True,'oracle'),
           VerdictCase('two','4','9',False,'observed_wrong_output')]
    source='def verify(input_text, candidate_output):\n    return str(2*int(input_text)) == candidate_output'
    captured=[]
    def sandbox(code,test,timeout):
        captured.append((code,test,timeout))
        assert '"label"' not in test['input']
        assert '"expected"' not in test['input']
        assert json.loads(test['input'])==[
             {'input':'4','candidate_output':'8'},
             {'input':'4','candidate_output':'9'}]
        assert 'def verify(' in code and 'try:' in code
        return Outcome('WRONG_OUTPUT','1 0')
    out=execute_verifier(source,cases,run_test_fn=sandbox)
    assert out==[True,False]
    assert len(captured)==1


def test_verifier_reflection_reports_real_mistakes_not_correct_examples():
    cases=[VerdictCase('a','2','2',True,'trusted_expected'),
           VerdictCase('b','2','3',False,'observed_wrong_output'),
           VerdictCase('c','2','1',False,'synthetic_output_perturbation')]
    mistakes=feedback_mistakes(cases,[False,True,None])
    assert len(mistakes)==3
    assert mistakes[0]['correct_verdict'] is True
    assert mistakes[1]['verifier_verdict'] is True
    assert mistakes[2]['verifier_verdict']=='ABSTAIN'


def test_candidate_gate_does_not_select_always_pass():
    cases=[VerdictCase(str(i),'x','o',i%2==0,'fixture') for i in range(8)]
    bad=audit_verdicts(cases,[True]*8)
    good=audit_verdicts(cases,[c.label for c in cases])
    index=choose_candidate([('def verify(a,b):return True',bad),
                            ('def verify(a,b):return a==b',good)],
                           max_fpr=.125,max_fnr=.25,min_per_class=4)
    assert index==1


def test_solver_feedback_labels_never_leak_expected():
    tests=[{'id':'t1','input':'1','expected':'2'},
           {'id':'t2','input':'2','expected':'4'}]
    def run_program(code,examples):
        assert code=='valid program'
        return [Outcome('PASS','2'),Outcome('WRONG_OUTPUT','7')]
    def run_verifier(code,cases):
        assert len(cases)==2
        assert all(c.origin=='solver' for c in cases)
        return [True,False]
    fb,diag=safe_feedback_for_solver('def verify(a,b):return True',
         'valid program',tests,run_program=run_program,run_verifier=run_verifier)
    assert [f['verifier_verdict'] for f in fb]==['PASS','FAIL']
    assert all('expected' not in f and 'runtime_status' not in f for f in fb)
    assert [f['program_execution_status'] for f in fb]==['EXECUTED','EXECUTED']
    assert diag['verifier_disagreements_with_local_oracle']==0


def test_class_imbalance_cannot_hide_false_rejections():
    cases=[VerdictCase(str(i),'x','o',i<8,'fixture') for i in range(10)]
    verdict=[False]*10
    a=audit_verdicts(cases,verdict)
    assert a.false_reject_rate==1.0
    assert not admit(a,min_per_class=2)


def test_unknown_or_unsafe_verifier_abstains_without_execution():
    called=[]
    cases=[VerdictCase('1','x','y',True,'trusted_expected')]
    out=execute_verifier('import subprocess\ndef verify(a,b):return True',cases,
         run_test_fn=lambda *a,**k:called.append(True))
    assert out==[None] and not called


def test_synthetic_negative_reproducible_and_not_actual_gold():
    t=[{'id':'1','input':'13','expected':'13\n'}]
    a=make_cases(t,seed=12)
    b=make_cases(t,seed=12)
    assert a==b and a[1].candidate_output.strip()!='13'


def test_rejects_nested_unapproved_import():
    code='def verify(input_text,candidate_output):\n    import os\n    return candidate_output==input_text'
    assert not verifier_source_sanity(code)[0]

def test_execution_failure_cannot_be_overruled_by_generated_verifier():
    cases=[{'id':'t1','input':'1','expected':'2'},
           {'id':'t2','input':'2','expected':'4'}]
    def run_program(code, tests):
        return [Outcome('RUNTIME_EXCEPTION',''),Outcome('WRONG_OUTPUT','5')]
    def run_verifier(code, cases):
        assert len(cases)==1 and cases[0].input_text=='2'
        return [False]
    feedback,stats=safe_feedback_for_solver('def verify(a,b):return True',
          'broken',cases,run_program=run_program,run_verifier=run_verifier)
    assert feedback[0]['verifier_verdict']=='FAIL'
    assert feedback[0]['program_execution_status']=='RUNTIME_EXCEPTION'
    assert feedback[1]['program_execution_status']=='EXECUTED'
    assert stats['verifier_disagreements_with_local_oracle']==0

def test_oracle_verdict_baseline_uses_identical_format_without_gold_stdout():
    cases=[{'id':'t1','input':'1','expected':'2'},
           {'id':'t2','input':'2','expected':'4'}]
    def run_program(code, tests):
        return [Outcome('PASS','2'),Outcome('WRONG_OUTPUT','5')]
    oracle=oracle_verdict_feedback('code',cases,run_program=run_program)
    assert [r['verifier_verdict'] for r in oracle]==['PASS','FAIL']
    assert all('expected' not in r and r['program_execution_status']=='EXECUTED' for r in oracle)
    assert all('verdict_origin' not in r for r in oracle)


def test_invalid_verifier_proposal_produces_actionable_training_signal():
    cases=[VerdictCase('one','2','4',True,'trusted_expected')]
    hints=verifier_training_mistakes('def verify((',False,'syntax',cases,[None],limit=2)
    assert len(hints)==1
    assert hints[0]['kind']=='invalid_verifier_source'
    assert hints[0]['static_reason']=='syntax'
    assert not any('correct_verdict' in h or 'verifier_excerpt' in h for h in hints)
