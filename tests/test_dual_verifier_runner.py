"""Import-only smoke for CLI/prompt invariants; no Transformers or GPU needed."""
from pathlib import Path
import importlib.util
import sys
import types
from unittest.mock import patch

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))


def _load():
    # Our runner imports existing RLVR modules in the real repository. Fake only
    # the name during this isolated pure-CPU test, not their behavior.
    strict=types.ModuleType('eesd.strict_execution')
    strict.run_test=lambda *a,**kw:None
    strict.run_tests_batched=lambda *a,**kw:None
    with patch.dict(sys.modules,{'run_joint_mix':types.ModuleType('run_joint_mix'),
                                 'eesd.strict_execution':strict}):
        spec=importlib.util.spec_from_file_location('test_dual_runner',root/'scripts'/'run_dual_verifier_opsd.py')
        obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj


def test_parser_arms_and_predeclared_modes():
    m=_load()
    assert set(m.ARMS)=={'dual','solver_fixed','solver_oracle_verdict','solver_static_verifier',
                         'solver_regenerated_verifier','verifier_only'}
    opts=m.build_parser().parse_args(['--data-dir','/tmp/data',
             '--init-adapter','/tmp/init','--output','/tmp/out','--arm','dual'])
    assert opts.rounds==5 and opts.verifier_proposals>=1 and opts.rollouts>=2
    assert opts.memory_limit==0  # Isolate LoRA gradient learning from reflection memory
    assert opts.missing_verifier=='skip'  # Never silently inject a trusted answer


def test_privileged_verifier_teacher_has_mistake_evidence_only():
    m=_load();row={'task':'Return twice the input.'}
    unpriv=m.verifier_messages(row)
    priv=m.verifier_messages(row,mistakes=[{'correct_verdict':False,'input':'3',
                              'candidate_output':'7'}])
    assert 'correct_verdict' not in str(unpriv)
    assert 'correct_verdict' in str(priv)
    assert 'input_text' in unpriv[1]['content']


def test_corrupted_role_adapter_blocks_resume(tmp_path):
    import pytest
    m=_load()
    s=tmp_path/'solver'; v=tmp_path/'verifier'
    s.mkdir();v.mkdir()
    (s/'weights').write_text('s');(v/'weights').write_text('v')
    import hashlib
    def digest(path):
        return hashlib.sha256((path/'weights').read_bytes()).hexdigest()
    receipt={'solver_adapter':str(s),'verifier_adapter':str(v),
             'solver_adapter_sha256':digest(s),'verifier_adapter_sha256':digest(v)}
    m._check_saved_adapter_integrity(receipt,digest)
    (v/'weights').write_text('altered')
    with pytest.raises(ValueError,match='verifier'):
        m._check_saved_adapter_integrity(receipt,digest)

def test_missing_verifier_policy_dryrun_defaults_to_skip():
    m=_load()
    args=m.build_parser().parse_args(['--data-dir','/tmp/d',
             '--init-adapter','/tmp/i','--output','/tmp/o','--arm','dual'])
    assert args.missing_verifier=='skip'
