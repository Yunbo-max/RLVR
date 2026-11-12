import importlib.util
from pathlib import Path
import numpy as np
import pytest

P=Path(__file__).resolve().parents[1]/'scripts'/'audit_behavioral_shapley.py'
s=importlib.util.spec_from_file_location('posthoc_behavioral_audit',P)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def test_posthoc_heldout_audit_may_disagree_with_training_selection():
    values=np.zeros((2,16),dtype=float)
    for mask in range(16):
        values[0,mask]=bool(mask&1)
        values[1,mask]=-bool(mask&1)
    # Mock withheld test says candidate 0 is actually wrong: the selected
    # training evidence now has NEGATIVE behavioral alignment on reserved audit.
    record={'source_id':'s1','selected_mask':1,'selection':{'reason':'positive_behavioral_evidence'},
            'candidate_sources':['a','b'],'candidate_shift_means':values.tolist()}
    fake_execute=lambda code,tests:[int(code=='b')]*len(tests)
    report=m.audit_selection(record,[{'id':str(j)} for j in range(4)],execute=fake_execute)
    assert report['selected_probe_alignment']<0
    assert report['full_probe_alignment']<0
    assert report['calibration_selection_reason']=='positive_behavioral_evidence'


def test_posthoc_denies_missing_score_provenance():
    assert m.audit_selection({'source_id':'s1'},[{'id':'a'}]) is None
