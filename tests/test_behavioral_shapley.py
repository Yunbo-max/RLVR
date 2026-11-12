import numpy as np
import pytest
from eesd.behavioral_shapley import (audit_value_game,select_feedback,
    directional_token_weights,full_feedback_audit_gate)


def example_values(coef):
    out=[]
    for u in coef:
        masks=np.arange(16)
        out.append(np.stack([np.full(3, sum(u[j] for j in range(4) if mask & (1<<j)),dtype=float) for mask in masks]))
    return out


def test_helpful_positive_and_harmful_negative_direction_are_both_rewarded():
    # Candidate 0 succeeds on the audit and gets higher confidence from T0.
    # Candidate 1 fails and gets lower confidence from T0.
    scores=example_values([[1.,0.,0.,0.],[-1.,0.,0.,0.]])
    audit=np.array([[1,1,1,1],[0,0,0,0]])
    game=audit_value_game(scores,audit)
    assert game[0].sum()==0 and game[1].mean()==.5
    selection=select_feedback(scores,audit,min_gain=.001)
    assert selection.mask==1
    weights=directional_token_weights(scores,audit,selection.mask)
    assert all((w>0).all() for w in weights)


def test_opposite_direction_feedback_is_rejected():
    scores=example_values([[-1.,0.,0.,0.],[1.,0.,0.,0.]])
    audit=np.array([[1,1,1,1],[0,0,0,0]])
    result=select_feedback(scores,audit)
    assert result.mask==0 and result.reason=='insufficient_behavioral_evidence'


def test_no_audit_disagreement_stops_updates_even_with_large_logp_change():
    scores=example_values([[10.,0,0,0],[-2.,0,0,0]])
    result=select_feedback(scores,np.ones((2,4)))
    assert result.mask==0 and result.reason=='no_cross_rollout_audit_disagreement'
    w=directional_token_weights(scores,np.ones((2,4)),1)
    assert sum(float(x.sum()) for x in w)==0


def test_redundant_and_interacting_tests_score_joint_subset():
    scores=example_values([[.4,.2,-.6,0],[-.4,-.2,.6,0]])
    audit=np.array([[1,1,1,1],[0,0,0,0]])
    r=select_feedback(scores,audit,subset_cost=.01)
    assert r.mask & (1<<2)==0
    assert r.mask & (1<<0)
    assert r.value_by_subset[0]==0


def test_per_rollout_weight_not_normalized_to_one_when_signal_tiny():
    scores=example_values([[.0001,0,0,0],[-.0001,0,0,0]])
    audit=np.array([[1,1,1,1],[0,0,0,0]])
    weights=directional_token_weights(scores,audit,1)
    assert max(float(x.max()) for x in weights)<.001


def test_audit_gate_full_feedback_no_exact_shapley_needed():
    scores=example_values([[1,0,0,0],[-1,0,0,0]])
    audit=np.array([[1,1,1,1],[0,0,0,0]])
    assert full_feedback_audit_gate(scores,audit).mask==15


def test_invalid_nonfinite_rejected():
    x=example_values([[0,0,0,0],[0,0,0,0]])
    x[0][3,1]=float('nan')
    with pytest.raises(ValueError):select_feedback(x,np.zeros((2,4)))
