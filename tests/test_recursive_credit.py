import numpy as np
import pytest
from eesd.recursive_credit import (exact_token_shapley, loo_token_attribution,
    credit_from_scores, weights_from_attribution, coalition_masks)


def test_additive_and_negative_attribution():
    coef=np.array([[1.,-2.,0.2],[-.4,3.,0.1],[.2,.5,-.3],[.8,2.,.3]])
    scores=np.array([np.sum(coef[[j for j in range(4) if m & (1<<j)]],axis=0) if m else np.zeros(3) for m in range(16)])
    np.testing.assert_allclose(exact_token_shapley(scores,4),coef,atol=1e-11)
    assert exact_token_shapley(scores,4)[0,1] < 0


def test_shapley_interaction_splits_equally_but_loo_differs():
    scores=np.array([[float(bool(m&1) and bool(m&2))] for m in range(4)])
    np.testing.assert_allclose(exact_token_shapley(scores,2)[:,0],[.5,.5])
    np.testing.assert_allclose(loo_token_attribution(scores,2)[:,0],[1.,1.])


def test_zero_feedback_does_not_create_confidence():
    p=np.zeros((4,8)); w,d=weights_from_attribution(p)
    assert (w==0).all() and d['zero_feedback_signal']


def test_signed_consensus_and_conflicts():
    phi=np.array([[.3,-.4,.2],[.2,-.2,-.2],[0,0,0],[0,0,0]])
    w,d=weights_from_attribution(phi)
    assert w[0]>0 and w[1]>0 and w[2]==0
    assert d['negative_shapley_fraction']>0


def test_token_weights_have_mean_one_if_nonzero():
    w,_=weights_from_attribution(np.array([[.2,0,0], [.1,.1,0]]))
    np.testing.assert_allclose(w.mean(),1,atol=1e-6)


def test_exact_requires_full_subsets():
    with pytest.raises(ValueError):exact_token_shapley(np.zeros((5,4)),4)
    assert len(coalition_masks(4,'shapley'))==16
    assert len(coalition_masks(4,'loo'))==5
    assert len(coalition_masks(4,'sdpo'))==1


def test_invalid_nan_rejected():
    with pytest.raises(ValueError):weights_from_attribution(np.array([[float('nan')]]))
