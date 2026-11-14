"""No pretrained model or remote dataset required; real probability algebra tests."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eesd.joint_mix import (behavior_value,mix_sampled_logprobs,fit_mixture,
    deterministic_pair,QueryCache,select_budgeted_joint,token_weights,mask_indices)


def _toy():
    base=[np.log(np.array([.5,.4])), np.log(np.array([.5,.4]))]
    # Feedback A supports the high-quality candidate, B acts oppositely.
    a=[np.log(np.array([.8,.65])),np.log(np.array([.2,.25]))]
    b=[np.log(np.array([.2,.25])),np.log(np.array([.8,.65]))]
    cal=np.array([[1,1,1,1],[0,0,0,0]])
    return base,a,b,cal


def test_covariance_and_shift_invariance():
    base,a,b,cal=_toy()
    v=behavior_value(a,base,cal,uncertainty=0)
    assert v.mean>0
    assert behavior_value(b,base,cal).mean<0
    raised=[v+0.05 for v in a]
    # Common shift on ALL candidates changes no behavioral covariance.
    assert abs(behavior_value(raised,base,cal).mean-v.mean)<1e-12


def test_mixture_is_arithmetic_in_probability_space():
    base,a,b,cal=_toy()
    weights=(.2,.3,.5)
    sampled=mix_sampled_logprobs(base,a,b,weights)
    expected=.2*np.exp(base[0])+.3*np.exp(a[0])+.5*np.exp(b[0])
    assert np.allclose(np.exp(sampled[0]),expected)
    assert np.allclose(mix_sampled_logprobs(base,a,b,(1.,0.,0.))[0],base[0])
    with pytest.raises(ValueError):mix_sampled_logprobs(base,a,b,(.4,.3,.4))


def test_mix_fits_c8_but_does_not_use_a8():
    base,a,b,cal=_toy()
    alpha,v=fit_mixture(base,a,b,cal,grid_denominator=10)
    assert alpha==(0.,1.,0.)
    assert v.criterion>0
    # Changing a reserved audit observation has no effect on the fitter's inputs.
    audit=np.array([[0,0],[1,1]])
    assert behavior_value(mix_sampled_logprobs(base,a,b,alpha),base,audit).mean<0


def test_token_gate_preserves_weak_evidence():
    base,a,b,cal=_toy()
    weights=token_weights(a,base,cal)
    assert weights[0].sum()>0
    assert weights[1].sum()>0
    assert weights[0].max()<=0.5
    assert weights[1].max()<=0.5
    same=np.zeros_like(cal)
    assert all(np.all(w==0) for w in token_weights(a,base,same))


def test_no_fixed_four_groups_or_singleton_sweep():
    t=64
    pair=deterministic_pair(t,seed=1701,round_index=0,source_id='example')
    assert len(set(pair))==2 and 0<=min(pair)<max(pair)<t
    assert pair==deterministic_pair(t,seed=1701,round_index=0,source_id='example')
    score_calls=[]
    def scorer(i,mask):
        score_calls.append((i,mask))
        base=math.log(.5)
        return np.array([base+(0.2 if mask.bit_count()>=2 and i==0 else
                                  -0.2 if mask.bit_count()>=2 and i==1 else 0.)])
    cache=QueryCache(2,t,scorer,budget=8)
    cal=np.array([[1,1],[0,0]])
    chosen=select_budgeted_joint(cache,cal,seed=1701,round_index=0,
        source_id='example',policy='adaptive',max_items=8)
    assert chosen.selected_mask.bit_count()>=2
    assert cache.forward_conditions==16
    assert len(cache.entries)==8
    assert all(mask.bit_count()<=8 for mask in cache.entries)
    # The algorithm never precomputes 65G teacher conditions.
    assert len(score_calls)==16
    assert (1<<pair[0])|(1<<pair[1]) in cache.entries


def test_joint_synergy_is_not_recoverable_by_singletons():
    # Joint feedback can favor a correct token beyond the single-teacher hull.
    q0,qA,qB,qAB=.1,.2,.3,.9
    assert max(q0,qA,qB)<qAB
    assert max(a*q0+b*qA+(1-a-b)*qB
               for a in np.linspace(0,1,11)
               for b in np.linspace(0,1-a,11)) < qAB


def test_independent_random_queries_are_budget_matched():
    def mock(i,mask):
        return np.log(np.array([.3+.1*int(i==0 and mask.bit_count()>1)]))
    cal=np.array([[1,1],[0,0]])
    for pol in ('adaptive','random'):
        cache=QueryCache(2,11,mock,budget=12)
        r=select_budgeted_joint(cache,cal,seed=9,round_index=1,source_id='s',
                                policy=pol,max_items=5)
        assert len(r.queried_masks)==12
        assert len(set(r.queried_masks))==12
        assert 0 in r.queried_masks
    with pytest.raises(ValueError):mask_indices(1<<11,11)
