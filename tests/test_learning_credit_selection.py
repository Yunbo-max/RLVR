import numpy as np
import pytest
from eesd.learning_credit_selection import (
    select_direct,select_shapley,selection_from_arm,
    deterministic_random_subset,sample_matched_update_sources,scoring_cost,
    value_and_criterion,
)


def synthetic(logp_effects,calibration):
    """Exact additive coalition token scores, one candidate per row."""
    scores=[]
    for coeff in logp_effects:
        scores.append(np.array([
            [float(sum(coeff[j] for j in range(4) if s & (1<<j)))]*2
            for s in range(16)],dtype=float))
    return scores,np.asarray(calibration,dtype=float)


def test_shapley_and_direct_compare_same_in_sample_objective():
    scores,cal=synthetic([[.2,.1,-.2,0],[-.2,-.1,.2,0]],[[1,1,1,1],[0,0,0,0]])
    sp=select_shapley(scores,cal)
    direct=select_direct(scores,cal)
    game,J,contrast=value_and_criterion(scores,cal)
    assert sp.mask!=0 and direct.mask!=0 and contrast>0
    assert J[direct.mask] >= J[sp.mask]-1e-12
    assert np.allclose(game[0],0)


def test_redundant_negative_feedback_can_be_excluded():
    scores,cal=synthetic([[.4,-.6,.2,0],[-.4,.6,-.2,0]],[[1,1,1,1],[0,0,0,0]])
    c=select_shapley(scores,cal)
    assert c.mask & 2 == 0
    assert c.mask & 1
    assert c.phi is not None and c.phi[1]<0


def test_direct_exhaustive_never_worse_in_sample_even_for_interacting_games():
    # Non-additive interactions may make Shapley eligibility conservative.
    values=[]
    for i,sign in enumerate((1.,-1.)):
        rows=[]
        for s in range(16):
            v=sign*(.05*bool(s&1)+.05*bool(s&2)+.4*bool((s&3)==3)-.1*bool(s&4))
            rows.append([v])
        values.append(np.asarray(rows))
    audit=np.asarray([[1,1,1,1],[0,0,0,0]])
    sp=select_shapley(values,audit)
    d=select_direct(values,audit)
    _,J,_=value_and_criterion(values,audit)
    assert J[d.mask]>=J[sp.mask]-1e-12


def test_empty_behavioral_signal_abstains_both_selectors():
    scores,cal=synthetic([[4,0,0,0],[-4,0,0,0]],[[1,1,1,1],[1,1,1,1]])
    assert select_direct(scores,cal).mask==0
    assert select_shapley(scores,cal).mask==0


def test_random_subset_matches_cardinality_but_not_reference_when_possible():
    for ref in (1,3,7):
        result=deterministic_random_subset(ref,seed=1701,round_index=1,source_id='abc')
        assert result.mask!=ref and result.mask.bit_count()==ref.bit_count()
        assert result.mask==deterministic_random_subset(ref,seed=1701,round_index=1,source_id='abc').mask
    assert deterministic_random_subset(15,seed=1,round_index=0,source_id='abc').rule.startswith('no_alternative')
    assert deterministic_random_subset(0,seed=1,round_index=0,source_id='abc').mask==0


def test_matched_update_count_deterministic_and_exact():
    source=[f'p{i}' for i in range(64)]
    a=sample_matched_update_sources(source,7,seed=1701,round_index=2)
    b=sample_matched_update_sources(source,7,seed=1701,round_index=2)
    assert a==b and len(a)==7 and set(a)<=set(source)
    assert sample_matched_update_sources(source,0,seed=1701,round_index=2)==[]
    with pytest.raises(ValueError):sample_matched_update_sources(source,65,seed=1701,round_index=2)


def test_all_seven_arm_interfaces():
    scores,cal=synthetic([[1,0,0,0],[-1,0,0,0]],[[1,1,1,1],[0,0,0,0]])
    for arm in ('sdpo','audit_gate','direct_max','shapley','random_subset','matched_random_updates','no_update'):
        x=selection_from_arm(arm,scores,cal,round_index=0,source_id='p1',seed=1701,
            reference_mask=1,reference_update_sources={'p1'})
        assert 0<=x.mask<16
    with pytest.raises(ValueError):selection_from_arm('random_subset',scores,cal,round_index=0,source_id='a',seed=1)


def test_feedback_coalitions_not_confused_with_rollout_count():
    for g in (8,16,32):
        cost=scoring_cost(4,g,64,5,'shapley')
        assert cost['exact_coalitions']==16
        assert cost['total_scoring_calls']==16*g*64*5
    assert scoring_cost(4,8,64,5,'no_update')['total_scoring_calls']==0
    assert scoring_cost(4,8,64,5,'sdpo')['per_candidate_scoring_calls']==2
    with pytest.raises(ValueError):scoring_cost(16,8,64,5,'shapley')
