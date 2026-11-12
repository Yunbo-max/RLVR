import math
import numpy as np
import pytest

from eesd.feedback_group_credit import (
    form_groups,compact_candidate_groups,marginal_pairs,
    approximate_individual_shapley,budgeted_random_subset_masks,
    nested_execution_discrimination
)


def records(t,g=8):
    tests=[{'id':str(j),'input':str(j),'expected':str(j*2)} for j in range(t)]
    feedback=[]
    for i in range(g):
        feedback.append([{
          'id':str(j),'outcome':('PASS' if (j+i)%4<2 else
                                'WRONG_OUTPUT' if (j+i)%4==2 else 'TIMEOUT'),
          'input':str(j),'expected':str(j*2),'actual':str(i+j),
          'stderr':''} for j in range(t)])
    return tests,feedback


@pytest.mark.parametrize('t',[4,8,16,32,64])
def test_64_executions_become_exactly_four_evidence_groups(t):
    tests,fb=records(t)
    grouping=form_groups(tests,fb,groups=4,seed=1701,source_id='p1')
    assert len(grouping.members)==4
    assert sorted(j for g in grouping.members for j in g)==list(range(t))
    assert all(g for g in grouping.members)
    assert grouping==form_groups(tests,fb,groups=4,seed=1701,source_id='p1')
    structured=compact_candidate_groups(grouping,fb[0])
    assert sum(z['tested'] for z in structured)==t
    assert len(structured)==4
    assert all('outcome_counts' in z for z in structured)
    assert all(len(__import__('json').dumps(z,sort_keys=True))<=440 for z in structured)


def test_individual_shapley_64_tests_256_budget_unbiased_for_additive_game():
    weights=np.linspace(-.2,.3,64)
    values=0
    def game(mask):
        nonlocal values
        values+=1
        effect=sum(weights[j] for j in range(64) if (mask>>j)&1)
        return np.array([effect,2*effect,3*effect],dtype=float)
    out=approximate_individual_shapley(64,256,game,seed=6,source_id='s1')
    assert out.total_marginal_pairs==128
    assert len(out.per_test_mean)==64
    assert min(out.observations)==max(out.observations)==2
    np.testing.assert_allclose(out.per_test_mean[:,0],weights,atol=1e-10)
    assert values==out.actual_unique_value_queries<=256
    assert np.all(out.per_test_se[:,0]<1e-10)


def test_one_marginal_per_test_is_not_a_confident_shapley_estimate():
    def game(mask):
        return np.asarray([mask.bit_count(),mask.bit_count()+10.],dtype=float)
    out=approximate_individual_shapley(64,128,game,seed=5,source_id='x')
    assert np.isnan(out.per_test_se).all()
    assert np.all(out.observations==1)
    assert out.actual_unique_value_queries<=128


def test_budget_and_test_count_different_dimensions():
    for t in (4,8,16,32,64):
        pairs=marginal_pairs(t,2*t,seed=1,source_id='fixed')
        assert len(pairs)==t
        assert set(p.test_index for p in pairs)==set(range(t))
        assert all((p.with_mask^p.without_mask)==1<<p.test_index for p in pairs)
        assert all(not p.without_mask & (1<<p.test_index) for p in pairs)
    with pytest.raises(ValueError):marginal_pairs(64,64,seed=1,source_id='x')


def test_grouping_never_mixes_calibration_or_heldout_tests():
    tests,fb=records(64)
    grouped=form_groups(tests,fb,groups=4)
    with pytest.raises(ValueError):compact_candidate_groups(grouped,
       [dict(x,id='other') for x in fb[0]])


def test_direct_search_is_subset_budgeted_not_exact():
    chosen=budgeted_random_subset_masks(64,128,seed=1701,source_id='train-1')
    assert len(chosen)==128
    assert chosen[0]==0 and chosen[-1]==(1<<64)-1
    assert chosen==budgeted_random_subset_masks(64,128,seed=1701,source_id='train-1')
    assert all(0<=s<(1<<64) for s in chosen)
    with pytest.raises(ValueError):budgeted_random_subset_masks(4,256,seed=1,source_id='bad')



def test_feedback_richness_report_detects_redundant_64_tests():
    _,fb=records(64,g=8)
    report=nested_execution_discrimination(fb)
    assert list(report)==['4','8','16','32','64']
    assert report['64']['unique_code_outcome_patterns']<=8
    assert report['64']['distinct_test_outcome_signatures']<=64
    assert 0<=report['64']['mean_test_outcome_entropy_bits']<=math.log2(8)
    assert report['64']['distinct_test_outcome_signatures']==4
    assert report['4']['unique_code_outcome_patterns']==4


def test_feedback_diversity_reports_zero_for_identical_candidates():
    _,fb=records(64,g=8)
    identical=[fb[0] for _ in range(8)]
    report=nested_execution_discrimination(identical)
    assert report['64']['all_candidates_identical'] is True
    assert report['64']['mean_test_outcome_entropy_bits']==0.0
