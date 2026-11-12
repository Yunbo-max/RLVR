"""Execution-rich feedback with bounded *group-level* Shapley cost.

Execute up to 64 input/output tests per program, aggregate outcomes into
K=4 behavioral evidence groups, then calculate exact 2**K=16 group Shapley
coalitions. This is GROUP attribution, not individual attribution to
64 tests. Run the same group builder for every compared selector.

Optional per-TEST Shapley uses budgeted, unbiased conditional-marginal
sampling: each pair V(S union {j})-V(S) is sampled with |S| uniform over
0..T-1 and S uniform conditional on j not in S. At 64 tests and
128 value queries, only ONE marginal sample per test is available; do
not report it as a precise test attribution.

Each V(S) teacher call costs one frozen-model teacher-forcing pass *per
rollout candidate*. We count actual unique coalition calls separately.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
import random
from typing import Callable, Sequence

import numpy as np


VALID_T=(4,8,16,32,64)
STATUS=('PASS','WRONG_OUTPUT','COMPILE_ERROR','RUNTIME_EXCEPTION','TIMEOUT')


def _sha_int(value:str)->int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:16],16)


def _status(item:dict)->str:
    status=item.get('outcome',item.get('status'))
    if status not in STATUS:
        raise ValueError(f'unknown/missing code execution status: {status!r}')
    return status


@dataclass(frozen=True)
class FeedbackGroups:
    members: tuple[tuple[int,...], ...]
    test_ids: tuple[str,...]
    distinct_outcome_signatures: int
    num_candidates: int

    @property
    def test_count(self)->int:
        return len(self.test_ids)

    @property
    def group_count(self)->int:
        return len(self.members)


def form_groups(tests:Sequence[dict],outcomes:Sequence[Sequence[dict]],
                *,groups:int=4,seed:int=1701,source_id:str='') -> FeedbackGroups:
    """Deterministically cluster tests by their OUTCOME vectors across G codes.

    Clusters are derived ONLY from public-feedback tests, never calibration,
    reserved audit or final benchmark. All compared arms share the same
    grouping and exact candidate pool at their common round-zero baseline.
    """
    t=len(tests);g=len(outcomes)
    if t not in VALID_T or not 1<=groups<=t or g<2:
        raise ValueError('T must be 4/8/16/32/64; >=2 rollouts; <=T groups')
    ids=tuple(str(x['id']) for x in tests)
    if len(set(ids))!=t:raise ValueError('duplicate public test ID')
    if any(len(line)!=t for line in outcomes):raise ValueError('test outcome matrix mismatch')
    for i,line in enumerate(outcomes):
        if [str(x['id']) for x in line]!=list(ids):
            raise ValueError(f'outcome/test order mismatch for rollout {i}')
    signatures=[tuple(_status(outcomes[i][j]) for i in range(g))
                for j in range(t)]
    if len(set(signatures))<1:raise ValueError('empty behavior signature')

    # Select diverse seed signatures (maximin Hamming), then assign the
    # remaining tests to closest signature with a small balance penalty.
    ordered=sorted(range(t),key=lambda j:(_sha_int(f'{seed}:{source_id}:{ids[j]}'),ids[j]))
    centers=[ordered[0]]
    while len(centers)<groups:
        def separation(j:int)->tuple:
            min_dist=min(sum(x!=y for x,y in zip(signatures[j],signatures[k]))
                         for k in centers)
            return (min_dist,-_sha_int(f'{seed}:{source_id}:seed:{ids[j]}'))
        centers.append(max((j for j in ordered if j not in centers),key=separation))
    members=[[c] for c in centers]
    for j in ordered:
        if j in centers:continue
        distances=[sum(a!=b for a,b in zip(signatures[j],signatures[c]))
                   for c in centers]
        # Balanced only as secondary criterion; distinct failure families
        # should remain meaningful groups, not arbitrary group sizes.
        group=min(range(groups),key=lambda idx:
                  (distances[idx],len(members[idx]),idx))
        members[group].append(j)
    return FeedbackGroups(tuple(tuple(sorted(x)) for x in members),ids,
                          len(set(signatures)),g)


def _clip(text:object,n:int)->str:
    value=str(text).replace('\r',' ').replace('\n',' ')
    return value[:n]


def compact_candidate_groups(grouping:FeedbackGroups,
                             candidate_outcomes:Sequence[dict], *,
                             per_group_chars:int=440)->list[dict]:
    """Summarize ALL executed tests as K bounded-size evidence units.

    Truncating examples means a group conveys aggregated observations rather
    than 64 raw records. This limit is identical for T4 and T64 arms.
    Check final tokenizer-based prompt length separately on the actual GPU.
    """
    if (len(candidate_outcomes)!=grouping.test_count or
        per_group_chars<200 or per_group_chars>2048):
        raise ValueError('invalid candidate outcome length / evidence budget')
    ids=[str(x['id']) for x in candidate_outcomes]
    if tuple(ids)!=grouping.test_ids:raise ValueError('candidate/test ID mismatch')
    result=[]
    for group_id,indices in enumerate(grouping.members):
        selected=[candidate_outcomes[j] for j in indices]
        counts=Counter(_status(v) for v in selected)
        # Stable representatives: show failure context and a PASS sanity
        # check when available, not 16 copies of the same failing input.
        failures=[v for v in selected if _status(v)!='PASS']
        passes=[v for v in selected if _status(v)=='PASS']
        examples=(failures[:2]+passes[:1])[:3]
        snippet=[]
        for v in examples:
            snippet.append({'id':str(v['id']),'result':_status(v),
                'input':_clip(v.get('input',''),28),
                'expected':_clip(v.get('expected',''),28),
                'actual':_clip(v.get('actual',''),28),
                'error':_clip(v.get('stderr',''),28)})
        base={'evidence_group':group_id,'tested':len(indices),
              'outcome_counts':dict(sorted(counts.items())),
              'examples':snippet}
        import json
        while len(json.dumps(base,sort_keys=True,ensure_ascii=False))>per_group_chars:
            if base['examples']:
                base['examples'].pop()
            else:
                raise ValueError('group summary exceeds reserved character budget')
        result.append(base)
    return result


@dataclass(frozen=True)
class MarginalPair:
    test_index: int
    without_mask: int
    with_mask: int
    coalition_size: int


def marginal_pairs(test_count:int,teacher_value_budget:int, *,
                   seed:int,source_id:str,round_idx:int=0)->tuple[MarginalPair,...]:
    """Each single-test marginal estimate is unbiased over Shapley coalitions.

    A 64-test run needs >=128 budgeted value evaluations for just one sample
    per test. With B=256, it samples 2 per test; B=512 gives 4. This is not
    sufficient for reliable uncertainty quantification in every case.
    """
    if test_count not in VALID_T or teacher_value_budget<2*test_count:
        raise ValueError('budget must provide at least one marginal pair/test')
    n_pairs=teacher_value_budget//2
    counts=[n_pairs//test_count]*test_count
    extra=n_pairs%test_count
    rng=random.Random(_sha_int(f'{seed}:{round_idx}:{source_id}:pair-budget'))
    indices=list(range(test_count));rng.shuffle(indices)
    for j in indices[:extra]:counts[j]+=1
    pairs=[]
    for j in range(test_count):
        possible=[k for k in range(test_count) if k!=j]
        for repetition in range(counts[j]):
            # Cardinality is uniform, as required by the Shapley formula.
            k=rng.randrange(test_count)
            selected=rng.sample(possible,k)
            mask=sum(1<<v for v in selected)
            pairs.append(MarginalPair(j,mask,mask|(1<<j),k))
    rng.shuffle(pairs)
    return tuple(pairs)


@dataclass
class BudgetedAttribution:
    per_test_mean: np.ndarray
    per_test_se: np.ndarray
    observations: np.ndarray
    actual_unique_value_queries: int
    total_marginal_pairs: int
    budget: int


def approximate_individual_shapley(
    test_count:int,teacher_value_budget:int,value:Callable[[int],np.ndarray], *,
    seed:int,source_id:str,round_idx:int=0,
)->BudgetedAttribution:
    """No gradients, just frozen teacher subset values V_h(S).

    Value is a function returning a vector of H behavior-calibration values.
    Caching prevents duplicate coalition calls; query budget is an upper bound.
    """
    pairs=marginal_pairs(test_count,teacher_value_budget,seed=seed,
                         source_id=source_id,round_idx=round_idx)
    cache={}
    values=[[] for _ in range(test_count)]
    h=None
    def get(mask:int)->np.ndarray:
        nonlocal h
        if mask not in cache:
            v=np.asarray(value(mask),dtype=np.float64)
            if v.ndim!=1 or len(v)<2 or not np.isfinite(v).all():
                raise ValueError('V(S) must be finite per-calibration-test vector')
            if h is not None and len(v)!=h:raise ValueError('inconsistent calibration size')
            h=len(v)
            cache[mask]=v
        return cache[mask]
    for pair in pairs:
        diff=get(pair.with_mask)-get(pair.without_mask)
        values[pair.test_index].append(diff)
    if len(cache)>teacher_value_budget:raise AssertionError('teacher budget exceeded')
    means=np.array([np.mean(v,axis=0) for v in values],dtype=float)
    ses=np.array([np.std(v,axis=0,ddof=1)/math.sqrt(len(v))
                  if len(v)>=2 else np.full(h,np.nan) for v in values],dtype=float)
    return BudgetedAttribution(means,ses,np.array([len(x) for x in values]),
          len(cache),len(pairs),teacher_value_budget)


def budgeted_random_subset_masks(test_count:int,teacher_value_budget:int, *,
                                 seed:int,source_id:str,round_idx:int=0)->tuple[int,...]:
    """Cost-matched DIRECT subset-search candidate set, without Shapley.

    These are subset queries, NOT 2^T exact direct maximization. All arms
    must report their query counts and optimize the same V/J when compared.
    """
    if test_count not in VALID_T or teacher_value_budget<2:
        raise ValueError('invalid direct search budget')
    rng=random.Random(_sha_int(f'{seed}:{round_idx}:{source_id}:direct-search'))
    full=(1<<test_count)-1
    masks={0,full}
    attempts=0
    while len(masks)<teacher_value_budget and attempts<teacher_value_budget*20:
        attempts+=1
        # Probe varying cardinalities including sparse and dense evidence.
        if attempts%3==0:
            size=rng.randint(1,min(test_count,8))
        elif attempts%3==1:
            size=rng.randint(max(1,test_count-8),test_count)
        else:
            size=rng.randint(1,test_count)
        mask=sum(1<<j for j in rng.sample(range(test_count),size))
        masks.add(mask)
    if len(masks)<teacher_value_budget:
        raise ValueError('cannot generate required distinct subsets within budget')
    return tuple(sorted(masks))



def nested_execution_discrimination(outcomes:Sequence[Sequence[dict]],
                                    *,levels:tuple[int,...]=VALID_T)->dict:
    """No model or hidden labels needed: how informative are 4..64 tests?

    - unique_code_outcome_patterns: number of distinct G-length code behaviors
      over the first T test cases (range 1..G).
    - distinct_test_outcome_signatures: number of distinct G-candidate outcome
      patterns among the first T tests (range 1..T).
    - mean_test_outcome_entropy: mean categorical entropy across G programs
      for each test, measured in bits.
    - all_candidates_identical_rate: indicator that every code sample behaves
      identically on that T-sized public set.
    Compute from PUBLIC tests only; no calibration/heldout exposure.
    """
    g=len(outcomes)
    if g<2:raise ValueError('at least two candidate programs required')
    width=len(outcomes[0])
    if width<max(levels) or any(len(v)!=width for v in outcomes):
        raise ValueError('need entire nested 64-test outcome matrix')
    signatures=[[ _status(v[j]) for j in range(width)] for v in outcomes]
    result={}
    for t in levels:
        if t<1 or t>width:raise ValueError('invalid nested test-prefix size')
        code_patterns={tuple(signatures[i][:t]) for i in range(g)}
        test_patterns={tuple(signatures[i][j] for i in range(g))
                       for j in range(t)}
        entropies=[]
        for j in range(t):
            hist=Counter(signatures[i][j] for i in range(g))
            entropies.append(-sum((n/g)*math.log2(n/g) for n in hist.values()))
        result[str(t)]={
            'unique_code_outcome_patterns':len(code_patterns),
            'distinct_test_outcome_signatures':len(test_patterns),
            'mean_test_outcome_entropy_bits':float(np.mean(entropies)),
            'all_candidates_identical':len(code_patterns)==1,
            'public_pass_fraction':sum(signatures[i][j]=='PASS'
                for i in range(g) for j in range(t))/(g*t),
        }
    return result
