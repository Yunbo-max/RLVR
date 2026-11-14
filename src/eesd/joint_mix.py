"""Ungrouped, behavior-calibrated joint-feedback selection and mixture controls.

All values are computed from one frozen model on fixed candidate-token prefixes.
This module has no Torch dependency; run-time teacher calls live in scripts/.
Calibration labels are training-side supervision, never final-test labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Value:
    mean: float
    stderr: float
    criterion: float
    by_probe: tuple[float, ...]


@dataclass(frozen=True)
class SearchResult:
    selected_mask: int
    selected_score: float
    queried_masks: tuple[int, ...]
    scored: tuple[tuple[int, float], ...]
    abstained: bool
    policy: str


def _logs(rows: Sequence[np.ndarray]) -> list[np.ndarray]:
    result = [np.asarray(a, dtype=np.float64) for a in rows]
    if len(result) < 2 or any(r.ndim != 1 or not len(r) or not np.isfinite(r).all() for r in result):
        raise ValueError('requires >=2 candidates with nonempty finite token log probabilities')
    if any((r > 1e-7).any() for r in result):
        raise ValueError('log probabilities cannot exceed zero')
    return result


def _calibration(data: np.ndarray, candidates: int) -> np.ndarray:
    a = np.asarray(data, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != candidates or a.shape[1] < 2:
        raise ValueError('need two or more independent training calibration probes')
    if not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
        raise ValueError('invalid calibration outcomes')
    return a


def behavior_value(
    target: Sequence[np.ndarray], baseline: Sequence[np.ndarray],
    calibration: np.ndarray, *, uncertainty: float = .5, cost: float = 0.,
) -> Value:
    """Probe-wise covariance: V_h = mean_i (c_ih - mean_i c_ih) delta_i.

    This is a fixed-candidate behavioral surrogate, not downstream reward.
    """
    p, q = _logs(target), _logs(baseline)
    if len(p) != len(q) or any(x.shape != y.shape for x,y in zip(p,q)):
        raise ValueError('teacher and baseline must score identical candidate tokens')
    if not np.isfinite([uncertainty,cost]).all() or min(uncertainty,cost) < 0:
        raise ValueError('invalid value-function hyperparameters')
    a = _calibration(calibration,len(p))
    centered = a - a.mean(axis=0,keepdims=True)
    shifts = np.array([np.mean(x-y) for x,y in zip(p,q)])
    by_probe = np.mean(centered*shifts[:,None],axis=0)
    mean = float(by_probe.mean())
    se = float(by_probe.std(ddof=1)/math.sqrt(len(by_probe)))
    return Value(mean,se,mean-uncertainty*se-cost,tuple(float(x) for x in by_probe))


def mix_sampled_logprobs(
    base: Sequence[np.ndarray], first: Sequence[np.ndarray],
    second: Sequence[np.ndarray], alpha: Sequence[float],
) -> list[np.ndarray]:
    """log(alpha_0 q_0(y) + alpha_A q_A(y) + alpha_B q_B(y))."""
    a = np.asarray(alpha,dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all() or np.min(a)<0 or not np.isclose(a.sum(),1.,atol=1e-10):
        raise ValueError('mixture coefficients must be nonnegative and sum to 1')
    parts=[_logs(x) for x in (base,first,second)]
    if not (len(parts[0])==len(parts[1])==len(parts[2])):
        raise ValueError('candidate count mismatch')
    output=[]
    for items in zip(*parts):
        if len({x.shape for x in items})!=1:
            raise ValueError('token alignment mismatch')
        out=np.full_like(items[0],-np.inf)
        for weight,vec in zip(a,items):
            if weight>0:
                out=np.logaddexp(out,math.log(weight)+vec)
        output.append(out)
    return output


def fit_mixture(
    base: Sequence[np.ndarray], first: Sequence[np.ndarray],
    second: Sequence[np.ndarray], calibration: np.ndarray, *,
    grid_denominator: int = 20, uncertainty: float = .5,
) -> tuple[tuple[float,float,float], Value]:
    """Search a reproducible simplex grid; include all three single teachers.

    Does not inspect reserved audit or final-test outcomes.
    """
    if not 1<=grid_denominator<=100:
        raise ValueError('grid denominator must be in 1..100')
    best=None
    # Prefer q0 on ties, then sparse feedback weights, for deterministic abstention.
    for n0 in range(grid_denominator,-1,-1):
        for n1 in range(grid_denominator-n0,-1,-1):
            n2=grid_denominator-n0-n1
            alpha=(n0/grid_denominator,n1/grid_denominator,n2/grid_denominator)
            value=behavior_value(mix_sampled_logprobs(base,first,second,alpha),
                                 base,calibration,uncertainty=uncertainty)
            key=(value.criterion,n0,-sum(int(v>0) for v in (n1,n2)),n1)
            if best is None or key>best[0]:
                best=(key,alpha,value)
    assert best is not None
    return best[1],best[2]


def token_weights(
    target: Sequence[np.ndarray], baseline: Sequence[np.ndarray],
    calibration: np.ndarray, *, scale: float = .15,
) -> list[np.ndarray]:
    target,baseline=_logs(target),_logs(baseline)
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('positive finite token-weight scale required')
    if len(target)!=len(baseline) or any(x.shape!=y.shape for x,y in zip(target,baseline)):
        raise ValueError('target and baseline alignment mismatch')
    a=_calibration(calibration,len(target))
    advantage=a.mean(axis=1)-a.mean()
    return [(abs(float(advantage[i]))*
             np.clip(np.maximum(0.,np.sign(advantage[i])*(p-q))/scale,0.,1.)).astype(np.float32)
            for i,(p,q) in enumerate(zip(target,baseline))]


def deterministic_pair(test_count:int, *, seed:int, round_index:int, source_id:str) -> tuple[int,int]:
    if test_count<2 or not source_id:
        raise ValueError('two tests and a nonempty source ID required')
    key=f'{seed}:{round_index}:{source_id}:joint-mix-pair'
    rng=random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16],16))
    return tuple(sorted(rng.sample(range(test_count),2)))


def mask_indices(mask:int, test_count:int) -> tuple[int,...]:
    if mask<0 or mask>>test_count:
        raise ValueError('subset outside the feedback pool')
    return tuple(i for i in range(test_count) if mask & (1<<i))


class QueryCache:
    """Lazy true-joint teacher queries, shared across all candidates.

    A query is one subset context across G candidate programs, not one GPU
    forward. No singleton teacher sweep or fixed group count is performed.
    """
    def __init__(self, candidate_count:int, test_count:int,
                 score:Callable[[int,int],np.ndarray], budget:int):
        if candidate_count<2 or test_count<2 or budget<1:
            raise ValueError('invalid query dimensions')
        self.candidates=candidate_count
        self.tests=test_count
        self.score=score
        self.budget=budget
        self.entries:dict[int,list[np.ndarray]]={}
        self.forward_conditions=0

    def get(self,mask:int)->list[np.ndarray]:
        mask_indices(mask,self.tests)
        if mask not in self.entries:
            if len(self.entries)>=self.budget:
                raise RuntimeError('joint teacher query budget exceeded')
            result=_logs([self.score(i,mask) for i in range(self.candidates)])
            self.entries[mask]=result
            self.forward_conditions+=self.candidates
        return self.entries[mask]


def select_budgeted_joint(
    cache:QueryCache, calibration:np.ndarray, *,
    seed:int, round_index:int, source_id:str,
    policy:str='adaptive', max_items:int=8, uncertainty:float=.5,
    subset_cost:float=.001, min_gain:float=.001,
) -> SearchResult:
    """Budgeted on-demand, true joint-prompt search (not Shapley).

    Explicit pair exploration prevents early stopping on zero singleton gains.
    Adaptive policy exploits add/drop/swap neighbors but continues random
    exploration; random policy proposes independent sets under equal budgets.
    No guarantee of global optimum or complete feedback-space coverage.
    """
    if policy not in ('adaptive','random') or not 2<=max_items<=cache.tests:
        raise ValueError('invalid search policy or size')
    if cache.budget<3 or cache.budget>256:
        raise ValueError('budget must be in 3..256')
    key=f'{seed}:{round_index}:{source_id}:{policy}:proposals'
    rng=random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16],16))
    baseline=cache.get(0)
    scored={0:0.}
    pair=deterministic_pair(cache.tests,seed=seed,round_index=round_index,source_id=source_id)
    anchor=(1<<pair[0])|(1<<pair[1])
    proposals=[anchor]
    best_mask=0; best_score=0.
    def random_mask(step:int)->int:
        max_k=min(max_items,cache.tests)
        # Every third query targets a pair, irrespective of prior singleton quality.
        if step%3==0:k=2
        elif step%3==1:k=rng.randint(1,min(4,max_k))
        else:k=rng.randint(2,max_k)
        return sum(1<<i for i in rng.sample(range(cache.tests),k))
    def neighbor(mask:int)->int:
        ids=list(mask_indices(mask,cache.tests))
        others=[j for j in range(cache.tests) if j not in ids]
        if not ids or (len(ids)<max_items and rng.random()<.4):
            return mask|(1<<rng.choice(others)) if others else mask
        if len(ids)>2 and rng.random()<.25:
            return mask&~(1<<rng.choice(ids))
        if ids and others:
            return (mask&~(1<<rng.choice(ids)))|(1<<rng.choice(others))
        return random_mask(len(scored))
    attempts=0
    while len(scored)<cache.budget:
        attempts+=1
        if attempts>cache.budget*100:
            raise RuntimeError('unable to construct enough distinct feedback subsets')
        step=len(scored)
        if proposals:
            mask=proposals.pop(0)
        elif policy=='adaptive' and best_mask and step%3!=0:
            mask=neighbor(best_mask)
        else:
            mask=random_mask(step+attempts)
        if not mask or mask in scored or mask.bit_count()>max_items:
            continue
        teacher=cache.get(mask)
        value=behavior_value(teacher,baseline,calibration,
              uncertainty=uncertainty,cost=subset_cost*mask.bit_count())
        scored[mask]=value.criterion
        if (value.criterion,-mask.bit_count(),-mask) > (best_score,-best_mask.bit_count(),-best_mask):
            best_mask=mask;best_score=value.criterion
    accept=best_mask!=0 and best_score>min_gain
    return SearchResult(best_mask if accept else 0,best_score,
                        tuple(cache.entries),tuple(scored.items()),not accept,policy)
