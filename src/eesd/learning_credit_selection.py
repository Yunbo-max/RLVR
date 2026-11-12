"""Falsifiable selection controls for five-round EESD learning credit.

F = 4 *public-feedback* tests => 16 exact coalitions. G = 8/16/32
candidate programs is an independent dimension. No method in this module
reads the held-out audit or unseen-problem outcomes. Model scores must come
from one frozen round teacher and the same candidate/feedback pool.

IMPORTANT: direct argmax J(S) always dominates constrained Shapley in its
IN-SAMPLE J(S). Shapley can win only in independent behavior/parameter-update
transfer. Never claim higher in-sample objective as a Shapley advantage.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import random

import numpy as np

from eesd.behavioral_shapley import audit_value_game, select_feedback


@dataclass(frozen=True)
class Choice:
    mask: int
    rule: str
    j_calibration: float | None
    phi: tuple[float, ...] | None
    abstained: bool
    audit_contrast: float | None


def value_and_criterion(scores: list[np.ndarray], calibration: np.ndarray,
                        *, uncertainty: float = .5, subset_cost: float = .001):
    """Return per-coalition behavioral value and calibrated conservative score."""
    if any(not math.isfinite(x) or x < 0 for x in (uncertainty, subset_cost)):
        raise ValueError('nonnegative finite score penalties required')
    game = audit_value_game(scores, calibration)
    if game.ndim != 2 or game.shape[1] < 2:
        raise ValueError('need >=2 calibration tests')
    means = game.mean(axis=1)
    errs = game.std(axis=1, ddof=1) / math.sqrt(game.shape[1])
    costs = np.array([s.bit_count() * subset_cost for s in range(len(game))])
    conservative = means - uncertainty * errs - costs
    contrast = float(np.mean(np.var(calibration, axis=0)))
    return game, conservative, contrast


def select_direct(scores: list[np.ndarray], calibration: np.ndarray, *,
                  uncertainty: float = .5, subset_cost: float = .001,
                  min_gain: float = .001) -> Choice:
    """Exhaustive direct maximization of J(S), with NO Shapley filter."""
    if not math.isfinite(min_gain) or min_gain < 0:
        raise ValueError('invalid min_gain')
    _, criterion, contrast = value_and_criterion(scores, calibration,
                uncertainty=uncertainty,subset_cost=subset_cost)
    if contrast < 1e-12:
        return Choice(0,'no_calibration_contrast',None,None,True,contrast)
    # Stable lexical tie-break: prefer fewer feedback messages, then smaller mask.
    best=max(range(1,len(criterion)),key=lambda s:(criterion[s],-s.bit_count(),-s))
    if float(criterion[best]) <= min_gain:
        return Choice(0,'low_direct_value',float(criterion[best]),None,True,contrast)
    return Choice(best,'direct_max',float(criterion[best]),None,False,contrast)


def select_shapley(scores: list[np.ndarray], calibration: np.ndarray, *,
                   uncertainty: float = .5, subset_cost: float = .001,
                   min_gain: float = .001) -> Choice:
    r=select_feedback(scores,calibration,uncertainty_penalty=uncertainty,
        subset_cost=subset_cost,min_gain=min_gain)
    c=float(r.score_by_subset[r.mask]) if r.mask else None
    return Choice(r.mask,r.reason,c,tuple(float(v) for v in r.phi),
                  r.mask==0,float(r.audit_contrast))


def deterministic_random_subset(reference_mask: int, *,
                                seed: int, round_index: int, source_id: str) -> Choice:
    """Match reference selected source AND number of feedback tests; not identity.

    This is explicitly a *reference-schedule* diagnostic: with diverging
    recursive policies the matched source does not imply equally good rollouts.
    """
    if not 0 <= reference_mask <= 15:
        raise ValueError('invalid reference feedback mask')
    if reference_mask == 0:
        return Choice(0,'reference_abstained',None,None,True,None)
    count=reference_mask.bit_count()
    choices=[s for s in range(1,16) if s.bit_count()==count and s!=reference_mask]
    if not choices:
        # all four feedback tests: only one possible choice, report no
        # identifiable subset randomization instead of claiming variation.
        return Choice(reference_mask,'no_alternative_same_size_subset',
                      None,None,False,None)
    digest=sha256(f'{seed}:{round_index}:{source_id}:subset'.encode()).digest()
    rng=random.Random(int.from_bytes(digest[:8],'big'))
    return Choice(rng.choice(choices),'matched_random_subset',None,None,False,None)


def sample_matched_update_sources(sources: list[str], count: int, *,
                                   seed: int, round_index: int) -> list[str]:
    """Same update COUNT as Shapley, random source IDs; never consult final eval."""
    if len(sources)!=len(set(sources)) or not 0<=count<=len(sources):
        raise ValueError('count/source mismatch')
    digest=sha256(f'{seed}:{round_index}:update-count'.encode()).digest()
    rng=random.Random(int.from_bytes(digest[:8],'big'))
    return sorted(rng.sample(sorted(sources),count))


def selection_from_arm(arm: str, scores: list[np.ndarray], calibration: np.ndarray,
                       *, round_index: int, source_id: str, seed: int,
                       reference_mask: int|None = None,
                       reference_update_sources: set[str]|None = None,
                       uncertainty: float=.5, subset_cost: float=.001,
                       min_gain: float=.001) -> Choice:
    if arm == 'shapley':
        return select_shapley(scores,calibration,uncertainty=uncertainty,
                subset_cost=subset_cost,min_gain=min_gain)
    if arm == 'direct_max':
        return select_direct(scores,calibration,uncertainty=uncertainty,
                subset_cost=subset_cost,min_gain=min_gain)
    if arm == 'audit_gate':
        # Matched audit-only gate: full feedback if its behavioral value positive,
        # no subset search and no Shapley.
        game,_,contrast=value_and_criterion(scores,calibration,uncertainty=0.,
                                            subset_cost=0.)
        val=float(game[-1].mean())
        accept=contrast>1e-12 and val>min_gain
        return Choice(15 if accept else 0,
            'audit_gate' if accept else 'audit_gate_abstain',val,None,not accept,contrast)
    if arm == 'random_subset':
        if reference_mask is None:
            raise ValueError('random subset needs frozen Shapley reference mask')
        return deterministic_random_subset(reference_mask,seed=seed,
                          round_index=round_index,source_id=source_id)
    if arm == 'matched_random_updates':
        if reference_update_sources is None:
            raise ValueError('need matched update source schedule')
        accept=source_id in reference_update_sources
        return Choice(15 if accept else 0,
           'matched_random_updates' if accept else 'matched_random_abstain',
           None,None,not accept,None)
    if arm == 'sdpo':
        return Choice(15,'all_feedback_sdpo',None,None,False,None)
    if arm == 'no_update':
        return Choice(0,'frozen',None,None,True,None)
    raise ValueError('unknown experimental arm')


def scoring_cost(feedback_tests: int, rollouts: int, sources: int,
                 rounds: int, arm: str) -> dict:
    """Number of teacher-forced *subset evaluations*, not generated tokens."""
    if feedback_tests<1 or rollouts<1 or sources<1 or rounds<1:
        raise ValueError('positive budgets required')
    if feedback_tests>8:
        raise ValueError('exact coalition scoring above 8 tests requires approximation')
    coalitions=(0 if arm=='no_update' else
                2**feedback_tests if arm in ('shapley','direct_max','random_subset') else 2)
    return {'feedback_tests':feedback_tests,'exact_coalitions':2**feedback_tests,
            'per_candidate_scoring_calls':coalitions,
            'total_scoring_calls':coalitions*rollouts*sources*rounds,
            'excludes_teacher_KL_for_selected_subset_and_generation':True}
