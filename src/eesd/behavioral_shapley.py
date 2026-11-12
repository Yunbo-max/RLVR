"""Behavior-aligned counterfactual feedback attribution for recursive EESD.

All scoring uses *one frozen model* and fixed sampled program tokens.
Audit outcomes are from training-source probes NOT included in teacher prompts.
This is partial-verification evidence, not a proof of correctness.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Selection:
    mask: int
    phi: np.ndarray  # signed [n_feedback]
    phi_stderr: np.ndarray
    value_by_subset: np.ndarray  # [2**n_feedback]
    score_by_subset: np.ndarray
    reason: str
    audit_contrast: float


def _test_scores(scores: Sequence[np.ndarray], outcomes: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Check finite [2**n_tests, n_tokens] scores and [n_candidates, n_audit] outcomes."""
    v = [np.asarray(x, dtype=np.float64) for x in scores]
    a = np.asarray(outcomes, dtype=np.float64)
    if len(v) < 2 or a.ndim != 2 or a.shape[0] != len(v) or a.shape[1] < 2:
        raise ValueError("need >=2 candidates and >=2 independent audit tests per candidate")
    if not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
        raise ValueError("audit correctness must be in [0,1]")
    count = v[0].shape[0]
    if count < 2 or count & (count - 1) or count > 256:
        raise ValueError("test coalitions must form 2**T, T in 1..8")
    for item in v:
        if item.ndim != 2 or item.shape[0] != count or item.shape[1] < 1 or not np.isfinite(item).all():
            raise ValueError("invalid token log-probability array")
    return v, a


def audit_value_game(scores: Sequence[np.ndarray], audit_outcomes: np.ndarray) -> np.ndarray:
    """Return V[S,audit_probe]: covariance of feedback shift and behavior.

    For candidate i, delta_i(S) is its *mean sampled-token logP shift*
    caused by its own public feedback subset S. Different candidates can
    have different feedback transcripts; V is a preference/behavior alignment
    surrogate, NOT a probability distribution over candidate programs.

    V_k(S) = mean_i [(audit_ik - mean_r audit_rk) * delta_i(S)].
    V(empty) = 0 exactly. Reducing confidence in failing programs helps.
    """
    values, audit = _test_scores(scores, audit_outcomes)
    centered = audit - audit.mean(axis=0, keepdims=True)
    shifts = np.stack([x.mean(axis=1) - x[0].mean() for x in values], axis=0)
    result = (shifts.T @ centered) / len(values)
    result[0] = 0.
    return result


def _attribution(game: np.ndarray, *, method: str) -> np.ndarray:
    """Per-test, per-audit-probe signed Shapley or LOO marginals."""
    if game.ndim != 2 or game.shape[0] & (game.shape[0] - 1):
        raise ValueError("bad value game shape")
    n = game.shape[0].bit_length() - 1
    full = game.shape[0] - 1
    if method == 'loo':
        return np.stack([game[full] - game[full ^ (1 << j)] for j in range(n)], axis=0)
    if method != 'shapley':
        raise ValueError('method must be shapley or loo')
    result = np.zeros((n, game.shape[1]))
    for j in range(n):
        for subset in range(1 << n):
            if subset & (1 << j):
                continue
            k = subset.bit_count()
            weight = math.factorial(k) * math.factorial(n-k-1) / math.factorial(n)
            result[j] += weight * (game[subset | (1 << j)] - game[subset])
    if not np.allclose(result.sum(axis=0), game[-1]-game[0], atol=1e-9):
        raise AssertionError('Shapley efficiency mismatch')
    return result


def _mean_se(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=-1)
    std_err = values.std(axis=-1, ddof=1) / math.sqrt(values.shape[-1])
    return mean, std_err


def select_feedback(
    scores: Sequence[np.ndarray], audit_outcomes: np.ndarray, *,
    method: str = 'shapley', uncertainty_penalty: float = 0.5,
    subset_cost: float = 0.001, min_gain: float = 0.001,
) -> Selection:
    """Select a beneficial feedback subset or abstain, using audit cross-checks.

    Positive per-test *lower* attribution determines the eligible evidence.
    Joint interactions are handled by scoring all admissible subsets, rather
    than summing Shapley numbers (which would merely reproduce the full effect).
    Independent audit probes are training-side verification, NOT final test data.
    """
    if any(not math.isfinite(v) or v < 0 for v in (uncertainty_penalty, subset_cost, min_gain)):
        raise ValueError('penalties/gain thresholds must be finite and >=0')
    game = audit_value_game(scores, audit_outcomes)
    phi_matrix = _attribution(game, method=method)
    phi, se_phi = _mean_se(phi_matrix)
    mean, se = _mean_se(game)
    cost = np.asarray([subset.bit_count()*subset_cost for subset in range(game.shape[0])])
    conservative = mean - uncertainty_penalty*se - cost
    contrast = float(np.mean(np.var(np.asarray(audit_outcomes), axis=0)))
    if contrast < 1e-12:
        reason = 'no_cross_rollout_audit_disagreement'
        mask = 0
    else:
        eligible = sum((1 << j) for j in range(len(phi)) if phi[j] > uncertainty_penalty*se_phi[j])
        choices = [s for s in range(1,game.shape[0]) if s & ~eligible == 0]
        mask = max(choices, key=lambda s: (conservative[s], -s.bit_count(), -s), default=0)
        reason = 'positive_behavioral_evidence' if mask and conservative[mask] > min_gain else 'insufficient_behavioral_evidence'
        if reason != 'positive_behavioral_evidence':
            mask = 0
    return Selection(mask=mask,phi=phi,phi_stderr=se_phi, value_by_subset=mean,
                     score_by_subset=conservative, reason=reason, audit_contrast=contrast)


def directional_token_weights(
    scores: Sequence[np.ndarray], audit_outcomes: np.ndarray, subset_mask: int, *,
    shift_scale: float=0.15,
) -> list[np.ndarray]:
    """Sparse-but-token-local, confidence-PRESERVING learning weights.

    Good relative audit candidate and higher teacher probability => promote;
    poor relative candidate and lower teacher probability => suppress.
    We do NOT normalize each rollout's weights back to unit mean; zero/weak
    audit evidence genuinely suppresses the entire update.
    """
    values, audit = _test_scores(scores, audit_outcomes)
    if not np.isfinite(shift_scale) or shift_scale <= 0:
        raise ValueError('shift_scale must be positive')
    if not 0 <= subset_mask < len(values[0]):
        raise ValueError('invalid subset mask')
    advantage = audit.mean(axis=1) - audit.mean()
    if subset_mask == 0:
        return [np.zeros(x.shape[1],dtype=np.float32) for x in values]
    result = []
    for i, x in enumerate(values):
        delta = x[subset_mask] - x[0]
        aligned = np.maximum(0., np.sign(advantage[i])*delta)
        result.append((abs(advantage[i])*np.clip(aligned/shift_scale,0.,1.)).astype(np.float32))
    return result


def full_feedback_audit_gate(scores: Sequence[np.ndarray], audit_outcomes: np.ndarray,
                             *, min_gain: float=0.001) -> Selection:
    """Matched audit-using baseline: use ALL feedback, without Shapley selection."""
    game = audit_value_game(scores,audit_outcomes)
    mean,_ = _mean_se(game)
    contrast = float(np.mean(np.var(np.asarray(audit_outcomes),axis=0)))
    full=len(game)-1
    ok=contrast>1e-12 and mean[full]>min_gain
    n=full.bit_length()
    return Selection(mask=full if ok else 0,phi=np.zeros(n),
        phi_stderr=np.zeros(n),value_by_subset=mean,
        score_by_subset=mean,reason='audit_full_feedback' if ok else 'audit_gate_abstain',
        audit_contrast=contrast)
