"""Counterfactual test-to-token credit for recursive EESD.

Score array V[S, t] holds a FROZEN teacher's log probability for the
ACTUALLY SAMPLED code token t given only feedback tests in subset S.
This is a model-preference attribution, not a correctness certificate.
"""
from __future__ import annotations

import math
import numpy as np


def exact_token_shapley(values: np.ndarray, tests: int) -> np.ndarray:
    """Return signed test-by-token Shapley, shape [tests, tokens]."""
    v = np.asarray(values, dtype=np.float64)
    if tests < 1 or tests > 8 or v.ndim != 2 or v.shape[0] != 1 << tests or v.shape[1] < 1:
        raise ValueError('require all 2**tests subset scores and >=1 token (tests in 1..8)')
    if not np.isfinite(v).all():
        raise ValueError('nonfinite teacher scores')
    out = np.zeros((tests, v.shape[1]), dtype=np.float64)
    for j in range(tests):
        for mask in range(1 << tests):
            if mask & (1 << j):
                continue
            k = mask.bit_count()
            coefficient = math.factorial(k) * math.factorial(tests-k-1) / math.factorial(tests)
            out[j] += coefficient * (v[mask | (1 << j)] - v[mask])
    if not np.allclose(out.sum(axis=0), v[-1]-v[0], rtol=1e-8, atol=1e-8):
        raise AssertionError('Shapley efficiency violated')
    return out


def loo_token_attribution(values: np.ndarray, tests: int) -> np.ndarray:
    """Leave-one-test-out, not Shapley. Input only needs full and each leave-out."""
    v = np.asarray(values, dtype=np.float64)
    if tests < 1 or v.ndim != 2 or v.shape[0] != 1 << tests or not np.isfinite(v).all():
        raise ValueError('invalid subset score array')
    full = (1 << tests)-1
    return np.stack([v[full]-v[full ^ (1 << j)] for j in range(tests)])


def weights_from_attribution(phi: np.ndarray, *, amplitude_floor: float = .15,
                             eps: float = 1e-8) -> tuple[np.ndarray, dict]:
    """Direction-preserving token gate: coherence x effective-support x strength.

    Negative phi is *not* interpreted as untrustworthy: signed contributions
    determine agreement/cancellation; teacher full-token distribution carries
    both positive and negative probability changes. Zero effect implies zero
    distillation weight (no uniform maximum-confidence fallback).
    """
    p = np.asarray(phi, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 1 or p.shape[1] < 1 or not np.isfinite(p).all():
        raise ValueError('expected finite [tests,tokens] attribution')
    if not np.isfinite(amplitude_floor) or amplitude_floor <= 0:
        raise ValueError('positive finite amplitude_floor required')
    mag = np.abs(p)
    total = mag.sum(axis=0)
    concentration = np.divide((mag**2).sum(axis=0), total**2,
                              out=np.zeros_like(total), where=total > eps)
    effective = np.divide(1., concentration, out=np.zeros_like(total), where=concentration > eps)
    coherence = np.divide(np.abs(p.sum(axis=0)), total, out=np.zeros_like(total), where=total > eps)
    strength = np.minimum(1., total/amplitude_floor)
    raw = effective / p.shape[0] * coherence * strength
    raw[total <= eps] = 0.
    # Same average token training weight as SDPO, while preserving selective credit.
    # If no meaningful feedback-induced change exists, skip distillation entirely.
    w = raw / raw.mean() if raw.mean() > eps else np.zeros_like(raw)
    return w.astype(np.float32), {
        'active_fraction': float(np.mean(raw > eps)),
        'effective_mass_mean': float(np.mean(effective)),
        'coherence_mean': float(np.mean(coherence)),
        'negative_shapley_fraction': float(np.mean(p < -eps)),
        'mean_absolute_feedback_shift': float(np.mean(total)),
        'zero_feedback_signal': bool(np.all(raw <= eps)),
    }


def credit_from_scores(values: np.ndarray, tests: int, *, method: str,
                       amplitude_floor: float = .15) -> tuple[np.ndarray, np.ndarray, dict]:
    if method == 'shapley':
        phi = exact_token_shapley(values, tests)
    elif method == 'loo':
        phi = loo_token_attribution(values, tests)
    else:
        raise ValueError('method must be shapley or loo')
    weights, report = weights_from_attribution(phi, amplitude_floor=amplitude_floor)
    return phi, weights, report


def coalition_masks(tests: int, method: str) -> list[int]:
    if tests < 1 or tests > 8:
        raise ValueError('tests must be in 1..8')
    full = (1 << tests)-1
    if method == 'shapley':
        return list(range(1 << tests))
    if method == 'loo':
        return [full] + [full ^ (1 << j) for j in range(tests)]
    if method == 'sdpo':
        return [full]
    raise ValueError('method not recognized')
