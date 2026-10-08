"""Overfitting controls: purged session folds, PSR, Deflated Sharpe Ratio.

References
  Bailey & Lopez de Prado (2012) "The Sharpe Ratio Efficient Frontier" (PSR).
  Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio" (DSR).
  Lopez de Prado (2018) AFML ch.7 (purging / embargo).

All Sharpe ratios here are PER-SESSION (non-annualised); PSR/DSR are defined
on the non-annualised estimate and T = number of sessions.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

EULER_GAMMA = 0.5772156649015329


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Acklam's rational approximation (|rel err| < 1.2e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > hi:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 1e-12 else 0.0


def moments(x: np.ndarray) -> Tuple[float, float]:
    """Sample skewness and (non-excess) kurtosis."""
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    if sd < 1e-12:
        return 0.0, 3.0
    z = (x - x.mean()) / sd
    return float(np.mean(z ** 3)), float(np.mean(z ** 4))


def psr(x: np.ndarray, sr_star: float = 0.0) -> float:
    """P[true SR > sr_star] given the sample (non-normal returns corrected)."""
    t = len(x)
    if t < 3:
        return 0.0
    sr = sharpe(x)
    g3, g4 = moments(x)
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr
    if denom <= 0:
        return 0.0
    return norm_cdf((sr - sr_star) * math.sqrt(t - 1) / math.sqrt(denom))


def expected_max_sr(n_trials: int, var_sr: float) -> float:
    """E[max SR] over n_trials independent zero-skill strategies (DSR eq. for SR0)."""
    if n_trials <= 1:
        return 0.0
    return math.sqrt(max(var_sr, 0.0)) * (
        (1 - EULER_GAMMA) * norm_ppf(1 - 1.0 / n_trials)
        + EULER_GAMMA * norm_ppf(1 - 1.0 / (n_trials * math.e)))


def dsr(x: np.ndarray, n_trials: int, var_sr_trials: float | None = None) -> float:
    """Deflated Sharpe Ratio: PSR against the SR a lucky search would produce.

    ``n_trials`` MUST be the cumulative number of parameter sets ever evaluated
    on overlapping data (persisted in state/trials.json), not just this run's.
    If the cross-trial variance of SR is unknown, use the variance of the SR
    estimator under the null, 1/(T-1).
    """
    t = len(x)
    v = var_sr_trials if var_sr_trials is not None else 1.0 / max(t - 1, 1)
    return psr(x, expected_max_sr(n_trials, v))


def purged_folds(n_sessions: int, k: int, embargo: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """K contiguous session blocks. For each fold returns (train_idx, test_idx),
    with ``embargo`` sessions removed from train on both sides of the test block.
    The parameter search has no in-fold fitting, so here the folds serve as a
    STABILITY test: a candidate must hold up on every block, not on average."""
    if k < 2 or n_sessions < k:
        raise ValueError("need k >= 2 and n_sessions >= k")
    edges = np.linspace(0, n_sessions, k + 1).astype(int)
    out = []
    allidx = np.arange(n_sessions)
    for i in range(k):
        a, b = edges[i], edges[i + 1]
        test = allidx[a:b]
        keep = (allidx < a - embargo) | (allidx >= b + embargo)
        out.append((allidx[keep], test))
    return out
