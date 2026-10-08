"""Causal order-flow / auction features.

Every function here is CAUSAL: the value at bar i depends only on bars <= i.
(tests/test_features.py::test_no_lookahead perturbs the future and asserts the
past is unchanged.) Each is mirrored line-for-line in MarketRockStrategy.cs;
change one, change both.

Parity rules shared with C#:
  * price -> tick level uses floor(p / tick + 0.5), never banker's rounding
    (Python round() and C# Math.Round both round half to even by default).
  * population std (divide by n), percentile rank = share of the previous L
    values strictly below the current one.
  * POC ties resolve to the LOWEST price; value-area expansion ties go UP.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._jit import jit
from .data import Bars

ATR_N = 14
DELTA_Z_N = 30
VOL_PCT_L = 1000
ER_N = 20


@jit
def atr_wilder(high, low, close, n):
    m = len(close)
    out = np.full(m, np.nan)
    if m < n:
        return out
    tr = np.empty(m)
    tr[0] = high[0] - low[0]
    for i in range(1, m):
        a = high[i] - low[i]
        b = abs(high[i] - close[i - 1])
        c = abs(low[i] - close[i - 1])
        tr[i] = max(a, max(b, c))
    s = 0.0
    for i in range(n):
        s += tr[i]
    out[n - 1] = s / n
    for i in range(n, m):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


@jit
def rolling_z(x, n):
    m = len(x)
    out = np.full(m, np.nan)
    for i in range(n - 1, m):
        s = 0.0
        ss = 0.0
        for k in range(i - n + 1, i + 1):
            s += x[k]
            ss += x[k] * x[k]
        mu = s / n
        var = ss / n - mu * mu
        out[i] = (x[i] - mu) / np.sqrt(var) if var > 1e-12 else 0.0
    return out


@jit
def pct_rank(x, L):
    m = len(x)
    out = np.full(m, np.nan)
    for i in range(L, m):
        if np.isnan(x[i]):
            continue
        cnt = 0
        valid = 0
        for k in range(i - L, i):
            if not np.isnan(x[k]):
                valid += 1
                if x[k] < x[i]:
                    cnt += 1
        if valid == L:
            out[i] = cnt / L
    return out


@jit
def efficiency_ratio(close, n):
    m = len(close)
    out = np.full(m, np.nan)
    for i in range(n, m):
        path = 0.0
        for k in range(i - n + 1, i + 1):
            path += abs(close[k] - close[k - 1])
        out[i] = abs(close[i] - close[i - n]) / path if path > 0 else 0.0
    return out


@jit
def _level(p, tick):
    return np.int64(np.floor(p / tick + 0.5))


@jit
def session_profiles(session, high, low, volume, tick, va_pct):
    """Per-session (POC, VAH, VAL) from uniformly distributing each bar's
    volume across its high-low ticks. Returns arrays aligned to bars holding
    the PRIOR session's values (NaN in the first session) - no lookahead."""
    m = len(session)
    poc_b = np.full(m, np.nan)
    vah_b = np.full(m, np.nan)
    val_b = np.full(m, np.nan)
    prev_poc = np.nan
    prev_vah = np.nan
    prev_val = np.nan
    start = 0
    while start < m:
        end = start
        while end < m and session[end] == session[start]:
            end += 1
        for i in range(start, end):
            poc_b[i] = prev_poc
            vah_b[i] = prev_vah
            val_b[i] = prev_val
        lo_lvl = _level(low[start], tick)
        hi_lvl = _level(high[start], tick)
        for i in range(start, end):
            lo_lvl = min(lo_lvl, _level(low[i], tick))
            hi_lvl = max(hi_lvl, _level(high[i], tick))
        nlev = hi_lvl - lo_lvl + 1
        hist = np.zeros(nlev)
        for i in range(start, end):
            a = _level(low[i], tick) - lo_lvl
            b = _level(high[i], tick) - lo_lvl
            per = volume[i] / (b - a + 1)
            for k in range(a, b + 1):
                hist[k] += per
        total = 0.0
        poc = 0
        for k in range(nlev):
            total += hist[k]
            if hist[k] > hist[poc]:
                poc = k
        lo_k = poc
        hi_k = poc
        acc = hist[poc]
        target = va_pct * total
        while acc < target and (lo_k > 0 or hi_k < nlev - 1):
            up = hist[hi_k + 1] if hi_k < nlev - 1 else -1.0
            dn = hist[lo_k - 1] if lo_k > 0 else -1.0
            if up >= dn:
                hi_k += 1
                acc += up
            else:
                lo_k -= 1
                acc += dn
        prev_poc = (poc + lo_lvl) * tick
        prev_vah = (hi_k + lo_lvl) * tick
        prev_val = (lo_k + lo_lvl) * tick
        start = end
    return poc_b, vah_b, val_b


@jit
def bar_in_session(session):
    """0 for the first bar of each session, then 1, 2, ..."""
    m = len(session)
    out = np.zeros(m, dtype=np.int64)
    for i in range(1, m):
        out[i] = 0 if session[i] != session[i - 1] else out[i - 1] + 1
    return out


@dataclass(frozen=True)
class Features:
    atr: np.ndarray
    delta: np.ndarray
    delta_z: np.ndarray
    vol_pct: np.ndarray
    er: np.ndarray
    poc: np.ndarray
    vah: np.ndarray
    val: np.ndarray
    bar_in_session: np.ndarray


def compute(b: Bars, tick: float, va_pct: float, vol_pct_l: int = VOL_PCT_L) -> Features:
    atr = atr_wilder(b.high, b.low, b.close, ATR_N)
    delta = b.buy_volume - b.sell_volume
    poc, vah, val = session_profiles(b.session, b.high, b.low, b.volume, tick, va_pct)
    return Features(
        atr=atr,
        delta=delta,
        delta_z=rolling_z(delta, DELTA_Z_N),
        vol_pct=pct_rank(atr / b.close, vol_pct_l),
        er=efficiency_ratio(b.close, ER_N),
        poc=poc, vah=vah, val=val,
        bar_in_session=bar_in_session(b.session),
    )
