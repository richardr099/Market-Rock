"""Portfolio backtester for genomes (one position at a time, like the live strategy).

Decision timing (identical in C#): conditions are evaluated when bar i is
complete; an entry fills at the open of bar i+1. Entries are never carried
across a session boundary. When several genomes signal on the same bar, the
first in priority order (portfolio order) wins.

Conventions (pessimistic on purpose):
  * stop and target touched in the same bar -> the stop is assumed hit;
  * slippage on every entry and exit, commission round-trip;
  * MTM equity is checked at each bar close against the daily loss limit
    and the trailing drawdown; a trailing breach ends the run ("blown").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import genome as G
from ._jit import jit
from .data import Bars
from .features import Features, compute

EXIT_STOP, EXIT_TARGET, EXIT_TIME, EXIT_SESSION, EXIT_DAILY, EXIT_TRAIL = 1, 2, 3, 4, 5, 6
VA_PCT = 0.70  # fixed globally: one shared volume profile for every genome


@dataclass(frozen=True)
class Costs:
    tick: float = 0.25           # ES
    tick_value: float = 12.50    # ES
    commission_rt: float = 4.50  # per contract, round trip
    slip_ticks: float = 1.0      # per side


@dataclass(frozen=True)
class Limits:
    """Mirror of the C# hard limits, used only to make backtests honest."""
    daily_loss_usd: float = 1000.0
    trailing_dd_usd: float = 2500.0


@jit
def _simulate(session, open_, high, low, close, atr, sig_dir, sig_g,
              g_stop, g_rr, g_hold, g_risk,
              tick, tick_value, comm_rt, slip, daily_loss, trail_dd):
    m = len(close)
    cap = m // 2 + 1
    t_entry = np.empty(cap, dtype=np.int64)
    t_exit = np.empty(cap, dtype=np.int64)
    t_dir = np.empty(cap, dtype=np.int64)
    t_qty = np.empty(cap, dtype=np.int64)
    t_pnl = np.empty(cap)
    t_risk = np.empty(cap)
    t_g = np.empty(cap, dtype=np.int64)
    t_why = np.empty(cap, dtype=np.int64)
    nt = 0
    pos = 0
    qty = 0
    entry_px = 0.0
    stop_px = 0.0
    tgt_px = 0.0
    entry_i = -1
    cur_g = -1
    cur_risk = 0.0
    pend_dir = 0
    pend_qty = 0
    pend_st = 0
    pend_tt = 0
    pend_g = -1
    realized = 0.0
    hwm = 0.0
    day_start = 0.0
    locked_day = False
    blown = False
    pv = tick_value / tick
    for j in range(m):
        new_session = j == 0 or session[j] != session[j - 1]
        last_of_session = j == m - 1 or session[j + 1] != session[j]
        if new_session:
            day_start = realized
            locked_day = False
        if pend_dir != 0:
            pos = pend_dir
            qty = pend_qty
            entry_px = open_[j] + pos * slip * tick
            stop_px = entry_px - pos * pend_st * tick
            tgt_px = entry_px + pos * pend_tt * tick
            entry_i = j
            cur_g = pend_g
            cur_risk = pend_st * tick_value * pend_qty
            pend_dir = 0
        exit_px = 0.0
        why = 0
        if pos != 0:
            if (pos == 1 and low[j] <= stop_px) or (pos == -1 and high[j] >= stop_px):
                exit_px = stop_px - pos * slip * tick
                why = EXIT_STOP
            elif (pos == 1 and high[j] >= tgt_px) or (pos == -1 and low[j] <= tgt_px):
                exit_px = tgt_px - pos * slip * tick
                why = EXIT_TARGET
            elif j - entry_i + 1 >= g_hold[cur_g]:
                exit_px = close[j] - pos * slip * tick
                why = EXIT_TIME
            elif last_of_session:
                exit_px = close[j] - pos * slip * tick
                why = EXIT_SESSION
        if pos != 0 and why == 0:
            mtm = realized + pos * (close[j] - entry_px) * pv * qty - comm_rt * qty
            if mtm - day_start <= -daily_loss:
                exit_px = close[j] - pos * slip * tick
                why = EXIT_DAILY
            elif mtm <= hwm - trail_dd:
                exit_px = close[j] - pos * slip * tick
                why = EXIT_TRAIL
        if why != 0:
            pnl = pos * (exit_px - entry_px) * pv * qty - comm_rt * qty
            realized += pnl
            t_entry[nt] = entry_i
            t_exit[nt] = j
            t_dir[nt] = pos
            t_qty[nt] = qty
            t_pnl[nt] = pnl
            t_risk[nt] = cur_risk
            t_g[nt] = cur_g
            t_why[nt] = why
            nt += 1
            pos = 0
        eq = realized if pos == 0 else realized + pos * (close[j] - entry_px) * pv * qty - comm_rt * qty
        if eq > hwm:
            hwm = eq
        if realized - day_start <= -daily_loss:
            locked_day = True
        if eq <= hwm - trail_dd:
            blown = True
            break
        if pos == 0 and not locked_day and not last_of_session and sig_dir[j] != 0:
            g = sig_g[j]
            st = max(1, np.int64(np.floor(g_stop[g] * atr[j] / tick + 0.5)))
            q = np.int64(np.floor(g_risk[g] / (st * tick_value)))
            if q >= 1:
                pend_dir = sig_dir[j]
                pend_qty = q
                pend_st = st
                pend_tt = max(1, np.int64(np.floor(st * g_rr[g] + 0.5)))
                pend_g = g
    return (t_entry[:nt], t_exit[:nt], t_dir[:nt], t_qty[:nt], t_pnl[:nt],
            t_risk[:nt], t_g[:nt], t_why[:nt], blown)


@dataclass(frozen=True)
class Result:
    entry: np.ndarray
    exit: np.ndarray
    direction: np.ndarray
    qty: np.ndarray
    pnl: np.ndarray
    risk: np.ndarray       # planned $ risk at entry (stop distance x qty)
    genome: np.ndarray     # index into the genome list
    why: np.ndarray
    blown: bool
    exit_session: np.ndarray
    entry_time: np.ndarray

    @property
    def n(self) -> int:
        return len(self.pnl)

    @property
    def r_multiple(self) -> np.ndarray:
        return self.pnl / np.maximum(self.risk, 1e-9)

    def daily_pnl(self, sessions: np.ndarray) -> np.ndarray:
        out = np.zeros(len(sessions))
        np.add.at(out, np.searchsorted(sessions, self.exit_session), self.pnl)
        return out


def combine(genomes: Sequence[G.Genome], close: np.ndarray, f: Features):
    """Priority combination: per bar, the first genome that signals."""
    warm = G.warm_mask(close, f)
    sig_dir = np.zeros(len(close), dtype=np.int64)
    sig_g = np.full(len(close), -1, dtype=np.int64)
    for k, g in enumerate(genomes):
        s = G.signal(g, close, f, warm)
        take = (sig_g < 0) & (s != 0)
        sig_dir[take] = s[take]
        sig_g[take] = k
    return sig_dir, sig_g


def run(b: Bars, genomes: Sequence[G.Genome], risk_usd: Sequence[float], costs: Costs = Costs(),
        limits: Limits = Limits(), feats: Features | None = None) -> Result:
    if not genomes:
        e = np.zeros(0, dtype=np.int64)
        return Result(e, e, e, e, np.zeros(0), np.zeros(0), e, e, False, e, e)
    f = feats if feats is not None else compute(b, costs.tick, VA_PCT)
    sig_dir, sig_g = combine(genomes, b.close, f)
    e, x, d, q, pnl, risk, gi, w, blown = _simulate(
        b.session, b.open, b.high, b.low, b.close, f.atr, sig_dir, sig_g,
        np.array([g.stop_atr for g in genomes]), np.array([g.target_rr for g in genomes]),
        np.array([g.max_hold for g in genomes], dtype=np.int64), np.asarray(risk_usd, dtype=np.float64),
        costs.tick, costs.tick_value, costs.commission_rt, costs.slip_ticks,
        limits.daily_loss_usd, limits.trailing_dd_usd)
    return Result(e, x, d, q, pnl, risk, gi, w, bool(blown), b.session[x], b.time[e])
