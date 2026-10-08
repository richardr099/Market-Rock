"""Regime filter, Auction-Market-Theory signals, and a conservative backtester.

Decision timing (identical in C#): features are evaluated when bar i is
complete; any entry fills at the open of bar i+1. Entries are never carried
across a session boundary.

Signals (prior session's value area, delta z-score confirmation):
  ROTATIONAL regime (low efficiency ratio) - failed auction back into value:
    long : close[i-1] < VAL and close[i] >= VAL and delta_z[i] >=  thr
    short: close[i-1] > VAH and close[i] <= VAH and delta_z[i] <= -thr
  TREND regime (high efficiency ratio) - initiative acceptance outside value:
    long : close[i-1] <= VAH and close[i] > VAH and delta_z[i] >=  thr
    short: close[i-1] >= VAL and close[i] < VAL and delta_z[i] <= -thr
  BLOCK regime: volatility percentile outside [vol_lo, vol_hi] or warm-up.

Backtest conventions (chosen to be pessimistic):
  * stop and target touched in the same bar -> the stop is assumed hit;
  * slippage applied on every entry and every exit, commission round-trip;
  * MTM equity checked at every bar close against the daily loss limit and
    the trailing drawdown; a trailing-drawdown breach ends the run ("blown").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ._jit import jit
from .data import Bars
from .features import Features, compute

BLOCK, ROTATIONAL, TREND = 0, 1, 2
EXIT_STOP, EXIT_TARGET, EXIT_TIME, EXIT_SESSION, EXIT_DAILY, EXIT_TRAIL = 1, 2, 3, 4, 5, 6


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
def regimes(vol_pct, er, atr, vah, val, vol_lo, vol_hi, er_trend):
    m = len(er)
    out = np.zeros(m, dtype=np.int64)
    for i in range(m):
        if (np.isnan(vol_pct[i]) or np.isnan(er[i]) or np.isnan(atr[i])
                or np.isnan(vah[i]) or np.isnan(val[i])):
            continue
        if vol_pct[i] < vol_lo or vol_pct[i] > vol_hi:
            continue
        out[i] = TREND if er[i] >= er_trend else ROTATIONAL
    return out


@jit
def signals(close, delta_z, vah, val, regime, thr, en_rot, en_trend):
    m = len(close)
    out = np.zeros(m, dtype=np.int64)
    for i in range(1, m):
        r = regime[i]
        if r == BLOCK or np.isnan(delta_z[i]):
            continue
        p, c, z = close[i - 1], close[i], delta_z[i]
        if r == ROTATIONAL and en_rot:
            if p < val[i] and c >= val[i] and z >= thr:
                out[i] = 1
            elif p > vah[i] and c <= vah[i] and z <= -thr:
                out[i] = -1
        elif r == TREND and en_trend:
            if p <= vah[i] and c > vah[i] and z >= thr:
                out[i] = 1
            elif p >= val[i] and c < val[i] and z <= -thr:
                out[i] = -1
    return out


@jit
def _simulate(session, open_, high, low, close, atr, sig, regime,
              stop_atr, target_rr, max_hold, risk_usd, max_qty, mult_rot, mult_trend,
              tick, tick_value, comm_rt, slip, daily_loss, trail_dd):
    m = len(close)
    cap = m // 2 + 1
    t_entry = np.empty(cap, dtype=np.int64)
    t_exit = np.empty(cap, dtype=np.int64)
    t_dir = np.empty(cap, dtype=np.int64)
    t_qty = np.empty(cap, dtype=np.int64)
    t_pnl = np.empty(cap)
    t_reg = np.empty(cap, dtype=np.int64)
    t_why = np.empty(cap, dtype=np.int64)
    nt = 0
    pos = 0
    qty = 0
    entry_px = 0.0
    stop_px = 0.0
    tgt_px = 0.0
    entry_i = -1
    entry_reg = 0
    pend_dir = 0
    pend_qty = 0
    pend_stop_t = 0
    pend_tgt_t = 0
    pend_reg = 0
    realized = 0.0
    hwm = 0.0
    day_start_eq = 0.0
    locked_day = False
    blown = False
    pv = tick_value / tick  # $ per point per contract
    for j in range(m):
        new_session = j == 0 or session[j] != session[j - 1]
        last_of_session = j == m - 1 or session[j + 1] != session[j]
        if new_session:
            day_start_eq = realized
            locked_day = False
        # 1) fill pending entry at this bar's open
        if pend_dir != 0:
            pos = pend_dir
            qty = pend_qty
            entry_px = open_[j] + pos * slip * tick
            stop_px = entry_px - pos * pend_stop_t * tick
            tgt_px = entry_px + pos * pend_tgt_t * tick
            entry_i = j
            entry_reg = pend_reg
            pend_dir = 0
        # 2) manage open position on this bar
        exit_px = 0.0
        why = 0
        if pos != 0:
            if (pos == 1 and low[j] <= stop_px) or (pos == -1 and high[j] >= stop_px):
                exit_px = stop_px - pos * slip * tick
                why = EXIT_STOP
            elif (pos == 1 and high[j] >= tgt_px) or (pos == -1 and low[j] <= tgt_px):
                exit_px = tgt_px - pos * slip * tick
                why = EXIT_TARGET
            elif j - entry_i + 1 >= max_hold:
                exit_px = close[j] - pos * slip * tick
                why = EXIT_TIME
            elif last_of_session:
                exit_px = close[j] - pos * slip * tick
                why = EXIT_SESSION
        # 3) MTM risk checks at close (only if still open)
        if pos != 0 and why == 0:
            mtm = realized + pos * (close[j] - entry_px) * pv * qty - comm_rt * qty
            if mtm - day_start_eq <= -daily_loss:
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
            t_reg[nt] = entry_reg
            t_why[nt] = why
            nt += 1
            pos = 0
        eq = realized if pos == 0 else realized + pos * (close[j] - entry_px) * pv * qty - comm_rt * qty
        if eq > hwm:
            hwm = eq
        if realized - day_start_eq <= -daily_loss:
            locked_day = True
        if eq <= hwm - trail_dd:
            blown = True
            break
        # 4) new signal -> pending entry for next bar
        if pos == 0 and not locked_day and not last_of_session and sig[j] != 0 and not np.isnan(atr[j]):
            mult = mult_rot if regime[j] == ROTATIONAL else mult_trend
            st = max(1, np.int64(np.floor(stop_atr * atr[j] / tick + 0.5)))
            q = np.int64(np.floor(risk_usd * mult / (st * tick_value)))
            q = min(q, max_qty)
            if q >= 1:
                pend_dir = sig[j]
                pend_qty = q
                pend_stop_t = st
                pend_tgt_t = max(1, np.int64(np.floor(st * target_rr + 0.5)))
                pend_reg = regime[j]
    return (t_entry[:nt], t_exit[:nt], t_dir[:nt], t_qty[:nt], t_pnl[:nt],
            t_reg[:nt], t_why[:nt], blown)


@dataclass(frozen=True)
class Result:
    entry: np.ndarray
    exit: np.ndarray
    direction: np.ndarray
    qty: np.ndarray
    pnl: np.ndarray
    regime: np.ndarray
    why: np.ndarray
    blown: bool
    exit_session: np.ndarray

    @property
    def n(self) -> int:
        return len(self.pnl)

    def daily_pnl(self, sessions: np.ndarray) -> np.ndarray:
        """PnL per session id in ``sessions`` (zero on days without trades)."""
        out = np.zeros(len(sessions))
        idx = np.searchsorted(sessions, self.exit_session)
        np.add.at(out, idx, self.pnl)
        return out


def run(b: Bars, p: Mapping[str, float], costs: Costs = Costs(), limits: Limits = Limits(),
        feats: Features | None = None) -> Result:
    f = feats if feats is not None else compute(b, costs.tick, p["va_pct"])
    reg = regimes(f.vol_pct, f.er, f.atr, f.vah, f.val, p["regime_vol_lo"], p["regime_vol_hi"], p["er_trend"])
    sig = signals(b.close, f.delta_z, f.vah, f.val, reg, p["delta_z_entry"],
                  bool(p["enable_rotational"]), bool(p["enable_trend"]))
    e, x, d, q, pnl, r, w, blown = _simulate(
        b.session, b.open, b.high, b.low, b.close, f.atr, sig, reg,
        p["stop_atr"], p["target_rr"], int(p["max_hold_bars"]), p["risk_per_trade_usd"],
        int(p["max_contracts"]), p["size_mult_rotational"], p["size_mult_trend"],
        costs.tick, costs.tick_value, costs.commission_rt, costs.slip_ticks,
        limits.daily_loss_usd, limits.trailing_dd_usd)
    return Result(e, x, d, q, pnl, r, w, bool(blown), b.session[x])
