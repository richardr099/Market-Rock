"""The self-improvement loop (math in docs/SELF_IMPROVEMENT.md).

Two independent learners, deliberately kept separate:

1. Trust-region evolution strategy over the ALPHA parameters (SEARCH_NAMES).
   Proposes; never applies. Its output goes through the promotion gate and
   then a human.

2. Bayesian regime posteriors from LIVE fills. Produces per-regime enable flags
   and size multipliers via a lower-confidence-bound Kelly fraction. These may
   be applied autonomously ONLY in the risk-reducing direction (autonomy.py).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

import numpy as np

from . import params as P
from .data import Bars
from .features import Features, compute, session_profiles
from .strategy import ROTATIONAL, TREND, Costs, Limits, Result, run
from .validation import dsr, purged_folds, sharpe

# ---------------------------------------------------------------------------
# Search configuration (constants, not learned)
# ---------------------------------------------------------------------------
N_CANDIDATES = 24        # lambda: candidates per session
K_FOLDS = 5
EMBARGO_SESSIONS = 1
LAMBDA_STD = 0.5         # stability penalty in fitness
MIN_TRADES = 60          # on the search set
SIGMA_MIN, SIGMA_MAX, SIGMA_INIT = 0.01, 0.10, 0.05

# Gate thresholds
DSR_MIN = 0.95
MIN_HOLDOUT_TRADES = 15
HOLDOUT_DD_FRACTION = 0.5  # holdout max drawdown must be <= this x trailing limit


@dataclass
class Evaluation:
    params: Dict[str, float]
    fitness: float
    fold_sharpes: List[float]
    n_trades: int
    blown: bool
    daily: np.ndarray = field(repr=False)


def _feats_for(base: Features, b: Bars, tick: float, va_pct: float) -> Features:
    poc, vah, val = session_profiles(b.session, b.high, b.low, b.volume, tick, va_pct)
    return dataclasses.replace(base, poc=poc, vah=vah, val=val)


def evaluate(b: Bars, p: Mapping[str, float], costs: Costs, limits: Limits,
             base: Features | None = None) -> Evaluation:
    f = _feats_for(base, b, costs.tick, p["va_pct"]) if base is not None else compute(b, costs.tick, p["va_pct"])
    r: Result = run(b, p, costs, limits, f)
    sessions = np.unique(b.session)
    daily = r.daily_pnl(sessions)
    folds = purged_folds(len(sessions), K_FOLDS, EMBARGO_SESSIONS)
    fs = [sharpe(daily[test]) for _, test in folds]
    if r.blown or r.n < MIN_TRADES:
        fit = -np.inf
    else:
        fit = float(np.mean(fs) - LAMBDA_STD * np.std(fs))
    return Evaluation(dict(p), fit, fs, r.n, r.blown, daily)


def evaluate_holdout(b_full: Bars, p: Mapping[str, float], hold_sessions: np.ndarray,
                     costs: Costs, limits: Limits) -> Evaluation:
    """Run on the FULL history (causal features need the warm-up) but score
    only trades that exit in the hold-out sessions. The search never sees
    these sessions: it runs on a slice that ends before them."""
    r = run(b_full, p, costs, limits)
    daily = r.daily_pnl(np.unique(b_full.session))
    mask = np.isin(np.unique(b_full.session), hold_sessions)
    n_hold = int(np.isin(r.exit_session, hold_sessions).sum())
    return Evaluation(dict(p), float("nan"), [], n_hold, r.blown, daily[mask])


def seed_for(*parts: bytes | str | int) -> int:
    h = hashlib.sha256()
    for x in parts:
        h.update(x if isinstance(x, bytes) else str(x).encode())
        h.update(b"\x00")
    return int.from_bytes(h.digest()[:8], "little")


def propose_candidates(incumbent: Mapping[str, float], n: int, sigma: float,
                       rng: np.random.Generator) -> List[Dict[str, float]]:
    """Gaussian perturbation in unit space, clipped to a per-parameter trust
    region |u_i - u0_i| <= max_step_i, then to [0, 1]. Invalid draws
    (vol_lo >= vol_hi) are resampled; the stream stays deterministic."""
    u0 = P.to_unit(incumbent)
    maxstep = np.array([P.SPEC_BY_NAME[nm].max_step for nm in P.SEARCH_NAMES])
    out: List[Dict[str, float]] = []
    while len(out) < n:
        step = np.clip(sigma * rng.standard_normal(len(u0)), -maxstep, maxstep)
        cand = P.from_unit(np.clip(u0 + step, 0.0, 1.0), incumbent)
        try:
            out.append(P.validate(cand))
        except ValueError:
            continue
    return out


def search(b: Bars, incumbent: Mapping[str, float], sigma: float, seed: int,
           costs: Costs, limits: Limits, n: int = N_CANDIDATES):
    rng = np.random.default_rng(seed)
    base = compute(b, costs.tick, incumbent["va_pct"])
    inc_eval = evaluate(b, incumbent, costs, limits, base)
    evals = [evaluate(b, c, costs, limits, base) for c in propose_candidates(incumbent, n, sigma, rng)]
    best = max(evals, key=lambda e: e.fitness)
    improved = best.fitness > inc_eval.fitness
    # 1/5th success rule on the step size (Rechenberg), bounded.
    new_sigma = float(np.clip(sigma * np.exp((float(improved) - 0.2) / 1.0), SIGMA_MIN, SIGMA_MAX))
    var_sr = float(np.var([np.mean(e.fold_sharpes) for e in evals if np.isfinite(e.fitness)] or [0.0]))
    return inc_eval, best, evals, new_sigma, var_sr


def max_drawdown(daily: np.ndarray) -> float:
    eq = np.concatenate([[0.0], np.cumsum(daily)])
    return float(np.max(np.maximum.accumulate(eq) - eq))


def gate(cand_search: Evaluation, cand_hold: Evaluation, inc_hold: Evaluation,
         n_trials_total: int, var_sr: float, limits: Limits) -> Dict[str, Dict]:
    """Every check must pass. Returned verbatim into the candidate report."""
    d = dsr(cand_search.daily, n_trials_total, var_sr if var_sr > 0 else None)
    pos_folds = sum(1 for s in cand_search.fold_sharpes if s > 0)
    hs_c, hs_i = sharpe(cand_hold.daily), sharpe(inc_hold.daily)
    hold_dd = max_drawdown(cand_hold.daily)
    hold_trades = cand_hold.n_trades
    checks = {
        "search_finite_fitness": {"pass": bool(np.isfinite(cand_search.fitness)), "value": cand_search.fitness},
        "deflated_sharpe": {"pass": d >= DSR_MIN, "value": d, "threshold": DSR_MIN, "n_trials": n_trials_total},
        "fold_stability": {"pass": pos_folds >= K_FOLDS - 1, "value": pos_folds, "threshold": K_FOLDS - 1},
        "holdout_not_blown": {"pass": not cand_hold.blown, "value": cand_hold.blown},
        "holdout_trades": {"pass": hold_trades >= MIN_HOLDOUT_TRADES, "value": hold_trades, "threshold": MIN_HOLDOUT_TRADES},
        "holdout_profitable": {"pass": float(cand_hold.daily.sum()) > 0, "value": float(cand_hold.daily.sum())},
        "holdout_beats_incumbent": {"pass": hs_c >= hs_i, "value": hs_c, "incumbent": hs_i},
        "holdout_drawdown": {"pass": hold_dd <= HOLDOUT_DD_FRACTION * limits.trailing_dd_usd,
                             "value": hold_dd, "threshold": HOLDOUT_DD_FRACTION * limits.trailing_dd_usd},
    }
    for v in checks.values():  # JSON-safe
        for k, x in list(v.items()):
            if isinstance(x, (np.floating, np.integer)):
                v[k] = x.item()
            if isinstance(x, float) and not np.isfinite(x):
                v[k] = str(x)
    return checks


# ---------------------------------------------------------------------------
# Bayesian regime layer
# ---------------------------------------------------------------------------
PRIOR_A, PRIOR_B = 2.0, 2.0
FORGET = 0.97            # per-session decay of evidence toward the prior
F_FULL = 0.10            # LCB-Kelly at which a regime earns full size
M_PROBATION = 0.25       # minimum size while evidence is not conclusively negative
Q_LO, Q_HI = 0.05, 0.95
_MC_DRAWS = 200_000


@dataclass
class RegimeState:
    a: float = PRIOR_A
    b: float = PRIOR_B
    win_sum: float = 0.0
    win_n: float = 0.0
    loss_sum: float = 0.0
    loss_n: float = 0.0
    last_session: int = 0


def _beta_q(a: float, b: float, qs: Sequence[float]) -> List[float]:
    rng = np.random.default_rng(1234567)  # fixed: results must be reproducible
    return [float(x) for x in np.quantile(rng.beta(a, b, _MC_DRAWS), qs)]


def update_regime(st: RegimeState, trades: Sequence[Mapping]) -> RegimeState:
    """trades: dicts with session, pnl_usd; chronological. Applies forgetting
    once per new session, then adds the session's wins/losses."""
    st = dataclasses.replace(st)
    for t in trades:
        s = int(t["session"])
        if s < st.last_session:
            raise ValueError("trades must be chronological")
        if s > st.last_session:
            if st.last_session:
                st.a = PRIOR_A + FORGET * (st.a - PRIOR_A)
                st.b = PRIOR_B + FORGET * (st.b - PRIOR_B)
                st.win_sum *= FORGET
                st.win_n *= FORGET
                st.loss_sum *= FORGET
                st.loss_n *= FORGET
            st.last_session = s
        pnl = float(t["pnl_usd"])
        if pnl > 0:
            st.a += 1
            st.win_sum += pnl
            st.win_n += 1
        else:
            st.b += 1
            st.loss_sum += -pnl
            st.loss_n += 1
    return st


def regime_sizing(st: RegimeState, default_payoff: float) -> Dict[str, float]:
    payoff = default_payoff
    if st.win_n > 0 and st.loss_n > 0 and st.loss_sum > 0:
        payoff = (st.win_sum / st.win_n) / (st.loss_sum / st.loss_n)
    p_lo, p_hi = _beta_q(st.a, st.b, (Q_LO, Q_HI))
    k_lo = p_lo - (1 - p_lo) / payoff
    k_hi = p_hi - (1 - p_hi) / payoff
    if k_hi <= 0:
        enable, mult = 0.0, 0.0
    else:
        enable, mult = 1.0, float(np.clip(k_lo / F_FULL, M_PROBATION, 1.0))
    return {"enable": enable, "mult": mult, "p_lo": p_lo, "p_hi": p_hi,
            "kelly_lo": k_lo, "kelly_hi": k_hi, "payoff": payoff}


REGIME_KEYS = {ROTATIONAL: ("enable_rotational", "size_mult_rotational"),
               TREND: ("enable_trend", "size_mult_trend")}


def bayes_proposal(incumbent: Mapping[str, float], states: Dict[int, RegimeState]) -> tuple[Dict[str, float], Dict]:
    out = dict(incumbent)
    report = {}
    for reg, (ek, mk) in REGIME_KEYS.items():
        z = regime_sizing(states.get(reg, RegimeState()), incumbent["target_rr"])
        out[ek] = z["enable"]
        out[mk] = round(z["mult"], 4)
        report[ek.replace("enable_", "")] = z
    return P.validate(out), report


def states_to_json(states: Dict[int, RegimeState]) -> str:
    return json.dumps({str(k): dataclasses.asdict(v) for k, v in sorted(states.items())}, sort_keys=True, indent=1)


def states_from_json(s: str) -> Dict[int, RegimeState]:
    return {int(k): RegimeState(**v) for k, v in json.loads(s).items()}
