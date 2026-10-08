"""Strategy invention: genetic search over genomes, then a statistical gate.

Fitness on the search slice (never the hold-out, never forward data):
    F = mean_k S_k - 0.5 * std_k S_k - 0.01 * n_conditions
S_k = per-session Sharpe in purged fold k. F = -inf if the account is blown
or the genome trades fewer than MIN_TRADES times. Every evaluation counts as
a trial for the Deflated Sharpe Ratio, cumulatively across nights.
"""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import numpy as np

from . import genome as G
from .data import Bars
from .features import Features
from .strategy import Costs, Limits, run
from .validation import dsr, purged_folds, sharpe

EVAL_RISK = 150.0          # fixed $ risk for RANKING only; live sizing is portfolio.py
POP, GENS, ELITE, TOURNAMENT = 32, 4, 4, 3
P_CROSS, P_MUT = 0.5, 0.8
MIN_TRADES = 60
LAMBDA_STD = 0.5
COMPLEXITY = 0.01
K_FOLDS, EMBARGO = 5, 1
DSR_MIN = 0.95
MIN_HOLDOUT_TRADES = 15
HOLDOUT_DD_FRACTION = 0.5


@dataclass
class Eval:
    genome: G.Genome
    fitness: float
    fold_sharpes: List[float]
    n: int
    blown: bool
    daily: np.ndarray = field(repr=False)
    r: np.ndarray = field(repr=False)


def slice_features(f: Features, a: int, b: int) -> Features:
    return Features(**{k.name: getattr(f, k.name)[a:b] for k in dataclasses.fields(f)})


def seed_for(*parts) -> int:
    h = hashlib.sha256()
    for x in parts:
        h.update(x if isinstance(x, bytes) else str(x).encode())
        h.update(b"\x00")
    return int.from_bytes(h.digest()[:8], "little")


def evaluate(b: Bars, g: G.Genome, f: Features, costs: Costs, limits: Limits) -> Eval:
    r = run(b, [g], [EVAL_RISK], costs, limits, f)
    sessions = np.unique(b.session)
    daily = r.daily_pnl(sessions)
    fs = [sharpe(daily[t]) for _, t in purged_folds(len(sessions), K_FOLDS, EMBARGO)]
    fit = -np.inf if (r.blown or r.n < MIN_TRADES) else float(np.mean(fs) - LAMBDA_STD * np.std(fs) - COMPLEXITY * len(g.conds))
    return Eval(g, fit, fs, r.n, r.blown, daily, r.r_multiple)


def evaluate_holdout(b_full: Bars, g: G.Genome, f_full: Features, hold_sessions: np.ndarray,
                     costs: Costs, limits: Limits) -> Eval:
    r = run(b_full, [g], [EVAL_RISK], costs, limits, f_full)
    sessions = np.unique(b_full.session)
    mask = np.isin(sessions, hold_sessions)
    in_hold = np.isin(r.exit_session, hold_sessions)
    return Eval(g, float("nan"), [], int(in_hold.sum()), r.blown, r.daily_pnl(sessions)[mask], r.r_multiple[in_hold])


class Search:
    """Caches evaluations by genome id; counts every new evaluation as a trial."""

    def __init__(self, b: Bars, f: Features, costs: Costs, limits: Limits):
        self.b, self.f, self.costs, self.limits = b, f, costs, limits
        self.cache: Dict[str, Eval] = {}

    def eval(self, g: G.Genome) -> Eval:
        if g.id not in self.cache:
            self.cache[g.id] = evaluate(self.b, g, self.f, self.costs, self.limits)
        return self.cache[g.id]

    @property
    def trials(self) -> int:
        return len(self.cache)

    def evolve(self, starters: Iterable[G.Genome], rng: np.random.Generator) -> List[Eval]:
        pop = list(dict.fromkeys(starters))
        while len(pop) < POP:
            pop.append(G.random_genome(rng))
        evals = [self.eval(g) for g in pop]
        for _ in range(GENS):
            evals.sort(key=lambda e: e.fitness, reverse=True)
            nxt = [e.genome for e in evals[:ELITE]]
            while len(nxt) < POP:
                a = self._tournament(evals, rng)
                child = G.crossover(a, self._tournament(evals, rng), rng) if rng.random() < P_CROSS else a
                if rng.random() < P_MUT or child == a:
                    child = G.mutate(child, rng)
                nxt.append(child)
            evals = [self.eval(g) for g in dict.fromkeys(nxt)]
        return sorted({e.genome.id: e for e in self.cache.values()}.values(), key=lambda e: e.fitness, reverse=True)

    def tune(self, parent: G.Genome, n: int, rng: np.random.Generator) -> List[Eval]:
        kids = {G.mutate(parent, rng, small=True) for _ in range(n)} - {parent}
        return sorted((self.eval(k) for k in kids), key=lambda e: e.fitness, reverse=True)

    @staticmethod
    def _tournament(evals: List[Eval], rng: np.random.Generator) -> G.Genome:
        pick = [evals[i] for i in rng.choice(len(evals), size=min(TOURNAMENT, len(evals)), replace=False)]
        return max(pick, key=lambda e: e.fitness).genome


def max_drawdown(daily: np.ndarray) -> float:
    eq = np.concatenate([[0.0], np.cumsum(daily)])
    return float(np.max(np.maximum.accumulate(eq) - eq))


def gate(es: Eval, eh: Eval, n_trials: int, var_sr: float, limits: Limits) -> Dict[str, Dict]:
    d = dsr(es.daily, n_trials, var_sr if var_sr > 0 else None) if np.isfinite(es.fitness) else 0.0
    pos = sum(1 for s in es.fold_sharpes if s > 0)
    hdd = max_drawdown(eh.daily)
    checks = {
        "search_finite_fitness": {"pass": bool(np.isfinite(es.fitness))},
        "deflated_sharpe": {"pass": bool(d >= DSR_MIN), "value": float(d), "n_trials": int(n_trials)},
        "fold_stability": {"pass": pos >= K_FOLDS - 1, "value": pos},
        "holdout_not_blown": {"pass": not eh.blown},
        "holdout_trades": {"pass": eh.n >= MIN_HOLDOUT_TRADES, "value": eh.n},
        "holdout_profitable": {"pass": bool(eh.daily.sum() > 0), "value": float(eh.daily.sum())},
        "holdout_drawdown": {"pass": hdd <= HOLDOUT_DD_FRACTION * limits.trailing_dd_usd, "value": hdd},
    }
    return checks


def var_sr_of(evals: List[Eval]) -> float:
    xs = [float(np.mean(e.fold_sharpes)) for e in evals if np.isfinite(e.fitness)]
    return float(np.var(xs)) if len(xs) > 1 else 0.0
