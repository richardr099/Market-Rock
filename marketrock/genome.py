"""Strategy genomes: strategies as DATA, so new ones deploy without new code.

A genome is a direction, a conjunction of 1..MAX_CONDS conditions over the
causal features, and an exit (stop in ATR multiples, target in R, max hold).
The NT8 strategy interprets the same text, so anything the evolution engine
invents runs on code that tests/test_parity.py has already verified.

Condition vocabulary (evaluated when bar i is complete; p = close[i-1], c = close[i]):
  CROSS_UP L    p <= L and c > L          L in {VAH, VAL, POC} (prior session profile)
  CROSS_DOWN L  p >= L and c < L
  ABOVE L       c > L
  BELOW L       c < L
  GE f x        feature f >= x            f in {delta_z, er, vol_pct}
  LE f x        feature f <= x
  TIME_IN a b   a <= bar_in_session < b

Canonical rule text (hashed; parsed by C#):
  dir=1;stop=1.25;rr=1.5;hold=30;c=CROSS_UP VAL|GE delta_z 1.0
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import List, Tuple

import numpy as np

from .features import Features

OPS = ("CROSS_UP", "CROSS_DOWN", "ABOVE", "BELOW", "GE", "LE", "TIME_IN")
LEVEL_OPS = ("CROSS_UP", "CROSS_DOWN", "ABOVE", "BELOW")
LEVELS = ("VAH", "VAL", "POC")
FEATS = ("delta_z", "er", "vol_pct")
FEAT_RANGE = {"delta_z": (-3.0, 3.0), "er": (0.0, 1.0), "vol_pct": (0.0, 1.0)}
FEAT_DIGITS = {"delta_z": 2, "er": 2, "vol_pct": 2}
TIME_MAX = 1440
MAX_CONDS = 4
STOP_RANGE = (0.5, 3.0)
RR_RANGE = (0.8, 3.0)
HOLD_RANGE = (5, 120)


@dataclass(frozen=True)
class Condition:
    op: str
    arg: str = ""
    x: float = 0.0
    y: float = 0.0

    def text(self) -> str:
        if self.op in LEVEL_OPS:
            return f"{self.op} {self.arg}"
        if self.op in ("GE", "LE"):
            return f"{self.op} {self.arg} {self.x!r}"
        return f"TIME_IN {int(self.x)} {int(self.y)}"

    def describe(self) -> str:
        if self.op in LEVEL_OPS:
            verb = {"CROSS_UP": "crosses up through", "CROSS_DOWN": "crosses down through",
                    "ABOVE": "closes above", "BELOW": "closes below"}[self.op]
            return f"price {verb} yesterday's {self.arg}"
        if self.op in ("GE", "LE"):
            return f"{self.arg} {'>=' if self.op == 'GE' else '<='} {self.x:g}"
        return f"bar {int(self.x)}-{int(self.y)} of the session"


@dataclass(frozen=True)
class Genome:
    direction: int
    conds: Tuple[Condition, ...]
    stop_atr: float
    target_rr: float
    max_hold: int

    def text(self) -> str:
        return (f"dir={self.direction};stop={self.stop_atr!r};rr={self.target_rr!r};hold={self.max_hold};"
                f"c={'|'.join(c.text() for c in self.conds)}")

    @property
    def id(self) -> str:
        return hashlib.sha256(self.text().encode()).hexdigest()[:12]

    def describe(self) -> str:
        side = "LONG" if self.direction > 0 else "SHORT"
        return (f"{side} when " + " AND ".join(c.describe() for c in self.conds)
                + f"; stop {self.stop_atr:g}xATR, target {self.target_rr:g}R, max {self.max_hold} bars")


def _q(v: float, digits: int) -> float:
    return float(round(float(v), digits))


def make_condition(op: str, arg: str = "", x: float = 0.0, y: float = 0.0) -> Condition:
    if op in LEVEL_OPS:
        return Condition(op, arg)
    if op in ("GE", "LE"):
        lo, hi = FEAT_RANGE[arg]
        return Condition(op, arg, _q(np.clip(x, lo, hi), FEAT_DIGITS[arg]))
    a = int(np.clip(round(x), 0, TIME_MAX - 1))
    b = int(np.clip(round(y), a + 1, TIME_MAX))
    return Condition("TIME_IN", "", float(a), float(b))


def make(direction: int, conds, stop_atr: float, target_rr: float, max_hold: int) -> Genome:
    g = Genome(int(direction), tuple(conds), _q(np.clip(stop_atr, *STOP_RANGE), 2),
               _q(np.clip(target_rr, *RR_RANGE), 2), int(np.clip(round(max_hold), *HOLD_RANGE)))
    validate(g)
    return g


def validate(g: Genome) -> None:
    if g.direction not in (1, -1):
        raise ValueError("direction must be +1/-1")
    if not 1 <= len(g.conds) <= MAX_CONDS:
        raise ValueError("1..MAX_CONDS conditions required")
    if len(set(g.conds)) != len(g.conds):
        raise ValueError("duplicate condition")
    for c in g.conds:
        if c.op not in OPS:
            raise ValueError(f"unknown op {c.op}")
        if c.op in LEVEL_OPS and c.arg not in LEVELS:
            raise ValueError(f"bad level {c.arg}")
        if c.op in ("GE", "LE"):
            lo, hi = FEAT_RANGE.get(c.arg, (np.nan, np.nan))
            if c.arg not in FEATS or not lo <= c.x <= hi:
                raise ValueError(f"bad feature condition {c.text()}")
        if c.op == "TIME_IN" and not 0 <= c.x < c.y <= TIME_MAX:
            raise ValueError(f"bad time window {c.text()}")
    if not (STOP_RANGE[0] <= g.stop_atr <= STOP_RANGE[1] and RR_RANGE[0] <= g.target_rr <= RR_RANGE[1]
            and HOLD_RANGE[0] <= g.max_hold <= HOLD_RANGE[1]):
        raise ValueError("exit parameters out of range")


def parse(text: str) -> Genome:
    kv = dict(part.split("=", 1) for part in text.split(";"))
    conds = []
    for ct in kv["c"].split("|"):
        tok = ct.split(" ")
        if tok[0] in LEVEL_OPS:
            conds.append(Condition(tok[0], tok[1]))
        elif tok[0] in ("GE", "LE"):
            conds.append(Condition(tok[0], tok[1], float(tok[2])))
        elif tok[0] == "TIME_IN":
            conds.append(Condition("TIME_IN", "", float(int(tok[1])), float(int(tok[2]))))
        else:
            raise ValueError(f"unknown op {tok[0]}")
    g = Genome(int(kv["dir"]), tuple(conds), float(kv["stop"]), float(kv["rr"]), int(kv["hold"]))
    validate(g)
    if g.text() != text:
        raise ValueError("non-canonical rule text")
    return g


# ------------------------------------------------------------------ signals
def warm_mask(close: np.ndarray, f: Features) -> np.ndarray:
    ok = np.ones(len(close), dtype=bool)
    ok[0] = False
    for a in (f.delta_z, f.vol_pct, f.er, f.atr, f.vah, f.val, f.poc):
        ok &= ~np.isnan(a)
    return ok


def condition_mask(c: Condition, close: np.ndarray, f: Features) -> np.ndarray:
    prev = np.r_[np.nan, close[:-1]]
    with np.errstate(invalid="ignore"):
        if c.op in LEVEL_OPS:
            L = {"VAH": f.vah, "VAL": f.val, "POC": f.poc}[c.arg]
            if c.op == "CROSS_UP":
                return (prev <= L) & (close > L)
            if c.op == "CROSS_DOWN":
                return (prev >= L) & (close < L)
            return close > L if c.op == "ABOVE" else close < L
        if c.op in ("GE", "LE"):
            v = getattr(f, c.arg)
            return v >= c.x if c.op == "GE" else v <= c.x
        return (f.bar_in_session >= c.x) & (f.bar_in_session < c.y)


def signal(g: Genome, close: np.ndarray, f: Features, warm: np.ndarray | None = None) -> np.ndarray:
    """int64 array: g.direction where every condition holds, else 0."""
    m = warm_mask(close, f) if warm is None else warm.copy()
    for c in g.conds:
        m &= condition_mask(c, close, f)
    return np.where(m, g.direction, 0).astype(np.int64)


# ------------------------------------------------------------- variation
def random_condition(rng: np.random.Generator) -> Condition:
    op = OPS[rng.integers(len(OPS))]
    if op in LEVEL_OPS:
        return make_condition(op, LEVELS[rng.integers(len(LEVELS))])
    if op in ("GE", "LE"):
        f = FEATS[rng.integers(len(FEATS))]
        lo, hi = FEAT_RANGE[f]
        return make_condition(op, f, rng.uniform(lo, hi))
    a = rng.integers(0, 400)
    return make_condition("TIME_IN", "", a, a + rng.integers(30, 400))


def random_genome(rng: np.random.Generator) -> Genome:
    while True:
        conds = [make_condition(rng.choice(["CROSS_UP", "CROSS_DOWN"]), LEVELS[rng.integers(3)])]
        for _ in range(rng.integers(0, MAX_CONDS)):
            conds.append(random_condition(rng))
        try:
            return make(rng.choice([1, -1]), dict.fromkeys(conds), rng.uniform(*STOP_RANGE),
                        rng.uniform(*RR_RANGE), rng.integers(*HOLD_RANGE))
        except ValueError:
            continue


def _jitter(c: Condition, rng: np.random.Generator) -> Condition:
    if c.op in ("GE", "LE"):
        lo, hi = FEAT_RANGE[c.arg]
        return make_condition(c.op, c.arg, c.x + rng.normal(0, 0.1 * (hi - lo)))
    if c.op == "TIME_IN":
        return make_condition("TIME_IN", "", c.x + rng.normal(0, 20), c.y + rng.normal(0, 20))
    return make_condition(c.op, LEVELS[rng.integers(3)])


def mutate(g: Genome, rng: np.random.Generator, small: bool = False) -> Genome:
    """small=True: parameter-only change (used to TUNE a live strategy)."""
    for _ in range(50):
        conds: List[Condition] = list(g.conds)
        stop, rr, hold, d = g.stop_atr, g.target_rr, g.max_hold, g.direction
        r = rng.random()
        if small or r < 0.4:
            k = rng.integers(len(conds) + 3)
            if k < len(conds):
                conds[k] = _jitter(conds[k], rng)
            elif k == len(conds):
                stop += rng.normal(0, 0.15)
            elif k == len(conds) + 1:
                rr += rng.normal(0, 0.15)
            else:
                hold += rng.normal(0, 8)
        elif r < 0.6 and len(conds) < MAX_CONDS:
            conds.append(random_condition(rng))
        elif r < 0.75 and len(conds) > 1:
            conds.pop(rng.integers(len(conds)))
        elif r < 0.9:
            conds[rng.integers(len(conds))] = random_condition(rng)
        else:
            d = -d
        try:
            child = make(d, dict.fromkeys(conds), stop, rr, hold)
            if child != g:
                return child
        except ValueError:
            continue
    return g


def crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
    pool = list(dict.fromkeys(a.conds + b.conds))
    for _ in range(20):
        k = int(rng.integers(1, min(MAX_CONDS, len(pool)) + 1))
        pick = [pool[i] for i in sorted(rng.choice(len(pool), size=k, replace=False))]
        src = a if rng.random() < 0.5 else b
        try:
            return make(src.direction, pick, (a.stop_atr + b.stop_atr) / 2, (a.target_rr + b.target_rr) / 2,
                        (a.max_hold + b.max_hold) / 2)
        except ValueError:
            continue
    return a


def seeds() -> List[Genome]:
    """The v1 Auction-Market-Theory rules, expressed as genomes."""
    return [
        make(1, [make_condition("CROSS_UP", "VAL"), make_condition("GE", "delta_z", 1.0), make_condition("LE", "er", 0.4)], 1.25, 1.5, 30),
        make(-1, [make_condition("CROSS_DOWN", "VAH"), make_condition("LE", "delta_z", -1.0), make_condition("LE", "er", 0.4)], 1.25, 1.5, 30),
        make(1, [make_condition("CROSS_UP", "VAH"), make_condition("GE", "delta_z", 1.0), make_condition("GE", "er", 0.4)], 1.25, 1.5, 30),
        make(-1, [make_condition("CROSS_DOWN", "VAL"), make_condition("LE", "delta_z", -1.0), make_condition("GE", "er", 0.4)], 1.25, 1.5, 30),
    ]
