"""Bar data container, CSV I/O, and a synthetic generator for tests.

Bar CSV schema (written by MarketRockStrategy.cs in realtime, or by its
ExportHistory mode):  time,session,open,high,low,close,volume,buy_volume,sell_volume
  time     bar end, epoch seconds (UTC)
  session  integer trading-session id (yyyymmdd of the session's trading date)
  buy/sell volume  tick-rule classified (same classifier as live; see docs)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COLUMNS = ("time", "session", "open", "high", "low", "close", "volume", "buy_volume", "sell_volume")


@dataclass(frozen=True)
class Bars:
    time: np.ndarray
    session: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    buy_volume: np.ndarray
    sell_volume: np.ndarray

    def __len__(self) -> int:
        return len(self.close)

    def slice(self, a: int, b: int) -> "Bars":
        return Bars(*(getattr(self, c)[a:b] for c in COLUMNS))

    def validate(self) -> None:
        n = len(self)
        for c in COLUMNS:
            if len(getattr(self, c)) != n:
                raise ValueError(f"column {c} length mismatch")
        if n == 0:
            raise ValueError("no bars")
        if np.any(np.diff(self.time) <= 0):
            raise ValueError("bar times must be strictly increasing")
        if np.any(np.diff(self.session) < 0):
            raise ValueError("session ids must be non-decreasing")
        if np.any(self.high < np.maximum(self.open, self.close)) or np.any(self.low > np.minimum(self.open, self.close)):
            raise ValueError("OHLC inconsistent")
        if not np.all(np.isfinite(self.close)):
            raise ValueError("non-finite prices")


def load_csv(path: str | Path) -> Bars:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if tuple(r.fieldnames or ()) != COLUMNS:
            raise ValueError(f"expected header {COLUMNS}, got {r.fieldnames}")
        rows = list(r)
    cols = {c: [row[c] for row in rows] for c in COLUMNS}
    b = Bars(
        time=np.asarray(cols["time"], dtype=np.int64),
        session=np.asarray(cols["session"], dtype=np.int64),
        **{c: np.asarray(cols[c], dtype=np.float64) for c in COLUMNS[2:]},
    )
    b.validate()
    return b


def save_csv(b: Bars, path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for i in range(len(b)):
            w.writerow([int(b.time[i]), int(b.session[i])] + [repr(float(getattr(b, c)[i])) for c in COLUMNS[2:]])


def synthetic(n_sessions: int = 60, bars_per_session: int = 390, tick: float = 0.25,
              seed: int = 0, edge: float = 0.0, start_price: float = 5000.0) -> Bars:
    """Regime-switching random walk on a tick grid.

    ``edge`` > 0 plants a weak, detectable dependency: delta at bar t predicts
    the sign of the return at t+1. ``edge`` = 0 is pure noise, which the
    promotion gate must reject (tests/test_gate.py).
    """
    rng = np.random.default_rng(seed)
    n = n_sessions * bars_per_session
    session = np.repeat(20250101 + np.arange(n_sessions, dtype=np.int64), bars_per_session)
    time = 1_735_700_000 + np.arange(n, dtype=np.int64) * 60
    vol_regime = np.repeat(rng.choice([0.5, 1.0, 2.0], size=n_sessions), bars_per_session)
    volume = np.maximum(1.0, rng.gamma(4.0, 250.0, size=n) * vol_regime).round()
    imbalance = rng.normal(0.0, 0.15, size=n)
    buy = np.clip(np.round(volume * (0.5 + imbalance / 2)), 0, volume)
    sell = volume - buy
    dz = (buy - sell) / np.maximum(volume, 1.0)
    shocks = rng.normal(0.0, 1.0, size=n) * 2.0 * vol_regime
    shocks[1:] += edge * 8.0 * vol_regime[1:] * dz[:-1]
    steps = np.round(shocks)  # ticks
    close = start_price + tick * np.cumsum(steps)
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    wick = tick * np.abs(np.round(rng.normal(0, 1.5, size=(2, n)) * vol_regime))
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    b = Bars(time, session, open_, high, low, close, volume, buy, sell)
    b.validate()
    return b
