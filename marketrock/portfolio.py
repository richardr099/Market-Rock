"""Strategy lifecycle and sizing (the autonomous part).

States:  PAPER -> LIVE -> RETIRED          (or PAPER -> REJECTED)

  PAPER    passed the gate; now forward-tested on bars recorded AFTER it was
           born (data no search has ever scored it on), with the same
           pessimistic fill model. No orders.
  LIVE     won its paper trial. Starts at `start_stage` (25%) of its computed
           size and grows linearly to 100% over `full_stage_trades` live trades.
  RETIRED  live results fell significantly below expectation, or it was
           replaced by a better-tuned child. A child that fails restores its
           parent automatically (auto-rollback).

Sizing (per LIVE genome, R = planned $ risk of a trade):
  pooled forward+live R-multiples, shrunk toward 0 with n0 pseudo-trades;
  mu_lo = mean - sd/sqrt(n+n0);  D = (trailing_dd - buffer) / n_live
  r_kelly = kelly_fraction * mu_lo / sd^2 * D
  r_ruin  = 2 * mu_lo * D / (sd^2 * ln(1/ruin_prob))   [P(hit floor) <= ruin_prob, drifted random walk]
  risk = min(stage * min(r_kelly, r_ruin), max_risk_per_trade_usd), floored at
         probe_risk_usd (never above the ceiling). max_risk_per_trade_usd is the user's ceiling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np

from . import genome as G
from .data import Bars
from .evolve import EVAL_RISK
from .features import Features
from .strategy import Costs, Limits, run

DEFAULT_CONFIG = {
    "max_risk_per_trade_usd": 250.0,   # YOUR ceiling on automatic sizing
    "probe_risk_usd": 10.0,            # smallest risk that can trade (~1 MES contract at an 8-tick stop)
    "trailing_dd_usd": 2500.0,
    "daily_loss_usd": 1000.0,
    "buffer_usd": 150.0,
    "ruin_prob": 0.05,
    "kelly_fraction": 0.5,
    "start_stage": 0.25,
    "full_stage_trades": 50,
    "max_live": 4,
    "max_paper": 6,
    "paper_min_trades": 20,
    "paper_max_sessions": 60,
    "holdout_sessions": 20,
    "tune_children": 12,
    "halted": False,
    # instrument: MES (Micro E-mini S&P 500) - must match the chart NT8 trades
    "tick": 0.25,
    "tick_value": 1.25,
    "commission_rt": 1.5,
    "slip_ticks": 1.0,
}
SHRINK_N0 = 20
SD_FLOOR = 0.5
PAPER_T_MIN = 1.0       # forward mean R at least 1 standard error above 0
PAPER_Z_MIN = -2.0      # forward not significantly worse than its HAIRCUT backtest
BACKTEST_HAIRCUT = 0.5  # the winner of a search is optimistic in-sample (Harvey & Liu 2015)
LIVE_MIN_N = 10
LIVE_Z_RETIRE = -2.5    # live significantly worse than its forward trial
LIVE_T_RETIRE = -1.0    # after 30 trades, live mean R clearly negative
LIVE_SUM_R_RETIRE = -10.0


@dataclass
class Stats:
    n: int = 0
    s: float = 0.0
    ss: float = 0.0

    @classmethod
    def of(cls, r: np.ndarray) -> "Stats":
        r = np.asarray(r, dtype=np.float64)
        return cls(int(len(r)), float(r.sum()), float((r * r).sum()))

    def add(self, x: float) -> None:
        self.n += 1
        self.s += x
        self.ss += x * x

    @property
    def mean(self) -> float:
        return self.s / self.n if self.n else 0.0

    @property
    def sd(self) -> float:
        if self.n < 2:
            return 1.0
        return math.sqrt(max(self.ss / self.n - self.mean ** 2, 0.0) * self.n / (self.n - 1))

    @property
    def t(self) -> float:
        return self.mean / (max(self.sd, 1e-9) / math.sqrt(self.n)) if self.n >= 2 else 0.0

    def to(self) -> dict:
        return {"n": self.n, "s": self.s, "ss": self.ss}

    @classmethod
    def frm(cls, d: Optional[Mapping]) -> "Stats":
        return cls(**d) if d else cls()


def new_member(g: G.Genome, born_time: int, born_session: int, bt_r: np.ndarray, fitness: float,
               parent: Optional[str], now: int, why: str) -> dict:
    return {"rule": g.text(), "status": "PAPER", "parent": parent, "born_time": int(born_time),
            "born_session": int(born_session), "bt": Stats.of(bt_r).to(), "fwd": None,
            "live": Stats().to(), "stage": 0.0, "risk_usd": 0.0, "fitness": float(fitness),
            "history": [{"t": now, "event": "PAPER", "why": why}]}


def _event(m: dict, now: int, event: str, why: str, log: List[dict], mid: str) -> None:
    m["history"].append({"t": now, "event": event, "why": why})
    log.append({"id": mid, "event": event, "why": why, "rule": G.parse(m["rule"]).describe()})


def forward_r(g: G.Genome, born_time: int, b: Bars, f: Features, costs: Costs, limits: Limits) -> np.ndarray:
    r = run(b, [g], [EVAL_RISK], costs, limits, f)
    return r.r_multiple[r.entry_time > born_time]


def ingest_trades(members: Dict[str, dict], rows: List[Mapping]) -> int:
    used = 0
    for t in rows:
        m = members.get(t["genome"])
        if m is None:
            continue
        risk = float(t["risk_usd"])
        if risk <= 0:
            continue
        st = Stats.frm(m["live"])
        st.add(float(t["pnl_usd"]) / risk)
        m["live"] = st.to()
        used += 1
    return used


def review_paper(members: Dict[str, dict], b: Bars, f: Features, costs: Costs, limits: Limits,
                 cfg: Mapping, now: int, log: List[dict]) -> None:
    last_session = int(b.session[-1])
    sessions = np.unique(b.session)
    n_live = sum(1 for m in members.values() if m["status"] == "LIVE")
    ready = []
    for mid, m in members.items():
        if m["status"] != "PAPER":
            continue
        g = G.parse(m["rule"])
        fr = forward_r(g, m["born_time"], b, f, costs, limits)
        fw = Stats.of(fr)
        m["fwd"] = fw.to()
        bt = Stats.frm(m["bt"])
        age = int(np.sum(sessions > m["born_session"])) if last_session > m["born_session"] else 0
        if fw.n < cfg["paper_min_trades"]:
            if age > cfg["paper_max_sessions"]:
                m["status"] = "REJECTED"
                _event(m, now, "REJECTED", f"paper trial expired: {fw.n} trades in {age} sessions", log, mid)
            continue
        z = (fw.mean - BACKTEST_HAIRCUT * bt.mean) / (max(bt.sd, 1e-9) / math.sqrt(fw.n))
        ok = fw.t >= PAPER_T_MIN and z >= PAPER_Z_MIN
        why = f"forward {fw.n} trades, mean {fw.mean:+.3f}R, t={fw.t:.2f}, vs haircut backtest z={z:.2f}"
        if ok and m["parent"] and m["parent"] in members:
            p = members[m["parent"]]
            pr = forward_r(G.parse(p["rule"]), m["born_time"], b, f, costs, limits)
            if len(pr) and float(np.mean(pr)) >= fw.mean:
                ok = False
                why += f"; not better than parent over same window ({float(np.mean(pr)):+.3f}R)"
        if not ok:
            m["status"] = "REJECTED"
            _event(m, now, "REJECTED", why, log, mid)
        else:
            ready.append((fw.t, mid, why))
    for _, mid, why in sorted(ready, reverse=True):
        m = members[mid]
        parent = members.get(m["parent"]) if m["parent"] else None
        if parent is not None and parent["status"] == "LIVE":
            # tuned version replaces its parent and keeps the parent's live stage
            m["stage"] = parent["stage"]
            parent["status"] = "RETIRED"
            parent["replaced_by"] = mid
            _event(parent, now, "RETIRED", f"replaced by tuned version {mid}", log, m["parent"])
        elif n_live >= cfg["max_live"]:
            continue  # wait for a slot; stays PAPER
        else:
            m["stage"] = cfg["start_stage"]
            n_live += 1
        m["status"] = "LIVE"
        _event(m, now, "LIVE", why, log, mid)


def monitor_live(members: Dict[str, dict], now: int, log: List[dict]) -> None:
    for mid, m in list(members.items()):
        if m["status"] != "LIVE":
            continue
        lv = Stats.frm(m["live"])
        if lv.n < LIVE_MIN_N:
            continue
        ref = Stats.frm(m["fwd"]) if m["fwd"] and m["fwd"]["n"] >= 2 else Stats.frm(m["bt"])
        z = (lv.mean - ref.mean) / (max(ref.sd, 1e-9) / math.sqrt(lv.n))
        reason = None
        if z < LIVE_Z_RETIRE:
            reason = f"live mean {lv.mean:+.3f}R over {lv.n} trades is {z:.2f} SE below its paper trial"
        elif lv.n >= 30 and lv.t < LIVE_T_RETIRE:
            reason = f"live mean {lv.mean:+.3f}R over {lv.n} trades, t={lv.t:.2f}"
        elif lv.s < LIVE_SUM_R_RETIRE:
            reason = f"cumulative live loss {lv.s:.1f}R"
        if reason is None:
            continue
        m["status"] = "RETIRED"
        _event(m, now, "RETIRED", reason, log, mid)
        par = members.get(m["parent"]) if m["parent"] else None
        if par is not None and par["status"] == "RETIRED" and par.get("replaced_by") == mid:
            par["status"] = "LIVE"
            par.pop("replaced_by", None)
            _event(par, now, "LIVE", f"auto-rollback: tuned child {mid} failed live", log, m["parent"])


def size(members: Dict[str, dict], cfg: Mapping) -> None:
    live = [m for m in members.values() if m["status"] == "LIVE"]
    if not live:
        return
    D = max(cfg["trailing_dd_usd"] - cfg["buffer_usd"], 0.0) / len(live)
    cap = cfg["max_risk_per_trade_usd"]
    for m in live:
        fw, lv = Stats.frm(m["fwd"]), Stats.frm(m["live"])
        n, s, ss = fw.n + lv.n, fw.s + lv.s, fw.ss + lv.ss
        mean = s / (n + SHRINK_N0)
        var = max(ss / n - (s / n) ** 2, SD_FLOOR ** 2) if n else 1.0
        sd = math.sqrt(var)
        mu_lo = mean - sd / math.sqrt(n + SHRINK_N0)
        stage = min(1.0, cfg["start_stage"] + (1 - cfg["start_stage"]) * lv.n / cfg["full_stage_trades"])
        stage = max(stage, m["stage"])  # never shrink stage just because a tune reset counters
        m["stage"] = stage
        if mu_lo > 0:
            r_kelly = cfg["kelly_fraction"] * mu_lo / var * D
            r_ruin = 2 * mu_lo * D / (var * math.log(1 / cfg["ruin_prob"]))
            r = min(stage * min(r_kelly, r_ruin), cap)
        else:
            r = 0.0
        m["risk_usd"] = round(min(max(r, cfg["probe_risk_usd"]), cap), 2)
        m["sizing"] = {"mu_lo": mu_lo, "sd": sd, "n": n, "D": D, "stage": stage}


def live_genomes(members: Dict[str, dict], cfg: Mapping):
    """Portfolio order = priority: highest lower-bound edge first."""
    if cfg.get("halted"):
        return []
    live = [(mid, m) for mid, m in members.items() if m["status"] == "LIVE"]
    live.sort(key=lambda x: (-(x[1].get("sizing") or {}).get("mu_lo", 0.0), x[0]))
    return [(mid, G.parse(m["rule"]), m["risk_usd"]) for mid, m in live]
