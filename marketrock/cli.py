"""marketrock CLI (v2: self-developing portfolio).

  init      create state/config.json (YOUR settings) and an empty live portfolio
  evolve    nightly, unattended: learn from live fills, retire what fails,
            promote what wins its paper trial, invent + tune strategies,
            size everything, write the live portfolio and a daily report
  backtest  run the current live portfolio (or a portfolio file) over bars
  halt      kill switch: empty live portfolio until `resume`
  resume    clear halt / pin
  rollback  restore an earlier live portfolio by hash and pin it
  verify    check the ledger chain and the live file hash
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np

from . import evolve as E, genome as G, portfolio as PF, report, store
from .data import load_csv
from .features import compute
from .strategy import VA_PCT, Costs, Limits, run
from .validation import sharpe

TRADE_COLUMNS = ("exit_time", "session", "genome", "direction", "qty", "pnl_usd", "risk_usd", "portfolio_sha256")
MAX_CANDIDATES_PER_NIGHT = 5
MAX_OVERLAP = 0.5  # Jaccard overlap of entry bars; above this a candidate is a clone


def _now() -> int:
    return int(time.time())  # metadata only; never feeds a computation


def _cfg(state: Path) -> dict:
    cfg = dict(PF.DEFAULT_CONFIG)
    cfg.update(store.read_json(state / "config.json", {}))
    return cfg


def _costs(cfg) -> Costs:
    return Costs(cfg["tick"], cfg["tick_value"], cfg["commission_rt"], cfg["slip_ticks"])


def _limits(cfg) -> Limits:
    return Limits(cfg["daily_loss_usd"], cfg["trailing_dd_usd"])


def _read_trades(path: Path) -> List[dict]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if tuple(r.fieldnames or ()) != TRADE_COLUMNS:
            raise ValueError(f"trade log header must be {TRADE_COLUMNS}")
        return list(r)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / u if u else 0.0


def _publish(live: Path, state: Path, entries, event: str, approver: str, extra=None) -> str:
    data = store.canonical(entries)
    try:
        _, cur = store.read_live(live)
    except FileNotFoundError:
        cur = None
    if cur == store.sha256(data):
        return cur
    h = store.write_live(live, entries)
    rec = {"event": event, "from": cur, "portfolio_sha256": h, "approver": approver, "t": _now()}
    rec.update(extra or {})
    store.ledger_append(state, rec)
    return h


def cmd_init(a) -> int:
    live, state = Path(a.live_dir), Path(a.state_dir)
    if (live / "portfolio.txt").exists():
        print("refusing: live portfolio already exists", file=sys.stderr)
        return 2
    if not (state / "config.json").exists():
        store.write_json(state / "config.json", PF.DEFAULT_CONFIG)
    _publish(live, state, [], "init", a.approver)
    print(f"initialised. Edit {state / 'config.json'} (max_risk_per_trade_usd is your sizing ceiling).")
    return 0


def cmd_evolve(a) -> int:
    live, state, rep_dir = Path(a.live_dir), Path(a.state_dir), Path(a.report_dir)
    cfg = _cfg(state)
    costs, limits = _costs(cfg), _limits(cfg)
    bars_bytes = Path(a.bars).read_bytes()
    b = load_csv(a.bars)
    f = compute(b, costs.tick, VA_PCT)
    pf = store.read_json(state / "portfolio.json", {"members": {}, "trades_meta": {"processed": 0, "head": None}})
    members = pf["members"]
    now = _now()
    events: List[dict] = []
    warnings: List[str] = []
    info = {"sessions": int(len(np.unique(b.session))), "last_session": int(b.session[-1])}

    # 1. learn from live fills
    if a.trades and Path(a.trades).exists():
        rows = _read_trades(Path(a.trades))
        head = store.sha256(json.dumps(rows[:1], sort_keys=True).encode())
        meta = pf["trades_meta"]
        if (meta["head"] not in (None, head) and rows) or len(rows) < meta["processed"]:
            print("refusing: trade log was rotated/rewritten since last run", file=sys.stderr)
            return 2
        PF.ingest_trades(members, rows[meta["processed"]:])
        pf["trades_meta"] = {"processed": len(rows), "head": head if rows else meta["head"]}

    # 2. retire live strategies that are failing (and roll back failed tunes)
    PF.monitor_live(members, now, events)
    # 3. paper trials on forward data
    PF.review_paper(members, b, f, costs, limits, cfg, now, events)

    # 4. research: invent + tune, gate, admit to paper
    sessions = np.unique(b.session)
    st = store.read_json(state / "search.json", {"n_trials": 0})
    H = int(cfg["holdout_sessions"])
    passed = 0
    if len(sessions) < H + 3 * E.K_FOLDS:
        warnings.append(f"Only {len(sessions)} sessions of data; research needs >= {H + 3 * E.K_FOLDS}. Export more history.")
        info["evaluated"] = 0
    else:
        hold = sessions[-H:]
        cut = int(np.searchsorted(b.session, hold[0]))
        srch = E.Search(b.slice(0, cut), E.slice_features(f, 0, cut), costs, limits)
        rng = np.random.default_rng(E.seed_for(bars_bytes, st["n_trials"]))
        active = [G.parse(m["rule"]) for m in members.values() if m["status"] in ("LIVE", "PAPER")]
        ranked = srch.evolve(G.seeds() + active, rng)
        tunes = []
        for mid, m in sorted(members.items()):
            if m["status"] == "LIVE":
                kids = srch.tune(G.parse(m["rule"]), int(cfg["tune_children"]), rng)
                if kids and np.isfinite(kids[0].fitness):
                    tunes.append((kids[0], mid))
        n_trials = st["n_trials"] + srch.trials
        st["n_trials"] = n_trials
        var_sr = E.var_sr_of(list(srch.cache.values()))
        cands = [(e, None) for e in ranked if e.genome.id not in members and np.isfinite(e.fitness)][:MAX_CANDIDATES_PER_NIGHT]
        cands += [(e, p) for e, p in tunes if e.genome.id not in members]
        n_paper = sum(1 for m in members.values() if m["status"] == "PAPER")
        close_s = b.close[:cut]
        warm = G.warm_mask(close_s, srch.f)
        masks = {mid: G.signal(G.parse(m["rule"]), close_s, srch.f, warm) != 0
                 for mid, m in members.items() if m["status"] in ("LIVE", "PAPER")}
        for e, parent in cands:
            if n_paper >= cfg["max_paper"]:
                break
            mk = G.signal(e.genome, close_s, srch.f, warm) != 0
            if any(_jaccard(mk, other) > MAX_OVERLAP for mid, other in masks.items() if mid != parent):
                continue  # a clone of something already live/paper adds no diversification
            eh = E.evaluate_holdout(b, e.genome, f, hold, costs, limits)
            checks = E.gate(e, eh, n_trials, var_sr, limits)
            if all(c["pass"] for c in checks.values()):
                why = ("tuned from " + parent if parent else "new strategy") + f"; passed gate (DSR {checks['deflated_sharpe']['value']:.3f})"
                masks[e.genome.id] = mk
                members[e.genome.id] = PF.new_member(e.genome, int(b.time[-1]), int(b.session[-1]), e.r, e.fitness, parent, now, why)
                events.append({"id": e.genome.id, "event": "PAPER", "why": why, "rule": e.genome.describe()})
                n_paper += 1
                passed += 1
        info["evaluated"] = srch.trials
    info["n_trials"] = st["n_trials"]
    info["passed"] = passed

    # 5. size, publish
    PF.size(members, cfg)
    entries = PF.live_genomes(members, cfg)[: store.MAX_GENOMES]
    if cfg.get("pinned"):
        warnings.append(f"Live portfolio is pinned to {cfg['pinned'][:12]} by a rollback; run `marketrock resume` to let the system manage it again.")
    else:
        _publish(live, state, entries, "auto", "auto:evolve", {"events": [{k: e[k] for k in ("id", "event", "why")} for e in events]})
    if cfg.get("halted"):
        warnings.append("Trading is halted.")

    store.write_json(state / "portfolio.json", pf)
    store.write_json(state / "search.json", st)
    date = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")
    rep_dir.mkdir(parents=True, exist_ok=True)
    path = rep_dir / f"{date}.md"
    path.write_text(report.render(date, members, events, cfg, dict(info, warnings=warnings)))
    print(json.dumps({"report": str(path), "events": [(e["event"], e["id"]) for e in events],
                      "live": [e[0] for e in entries], "passed_gate": passed}, indent=1))
    return 0


def cmd_backtest(a) -> int:
    state = Path(a.state_dir)
    cfg = _cfg(state)
    b = load_csv(a.bars)
    entries = store.parse(Path(a.portfolio).read_bytes()) if a.portfolio else store.read_live(Path(a.live_dir))[0]
    r = run(b, [g for _, g, _ in entries], [x for _, _, x in entries], _costs(cfg), _limits(cfg))
    daily = r.daily_pnl(np.unique(b.session))
    print(json.dumps({"genomes": len(entries), "trades": r.n, "net_usd": round(float(r.pnl.sum()), 2),
                      "blown": r.blown, "sharpe_per_session": round(sharpe(daily), 4),
                      "max_drawdown_usd": round(E.max_drawdown(daily), 2)}, indent=1))
    return 0


def _need_name(a) -> bool:
    if not a.approver.strip():
        print("refusing: --approver must name a person", file=sys.stderr)
        return False
    return True


def cmd_halt(a) -> int:
    if not _need_name(a):
        return 2
    state = Path(a.state_dir)
    cfg = store.read_json(state / "config.json", dict(PF.DEFAULT_CONFIG))
    cfg["halted"] = True
    store.write_json(state / "config.json", cfg)
    _publish(Path(a.live_dir), state, [], "halt", a.approver)
    print("halted: live portfolio is empty; NT8 stops opening trades at its next parameter check")
    return 0


def cmd_resume(a) -> int:
    if not _need_name(a):
        return 2
    state = Path(a.state_dir)
    cfg = store.read_json(state / "config.json", dict(PF.DEFAULT_CONFIG))
    cfg["halted"] = False
    cfg.pop("pinned", None)
    store.write_json(state / "config.json", cfg)
    store.ledger_append(state, {"event": "resume", "approver": a.approver, "t": _now()})
    print("resumed: the next `evolve` run manages the live portfolio again")
    return 0


def cmd_rollback(a) -> int:
    if not _need_name(a):
        return 2
    live, state = Path(a.live_dir), Path(a.state_dir)
    entries = store.read_history(live, a.to)
    cfg = store.read_json(state / "config.json", dict(PF.DEFAULT_CONFIG))
    cfg["pinned"] = a.to
    store.write_json(state / "config.json", cfg)
    _publish(live, state, entries, "rollback", a.approver)
    print(f"rolled back and pinned to {a.to[:12]}; `marketrock resume` to unpin")
    return 0


def cmd_verify(a) -> int:
    recs = store.ledger_verify(Path(a.state_dir))
    _, h = store.read_live(Path(a.live_dir))
    last = next((r["portfolio_sha256"] for r in reversed(recs) if "portfolio_sha256" in r), None)
    if last != h:
        print(f"FAIL: live portfolio {h} is not the last ledgered version {last}", file=sys.stderr)
        return 1
    print(f"OK: {len(recs)} ledger entries, live={h}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="marketrock")
    ap.add_argument("--live-dir", default="live")
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--report-dir", default="reports")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init"); s.add_argument("--approver", required=True); s.set_defaults(fn=cmd_init)
    s = sub.add_parser("evolve"); s.add_argument("--bars", required=True); s.add_argument("--trades")
    s.set_defaults(fn=cmd_evolve)
    s = sub.add_parser("backtest"); s.add_argument("--bars", required=True); s.add_argument("--portfolio")
    s.set_defaults(fn=cmd_backtest)
    for name, fn in (("halt", cmd_halt), ("resume", cmd_resume)):
        s = sub.add_parser(name); s.add_argument("--approver", required=True); s.set_defaults(fn=fn)
    s = sub.add_parser("rollback"); s.add_argument("--to", required=True); s.add_argument("--approver", required=True)
    s.set_defaults(fn=cmd_rollback)
    s = sub.add_parser("verify"); s.set_defaults(fn=cmd_verify)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
