"""marketrock CLI.

  init      write default params as the first live version (human, named)
  backtest  run the current or a given parameter file over a bar CSV
  propose   nightly job: learn from live fills + search; auto-apply DE-RISK
            changes only; write a candidate for a human to promote
  promote   human approval of a gated candidate -> live
  rollback  human: restore any earlier live version by hash
  verify    check the ledger chain and the live params hash
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

from . import autonomy, optimizer as O, params as P, store
from .data import load_csv
from .strategy import Costs, Limits, run
from .validation import sharpe

TRADE_COLUMNS = ("exit_time", "session", "regime", "direction", "qty", "pnl_usd", "params_sha256")


def _now() -> int:
    return int(time.time())  # metadata only; never feeds a computation


def _costs(a) -> Costs:
    return Costs(a.tick, a.tick_value, a.commission, a.slippage)


def _limits(a) -> Limits:
    return Limits(a.daily_loss, a.trailing_dd)


def _read_trades(path: Path) -> List[dict]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if tuple(r.fieldnames or ()) != TRADE_COLUMNS:
            raise ValueError(f"trade log header must be {TRADE_COLUMNS}")
        return list(r)


def cmd_init(a) -> int:
    live, state = Path(a.live_dir), Path(a.state_dir)
    if (live / "params.json").exists():
        print("refusing: live params already exist (use promote/rollback)", file=sys.stderr)
        return 2
    h = store.write_live(live, P.defaults())
    store.ledger_append(state, {"event": "init", "params_sha256": h, "approver": a.approver, "t": _now()})
    print(h)
    return 0


def cmd_backtest(a) -> int:
    b = load_csv(a.bars)
    p = P.validate(json.loads(Path(a.params).read_text())) if a.params else store.read_live(Path(a.live_dir))[0]
    r = run(b, p, _costs(a), _limits(a))
    daily = r.daily_pnl(np.unique(b.session))
    print(json.dumps({
        "trades": r.n, "net_usd": round(float(r.pnl.sum()), 2), "blown": r.blown,
        "win_rate": round(float((r.pnl > 0).mean()), 4) if r.n else None,
        "sharpe_per_session": round(sharpe(daily), 4),
        "max_drawdown_usd": round(O.max_drawdown(daily), 2),
    }, indent=1))
    return 0


def cmd_propose(a) -> int:
    live, state, out = Path(a.live_dir), Path(a.state_dir), Path(a.out_dir)
    costs, limits = _costs(a), _limits(a)
    inc, inc_hash = store.read_live(live)
    bars_bytes = Path(a.bars).read_bytes()
    b = load_csv(a.bars)
    sessions = np.unique(b.session)
    if len(sessions) < a.holdout_sessions + 3 * O.K_FOLDS:
        print(f"refusing: need >= {a.holdout_sessions + 3 * O.K_FOLDS} sessions, have {len(sessions)}", file=sys.stderr)
        return 2
    report: dict = {"base_params_sha256": inc_hash, "bars_sha256": store.sha256(bars_bytes), "t": _now()}

    # ---- 1. Bayesian regime layer from live fills (autonomous, de-risk only)
    regimes = O.states_from_json(store.read_text_opt(state / "regimes.json") or "{}")
    rmeta = store.read_json(state / "regimes_meta.json", {"processed": 0, "head_sha256": None})
    derisk_applied = {}
    if a.trades:
        rows = _read_trades(Path(a.trades))
        head = store.sha256(json.dumps(rows[:1], sort_keys=True).encode())
        if rmeta["head_sha256"] not in (None, head) or len(rows) < rmeta["processed"]:
            print("refusing: trade log was rotated/rewritten since last run; human review required", file=sys.stderr)
            return 2
        new = rows[rmeta["processed"]:]
        for reg in O.REGIME_KEYS:
            regimes[reg] = O.update_regime(regimes.get(reg, O.RegimeState()),
                                           [t for t in new if int(t["regime"]) == reg])
        bayes, breport = O.bayes_proposal(inc, regimes)
        report["regime_posteriors"] = breport
        if not a.no_auto_derisk:
            derisked = autonomy.apply_derisk(inc, bayes)
            if derisked != inc:
                derisk_applied, _ = autonomy.split_changes(inc, derisked)
                h = store.write_live(live, derisked)
                store.ledger_append(state, {"event": "auto_derisk", "from": inc_hash, "params_sha256": h,
                                            "changes": derisk_applied, "approver": "auto:derisk", "t": _now()})
                inc, inc_hash = derisked, h
        store.write_json(state / "regimes_meta.json", {"processed": len(rows), "head_sha256": head})
        store._atomic_write(state / "regimes.json", O.states_to_json(regimes).encode())
        # Regime re-enables / size increases go to the human candidate below.
        report["regime_increase_suggestions"] = autonomy.split_changes(inc, bayes)[1]
    report["auto_derisk_applied"] = derisk_applied

    # ---- 2. Evolutionary search on the search slice (never sees hold-out)
    hold = sessions[-a.holdout_sessions:]
    n_search_bars = int(np.searchsorted(b.session, hold[0]))
    b_search = b.slice(0, n_search_bars)
    st = store.read_json(state / "search.json", {"n_trials": 0, "sigma": O.SIGMA_INIT})
    seed = O.seed_for(bars_bytes, inc_hash, st["n_trials"])
    inc_eval, best, evals, new_sigma, var_sr = O.search(b_search, inc, st["sigma"], seed, costs, limits)
    n_trials = st["n_trials"] + len(evals) + 1
    store.write_json(state / "search.json", {"n_trials": n_trials, "sigma": new_sigma})

    # ---- 3. Gate on hold-out
    cand_hold = O.evaluate_holdout(b, best.params, hold, costs, limits)
    inc_hold = O.evaluate_holdout(b, inc, hold, costs, limits)
    checks = O.gate(best, cand_hold, inc_hold, n_trials, var_sr, limits)
    passed = all(c["pass"] for c in checks.values()) and best.params != inc
    report.update({
        "status": "AWAITING_APPROVAL" if passed else "REJECTED",
        "gates": checks, "seed": seed, "n_trials_cumulative": n_trials, "sigma_next": new_sigma,
        "incumbent_fitness": inc_eval.fitness if np.isfinite(inc_eval.fitness) else str(inc_eval.fitness),
        "candidate_fitness": best.fitness if np.isfinite(best.fitness) else str(best.fitness),
        "changes": autonomy.split_changes(inc, best.params)[1],
    })
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidate.json").write_bytes(store.canonical(best.params))
    store.write_json(out / "report.json", report)
    store.ledger_append(state, {"event": "proposal", "status": report["status"], "base": inc_hash,
                                "candidate_sha256": store.sha256(store.canonical(best.params)),
                                "n_trials": n_trials, "t": _now()})
    print(json.dumps({"status": report["status"], "auto_derisk_applied": derisk_applied,
                      "failed_gates": [k for k, v in checks.items() if not v["pass"]]}, indent=1))
    return 0


def cmd_promote(a) -> int:
    live, state, out = Path(a.live_dir), Path(a.state_dir), Path(a.out_dir)
    if not a.approver.strip():
        print("refusing: --approver must name a person", file=sys.stderr)
        return 2
    report = json.loads((out / "report.json").read_text())
    cand_bytes = (out / "candidate.json").read_bytes()
    cand = P.validate(json.loads(cand_bytes))
    _, live_hash = store.read_live(live)
    if report.get("status") != "AWAITING_APPROVAL":
        print(f"refusing: candidate status is {report.get('status')}", file=sys.stderr)
        return 2
    if report.get("base_params_sha256") != live_hash:
        # Live params changed after this candidate was evaluated (e.g. an
        # auto-derisk ran). Promoting would silently undo that change.
        print("refusing: candidate was built on a different live version; re-run propose", file=sys.stderr)
        return 2
    h = store.write_live(live, cand)
    store.ledger_append(state, {"event": "promote", "from": live_hash, "params_sha256": h,
                                "approver": a.approver, "report_sha256": store.sha256((out / "report.json").read_bytes()),
                                "t": _now()})
    print(h)
    return 0


def cmd_rollback(a) -> int:
    live, state = Path(a.live_dir), Path(a.state_dir)
    if not a.approver.strip():
        print("refusing: --approver must name a person", file=sys.stderr)
        return 2
    p = store.read_history(live, a.to)
    _, cur = store.read_live(live)
    h = store.write_live(live, p)
    store.ledger_append(state, {"event": "rollback", "from": cur, "params_sha256": h, "approver": a.approver, "t": _now()})
    print(h)
    return 0


def cmd_verify(a) -> int:
    recs = store.ledger_verify(Path(a.state_dir))
    _, h = store.read_live(Path(a.live_dir))
    last = next((r["params_sha256"] for r in reversed(recs) if "params_sha256" in r), None)
    if last != h:
        print(f"FAIL: live params {h} not the last ledgered version {last}", file=sys.stderr)
        return 1
    print(f"OK: {len(recs)} ledger entries, live={h}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="marketrock")
    ap.add_argument("--live-dir", default="live")
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--out-dir", default="proposals")
    ap.add_argument("--tick", type=float, default=0.25)
    ap.add_argument("--tick-value", type=float, default=12.5)
    ap.add_argument("--commission", type=float, default=4.5, help="USD per contract round trip")
    ap.add_argument("--slippage", type=float, default=1.0, help="ticks per side")
    ap.add_argument("--daily-loss", type=float, default=1000.0)
    ap.add_argument("--trailing-dd", type=float, default=2500.0)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init"); s.add_argument("--approver", required=True); s.set_defaults(fn=cmd_init)
    s = sub.add_parser("backtest"); s.add_argument("--bars", required=True); s.add_argument("--params")
    s.set_defaults(fn=cmd_backtest)
    s = sub.add_parser("propose"); s.add_argument("--bars", required=True); s.add_argument("--trades")
    s.add_argument("--holdout-sessions", type=int, default=20)
    s.add_argument("--no-auto-derisk", action="store_true")
    s.set_defaults(fn=cmd_propose)
    s = sub.add_parser("promote"); s.add_argument("--approver", required=True); s.set_defaults(fn=cmd_promote)
    s = sub.add_parser("rollback"); s.add_argument("--to", required=True); s.add_argument("--approver", required=True)
    s.set_defaults(fn=cmd_rollback)
    s = sub.add_parser("verify"); s.set_defaults(fn=cmd_verify)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
