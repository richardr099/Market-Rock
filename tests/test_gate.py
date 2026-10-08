"""End-to-end: the nightly loop on noise must not produce a promotable candidate,
and promote must refuse anything ungated, unnamed, or stale."""
import json

import numpy as np

from marketrock import cli, store
from marketrock.data import save_csv, synthetic


def _dirs(tmp_path):
    return ["--live-dir", str(tmp_path / "live"), "--state-dir", str(tmp_path / "state"),
            "--out-dir", str(tmp_path / "prop")]


def test_noise_is_rejected_and_promote_refuses(tmp_path):
    bars = tmp_path / "bars.csv"
    save_csv(synthetic(n_sessions=60, seed=11, edge=0.0), bars)
    d = _dirs(tmp_path)
    assert cli.main(d + ["init", "--approver", "Test Operator"]) == 0
    assert cli.main(d + ["propose", "--bars", str(bars)]) == 0
    rep = json.loads((tmp_path / "prop" / "report.json").read_text())
    assert rep["status"] == "REJECTED"
    assert not rep["gates"]["deflated_sharpe"]["pass"]
    assert cli.main(d + ["promote", "--approver", "Someone"]) == 2
    # trial counter is cumulative across runs (DSR depends on it)
    n1 = json.loads((tmp_path / "state" / "search.json").read_text())["n_trials"]
    assert cli.main(d + ["propose", "--bars", str(bars)]) == 0
    n2 = json.loads((tmp_path / "state" / "search.json").read_text())["n_trials"]
    assert n2 > n1
    assert cli.main(d + ["verify"]) == 0


def test_promote_refuses_stale_base_and_blank_approver(tmp_path):
    d = _dirs(tmp_path)
    assert cli.main(d + ["init", "--approver", "Op"]) == 0
    prop = tmp_path / "prop"; prop.mkdir()
    from marketrock import params as P
    cand = dict(P.defaults(), va_pct=0.72)
    (prop / "candidate.json").write_bytes(store.canonical(cand))
    store.write_json(prop / "report.json", {"status": "AWAITING_APPROVAL", "base_params_sha256": "f" * 64})
    assert cli.main(d + ["promote", "--approver", "Op"]) == 2          # stale base
    _, live_hash = store.read_live(tmp_path / "live")
    store.write_json(prop / "report.json", {"status": "AWAITING_APPROVAL", "base_params_sha256": live_hash})
    assert cli.main(d + ["promote", "--approver", "  "]) == 2          # unnamed
    assert cli.main(d + ["promote", "--approver", "Op"]) == 0
    p, _ = store.read_live(tmp_path / "live")
    assert p["va_pct"] == 0.72
    assert cli.main(d + ["verify"]) == 0


def test_auto_derisk_from_losing_live_fills(tmp_path):
    bars = tmp_path / "bars.csv"
    save_csv(synthetic(n_sessions=40, seed=2), bars)
    trades = tmp_path / "trades.csv"
    rows = ["exit_time,session,regime,direction,qty,pnl_usd,params_sha256"]
    for i in range(150):  # trend regime (2) loses consistently
        rows.append(f"{1000 + i},{20250101 + i // 5},2,1,1,{-120.0 if i % 4 else 60.0},x")
    trades.write_text("\n".join(rows) + "\n")
    d = _dirs(tmp_path)
    cli.main(d + ["init", "--approver", "Op"])
    assert cli.main(d + ["propose", "--bars", str(bars), "--trades", str(trades), "--holdout-sessions", "10"]) == 0
    p, _ = store.read_live(tmp_path / "live")
    assert p["enable_trend"] == 0.0 and p["size_mult_trend"] == 0.0
    assert p["enable_rotational"] == 1.0          # no evidence -> untouched (probation mult only lowers size)
    assert p["size_mult_rotational"] <= 1.0
    events = [r["event"] for r in store.ledger_verify(tmp_path / "state")]
    assert "auto_derisk" in events
    # rewriting the trade log is detected
    trades.write_text("\n".join(rows[:1] + rows[5:]) + "\n")
    assert cli.main(d + ["propose", "--bars", str(bars), "--trades", str(trades), "--holdout-sessions", "10"]) == 2


def test_planted_edge_passes_gate_then_human_promotes(tmp_path):
    """Positive path: with a real (planted) edge and ~1 year of sessions the
    gate can pass. Guards against a gate so strict it can never approve."""
    bars = tmp_path / "bars.csv"
    save_csv(synthetic(n_sessions=250, seed=4, edge=3.0), bars)
    d = _dirs(tmp_path)
    assert cli.main(d + ["init", "--approver", "Op"]) == 0
    assert cli.main(d + ["propose", "--bars", str(bars), "--holdout-sessions", "50"]) == 0
    rep = json.loads((tmp_path / "prop" / "report.json").read_text())
    assert rep["status"] == "AWAITING_APPROVAL", rep["gates"]
    before, _ = store.read_live(tmp_path / "live")
    assert cli.main(d + ["promote", "--approver", "Op"]) == 0
    after, h = store.read_live(tmp_path / "live")
    assert after != before
    assert store.ledger_verify(tmp_path / "state")[-1]["approver"] == "Op"
