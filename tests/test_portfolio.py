import math

import numpy as np

from marketrock import genome as G, portfolio as PF

CFG = dict(PF.DEFAULT_CONFIG)


def _live(g, fwd_r, live_r=(), stage=0.25, parent=None):
    m = PF.new_member(g, 0, 0, np.array(fwd_r), 0.0, parent, 0, "test")
    m["status"] = "LIVE"
    m["fwd"] = PF.Stats.of(np.array(fwd_r)).to()
    m["live"] = PF.Stats.of(np.array(live_r)).to()
    m["stage"] = stage
    return m


def test_sizing_is_capped_staged_and_monotone_in_edge():
    rng = np.random.default_rng(0)
    g1, g2 = G.seeds()[:2]
    weak = _live(g1, rng.normal(0.15, 1.0, 200))
    strong = _live(g2, rng.normal(0.60, 1.0, 200))
    cfg = dict(CFG, probe_risk_usd=0.0, max_risk_per_trade_usd=10_000.0)
    PF.size({"a": weak}, cfg)
    PF.size({"b": strong}, cfg)
    assert 0 < weak["risk_usd"] < strong["risk_usd"]
    # never above the ruin-constrained size: P(hit floor) <= ruin_prob
    s = strong["sizing"]
    r_ruin = 2 * s["mu_lo"] * s["D"] / (s["sd"] ** 2 * math.log(1 / cfg["ruin_prob"]))
    assert strong["risk_usd"] <= s["stage"] * r_ruin + 1e-6
    # the user's ceiling always wins
    PF.size({"b": strong}, dict(cfg, max_risk_per_trade_usd=40.0))
    assert strong["risk_usd"] == 40.0
    # stage grows with live trades
    grown = _live(g2, rng.normal(0.6, 1.0, 200), live_r=rng.normal(0.6, 1.0, 50))
    PF.size({"c": grown}, cfg)
    assert grown["stage"] == 1.0 and grown["risk_usd"] > strong["sizing"]["stage"] * 0  # sanity


def test_no_edge_sizes_to_probe_only():
    g = G.seeds()[0]
    m = _live(g, np.r_[np.ones(10), -np.ones(10)])
    PF.size({"x": m}, dict(CFG, probe_risk_usd=60.0))
    assert m["risk_usd"] == 60.0 and m["sizing"]["mu_lo"] <= 0


def test_live_failure_retires_and_rolls_back_tuned_child():
    parent_g, child_g = G.seeds()[0], G.mutate(G.seeds()[0], np.random.default_rng(3), small=True)
    parent = _live(parent_g, np.full(50, 0.4) + np.r_[0.5, -0.5] .repeat(25))
    parent["status"] = "RETIRED"
    parent["replaced_by"] = child_g.id
    child = _live(child_g, np.full(50, 0.4) + np.r_[0.5, -0.5].repeat(25), live_r=-np.ones(12), parent=parent_g.id)
    members = {parent_g.id: parent, child_g.id: child}
    log = []
    PF.monitor_live(members, 0, log)
    assert child["status"] == "RETIRED"
    assert parent["status"] == "LIVE"
    assert [e["event"] for e in log] == ["RETIRED", "LIVE"]
    assert "auto-rollback" in log[1]["why"]


def test_live_on_track_is_kept():
    g = G.seeds()[0]
    rng = np.random.default_rng(4)
    m = _live(g, rng.normal(0.3, 1, 100), live_r=rng.normal(0.3, 1, 40))
    PF.monitor_live({"a": m}, 0, [])
    assert m["status"] == "LIVE"


def test_ingest_trades_by_genome_id():
    g = G.seeds()[0]
    m = _live(g, [0.1, 0.2])
    n = PF.ingest_trades({g.id: m}, [{"genome": g.id, "pnl_usd": "-50", "risk_usd": "100"},
                                     {"genome": "unknown", "pnl_usd": "10", "risk_usd": "100"},
                                     {"genome": g.id, "pnl_usd": "150", "risk_usd": "100"}])
    assert n == 2
    st = PF.Stats.frm(m["live"])
    assert st.n == 2 and st.s == 1.0


def test_paper_trial_rejects_strategy_without_forward_edge():
    from marketrock.data import synthetic
    from marketrock.features import compute
    from marketrock.strategy import Costs, Limits
    b = synthetic(n_sessions=120, seed=21, edge=0.0)
    f = compute(b, 0.25, 0.7)
    cut = 40 * 390
    g = G.make(-1, [G.make_condition("LE", "delta_z", -1.0)], 1.2, 1.5, 30)
    # pretend it looked great in its backtest (+0.4R) and was born at session 40
    m = PF.new_member(g, int(b.time[cut]), int(b.session[cut]), np.full(300, 0.4) + np.r_[1, -1].repeat(150),
                      1.0, None, 0, "test")
    log = []
    PF.review_paper({g.id: m}, b, f, Costs(), Limits(1e9, 1e9), dict(CFG), 0, log)
    assert m["status"] == "REJECTED", m
    assert PF.Stats.frm(m["fwd"]).n >= CFG["paper_min_trades"]
