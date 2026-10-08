import json
import math

import numpy as np
import pytest

from marketrock import autonomy, optimizer as O, params as P, store
from marketrock.data import Bars, synthetic
from marketrock.features import atr_wilder, compute, efficiency_ratio, pct_rank, rolling_z, session_profiles
from marketrock.strategy import EXIT_DAILY, EXIT_STOP, Costs, Limits, _simulate, run
from marketrock.validation import dsr, expected_max_sr, norm_ppf, psr, purged_folds


# ----------------------------------------------------------------- features
def test_no_lookahead():
    b = synthetic(n_sessions=8, seed=1)
    cut = 5 * 390 + 17
    rng = np.random.default_rng(9)
    noise = rng.normal(0, 5, len(b) - cut)
    hi = b.high.copy(); lo = b.low.copy(); cl = b.close.copy(); op = b.open.copy()
    for arr in (hi, lo, cl, op):
        arr[cut:] += noise
    hi[cut:] = np.maximum.reduce([hi[cut:], op[cut:], cl[cut:]])
    lo[cut:] = np.minimum.reduce([lo[cut:], op[cut:], cl[cut:]])
    buy = b.buy_volume.copy(); buy[cut:] = b.volume[cut:]
    b2 = Bars(b.time, b.session, op, hi, lo, cl, b.volume, buy, b.volume - buy)
    f1, f2 = compute(b, 0.25, 0.7, vol_pct_l=200), compute(b2, 0.25, 0.7, vol_pct_l=200)
    for name in ("atr", "delta_z", "vol_pct", "er", "poc", "vah", "val"):
        np.testing.assert_array_equal(getattr(f1, name)[:cut], getattr(f2, name)[:cut], err_msg=name)


def test_profile_uses_prior_session_and_known_values():
    # session 1: three bars, all volume concentrated -> POC at 100.00
    s = np.array([1, 1, 1, 2], dtype=np.int64)
    high = np.array([100.0, 100.25, 100.0, 101.0])
    low = np.array([100.0, 99.75, 100.0, 101.0])
    vol = np.array([100.0, 30.0, 100.0, 5.0])
    poc, vah, val = session_profiles(s, high, low, vol, 0.25, 0.70)
    assert np.isnan(poc[:3]).all()                  # first session has no prior
    assert poc[3] == 100.0
    # hist: 99.75:10, 100.00:210, 100.25:10 ; total 230; 0.7*230=161 <= 210
    assert vah[3] == 100.0 and val[3] == 100.0
    poc, vah, val = session_profiles(s, high, low, vol, 0.25, 0.95)
    # needs 218.5 -> expand; up/down tie (10 vs 10) goes UP first
    assert vah[3] == 100.25 and val[3] == 100.0


def test_level_rounding_is_half_up_not_bankers():
    s = np.array([1, 2], dtype=np.int64)
    # 100.125/0.25 = 400.5 -> half-up gives level 401 = 100.25 (banker's: 400 = 100.00)
    poc, _, _ = session_profiles(s, np.array([100.125, 1.0]), np.array([100.125, 1.0]), np.array([1.0, 1.0]), 0.25, 0.7)
    assert poc[1] == 100.25


def test_indicator_reference_values():
    h = np.array([10, 11, 12, 11, 13.0]); l = h - 1; c = h - 0.5
    a = atr_wilder(h, l, c, 3)
    assert np.isnan(a[:2]).all()
    tr = [1, 1.5, 1.5, 1.5, 2.5]
    assert a[2] == pytest.approx(np.mean(tr[:3]))
    assert a[3] == pytest.approx((a[2] * 2 + tr[3]) / 3)
    z = rolling_z(np.array([1.0, 2, 3]), 3)
    assert z[2] == pytest.approx((3 - 2) / math.sqrt(2 / 3))
    assert pct_rank(np.array([1.0, 3, 2, 2.5]), 3)[3] == pytest.approx(2 / 3)
    er = efficiency_ratio(np.array([1.0, 2, 1, 2, 3]), 4)
    assert er[4] == pytest.approx(2 / 4)


# ----------------------------------------------------------------- backtest
def _one_session(op, hi, lo, cl):
    n = len(cl)
    return (np.ones(n, dtype=np.int64), np.array(op, float), np.array(hi, float),
            np.array(lo, float), np.array(cl, float))


def test_same_bar_stop_and_target_assumes_stop():
    s, o, h, l, c = _one_session([100, 100, 100, 100], [100, 100, 110, 100], [100, 100, 90, 100], [100, 100, 100, 100])
    atr = np.full(4, 4.0); sig = np.array([0, 1, 0, 0]); reg = np.ones(4, dtype=np.int64)
    out = _simulate(s, o, h, l, c, atr, sig, reg, 1.0, 1.0, 50, 1000.0, 1, 1.0, 1.0,
                    0.25, 12.5, 0.0, 0.0, 1e9, 1e9)
    assert out[6][0] == EXIT_STOP
    assert out[4][0] == pytest.approx(-4.0 * 50)  # 4 points * $50


def test_costs_reduce_pnl_and_daily_lock_stops_trading():
    b = synthetic(n_sessions=30, seed=3)
    p = P.defaults()
    free = run(b, p, Costs(commission_rt=0, slip_ticks=0), Limits(1e9, 1e9))
    paid = run(b, p, Costs(), Limits(1e9, 1e9))
    assert free.n > 0
    assert paid.pnl.sum() < free.pnl.sum()
    tight = run(b, p, Costs(), Limits(daily_loss_usd=50.0, trailing_dd_usd=1e9))
    # after a daily-loss exit, no further trade exits in that session
    for i in np.where(tight.why == EXIT_DAILY)[0]:
        assert not np.any(tight.exit_session[i + 1:] == tight.exit_session[i])
    assert tight.n < paid.n


def test_trailing_drawdown_ends_run():
    b = synthetic(n_sessions=30, seed=3)
    r = run(b, P.defaults(), Costs(slip_ticks=4), Limits(1e9, 100.0))
    assert r.blown


def test_no_entry_on_last_bar_of_session():
    b = synthetic(n_sessions=20, seed=5)
    r = run(b, P.defaults())
    assert np.all(b.session[r.entry] == b.session[r.exit])


# --------------------------------------------------------------- statistics
def test_psr_dsr_properties():
    rng = np.random.default_rng(0)
    x = rng.normal(0.1, 1.0, 250)
    assert psr(x, sr_star=float(x.mean() / x.std(ddof=1))) == pytest.approx(0.5, abs=1e-9)
    assert dsr(x, 1000) < dsr(x, 10) < psr(x)
    assert expected_max_sr(1, 1.0) == 0.0
    assert norm_ppf(0.975) == pytest.approx(1.959963985, abs=1e-8)


def test_purged_folds_embargo():
    folds = purged_folds(20, 4, 1)
    for train, test in folds:
        assert not set(train) & set(test)
        for t in test:
            assert t - 1 not in train or t - 1 in test
            assert t + 1 not in train or t + 1 in test


# ---------------------------------------------------------------- optimizer
def test_candidates_respect_trust_region_and_are_deterministic():
    inc = P.defaults()
    c1 = O.propose_candidates(inc, 50, 0.5, np.random.default_rng(7))
    c2 = O.propose_candidates(inc, 50, 0.5, np.random.default_rng(7))
    assert c1 == c2
    u0 = P.to_unit(inc)
    lim = np.array([P.SPEC_BY_NAME[n].max_step for n in P.SEARCH_NAMES])
    for c in c1:
        # integer rounding can move up to half a unit step beyond the cap
        slack = np.array([0.5 / (P.SPEC_BY_NAME[n].hi - P.SPEC_BY_NAME[n].lo) if P.SPEC_BY_NAME[n].integer else 1e-12
                          for n in P.SEARCH_NAMES])
        assert np.all(np.abs(P.to_unit(c) - u0) <= lim + slack)
        for n in P.NAMES:
            if n not in P.SEARCH_NAMES:
                assert c[n] == inc[n]


def test_bayes_disables_confidently_losing_regime_only():
    loser = O.update_regime(O.RegimeState(), [{"session": 1 + i // 5, "pnl_usd": -100.0 if i % 4 else 80.0}
                                              for i in range(120)])
    z = O.regime_sizing(loser, 1.5)
    assert z["enable"] == 0.0 and z["mult"] == 0.0
    fresh = O.regime_sizing(O.RegimeState(), 1.5)  # no evidence: probation, not disabled
    assert fresh["enable"] == 1.0 and fresh["mult"] == O.M_PROBATION


def test_forgetting_decays_toward_prior():
    st = O.update_regime(O.RegimeState(), [{"session": 1, "pnl_usd": 10.0}] * 10)
    st2 = O.update_regime(st, [{"session": 2, "pnl_usd": -1.0}])
    assert st2.a == pytest.approx(O.PRIOR_A + O.FORGET * (st.a - O.PRIOR_A))
    with pytest.raises(ValueError):
        O.update_regime(st2, [{"session": 1, "pnl_usd": 1.0}])


# ----------------------------------------------------------------- autonomy
def test_autonomy_only_auto_applies_risk_reductions():
    inc = P.defaults()
    prop = dict(inc, max_contracts=1.0, size_mult_trend=0.5, enable_rotational=0.0,  # de-risk
                regime_vol_hi=0.99, va_pct=0.75)                                    # risk-up, alpha
    auto, human = autonomy.split_changes(inc, prop)
    assert set(auto) == {"max_contracts", "size_mult_trend", "enable_rotational"}
    assert set(human) == {"regime_vol_hi", "va_pct"}
    applied = autonomy.apply_derisk(inc, prop)
    assert applied["regime_vol_hi"] == inc["regime_vol_hi"] and applied["va_pct"] == inc["va_pct"]
    assert applied["max_contracts"] == 1.0
    # re-enabling is a risk increase -> never automatic
    off = dict(inc, enable_trend=0.0)
    assert autonomy.split_changes(off, inc)[0] == {}


# -------------------------------------------------------------------- store
def test_live_params_hash_and_ledger_tamper(tmp_path):
    live, state = tmp_path / "live", tmp_path / "state"
    h = store.write_live(live, P.defaults())
    p, h2 = store.read_live(live)
    assert h == h2 and p == P.validate(P.defaults())
    store.ledger_append(state, {"event": "a"})
    store.ledger_append(state, {"event": "b"})
    assert len(store.ledger_verify(state)) == 2
    lines = (state / "ledger.jsonl").read_bytes().splitlines()
    lines[0] = lines[0].replace(b'"a"', b'"x"')
    (state / "ledger.jsonl").write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(ValueError, match="chain broken"):
        store.ledger_verify(state)
    (live / "params.json").write_bytes((live / "params.json").read_bytes().replace(b"2.0", b"9.0", 1))
    with pytest.raises(ValueError, match="hash mismatch"):
        store.read_live(live)


def test_validate_rejects_out_of_range_and_unknown():
    with pytest.raises(ValueError):
        P.validate(dict(P.defaults(), max_contracts=11.0))
    with pytest.raises(ValueError):
        P.validate(dict(P.defaults(), surprise=1.0))
    with pytest.raises(ValueError):
        P.validate(dict(P.defaults(), max_hold_bars=10.5))
