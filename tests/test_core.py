import json
import math

import numpy as np
import pytest

from marketrock import genome as G, store
from marketrock.data import Bars, synthetic
from marketrock.features import atr_wilder, bar_in_session, compute, efficiency_ratio, pct_rank, rolling_z, session_profiles
from marketrock.strategy import EXIT_DAILY, EXIT_STOP, Costs, Limits, _simulate, combine, run
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
    for name in ("atr", "delta_z", "vol_pct", "er", "poc", "vah", "val", "bar_in_session"):
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
SEEDS = G.seeds()


def test_bar_in_session():
    assert list(bar_in_session(np.array([5, 5, 5, 6, 6, 7], dtype=np.int64))) == [0, 1, 2, 0, 1, 0]


def test_same_bar_stop_and_target_assumes_stop():
    s = np.ones(4, dtype=np.int64)
    o = np.full(4, 100.0); c = np.full(4, 100.0)
    h = np.array([100, 100, 110, 100.0]); l = np.array([100, 100, 90, 100.0])
    out = _simulate(s, o, h, l, c, np.full(4, 4.0), np.array([0, 1, 0, 0]), np.array([-1, 0, -1, -1]),
                    np.array([1.0]), np.array([1.0]), np.array([50]), np.array([100.0]),
                    0.25, 1.25, 0.0, 0.0, 1e9, 1e9)
    # MES: stop = 1.0 x ATR(4) = 16 ticks; qty = floor(100 / (16 x 1.25)) = 5; loss = 4 pts x $5 x 5
    assert out[7][0] == EXIT_STOP
    assert out[3][0] == 5
    assert out[4][0] == pytest.approx(-100.0)
    assert out[5][0] == pytest.approx(100.0)         # planned risk recorded -> R = -1


def test_costs_reduce_pnl_and_daily_lock_stops_trading():
    b = synthetic(n_sessions=30, seed=3)
    free = run(b, SEEDS, [150.0] * 4, Costs(commission_rt=0, slip_ticks=0), Limits(1e9, 1e9))
    paid = run(b, SEEDS, [150.0] * 4, Costs(), Limits(1e9, 1e9))
    assert free.n > 0
    assert paid.pnl.sum() < free.pnl.sum()
    tight = run(b, SEEDS, [150.0] * 4, Costs(), Limits(daily_loss_usd=50.0, trailing_dd_usd=1e9))
    for i in np.where(tight.why == EXIT_DAILY)[0]:
        assert not np.any(tight.exit_session[i + 1:] == tight.exit_session[i])
    assert tight.n < paid.n


def test_trailing_drawdown_ends_run():
    b = synthetic(n_sessions=30, seed=3)
    assert run(b, SEEDS, [150.0] * 4, Costs(slip_ticks=4), Limits(1e9, 100.0)).blown


def test_no_entry_on_last_bar_of_session_and_priority():
    b = synthetic(n_sessions=20, seed=5)
    r = run(b, SEEDS, [150.0] * 4)
    assert r.n > 0 and np.all(b.session[r.entry] == b.session[r.exit])
    f = compute(b, 0.25, 0.7)
    sd, sg = combine(SEEDS, b.close, f)
    s0 = G.signal(SEEDS[0], b.close, f)
    assert np.all(sg[s0 != 0] == 0)            # genome 0 has priority wherever it fires


# --------------------------------------------------------------- statistics
def test_psr_dsr_properties():
    rng = np.random.default_rng(0)
    x = rng.normal(0.1, 1.0, 250)
    assert psr(x, sr_star=float(x.mean() / x.std(ddof=1))) == pytest.approx(0.5, abs=1e-9)
    assert dsr(x, 1000) < dsr(x, 10) < psr(x)
    assert expected_max_sr(1, 1.0) == 0.0
    assert norm_ppf(0.975) == pytest.approx(1.959963985, abs=1e-8)


def test_purged_folds_embargo():
    for train, test in purged_folds(20, 4, 1):
        assert not set(train) & set(test)
        for t in test:
            assert t - 1 not in train or t - 1 in test
            assert t + 1 not in train or t + 1 in test


# -------------------------------------------------------------------- store
def test_portfolio_file_hash_ids_and_ledger_tamper(tmp_path):
    live, state = tmp_path / "live", tmp_path / "state"
    entries = [(g.id, g, 150.0) for g in SEEDS[:2]]
    h = store.write_live(live, entries)
    got, h2 = store.read_live(live)
    assert h == h2 and [e[0] for e in got] == [e[0] for e in entries]
    with pytest.raises(ValueError, match="id mismatch"):
        store.canonical([("deadbeef0000", SEEDS[0], 1.0)])
    store.ledger_append(state, {"event": "a"})
    store.ledger_append(state, {"event": "b"})
    assert len(store.ledger_verify(state)) == 2
    lines = (state / "ledger.jsonl").read_bytes().splitlines()
    lines[0] = lines[0].replace(b'"a"', b'"x"')
    (state / "ledger.jsonl").write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(ValueError, match="chain broken"):
        store.ledger_verify(state)
    (live / "portfolio.txt").write_bytes((live / "portfolio.txt").read_bytes().replace(b"150.0", b"950.0", 1))
    with pytest.raises(ValueError, match="hash mismatch"):
        store.read_live(live)
