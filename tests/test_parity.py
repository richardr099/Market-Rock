"""Python <-> C# parity.

1. Static: constants and the parameter bounds table in MarketRockStrategy.cs
   equal those in features.py / params.py.
2. Dynamic (needs the .NET SDK; skipped otherwise): synthetic ticks are
   replayed through the REAL MarketRockStrategy.cs (against tools/ stubs, in
   NT8's primary-before-secondary event order). Its exported bars (incl.
   tick-rule buy/sell volume) and every entry decision must equal the Python
   model's exactly.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from marketrock import features as F, params as P, store
from marketrock.data import Bars, load_csv
from marketrock.features import compute
from marketrock.strategy import regimes, signals

ROOT = Path(__file__).resolve().parents[1]
CS = (ROOT / "ninjatrader" / "MarketRockStrategy.cs").read_text()


def test_constants_match():
    for name, py in (("AtrN", F.ATR_N), ("DeltaZN", F.DELTA_Z_N), ("VolPctL", F.VOL_PCT_L), ("ErN", F.ER_N)):
        m = re.search(rf"const int {name} = (\d+);", CS)
        assert m and int(m.group(1)) == py, name


def test_param_bounds_match():
    rows = re.findall(r'new object\[\] \{ "([a-z_]+)", ([0-9.]+), ([0-9.]+) \}', CS)
    assert [r[0] for r in rows] == list(P.NAMES)
    for name, lo, hi in rows:
        s = P.SPEC_BY_NAME[name]
        assert (float(lo), float(hi)) == (s.lo, s.hi), name


def _dotnet():
    return os.environ.get("DOTNET") or shutil.which("dotnet")


def _ticks(n_sessions=30, bars=200, seed=21):
    rng = np.random.default_rng(seed)
    t, px, v, s = [], [], [], []
    price = 5000.0
    for d in range(n_sessions):
        day0 = 1_735_725_600 + d * 86400          # 2025-01-01 10:00 UTC + d days
        sid = int(np.datetime64(day0, "s").astype("datetime64[D]").astype(str).replace("-", ""))
        drift = rng.choice([-1, 0, 1]) * 0.3
        for b in range(bars):
            end = day0 + 60 * (b + 1)
            k = int(rng.poisson(14)) + 1
            secs = np.sort(rng.integers(end - 59, end + 1, size=k))  # includes ticks exactly at the bar end
            for x in secs:
                price += 0.25 * round(rng.normal(drift, 1.6))
                t.append(int(x)); px.append(price); v.append(float(rng.integers(1, 30))); s.append(sid)
    return np.array(t), np.array(px), np.array(v), np.array(s)


def _py_bars(t, px, v, s):
    end = (t + 59) // 60 * 60
    buy = np.zeros(len(t)); sell = np.zeros(len(t))
    last, ldir = np.nan, 0
    for i in range(len(t)):
        d = 0 if np.isnan(last) else (1 if px[i] > last else (-1 if px[i] < last else ldir))
        if d > 0: buy[i] = v[i]
        elif d < 0: sell[i] = v[i]
        else: buy[i] = sell[i] = v[i] / 2
        ldir, last = d, px[i]
    ue, first = np.unique(end, return_index=True)
    last_idx = np.r_[first[1:], len(t)] - 1
    red = lambda a, f: np.array([f(a[i:j + 1]) for i, j in zip(first, last_idx)])
    return Bars(ue.astype(np.int64), s[first].astype(np.int64), px[first], red(px, np.max), red(px, np.min),
                px[last_idx], red(v, np.sum), red(buy, np.sum), red(sell, np.sum))


@pytest.mark.skipif(_dotnet() is None, reason=".NET SDK not installed")
def test_csharp_replay_matches_python(tmp_path):
    p = P.validate(dict(P.defaults(), delta_z_entry=0.8, regime_vol_lo=0.0, regime_vol_hi=1.0,
                        risk_per_trade_usd=500.0, max_contracts=10.0))
    store.write_live(tmp_path, p)
    t, px, v, s = _ticks()
    with open(tmp_path / "ticks.csv", "w") as f:
        f.write("time,price,volume,session\n")
        for row in zip(t, px, v, s):
            f.write("%d,%r,%r,%d\n" % (int(row[0]), float(row[1]), float(row[2]), int(row[3])))
    out = tmp_path / "bin"
    subprocess.run([_dotnet(), "build", str(ROOT / "tools" / "parity" / "parity.csproj"), "-nologo", "-v", "q",
                    "-o", str(out)], check=True, capture_output=True,
                   env=dict(os.environ, DOTNET_CLI_TELEMETRY_OPTOUT="1"))
    entries_csv = tmp_path / "entries.csv"
    subprocess.run([_dotnet(), str(out / "parity.dll"), str(tmp_path / "ticks.csv"), str(tmp_path), str(entries_csv)],
                   check=True)

    pyb = _py_bars(t, px, v, s)
    csb = load_csv(tmp_path / "bars_history.csv")
    n = len(csb)
    assert n == len(pyb) - 1                     # the final bar is never finalized (no next tick)
    for c in ("time", "session", "open", "high", "low", "close", "volume", "buy_volume", "sell_volume"):
        np.testing.assert_allclose(getattr(csb, c), getattr(pyb, c)[:n], rtol=0, atol=1e-9, err_msg=c)

    f = compute(pyb, 0.25, p["va_pct"])
    feat = np.genfromtxt(str(entries_csv) + ".features.csv", delimiter=",", names=True)
    assert len(feat) == n
    for c in ("atr", "delta_z", "vol_pct", "er", "poc", "vah", "val"):
        np.testing.assert_allclose(feat[c], getattr(f, c)[:n], rtol=1e-9, atol=1e-9, equal_nan=True, err_msg=c)
    reg = regimes(f.vol_pct, f.er, f.atr, f.vah, f.val, p["regime_vol_lo"], p["regime_vol_hi"], p["er_trend"])
    sig = signals(pyb.close, f.delta_z, f.vah, f.val, reg, p["delta_z_entry"], True, True)
    exp = []
    for j in range(n):
        last_of_session = j == len(pyb) - 1 or pyb.session[j + 1] != pyb.session[j]
        if sig[j] == 0 or last_of_session:
            continue
        mult = p["size_mult_rotational"] if reg[j] == 1 else p["size_mult_trend"]
        st = max(1, int(np.floor(p["stop_atr"] * f.atr[j] / 0.25 + 0.5)))
        q = min(int(np.floor(p["risk_per_trade_usd"] * mult / (st * 12.5))), int(p["max_contracts"]))
        if q >= 1:
            exp.append((int(pyb.time[j]), int(sig[j]), q, st, max(1, int(np.floor(st * p["target_rr"] + 0.5)))))
    got = [tuple(int(float(x)) for x in line.split(",")) for line in entries_csv.read_text().splitlines()[1:]]
    assert len(exp) >= 5, "fixture produced too few signals to be a meaningful parity test"
    assert got == exp
