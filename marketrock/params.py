"""Parameter schema: the only surface the learning loop is allowed to mutate.

Hard account limits (trailing drawdown, daily loss, consistency cap) are NOT
here on purpose. They live as user-set NinjaScript properties on the C# side,
so no output of this package can loosen them.

Every parameter declares ``risk_dir``:
  +1  increasing the value increases risk      (e.g. max_contracts)
  -1  decreasing the value increases risk      (e.g. regime_vol_lo)
   0  alpha parameter, no monotone risk effect (e.g. va_pct)
The autonomy policy (``autonomy.py``) uses this to decide which changes may be
applied without a human: only changes that move every touched risk parameter
in the risk-reducing direction, and touch no alpha parameter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np


@dataclass(frozen=True)
class ParamSpec:
    name: str
    lo: float
    hi: float
    default: float
    max_step: float      # max |change| per session, as a fraction of (hi - lo)
    risk_dir: int        # +1 / -1 / 0, see module docstring
    integer: bool = False


SPECS: tuple[ParamSpec, ...] = (
    # --- alpha (Auction Market Theory) ---
    ParamSpec("va_pct",            0.60, 0.80, 0.70, 0.10, 0),
    ParamSpec("delta_z_entry",     0.00, 3.00, 1.00, 0.10, 0),
    ParamSpec("stop_atr",          0.50, 3.00, 1.25, 0.10, 0),
    ParamSpec("target_rr",         0.80, 3.00, 1.50, 0.10, 0),
    ParamSpec("er_trend",          0.20, 0.70, 0.40, 0.10, 0),
    ParamSpec("max_hold_bars",     5.0,  120.0, 30.0, 0.10, 0, integer=True),
    # --- regime filter (risk-bearing: widening the tradable band adds risk) ---
    ParamSpec("regime_vol_lo",     0.00, 0.40, 0.15, 0.10, -1),
    ParamSpec("regime_vol_hi",     0.60, 1.00, 0.95, 0.10, +1),
    # --- sizing ---
    ParamSpec("risk_per_trade_usd", 50.0, 500.0, 150.0, 0.10, +1),
    ParamSpec("max_contracts",      1.0,  10.0,  2.0,  0.12, +1, integer=True),
    # --- per-regime gates written by the Bayesian sizing layer ---
    ParamSpec("enable_rotational", 0.0, 1.0, 1.0, 1.0, +1, integer=True),
    ParamSpec("enable_trend",      0.0, 1.0, 1.0, 1.0, +1, integer=True),
    ParamSpec("size_mult_rotational", 0.0, 1.0, 1.0, 1.0, +1),
    ParamSpec("size_mult_trend",      0.0, 1.0, 1.0, 1.0, +1),
)

SPEC_BY_NAME: Dict[str, ParamSpec] = {s.name: s for s in SPECS}
NAMES: tuple[str, ...] = tuple(s.name for s in SPECS)

# Parameters the evolutionary search explores. The regime gates / size
# multipliers are owned by the Bayesian layer (optimizer.regime_posteriors),
# not by random search.
SEARCH_NAMES: tuple[str, ...] = (
    "va_pct", "delta_z_entry", "stop_atr", "target_rr", "er_trend",
    "max_hold_bars", "regime_vol_lo", "regime_vol_hi",
)


def defaults() -> Dict[str, float]:
    return {s.name: s.default for s in SPECS}


def validate(p: Mapping[str, float]) -> Dict[str, float]:
    """Return a canonical copy; raise ValueError on any missing/extra/out-of-range key."""
    missing = set(NAMES) - set(p)
    extra = set(p) - set(NAMES)
    if missing or extra:
        raise ValueError(f"param keys mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    out: Dict[str, float] = {}
    for s in SPECS:
        v = float(p[s.name])
        if not np.isfinite(v) or v < s.lo - 1e-12 or v > s.hi + 1e-12:
            raise ValueError(f"{s.name}={v} outside [{s.lo}, {s.hi}]")
        if s.integer and abs(v - round(v)) > 1e-9:
            raise ValueError(f"{s.name}={v} must be an integer")
        out[s.name] = float(round(v)) if s.integer else v
    if out["regime_vol_lo"] >= out["regime_vol_hi"]:
        raise ValueError("regime_vol_lo must be < regime_vol_hi")
    return out


def to_unit(p: Mapping[str, float], names=SEARCH_NAMES) -> np.ndarray:
    return np.array([(p[n] - SPEC_BY_NAME[n].lo) / (SPEC_BY_NAME[n].hi - SPEC_BY_NAME[n].lo)
                     for n in names], dtype=np.float64)


def from_unit(u: np.ndarray, base: Mapping[str, float], names=SEARCH_NAMES) -> Dict[str, float]:
    out = dict(base)
    for x, n in zip(u, names):
        s = SPEC_BY_NAME[n]
        v = s.lo + float(np.clip(x, 0.0, 1.0)) * (s.hi - s.lo)
        out[n] = float(round(v)) if s.integer else v
    return out
