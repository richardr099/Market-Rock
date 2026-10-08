"""Semi-autonomy policy: the machine may only make itself SAFER on its own.

A change is auto-applicable iff it touches no alpha parameter (risk_dir == 0)
and every touched risk parameter moves in its risk-reducing direction.
Everything else (any alpha change, any risk increase, re-enabling a regime)
requires `marketrock promote --approver <name>`.
"""
from __future__ import annotations

from typing import Dict, Mapping, Tuple

from . import params as P

EPS = 1e-12


def split_changes(incumbent: Mapping[str, float], proposed: Mapping[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return (auto, human): dicts of {name: proposed_value} for changed keys."""
    auto: Dict[str, float] = {}
    human: Dict[str, float] = {}
    for s in P.SPECS:
        old, new = float(incumbent[s.name]), float(proposed[s.name])
        if abs(new - old) <= EPS:
            continue
        if s.risk_dir != 0 and (new - old) * s.risk_dir < 0:
            auto[s.name] = new
        else:
            human[s.name] = new
    return auto, human


def apply_derisk(incumbent: Mapping[str, float], proposed: Mapping[str, float]) -> Dict[str, float]:
    auto, _ = split_changes(incumbent, proposed)
    out = dict(incumbent)
    out.update(auto)
    out = P.validate(out)
    # Defence in depth: re-check the result is risk-monotone vs the incumbent.
    _, human = split_changes(incumbent, out)
    if human:
        raise AssertionError(f"auto-derisk produced non-derisk changes: {human}")
    return out
