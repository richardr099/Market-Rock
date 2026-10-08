"""Plain-language daily report."""
from __future__ import annotations

from typing import Dict, List, Mapping

from . import genome as G
from .portfolio import Stats


def render(date: str, members: Dict[str, dict], events: List[dict], cfg: Mapping, info: Mapping) -> str:
    out = [f"# Market-Rock daily report — {date}", ""]
    if cfg.get("halted"):
        out += ["**TRADING IS HALTED** (`marketrock resume` to re-enable). No strategies are live.", ""]
    live = [(k, m) for k, m in members.items() if m["status"] == "LIVE"]
    paper = [(k, m) for k, m in members.items() if m["status"] == "PAPER"]
    out += ["## What changed today", ""]
    if events:
        for e in events:
            out.append(f"- **{e['event']}** `{e['id']}` — {e['why']}  \n  _{e['rule']}_")
    else:
        out.append("- Nothing. No strategy was promoted, retired or rolled back.")
    out += ["", "## Live strategies", ""]
    if live:
        out += ["| id | rule | size stage | risk/trade | live trades | live avg | live total |",
                "|---|---|---|---|---|---|---|"]
        for k, m in live:
            lv = Stats.frm(m["live"])
            out.append(f"| `{k}` | {G.parse(m['rule']).describe()} | {m['stage']:.0%} | ${m['risk_usd']:.0f} | "
                       f"{lv.n} | {lv.mean:+.2f}R | {lv.s:+.1f}R |")
    else:
        out.append("None. The system is not trading. That is the correct state until a strategy proves itself on forward data.")
    out += ["", "## On paper trial", ""]
    if paper:
        for k, m in paper:
            fw = Stats.frm(m["fwd"])
            out.append(f"- `{k}`: {fw.n}/{cfg['paper_min_trades']} forward trades, avg {fw.mean:+.2f}R — "
                       f"{G.parse(m['rule']).describe()}")
    else:
        out.append("None.")
    out += ["", "## Research", "",
            f"- Strategies evaluated tonight: {info.get('evaluated', 0)} (all-time: {info.get('n_trials', 0)}; "
            "every one counts against the significance test)",
            f"- Candidates that passed the gate tonight: {info.get('passed', 0)}",
            f"- Data: {info.get('sessions', 0)} sessions up to session {info.get('last_session', '-')}",
            "", "## Needs you", ""]
    needs = list(info.get("warnings", []))
    out += [f"- {w}" for w in needs] if needs else ["- Nothing."]
    out += ["", f"_Your automatic-sizing ceiling: ${cfg['max_risk_per_trade_usd']:.0f} per trade "
            "(state/config.json `max_risk_per_trade_usd`)._", ""]
    return "\n".join(out)
