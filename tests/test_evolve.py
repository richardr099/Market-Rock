"""End-to-end: unattended nightly runs over growing data."""
import json

import pytest

from marketrock import cli, store
from marketrock.data import save_csv, synthetic


def _d(tmp):
    return ["--live-dir", str(tmp / "live"), "--state-dir", str(tmp / "state"), "--report-dir", str(tmp / "rep")]


def _nights(tmp, edge, seed, ends=(250, 270, 290, 310)):
    b = synthetic(n_sessions=ends[-1], seed=seed, edge=edge)
    paths = []
    for k, n in enumerate(ends):
        p = tmp / f"night{k}.csv"
        save_csv(b.slice(0, n * 390), p)
        paths.append(p)
    return paths


def _ledger_events(tmp):
    return [e for r in store.ledger_verify(tmp / "state") for e in r.get("events", [])]


def test_noise_never_goes_live(tmp_path):
    d = _d(tmp_path)
    assert cli.main(d + ["init", "--approver", "Op"]) == 0
    for p in _nights(tmp_path, edge=0.0, seed=11):
        assert cli.main(d + ["evolve", "--bars", str(p)]) == 0
    entries, _ = store.read_live(tmp_path / "live")
    assert entries == []
    assert not any(e["event"] == "LIVE" for e in _ledger_events(tmp_path))
    n = json.loads((tmp_path / "state" / "search.json").read_text())["n_trials"]
    assert n > 300                                    # cumulative trial count kept
    assert cli.main(d + ["verify"]) == 0


def test_planted_edge_is_discovered_paper_traded_and_promoted(tmp_path):
    d = _d(tmp_path)
    cli.main(d + ["init", "--approver", "Op"])
    for p in _nights(tmp_path, edge=3.0, seed=4):
        assert cli.main(d + ["evolve", "--bars", str(p)]) == 0
    entries, _ = store.read_live(tmp_path / "live")
    assert len(entries) >= 1
    members = json.loads((tmp_path / "state" / "portfolio.json").read_text())["members"]
    for gid, _, risk in entries:
        hist = [h["event"] for h in members[gid]["history"]]
        assert hist[:2] == ["PAPER", "LIVE"]          # nothing skips the paper trial
        assert 0 < risk <= 250.0                      # within the user's ceiling
    assert any((tmp_path / "rep").glob("*.md"))
    # no clones: live strategies must not share most of their entry bars
    import numpy as np
    from marketrock import genome as G
    from marketrock.data import load_csv
    from marketrock.features import compute
    b = load_csv(tmp_path / "night3.csv")
    f = compute(b, 0.25, 0.7)
    masks = [G.signal(g, b.close, f) != 0 for _, g, _ in entries]
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            u = np.count_nonzero(masks[i] | masks[j])
            assert np.count_nonzero(masks[i] & masks[j]) / u <= 0.6
    assert cli.main(d + ["verify"]) == 0


def test_halt_resume_rollback_and_log_rotation(tmp_path):
    d = _d(tmp_path)
    cli.main(d + ["init", "--approver", "Op"])
    nights = _nights(tmp_path, edge=3.0, seed=4)
    for p in nights:
        cli.main(d + ["evolve", "--bars", str(p)])
    live_before, h_before = store.read_live(tmp_path / "live")
    assert live_before
    assert cli.main(d + ["halt", "--approver", " "]) == 2
    assert cli.main(d + ["halt", "--approver", "Op"]) == 0
    assert store.read_live(tmp_path / "live")[0] == []
    cli.main(d + ["evolve", "--bars", str(nights[-1])])
    assert store.read_live(tmp_path / "live")[0] == []     # stays halted
    assert cli.main(d + ["resume", "--approver", "Op"]) == 0
    cli.main(d + ["evolve", "--bars", str(nights[-1])])
    assert store.read_live(tmp_path / "live")[0] != []
    assert cli.main(d + ["rollback", "--to", h_before, "--approver", "Op"]) == 0
    cli.main(d + ["evolve", "--bars", str(nights[-1])])
    assert store.read_live(tmp_path / "live")[1] == h_before  # pinned
    assert cli.main(d + ["verify"]) == 0
    # trade log rewrite is detected
    t = tmp_path / "trades.csv"
    hdr = ",".join(cli.TRADE_COLUMNS)
    t.write_text(hdr + "\n1,20250101,abc,1,1,10,100,x\n2,20250101,abc,1,1,-10,100,x\n")
    assert cli.main(d + ["evolve", "--bars", str(nights[-1]), "--trades", str(t)]) == 0
    t.write_text(hdr + "\n2,20250101,abc,1,1,-10,100,x\n")
    assert cli.main(d + ["evolve", "--bars", str(nights[-1]), "--trades", str(t)]) == 2
