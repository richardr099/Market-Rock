import numpy as np
import pytest

from marketrock import genome as G
from marketrock.data import synthetic
from marketrock.features import compute


def test_text_roundtrip_and_id_stable():
    rng = np.random.default_rng(0)
    for _ in range(300):
        g = G.random_genome(rng)
        assert G.parse(g.text()) == g
        assert g.id == G.parse(g.text()).id and len(g.id) == 12


def test_parse_rejects_bad_rules():
    good = G.seeds()[0].text()
    for bad in (good.replace("dir=1", "dir=2"), good.replace("CROSS_UP", "TELEPORT"),
                good.replace("stop=1.25", "stop=9.0"), good + "|GE er 0.4|GE er 0.5|GE er 0.6|GE er 0.7",
                good.replace("GE delta_z 1.0", "GE delta_z 7.0"), good.replace("stop=1.25", "stop=1.250")):
        with pytest.raises((ValueError, KeyError)):
            G.parse(bad)


def test_variation_stays_valid_and_deterministic():
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    a, b = G.seeds()[0], G.seeds()[2]
    out1 = [G.mutate(a, r1) for _ in range(100)] + [G.crossover(a, b, r1) for _ in range(100)]
    out2 = [G.mutate(a, r2) for _ in range(100)] + [G.crossover(a, b, r2) for _ in range(100)]
    assert out1 == out2
    for g in out1:
        G.validate(g)


def test_small_mutation_keeps_structure():
    rng = np.random.default_rng(1)
    g = G.seeds()[0]
    for _ in range(100):
        k = G.mutate(g, rng, small=True)
        assert k.direction == g.direction and len(k.conds) == len(g.conds)
        assert [c.op for c in k.conds] == [c.op for c in g.conds]


def test_condition_semantics():
    b = synthetic(n_sessions=6, seed=2)
    f = compute(b, 0.25, 0.7, vol_pct_l=200)
    prev = np.r_[np.nan, b.close[:-1]]
    m = G.condition_mask(G.make_condition("CROSS_UP", "VAL"), b.close, f)
    i = np.where(m)[0]
    assert len(i) and np.all(prev[i] <= f.val[i]) and np.all(b.close[i] > f.val[i])
    m = G.condition_mask(G.make_condition("TIME_IN", "", 10, 20), b.close, f)
    assert set(f.bar_in_session[m]) == set(range(10, 20))
    sig = G.signal(G.seeds()[1], b.close, f)
    assert set(np.unique(sig)) <= {0, -1}
