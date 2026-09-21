import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from slang_half_life import shapes as sh


def _curve(kind, rng, n=80, peak_at=30):
    t = np.arange(n) - peak_at
    rise = np.exp(np.minimum(t, 0) / 3)
    if kind == "flash":
        after = 0.5 ** (np.maximum(t, 0) / 1.5)
    elif kind == "slow":
        after = 0.5 ** (np.maximum(t, 0) / 10)
    else:  # stuck: fades to a 60% plateau
        after = 0.6 + 0.4 * 0.5 ** (np.maximum(t, 0) / 2)
    y = np.where(t <= 0, rise, after) * 1000
    return y * (1 + 0.03 * rng.standard_normal(n)).clip(0.9, 1.1)


def _table(kinds, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2017-01", periods=80, freq="M")
    return pd.DataFrame({f"{k}{i}": _curve(k, rng) for i, k in enumerate(kinds)}, index=idx)


def test_alignment_puts_peak_at_zero_scaled_to_one():
    prepared = _table(["slow"])
    shapes = sh.aligned_shapes(prepared, prepared.columns)
    assert list(shapes.columns) == list(range(-12, 25))
    assert shapes.iloc[0].idxmax() == 0
    assert shapes.iloc[0].max() == 1.0


def test_too_recent_terms_are_dropped_not_guessed():
    short = _table(["slow"]).iloc[:45]  # peak at month 30: only 14 months after it
    assert sh.aligned_shapes(short, short.columns).empty


def test_months_before_page_existed_count_as_zero():
    prepared = _table(["slow"])
    prepared.iloc[:25, 0] = np.nan  # page created 5 months before the peak
    shapes = sh.aligned_shapes(prepared, prepared.columns)
    assert (shapes.iloc[0].loc[-12:-7] == 0).all()
    assert shapes.iloc[0].loc[-4] > 0


def test_recovers_and_names_planted_shapes():
    kinds = ["flash"] * 8 + ["slow"] * 8 + ["stuck"] * 8
    prepared = _table(kinds, seed=3)
    shapes, named = sh.classify(prepared, prepared.columns)
    truth = [c.rstrip("0123456789") for c in shapes.index]
    assert adjusted_rand_score(truth, named) == 1.0
    expected = {"flash": "flash in the pan", "slow": "slow burn", "stuck": "stuck around"}
    assert all(named[c] == expected[t] for c, t in zip(shapes.index, truth))


def test_silhouette_prefers_the_true_k():
    prepared = _table(["flash"] * 8 + ["slow"] * 8 + ["stuck"] * 8, seed=4)
    scores = sh.silhouettes(sh.aligned_shapes(prepared, prepared.columns), ks=range(2, 6))
    assert max(scores, key=scores.get) == 3


def test_centroids_one_row_per_type():
    prepared = _table(["flash"] * 5 + ["slow"] * 5 + ["stuck"] * 5, seed=5)
    shapes, named = sh.classify(prepared, prepared.columns)
    c = sh.centroids(shapes, named)
    assert set(c.index) == {"flash in the pan", "slow burn", "stuck around"}
    assert c.loc["stuck around", 24] > c.loc["slow burn", 24] > c.loc["flash in the pan", 24]
