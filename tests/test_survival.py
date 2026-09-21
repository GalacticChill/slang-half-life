import numpy as np
import pandas as pd
import pytest
from scipy import stats

from slang_half_life import survival as sv


def test_kaplan_meier_hand_worked_example():
    # 5 terms: half-lives 1, 2, 3 (decayed) and 2, 4 (still alive when last seen).
    km = sv.kaplan_meier([1, 2, 2, 3, 4], [True, True, False, True, False])
    # t=1: 1 of 5 fades -> 0.8. t=2: 1 of 4 fades -> 0.6 (the censored 2 still counts at risk).
    # t=3: 1 of 2 fades -> 0.3. t=4: censored only, stays 0.3.
    assert km["at_risk"].tolist() == [5, 4, 2, 1]
    assert km["survival"].tolist() == pytest.approx([0.8, 0.6, 0.3, 0.3])
    assert sv.median_survival(km) == 3
    assert sv.survival_at(km, 2.5) == pytest.approx(0.6)
    assert sv.survival_at(km, 0) == 1.0


def test_nothing_decayed_means_no_median():
    km = sv.kaplan_meier([5, 8, 12], [False, False, False])
    assert (km["survival"] == 1).all()
    assert np.isnan(sv.median_survival(km))


def test_dropping_censored_terms_would_bias_toward_fast():
    # The slow faders are still alive; ignoring them halves the apparent median.
    d = [2, 3, 4, 20, 20, 20, 20]
    e = [True, True, True, False, False, False, False]
    naive = np.median([x for x, ev in zip(d, e) if ev])
    assert naive == 3
    assert np.isnan(sv.median_survival(sv.kaplan_meier(d, e)))  # >50% never faded


def test_logrank_two_groups_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.exponential(6, 30).round() + 1, rng.exponential(10, 30).round() + 1
    ea, eb = rng.random(30) < 0.8, rng.random(30) < 0.8
    ours = sv.logrank(np.r_[a, b], np.r_[ea, eb], ["a"] * 30 + ["b"] * 30)
    ref = stats.logrank(stats.CensoredData(uncensored=a[ea], right=a[~ea]),
                        stats.CensoredData(uncensored=b[eb], right=b[~eb]))
    assert ours.statistic == pytest.approx(ref.statistic ** 2, rel=1e-6)
    assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-6)


def _eras(scales, n=40, seed=1):
    rng = np.random.default_rng(seed)
    d, e, g = [], [], []
    for era, scale in scales.items():
        t = rng.exponential(scale, n).round() + 1
        cens = rng.uniform(5, 60, n).round()
        d += list(np.minimum(t, cens)); e += list(t <= cens); g += [era] * n
    return d, e, g


def test_logrank_detects_a_planted_speedup():
    lr = sv.logrank(*_eras({"early": 14, "mid": 7, "late": 3}))
    assert lr.df == 2 and lr.p_value < 0.001


def test_logrank_finds_nothing_when_eras_are_the_same():
    ps = [sv.logrank(*_eras({"early": 8, "mid": 8, "late": 8}, seed=s)).p_value for s in range(20)]
    assert np.median(ps) > 0.2  # no difference: p-values spread out, not near 0


def test_sidak():
    assert sv.sidak(0.05, 1) == pytest.approx(0.05)
    assert sv.sidak(0.05, 3) == pytest.approx(1 - 0.95 ** 3)


def test_peak_era():
    assert sv.peak_era(2015) == "2016-2018"
    assert sv.peak_era(2021) == "2019-2021"
    assert sv.peak_era(2026) == "2022+"


def test_compare_eras_uses_censored_followup():
    m = pd.DataFrame({
        "usable": [True, True, True, True, False],
        "decayed": [True, True, False, True, True],
        "half_life_months": [4, 6, np.nan, 2, 1],
        "followup_months": [50, 40, 30, 10, 99],
        "peak_era": ["2016-2018", "2016-2018", "2022+", "2022+", "2022+"],
    })
    out = sv.compare_eras(m)
    s = out["summary"]
    assert s.loc["2022+", "terms"] == 2 and s.loc["2022+", "still_alive"] == 1
    assert s.loc["2016-2018", "median_half_life"] == 4
    assert len(out["pairs"]) == 1
