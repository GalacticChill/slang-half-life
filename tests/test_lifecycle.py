import numpy as np
import pandas as pd
import pytest

from slang_half_life import lifecycle as lc


def _series(values, start="2018-01", name="t"):
    idx = pd.period_range(start, periods=len(values), freq="M")
    return pd.Series(values, index=idx, name=name, dtype=float)


def _rise_then_decay(rise, half_life, tail, peak=1000.0, flat=6):
    """Flat low start, linear rise to the peak, then exact exponential decay."""
    up = np.linspace(peak * 0.01, peak, rise + 1)
    down = peak * 0.5 ** (np.arange(1, tail + 1) / half_life)
    return np.concatenate([np.full(flat, peak * 0.01), up, down])


def test_recovers_known_half_life_exactly():
    for hl in (3, 6, 11):
        m = lc.measure(_series(_rise_then_decay(rise=8, half_life=hl, tail=60)), window=1)
        assert m.decayed
        assert m.half_life_months == hl


def test_recovers_known_rise_time():
    # Rise from 1% to 100% over 8 steps: the last month below 10% is step 0.
    m = lc.measure(_series(_rise_then_decay(rise=8, half_life=6, tail=60)), window=1)
    assert m.rise_months == 8
    assert str(m.peak_month) == str(pd.Period("2018-01", "M") + 6 + 8)


def test_still_alive_is_censored_not_dropped():
    values = _rise_then_decay(rise=8, half_life=40, tail=20)  # only fell to ~70%
    m = lc.measure(_series(values), window=1)
    assert not m.decayed
    assert np.isnan(m.half_life_months)
    assert m.followup_months == 20


def test_brief_dip_does_not_count_as_decay():
    values = [10, 100, 90, 40, 95, 90, 85, 80]  # one month below half, then recovers
    m = lc.measure(_series(values), window=1)
    assert not m.decayed


def test_decay_must_be_sustained_three_months():
    values = [10, 100, 90, 40, 40, 40, 30]
    assert lc.measure(_series(values), window=1).half_life_months == 2


def test_smoothing_ignores_a_one_month_spike():
    values = np.array([50.0] * 20)
    values[3:6] = 200  # a genuine three-month bump
    values[14] = 400  # a lone one-month spike, twice as tall
    # Raw, the spike is the peak; smoothed (50,400,50 -> 167 vs 200), the bump wins.
    assert lc.measure(_series(values), window=1).peak_month == pd.Period("2018-03", "M") + 12
    assert lc.measure(_series(values)).peak_month == pd.Period("2018-05", "M")


def test_rise_unobserved_when_already_popular():
    m = lc.measure(_series([900, 950, 1000, 800, 400, 300, 200, 100]), window=1)
    assert np.isnan(m.rise_months)
    assert not m.peak_observed  # peak in month 2 of the data: may not be the real peak


def test_months_before_creation_are_ignored():
    values = [np.nan] * 5 + [10, 100, 40, 30, 20]
    m = lc.measure(_series(values), window=1)
    assert str(m.peak_month) == "2018-07"
    assert m.half_life_months == 1


def test_stickiness():
    values = [10, 100] + [25] * 12
    assert lc.measure(_series(values), window=1).stickiness == pytest.approx(0.25)


def test_measure_all_applies_size_and_edge_rules():
    idx = pd.period_range("2018-01", periods=20, freq="M")
    shape = np.array([1, 2, 5, 10, 20, 50, 100, 80, 50, 30, 20, 10, 8, 6, 5, 5, 4, 4, 3, 3], float)
    prepared = pd.DataFrame({"big": shape, "small": shape}, index=idx)
    raw = pd.DataFrame({"big": shape * 50, "small": shape}, index=idx)  # small peaks ~100 raw
    out = lc.measure_all(prepared, raw)
    assert out.loc["big", "measurable"] and out.loc["big", "usable"]
    assert not out.loc["small", "measurable"] and not out.loc["small", "usable"]
    assert out.loc["big", "peak_year"] == 2018


def test_empty_series_is_an_error():
    with pytest.raises(ValueError):
        lc.measure(_series([np.nan, np.nan]))
