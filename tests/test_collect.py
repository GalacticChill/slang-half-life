import io
import urllib.error

import numpy as np
import pandas as pd
import pytest

from slang_half_life import collect as cl


def _items(pairs):
    return {"items": [{"timestamp": d.replace("-", "") + "00", "views": v} for d, v in pairs]}


def test_url_uses_slug_and_full_day_range():
    url = cl.pageviews_url("OK boomer", "2015-07", "2026-08")
    assert url.endswith("/en.wiktionary.org/all-access/user/OK_boomer/daily/20150701/20260831")


def test_last_full_month():
    assert cl.last_full_month(pd.Timestamp("2026-09-21")) == "2026-08"
    assert cl.last_full_month(pd.Timestamp("2026-01-05")) == "2025-12"


def test_missing_days_filled_with_zero():
    s = cl.fetch_daily("rizz", "2023-02", "2023-02",
                       fetch=lambda url: _items([("2023-02-03", 50), ("2023-02-28", 9)]))
    assert len(s) == 28
    assert s.sum() == 59 and s["2023-02-01"] == 0


def _raise(code):
    def fetch(url):
        raise urllib.error.HTTPError(url, code, "err", {}, io.BytesIO())
    return fetch


def test_404_means_no_views_at_all():
    s = cl.fetch_daily("brand new", "2025-01", "2025-01", fetch=_raise(404))
    assert s.sum() == 0 and len(s) == 31


def test_other_errors_are_not_swallowed():
    with pytest.raises(urllib.error.HTTPError):
        cl.fetch_daily("rizz", "2025-01", "2025-03", fetch=_raise(403))


def _days(values, start="2022-08-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values)), dtype=float)


def test_despike_clips_a_two_day_burst():
    # Like "goated" on 25-26 Aug 2022: ~50 a day, then 4,000 and 1,600.
    values = np.full(60, 50.0)
    values[30], values[31] = 4000, 1600
    clean = cl.despike(_days(values))
    assert clean.iloc[30] == 500 and clean.iloc[31] == 500  # 10x the 50/day median
    assert (clean.drop(clean.index[[30, 31]]) == 50).all()


def test_despike_keeps_a_real_sustained_rise():
    # A word that jumps from 10 a day to 3,000 a day and stays there for weeks.
    values = np.concatenate([np.full(60, 10.0), np.full(60, 3000.0)])
    clean = cl.despike(_days(values))
    assert clean.sum() == values.sum()


def test_despike_leaves_tiny_pages_alone():
    values = np.zeros(40)
    values[20] = 40  # a lone 40 on a page with a median of 0
    assert cl.despike(_days(values)).iloc[20] == 40


def test_collect_filters_per_title_adds_variants_and_roundtrips(tmp_path):
    feb = [f"2023-02-{d:02d}" for d in range(1, 29)]
    data = {
        "rizz": [(d, 100) for d in feb[:-1]] + [(feb[-1], 5000)],  # burst on the last day
        "gyatt": [(d, 10) for d in feb],
        "gyat": [(d, 2) for d in feb],
    }

    def fetch(url):
        title = url.split("/user/")[1].split("/")[0]
        return _items(data[title])

    terms = pd.DataFrame({"term": ["rizz", "gyatt"], "variants": [[], ["gyat"]]})
    long = cl.collect(terms, "2023-01", "2023-02", fetch=fetch, pause=0)
    path = tmp_path / "pv.csv"
    cl.save(long, path)

    views, raw = cl.load(path), cl.load(path, column="views_raw")
    assert list(views.columns) == ["gyatt", "rizz"]
    feb_ = pd.Period("2023-02", "M")
    assert views.loc[feb_, "gyatt"] == 28 * 12  # variant summed in
    assert raw.loc[feb_, "rizz"] == 27 * 100 + 5000
    assert views.loc[feb_, "rizz"] == 27 * 100 + 1000  # burst capped at 10x the median
    assert views.loc[pd.Period("2023-01", "M"), "rizz"] == 0
