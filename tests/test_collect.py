import io
import urllib.error

import pandas as pd
import pytest

from slang_half_life import collect as cl


def _items(pairs):
    return {"items": [{"timestamp": f"{m.replace('-', '')}0100", "views": v} for m, v in pairs]}


def test_url_uses_slug_and_month_bounds():
    url = cl.pageviews_url("OK boomer", "2015-07", "2026-08")
    assert url.endswith("/en.wiktionary.org/all-access/user/OK_boomer/monthly/2015070100/2026080100")


def test_last_full_month():
    assert cl.last_full_month(pd.Timestamp("2026-09-21")) == "2026-08"
    assert cl.last_full_month(pd.Timestamp("2026-01-05")) == "2025-12"


def test_missing_months_filled_with_zero():
    s = cl.fetch_term("rizz", "2022-10", "2023-01",
                      fetch=lambda url: _items([("2022-11", 50), ("2023-01", 900)]))
    assert s.tolist() == [0, 50, 0, 900]
    assert str(s.index[0]) == "2022-10" and str(s.index[-1]) == "2023-01"


def _raise(code):
    def fetch(url):
        raise urllib.error.HTTPError(url, code, "err", {}, io.BytesIO())
    return fetch


def test_404_means_no_views_at_all():
    s = cl.fetch_term("brand new", "2025-01", "2025-03", fetch=_raise(404))
    assert s.tolist() == [0, 0, 0]


def test_other_errors_are_not_swallowed():
    with pytest.raises(urllib.error.HTTPError):
        cl.fetch_term("rizz", "2025-01", "2025-03", fetch=_raise(403))


def test_collect_adds_variants_and_roundtrips(tmp_path):
    data = {"rizz": [("2023-01", 900)], "gyatt": [("2022-12", 10), ("2023-01", 20)],
            "gyat": [("2023-01", 5)]}

    def fetch(url):
        title = url.split("/user/")[1].split("/")[0]
        return _items(data[title])

    terms = pd.DataFrame({"term": ["rizz", "gyatt"], "variants": [[], ["gyat"]]})
    long = cl.collect(terms, "2022-12", "2023-01", fetch=fetch, pause=0)
    assert len(long) == 4
    path = tmp_path / "pv.csv"
    cl.save(long, path)
    wide = cl.load(path)
    assert list(wide.columns) == ["gyatt", "rizz"]
    assert wide["gyatt"].tolist() == [10, 25]
    assert wide.loc[pd.Period("2023-01", "M"), "rizz"] == 900
    assert wide.loc[pd.Period("2022-12", "M"), "rizz"] == 0
