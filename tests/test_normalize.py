import numpy as np
import pandas as pd
import pytest

from slang_half_life import normalize as nm

M = pd.period_range("2020-01", "2020-04", freq="M")


def test_per_million_divides_by_same_month_total():
    views = pd.DataFrame({"rizz": [10, 20, 30, 40]}, index=M)
    totals = pd.Series([1e6, 2e6, 3e6, 4e6], index=M)
    # Raw views quadruple, but so does site traffic: the share is flat.
    assert nm.per_million(views, totals)["rizz"].tolist() == [10, 10, 10, 10]


def test_per_million_rejects_missing_totals():
    views = pd.DataFrame({"rizz": [1, 2, 3, 4]}, index=M)
    with pytest.raises(ValueError):
        nm.per_million(views, pd.Series([1.0, 1.0], index=M[:2]))


def test_mask_keeps_creation_month_and_later():
    table = pd.DataFrame({"a": [0, 0, 5, 6], "b": [1, 2, 3, 4]}, index=M)
    created = pd.Series({"a": pd.Period("2020-03", "M"), "b": pd.Period("2019-06", "M")})
    out = nm.mask_before_creation(table, created)
    assert np.isnan(out["a"].iloc[:2]).all()
    assert out["a"].iloc[2:].tolist() == [5, 6]
    assert out["b"].tolist() == [1, 2, 3, 4]


def test_fetch_totals_parses_and_checks_gaps():
    items = [{"timestamp": f"2020{m:02d}0100", "views": m * 100} for m in (1, 2, 3)]
    s = nm.fetch_totals("2020-01", "2020-03", fetch=lambda url: {"items": items})
    assert s.tolist() == [100, 200, 300]
    with pytest.raises(ValueError, match="gaps"):
        nm.fetch_totals("2020-01", "2020-04", fetch=lambda url: {"items": items})


def _rev(ts):
    return {"query": {"pages": [{"title": "x", "revisions": [{"timestamp": ts}]}]}}


def test_created_is_earliest_across_variants():
    dates = {"gyatt": "2023-05-10T12:00:00Z", "gyat": "2023-02-01T00:00:00Z",
             "rizz": "2022-08-03T00:00:00Z"}

    def fetch(url):
        title = url.split("titles=")[1].split("&")[0]
        return _rev(dates[title])

    terms = pd.DataFrame({"term": ["gyatt", "rizz"], "variants": [["gyat"], []]})
    out = nm.collect_created(terms, fetch=fetch, pause=0)
    assert out.set_index("term")["created"].to_dict() == {"gyatt": "2023-02-01", "rizz": "2022-08-03"}


def test_missing_page_is_an_error():
    with pytest.raises(ValueError):
        nm.fetch_created("nope", fetch=lambda url: {"query": {"pages": [{"title": "nope", "missing": True}]}})


def test_save_load_roundtrip(tmp_path):
    s = pd.Series([5, 6], index=pd.period_range("2020-01", "2020-02", freq="M"))
    nm.save_totals(s, tmp_path / "t.csv")
    assert nm.load_totals(tmp_path / "t.csv").tolist() == [5, 6]
