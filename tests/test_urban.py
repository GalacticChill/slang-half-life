import numpy as np
import pandas as pd
import pytest

from slang_half_life import urban as ud


def _entry(word, defid, date):
    return {"word": word, "defid": defid, "written_on": f"{date}T12:00:00.000Z"}


def _pager(pages):
    calls = []

    def fetch(url):
        page = int(url.split("page=")[1])
        calls.append(page)
        return {"list": pages[page - 1] if page <= len(pages) else []}
    return fetch, calls


def test_pages_until_exact_matches_run_out():
    full = [_entry("Rizz", i, "2023-01-01") for i in range(10)]
    mixed = [_entry("rizz", 10, "2023-02-01"), _entry("Rizz rag", 11, "2023-02-01")] + \
            [_entry("Burger Rizz", 12 + i, "2023-02-01") for i in range(8)]
    fetch, calls = _pager([full, mixed, full])
    defs, truncated = ud.fetch_definitions("rizz", fetch, pause=0)
    assert calls == [1, 2]  # stopped at the first page that wasn't all exact
    assert [d["defid"] for d in defs] == list(range(11))
    assert not truncated
    assert defs[0]["written_on"] == "2023-01-01"


def test_max_pages_flags_truncation():
    full = [_entry("lit", i, "2020-01-01") for i in range(10)]
    fetch, calls = _pager([full] * 5)
    defs, truncated = ud.fetch_definitions("lit", fetch, pause=0, max_pages=3)
    assert len(defs) == 30 and truncated and calls == [1, 2, 3]


def test_collect_includes_variants_and_dedupes():
    pages = {"gyatt": [[_entry("gyatt", 1, "2023-05-01"), _entry("GYATT", 2, "2023-06-01")]],
             "gyat": [[_entry("gyat", 2, "2023-06-01"), _entry("gyat", 3, "2023-07-01")]]}

    def fetch(url):
        title = url.split("term=")[1].split("&")[0]
        page = int(url.split("page=")[1])
        return {"list": pages[title][0] if page == 1 else []}

    terms = pd.DataFrame({"term": ["gyatt"], "variants": [["gyat"]]})
    df = ud.collect(terms, fetch=fetch, pause=0)
    assert sorted(df["defid"]) == [1, 2, 3]


def test_checkpoint_resumes_without_refetching(tmp_path):
    fetched = []

    def fetch(url):
        title = url.split("term=")[1].split("&")[0]
        fetched.append(title)
        if title == "b" and fetched.count("b") == 1:
            raise TimeoutError("connection dropped")
        return {"list": [_entry(title, hash(title) % 1000, "2023-01-01")] if "page=1" in url else []}

    terms = pd.DataFrame({"term": ["a", "b"], "variants": [[], []]})
    ckpt = tmp_path / "ud.csv"
    with pytest.raises(TimeoutError):
        ud.collect(terms, fetch=fetch, pause=0, checkpoint=ckpt)
    df = ud.collect(terms, fetch=fetch, pause=0, checkpoint=ckpt)
    assert fetched == ["a", "b", "b"]  # "a" was not fetched again
    assert sorted(df["term"]) == ["a", "b"]


def test_onset_ignores_a_few_stray_old_definitions():
    dates = pd.Series(["2008-05-09", "2011-09-20"] + ["2022-06-15"] * 10 + ["2023-02-01"] * 30)
    assert ud.onset_month(dates) == pd.Period("2022-06", "M")
    assert ud.onset_month(pd.Series(["2020-01-05"])) == pd.Period("2020-01", "M")


def test_lags_sign_and_minimum():
    months = pd.period_range("2021-01", "2024-12", freq="M")
    defs = pd.DataFrame({
        "term": ["rizz"] * 30 + ["tiny"] * 5,
        "defid": range(35),
        "written_on": ["2022-09-10"] * 5 + ["2022-10-10"] * 20 + ["2022-11-10"] * 5
                      + ["2022-01-01"] * 5,
        "truncated": False,
    })
    metrics = pd.DataFrame({"peak_month": [pd.Period("2023-02", "M"), pd.Period("2022-02", "M")]},
                           index=["rizz", "tiny"])
    out = ud.lags(defs, metrics, months)
    assert list(out.index) == ["rizz"]  # tiny has fewer than 20 definitions
    assert out.loc["rizz", "peak_lead"] == 4  # UD peaked Oct 2022, lookups Feb 2023
    assert out.loc["rizz", "onset_lead"] == 5  # 10% of definitions written by Sep 2022


def test_sign_test():
    r = ud.sign_test(pd.Series([3, 5, 2, 4, 6, 1, 2, 3, 0, -1]))
    assert r["ud_first"] == 8 and r["wiktionary_first"] == 1 and r["same_month"] == 1
    assert r["p_value"] == pytest.approx(0.0390625)
    assert r["median_lead"] == 2.5
