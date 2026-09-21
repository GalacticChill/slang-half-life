import pandas as pd
import pytest

from slang_half_life import terms as tm


def test_shipped_list_is_valid():
    df = tm.load_terms()
    assert len(df) >= 60
    assert df["ambiguous"].dtype == bool
    assert set(df["era"]) == {"2016-2018", "2019-2021", "2022+"}
    # Every era needs enough terms to compare.
    assert df["era"].value_counts().min() >= 15


def test_era_boundaries():
    assert tm.era_of(2016) == "2016-2018"
    assert tm.era_of(2018) == "2016-2018"
    assert tm.era_of(2019) == "2019-2021"
    assert tm.era_of(2025) == "2022+"
    with pytest.raises(ValueError):
        tm.era_of(2015)


def _table(**overrides):
    row = {"term": ["rizz"], "variants": [""], "takeoff_year": [2022], "ambiguous": ["no"], "origin": ["x"]}
    row.update(overrides)
    return pd.DataFrame(row)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"term": ["rizz "]}, "whitespace"),
        ({"ambiguous": ["maybe"]}, "ambiguous"),
        ({"takeoff_year": [2014]}, "takeoff_year"),
    ],
)
def test_validate_rejects_bad_rows(overrides, message):
    with pytest.raises(ValueError, match=message):
        tm.validate(_table(**overrides))


def test_validate_rejects_duplicates():
    df = pd.concat([_table(), _table()], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        tm.validate(df)


def test_variants_parsed_and_checked():
    df = tm.load_terms().set_index("term", drop=False)
    assert df.loc["gyatt", "variants"] == ["gyat"]
    assert tm.titles_for(df.loc["yeet"]) == ["yeet", "yoit", "yait"]
    assert tm.titles_for(df.loc["rizz"]) == ["rizz"]
    assert tm.split_variants("") == []
    assert tm.split_variants(" a ; b;") == ["a", "b"]


def test_validate_rejects_variant_that_is_a_term():
    df = pd.concat([_table(), _table(term=["gyatt"], variants=["rizz"])], ignore_index=True)
    with pytest.raises(ValueError, match="also terms"):
        tm.validate(df)


def test_page_slug():
    assert tm.page_slug("rizz") == "rizz"
    assert tm.page_slug("OK boomer") == "OK_boomer"
    assert tm.page_slug("ate and left no crumbs") == "ate_and_left_no_crumbs"
    assert tm.page_slug("caught in 4K") == "caught_in_4K"
    assert tm.page_slug("OK, boomer") == "OK%2C_boomer"


def _page(title, content="==English==\n# a sense"):
    return {"title": title, "revisions": [{"slots": {"main": {"content": content}}}]}


def test_parse_status_classifies_every_case():
    response = {
        "query": {
            "normalized": [{"from": "OK_boomer", "to": "OK boomer"}],
            "redirects": [{"from": "let him cook", "to": "cook"}],
            "pages": [
                _page("rizz"),
                _page("OK boomer"),
                _page("cook"),
                {"title": "fanum tax", "missing": True},
                _page("6-7", "==English==\n{{no entry|en|because=unattested}}"),
            ],
        }
    }
    assert tm.parse_status(response) == {
        "rizz": "ok",
        "OK_boomer": "ok",
        "let him cook": "redirect",
        "fanum tax": "missing",
        "6-7": "no-entry",
    }


def test_check_entries_batches_by_fifty():
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return {"query": {"pages": []}}

    tm.check_entries([f"t{i}" for i in range(120)], fetch=fake_fetch)
    assert len(calls) == 3
