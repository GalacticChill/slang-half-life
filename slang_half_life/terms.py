"""The slang terms under study.

The list in ``data/terms.csv`` was hand-picked, which makes it the one place
where judgment shapes the data. The rules were fixed before looking at any
lookup counts, so no term was kept or dropped because of its curve:

* **Took off in 2016 or later.** Wikimedia's pageview data starts in July
  2015, so an earlier term's rise would be invisible.
* **Has its own Wiktionary entry.** Redirects and "no entry" pages are left
  out. For a spelling variant (``gyat``), the main entry (``gyatt``) is used;
  otherwise the form people actually use (``stonks``, ``yapping``).
* **Spelling variants are counted too.** Wiktionary gives variants their own
  "alternative form" pages instead of redirects, so lookups get split across
  spellings (``gyat`` / ``gyatt``). ``variants`` lists the extra titles
  (separated by ``;``) whose views are added to the term. They were found by
  searching each entry's alternative forms, then hand-checked so only
  spellings of the *slang* sense are kept (``karen`` yes, ``Karin`` no).
* **Flagged as ambiguous** when the page also covers a common non-slang
  meaning (``cap``, ``lit``, ``sigma``). Those lookups mix both meanings, so
  the robustness check reruns the analysis without them.

``takeoff_year`` is an approximate, hand-assigned label. The analysis measures
each term's timing from the data, and this label is only a sanity check.
"""

from __future__ import annotations

import urllib.parse
from pathlib import Path

import pandas as pd

from .http import get_json

DEFAULT_TERMS = Path(__file__).resolve().parent.parent / "data" / "terms.csv"

ERAS = [("2016-2018", 2016, 2018), ("2019-2021", 2019, 2021), ("2022+", 2022, 9999)]

WIKTIONARY_API = "https://en.wiktionary.org/w/api.php"


def era_of(year: int) -> str:
    """Map a year to its era label."""
    for label, lo, hi in ERAS:
        if lo <= year <= hi:
            return label
    raise ValueError(f"year {year} is before the first era")


def load_terms(path: str | Path = DEFAULT_TERMS) -> pd.DataFrame:
    """Load and validate the term list, adding a boolean ``ambiguous`` and an ``era``."""
    df = pd.read_csv(path, dtype={"term": str, "variants": str}, keep_default_na=False)
    df["takeoff_year"] = pd.to_numeric(df["takeoff_year"])
    validate(df)
    df["ambiguous"] = df["ambiguous"].eq("yes")
    df["variants"] = df["variants"].map(split_variants)
    df["era"] = df["takeoff_year"].map(era_of)
    return df


def split_variants(cell) -> list[str]:
    """Parse a ``;``-separated variants cell into a list of titles."""
    if not isinstance(cell, str):
        return []
    return [v.strip() for v in cell.split(";") if v.strip()]


def titles_for(row) -> list[str]:
    """Every Wiktionary title whose lookups count toward a term: main entry first."""
    return [row["term"], *row["variants"]]


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError if the term table breaks any of the list's rules."""
    required = {"term", "variants", "takeoff_year", "ambiguous", "origin"}
    if missing := required - set(df.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    if df["term"].isna().any() or (df["term"].str.strip() != df["term"]).any():
        raise ValueError("terms must be non-empty with no surrounding whitespace")
    if dupes := sorted(df.loc[df["term"].duplicated(), "term"]):
        raise ValueError(f"duplicate terms: {dupes}")
    variants = [v for cell in df["variants"] for v in split_variants(cell)]
    if clash := sorted(set(variants) & set(df["term"])):
        raise ValueError(f"variants that are also terms: {clash}")
    if len(variants) != len(set(variants)):
        raise ValueError("a variant is listed more than once")
    if not df["ambiguous"].isin(["yes", "no"]).all():
        raise ValueError("ambiguous must be 'yes' or 'no'")
    years = df["takeoff_year"]
    if not pd.api.types.is_integer_dtype(years) or (years < 2016).any() or (years > 2026).any():
        raise ValueError("takeoff_year must be an integer from 2016 to 2026")


def page_slug(term: str) -> str:
    """The URL path segment for a Wiktionary title (spaces become underscores)."""
    return urllib.parse.quote(term.replace(" ", "_"), safe="")


def parse_status(response: dict) -> dict[str, str]:
    """Classify each requested title as 'ok', 'missing', 'redirect', or 'no-entry'.

    ``response`` is a MediaWiki ``action=query`` result (formatversion=2) that
    requested ``prop=revisions`` content with ``redirects`` resolution.
    Titles are keyed by their original spelling, undoing MediaWiki's
    normalization (e.g. first-letter capitalization is not applied on
    Wiktionary, but underscores become spaces).
    """
    q = response["query"]
    original = {n["to"]: n["from"] for n in q.get("normalized", [])}
    status = {}
    for r in q.get("redirects", []):
        status[original.get(r["from"], r["from"])] = "redirect"
    redirect_targets = {r["to"] for r in q.get("redirects", [])}
    for page in q["pages"]:
        title = page["title"]
        if title in redirect_targets:
            continue
        key = original.get(title, title)
        if page.get("missing"):
            status[key] = "missing"
            continue
        content = page["revisions"][0]["slots"]["main"]["content"]
        status[key] = "no-entry" if "{{no entry" in content else "ok"
    return status


def check_entries(terms: list[str], fetch=get_json) -> dict[str, str]:
    """Look up each term on Wiktionary (live) and report its status."""
    status = {}
    for i in range(0, len(terms), 50):
        params = {
            "action": "query",
            "titles": "|".join(terms[i : i + 50]),
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "redirects": "1",
            "format": "json",
            "formatversion": "2",
        }
        status.update(parse_status(fetch(f"{WIKTIONARY_API}?{urllib.parse.urlencode(params)}")))
    return status
