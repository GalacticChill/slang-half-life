"""Download Wiktionary lookups for each term.

Source: the Wikimedia pageview API (no key needed). We count only
``agent=user`` traffic, which excludes known crawlers and traffic Wikimedia
classifies as automated. Data starts in July 2015.

Lookups are collected **daily** and then added up into months, so that short
bursts can be filtered first. On 25–26 August 2022, for example, the pages for
"OK, boomer", "bussin" and "goated" each got thousands of lookups in two days
against a normal few dozen, while the rest of Wiktionary was quiet. A burst
like that comes from a single link or a bot, not from a word spreading, but
summed into a month it looks like a peak. So each day is capped at
``SPIKE_FACTOR`` times the median of the surrounding ``SPIKE_WINDOW`` days.
A real rise lasts weeks, which lifts that median with it, so only bursts
shorter than about two weeks get clipped.

The API leaves out days with zero views and returns 404 for a page with no
views at all, so every title is filled out to the full date range with zeros.
Each term's count includes its spelling variants (see ``terms.py``): views
are counted under the exact title a reader lands on, so ``gyat`` lookups never
reach ``gyatt`` on their own. Bursts are filtered per title, before summing.

A zero before a page existed means "no page yet", not "nobody cared"; the
normalization step handles that using each entry's creation date.
"""

from __future__ import annotations

import time
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd

from .http import get_json
from .terms import page_slug, titles_for

PAGEVIEWS_API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wiktionary.org/all-access/user/{slug}/daily/{start}/{end}"
)

FIRST_MONTH = "2015-07"
DEFAULT_PAGEVIEWS = Path(__file__).resolve().parent.parent / "data" / "pageviews.csv"

SPIKE_WINDOW = 29  # days, centered: two weeks either side
SPIKE_FACTOR = 10
SPIKE_MIN_BASE = 5  # so pages with a median of 0-1 lookups a day aren't clipped at 10


def last_full_month(today: pd.Timestamp | None = None) -> str:
    """The most recent month that has fully ended, as 'YYYY-MM'."""
    today = pd.Timestamp.today() if today is None else pd.Timestamp(today)
    return (today.to_period("M") - 1).strftime("%Y-%m")


def _days(start: str, end: str) -> pd.DatetimeIndex:
    """Every day from the first of ``start`` to the last of ``end`` (months as 'YYYY-MM')."""
    return pd.date_range(pd.Period(start, "M").start_time.normalize(),
                         pd.Period(end, "M").end_time.normalize(), freq="D")


def pageviews_url(title: str, start: str, end: str) -> str:
    """API URL for one title's daily views across the months ``start`` to ``end``."""
    days = _days(start, end)
    return PAGEVIEWS_API.format(slug=page_slug(title), start=days[0].strftime("%Y%m%d"),
                                end=days[-1].strftime("%Y%m%d"))


def fetch_daily(title: str, start: str, end: str, fetch=get_json) -> pd.Series:
    """One title's daily views, with missing days filled as 0."""
    days = _days(start, end)
    try:
        items = fetch(pageviews_url(title, start, end))["items"]
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        items = []
    views = {pd.Timestamp(i["timestamp"][:8]): i["views"] for i in items}
    return pd.Series([views.get(d, 0) for d in days], index=days, name=title, dtype="int64")


def despike(daily: pd.Series, window: int = SPIKE_WINDOW, factor: float = SPIKE_FACTOR,
            min_base: float = SPIKE_MIN_BASE) -> pd.Series:
    """Cap each day at ``factor`` times the median of the surrounding ``window`` days."""
    base = daily.rolling(window, center=True, min_periods=1).median().clip(lower=min_base)
    return np.minimum(daily, factor * base)


def to_monthly(daily: pd.Series) -> pd.Series:
    return daily.groupby(daily.index.to_period("M")).sum()


def collect(terms: pd.DataFrame, start: str = FIRST_MONTH, end: str | None = None,
            fetch=get_json, pause: float = 0.1, progress=None) -> pd.DataFrame:
    """Monthly views for every term (variants included), in long form.

    One row per (term, month) with ``views`` (bursts filtered) and
    ``views_raw`` (unfiltered). ``terms`` is the table from ``terms.load_terms``.
    """
    end = end or last_full_month()
    frames = []
    for n, (_, row) in enumerate(terms.iterrows(), 1):
        clean = raw = 0
        for title in titles_for(row):
            daily = fetch_daily(title, start, end, fetch=fetch)
            clean = clean + to_monthly(despike(daily))
            raw = raw + to_monthly(daily)
            if pause:
                time.sleep(pause)
        frames.append(pd.DataFrame({"term": row["term"], "month": raw.index.strftime("%Y-%m"),
                                    "views": clean.round().astype("int64").to_numpy(),
                                    "views_raw": raw.to_numpy()}))
        if progress:
            progress(n, len(terms), row["term"])
    return pd.concat(frames, ignore_index=True)


def save(df: pd.DataFrame, path: str | Path = DEFAULT_PAGEVIEWS) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load(path: str | Path = DEFAULT_PAGEVIEWS, column: str = "views") -> pd.DataFrame:
    """Load cached monthly views as a wide table: rows are months, columns are terms.

    ``column="views_raw"`` gives the counts before burst filtering.
    """
    df = pd.read_csv(path, dtype={"term": str})
    wide = df.pivot(index="month", columns="term", values=column)
    wide.index = pd.PeriodIndex(wide.index, freq="M")
    return wide.sort_index()
