"""Download monthly Wiktionary lookups for each term.

Source: the Wikimedia pageview API (no key needed). We count only
``agent=user`` traffic, which excludes known crawlers and traffic Wikimedia
classifies as automated. Data starts in July 2015.

The API leaves out months with zero views and returns 404 for a page with no
views at all, so every term is filled out to the full month range with zeros.
Each term's count includes its spelling variants (see ``terms.py``): views
are counted under the exact title a reader lands on, so ``gyat`` lookups never
reach ``gyatt`` on their own.

A zero before a page existed means "no page yet", not "nobody cared"; the
normalization step handles that using each entry's creation date.
"""

from __future__ import annotations

import time
import urllib.error
from pathlib import Path

import pandas as pd

from .http import get_json
from .terms import page_slug, titles_for

PAGEVIEWS_API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wiktionary.org/all-access/user/{slug}/monthly/{start}/{end}"
)

FIRST_MONTH = "2015-07"
DEFAULT_PAGEVIEWS = Path(__file__).resolve().parent.parent / "data" / "pageviews.csv"


def last_full_month(today: pd.Timestamp | None = None) -> str:
    """The most recent month that has fully ended, as 'YYYY-MM'."""
    today = pd.Timestamp.today() if today is None else pd.Timestamp(today)
    return (today.to_period("M") - 1).strftime("%Y-%m")


def pageviews_url(term: str, start: str, end: str) -> str:
    """API URL for one term's monthly views from ``start`` to ``end`` ('YYYY-MM')."""
    return PAGEVIEWS_API.format(
        slug=page_slug(term),
        start=start.replace("-", "") + "0100",
        end=end.replace("-", "") + "0100",
    )


def fetch_term(term: str, start: str, end: str, fetch=get_json) -> pd.Series:
    """One term's monthly views, indexed by month, with missing months filled as 0."""
    months = pd.period_range(start, end, freq="M")
    try:
        items = fetch(pageviews_url(term, start, end))["items"]
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        items = []
    views = {pd.Period(i["timestamp"][:6], freq="M"): i["views"] for i in items}
    return pd.Series([views.get(m, 0) for m in months], index=months, name=term, dtype="int64")


def collect(terms: pd.DataFrame, start: str = FIRST_MONTH, end: str | None = None,
            fetch=get_json, pause: float = 0.1, progress=None) -> pd.DataFrame:
    """Views for every term (variants included), in long form: one row per (term, month).

    ``terms`` is the table from ``terms.load_terms``.
    """
    end = end or last_full_month()
    frames = []
    for n, (_, row) in enumerate(terms.iterrows(), 1):
        total = None
        for title in titles_for(row):
            s = fetch_term(title, start, end, fetch=fetch)
            total = s if total is None else total + s
            if pause:
                time.sleep(pause)
        frames.append(pd.DataFrame({"term": row["term"], "month": total.index.strftime("%Y-%m"),
                                    "views": total.to_numpy()}))
        if progress:
            progress(n, len(terms), row["term"])
    return pd.concat(frames, ignore_index=True)


def save(df: pd.DataFrame, path: str | Path = DEFAULT_PAGEVIEWS) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load(path: str | Path = DEFAULT_PAGEVIEWS) -> pd.DataFrame:
    """Load the cached views as a wide table: rows are months, columns are terms."""
    df = pd.read_csv(path, dtype={"term": str})
    wide = df.pivot(index="month", columns="term", values="views")
    wide.index = pd.PeriodIndex(wide.index, freq="M")
    return wide.sort_index()
