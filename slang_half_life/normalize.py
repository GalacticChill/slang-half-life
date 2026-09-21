"""Put every term's lookups on a fair footing.

Two corrections before any curve is measured:

1. **Site traffic.** Wiktionary's own traffic rises and falls over the years,
   which would look like every term rising or fading together. Each term's
   views are divided by total Wiktionary views that month and reported as
   *lookups per million Wiktionary pageviews*.
2. **Entry creation.** A month with zero views before a page existed means
   "no page yet", not "nobody cared". Months before the earliest creation date
   among a term's titles (main entry and variants) are marked missing.
"""

from __future__ import annotations

import time
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd

from .collect import FIRST_MONTH, last_full_month
from .http import get_json
from .terms import titles_for

TOTALS_API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/"
    "en.wiktionary.org/all-access/user/monthly/{start}/{end}"
)
WIKTIONARY_API = "https://en.wiktionary.org/w/api.php"

DATA = Path(__file__).resolve().parent.parent / "data"
DEFAULT_TOTALS = DATA / "totals.csv"
DEFAULT_CREATED = DATA / "created.csv"


def fetch_totals(start: str = FIRST_MONTH, end: str | None = None, fetch=get_json) -> pd.Series:
    """Total human Wiktionary pageviews per month."""
    end = end or last_full_month()
    url = TOTALS_API.format(start=start.replace("-", "") + "0100",
                            end=end.replace("-", "") + "0100")
    items = fetch(url)["items"]
    s = pd.Series({pd.Period(i["timestamp"][:6], freq="M"): i["views"] for i in items},
                  name="total", dtype="int64")
    expected = pd.period_range(start, end, freq="M")
    if not s.index.equals(expected):
        raise ValueError("total-traffic series has gaps")
    return s


def first_revision_url(title: str) -> str:
    params = {"action": "query", "titles": title, "prop": "revisions", "rvdir": "newer",
              "rvlimit": "1", "rvprop": "timestamp", "format": "json", "formatversion": "2"}
    return f"{WIKTIONARY_API}?{urllib.parse.urlencode(params)}"


def fetch_created(title: str, fetch=get_json) -> pd.Timestamp:
    """When a title's current page was created (its first revision)."""
    page = fetch(first_revision_url(title))["query"]["pages"][0]
    if page.get("missing"):
        raise ValueError(f"no Wiktionary page for {title!r}")
    return pd.Timestamp(page["revisions"][0]["timestamp"]).tz_localize(None)


def collect_created(terms: pd.DataFrame, fetch=get_json, pause: float = 0.1) -> pd.DataFrame:
    """Earliest creation date among each term's titles."""
    rows = []
    for _, row in terms.iterrows():
        dates = []
        for title in titles_for(row):
            dates.append(fetch_created(title, fetch=fetch))
            if pause:
                time.sleep(pause)
        rows.append({"term": row["term"], "created": min(dates).strftime("%Y-%m-%d")})
    return pd.DataFrame(rows)


def save_totals(s: pd.Series, path: str | Path = DEFAULT_TOTALS) -> None:
    pd.DataFrame({"month": s.index.strftime("%Y-%m"), "total": s.to_numpy()}).to_csv(path, index=False)


def load_totals(path: str | Path = DEFAULT_TOTALS) -> pd.Series:
    df = pd.read_csv(path)
    return pd.Series(df["total"].to_numpy(), index=pd.PeriodIndex(df["month"], freq="M"), name="total")


def load_created(path: str | Path = DEFAULT_CREATED) -> pd.Series:
    df = pd.read_csv(path, dtype={"term": str})
    return pd.Series(pd.to_datetime(df["created"]).dt.to_period("M").to_numpy(),
                     index=df["term"], name="created")


def per_million(views: pd.DataFrame, totals: pd.Series) -> pd.DataFrame:
    """Views as lookups per million total Wiktionary pageviews in the same month."""
    totals = totals.reindex(views.index)
    if totals.isna().any():
        raise ValueError("total traffic missing for some months")
    return views.div(totals, axis=0) * 1e6


def mask_before_creation(table: pd.DataFrame, created: pd.Series) -> pd.DataFrame:
    """Set months before each term's creation month to NaN (the creation month is kept)."""
    out = table.astype(float).copy()
    for term in out.columns:
        out.loc[out.index < created[term], term] = np.nan
    return out


def prepare(views: pd.DataFrame, totals: pd.Series, created: pd.Series) -> pd.DataFrame:
    """The analysis-ready table: per-million lookups, missing before each entry existed."""
    return mask_before_creation(per_million(views, totals), created)
