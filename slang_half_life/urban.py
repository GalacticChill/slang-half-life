"""Do insiders write about a word before the mainstream looks it up?

Urban Dictionary definitions are written by people who already use a word;
Wiktionary lookups come from people who've just run into it. Comparing when
each peaks gives the lag between the two.

Source: the unofficial Urban Dictionary API (``api.urbandictionary.com/v0``),
queried gently (one request a second) and saved as a snapshot, so the analysis
never depends on it staying up. Only each definition's ID and date are saved,
never its text or author.

How the data is read:

* **Exact matches only.** Search also returns near-matches ("rizz rag",
  "well lit"). Exact matches (ignoring case) come first, so paging stops at the
  first page that isn't all exact matches.
* **Onset, not first definition.** A few definitions predate the slang and mean
  something else (``rizz`` has a 2008 one meaning cocaine). So a term's onset
  is the month by which ``ONSET_SHARE`` of all its definitions had been
  written, which such strays can't move.
* **Peak**: the month with the most definitions written (3-month smoothed).
* A term needs at least ``MIN_DEFINITIONS`` definitions to be measured.

Spelling variants are included, as in the pageview data. For ambiguous words
the definitions mix meanings too, so the robustness step checks them separately.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .http import get_json
from .lifecycle import smooth
from .terms import titles_for

UD_API = "https://api.urbandictionary.com/v0/define"
PAGE_SIZE = 10
MAX_PAGES = 150
ONSET_SHARE = 0.10
MIN_DEFINITIONS = 20

DEFAULT_UD = Path(__file__).resolve().parent.parent / "data" / "urban_definitions.csv"


def page_url(title: str, page: int) -> str:
    return f"{UD_API}?{urllib.parse.urlencode({'term': title, 'page': page})}"


def _patient_get(url: str) -> dict:
    """The Urban Dictionary API drops connections now and then: retry longer."""
    return get_json(url, retries=6, backoff=5.0)


def fetch_definitions(title: str, fetch=_patient_get, pause: float = 1.0,
                      max_pages: int = MAX_PAGES) -> tuple[list[dict], bool]:
    """Exact-match definitions for a title, as ``{defid, written_on}`` dicts.

    Returns the definitions and whether ``max_pages`` was hit before the
    exact matches ran out (i.e. the list may be truncated).
    """
    want = title.strip().lower()
    found = []
    for page in range(1, max_pages + 1):
        entries = fetch(page_url(title, page))["list"]
        exact = [e for e in entries if e["word"].strip().lower() == want]
        found += [{"defid": e["defid"], "written_on": e["written_on"][:10]} for e in exact]
        if pause:
            time.sleep(pause)
        if len(exact) < PAGE_SIZE:
            return found, False
    return found, True


COLUMNS = ["term", "title", "defid", "written_on", "truncated"]


def collect(terms: pd.DataFrame, fetch=_patient_get, pause: float = 1.0,
            max_pages: int = MAX_PAGES, progress=None, checkpoint: str | Path | None = None) -> pd.DataFrame:
    """Definition dates for every term (variants included), one row per definition.

    With ``checkpoint``, rows are saved after every term (plus a ``.done.json``
    list of finished terms), and a rerun skips terms already finished, so a
    dropped connection costs one term, not the whole run.
    """
    rows, done = [], set()
    done_path = Path(f"{checkpoint}.done.json") if checkpoint else None
    if checkpoint and Path(checkpoint).exists() and done_path.exists():
        rows = load(checkpoint).to_dict("records")
        done = set(json.loads(done_path.read_text()))
    for n, (_, row) in enumerate(terms.iterrows(), 1):
        if row["term"] in done:
            continue
        for title in titles_for(row):
            defs, truncated = fetch_definitions(title, fetch, pause, max_pages)
            rows += [{"term": row["term"], "title": title, **d, "truncated": truncated} for d in defs]
        done.add(row["term"])
        if checkpoint:
            save(pd.DataFrame(rows, columns=COLUMNS), checkpoint)
            done_path.write_text(json.dumps(sorted(done)))
        if progress:
            progress(n, len(terms), row["term"])
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.drop_duplicates(["term", "defid"]).reset_index(drop=True)


def save(df: pd.DataFrame, path: str | Path = DEFAULT_UD) -> None:
    df.to_csv(path, index=False)


def load(path: str | Path = DEFAULT_UD) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"term": str, "title": str})


def monthly_counts(defs: pd.DataFrame, months: pd.PeriodIndex) -> pd.DataFrame:
    """Definitions written per month, per term, over ``months`` (earlier ones dropped)."""
    m = pd.to_datetime(defs["written_on"]).dt.to_period("M")
    counts = defs.assign(month=m).groupby(["month", "term"]).size().unstack(fill_value=0)
    return counts.reindex(months, fill_value=0)


def onset_month(dates: pd.Series, share: float = ONSET_SHARE) -> pd.Period:
    """The month by which ``share`` of all definitions had been written."""
    months = pd.to_datetime(dates).dt.to_period("M").sort_values().reset_index(drop=True)
    k = int(np.ceil(share * len(months))) - 1
    return months.iloc[max(k, 0)]


def lags(defs: pd.DataFrame, metrics: pd.DataFrame, months: pd.PeriodIndex) -> pd.DataFrame:
    """Per-term Urban Dictionary onset and peak, and their lead over the Wiktionary peak.

    A positive ``peak_lead`` means Urban Dictionary writing peaked that many
    months *before* Wiktionary lookups did.
    """
    counts = monthly_counts(defs, months)
    rows = []
    for term, grp in defs.groupby("term"):
        if len(grp) < MIN_DEFINITIONS or term not in metrics.index:
            continue
        ud_peak = smooth(counts[term].astype(float)).idxmax()
        onset = onset_month(grp["written_on"])
        wk_peak = metrics.loc[term, "peak_month"]
        rows.append({"term": term, "definitions": len(grp), "ud_onset": onset,
                     "ud_peak": ud_peak, "wiktionary_peak": wk_peak,
                     "onset_lead": (wk_peak - onset).n, "peak_lead": (wk_peak - ud_peak).n,
                     "truncated": bool(grp["truncated"].any())})
    return pd.DataFrame(rows).set_index("term")


def sign_test(leads: pd.Series) -> dict:
    """Does Urban Dictionary peak first more often than chance? (ties dropped)"""
    ahead, behind = int((leads > 0).sum()), int((leads < 0).sum())
    p = stats.binomtest(ahead, ahead + behind, 0.5).pvalue if ahead + behind else np.nan
    return {"ud_first": ahead, "wiktionary_first": behind, "same_month": int((leads == 0).sum()),
            "p_value": float(p), "median_lead": float(leads.median())}
