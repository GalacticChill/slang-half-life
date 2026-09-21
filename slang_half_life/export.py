"""Write the dashboard's data file, ``docs/latest.json``.

Everything comes from ``report.compute``, the same results the README prints,
so the site and the README can't disagree.

**Rising now** ranks terms by growth: average lookups (per million
Wiktionary pageviews) over the last ``RECENT`` months divided by the
``RECENT`` months before that. Only terms averaging at least
``RISING_MIN_VIEWS`` raw lookups a month recently are eligible, so a jump from
3 to 9 lookups doesn't count as a trend.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import pandas as pd

from . import terms as tm

DOCS = Path(__file__).resolve().parent.parent / "docs"
DEFAULT_OUT = DOCS / "latest.json"
RECENT = 3
RISING_MIN_VIEWS = 100
RISING_TOP = 8


def rising(prepared: pd.DataFrame, raw: pd.DataFrame, recent: int = RECENT,
           min_views: float = RISING_MIN_VIEWS, top: int = RISING_TOP) -> list[dict]:
    """Terms whose lookups grew over the last ``recent`` months, fastest first."""
    rows = []
    for term in prepared.columns:
        s = prepared[term]
        now, before = s.iloc[-recent:].mean(), s.iloc[-2 * recent:-recent].mean()
        if raw[term].iloc[-recent:].mean() < min_views or not before > 0 or math.isnan(now):
            continue
        growth = now / before
        if growth > 1:
            rows.append({"term": term, "growth": round(float(growth), 3),
                         "recent_views": int(round(raw[term].iloc[-recent:].mean()))})
    return sorted(rows, key=lambda r: -r["growth"])[:top]


def _num(x, digits=None):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), digits) if digits is not None else x


def _eras(eras: dict) -> dict:
    s = eras["summary"]
    return {
        "p_value": _num(eras["logrank"].p_value, 4),
        "eras": [{"era": era, "terms": int(r["terms"]), "median_half_life": _num(r["median_half_life"]),
                  "faded_3m": _num(1 - r["alive_at_3m"], 3)} for era, r in s.iterrows()],
        "curves": {era: [[float(t), round(float(v), 4)] for t, v in zip(km["time"], km["survival"])]
                   for era, km in eras["curves"].items()},
    }


def build(res: dict) -> dict:
    """The JSON document for the dashboard."""
    m, prepared, raw = res["metrics"], res["prepared"], res["raw"]
    table = tm.load_terms().set_index("term")
    lags = res["lags"]
    named = res["named"]
    months = [str(p) for p in raw.index]
    terms_out = []
    for term in sorted(m.index, key=str.lower):
        r = m.loc[term]
        first = prepared[term].first_valid_index()
        start = months.index(str(first)) if first is not None else len(months)
        terms_out.append({
            "term": term,
            "origin": table.loc[term, "origin"],
            "variants": table.loc[term, "variants"],
            "takeoff_year": int(table.loc[term, "takeoff_year"]),
            "ambiguous": bool(r["ambiguous"]),
            "measured": bool(r["usable"]),
            "big_enough": bool(r["measurable"]),
            "peak_observed": bool(r["peak_observed"]),
            "peak_month": str(r["peak_month"]),
            "half_life": _num(r["half_life_months"]) if r["decayed"] else None,
            "still_alive": bool(not r["decayed"]),
            "followup": int(r["followup_months"]),
            "stickiness": _num(r["stickiness"], 3),
            "shape": named.get(term),
            "ud_lead": int(lags.loc[term, "peak_lead"]) if term in lags.index else None,
            "start": start,
            "views": [int(v) for v in raw[term].iloc[start:]],
            "adjusted": [round(float(v), 2) for v in prepared[term].iloc[start:]],
        })
    return {
        "generated": date.today().isoformat(),
        "first_month": months[0],
        "last_month": months[-1],
        "months": months,
        "counts": res["counts"],
        "headline": {
            "eras": _eras(res["eras"]),
            "lag": {k: _num(v, 5) for k, v in res["lag_non_ambiguous"].items()},
            "lag_all": {k: _num(v, 5) for k, v in res["lag_all"].items()},
            "shapes": {name: {"terms": int((named == name).sum()),
                              "curve": [round(float(v), 4) for v in res["shape_centroids"].loc[name]]}
                       for name in res["shape_centroids"].index},
            "shape_months": [int(c) for c in res["shape_centroids"].columns],
        },
        "rising": rising(prepared, raw),
        "terms": terms_out,
    }


def data_changed(old: dict | None, new: dict) -> bool:
    """Whether the dashboard's data differs, ignoring the date it was generated.

    The weekly refresh commits only when this is true, so weeks with no new
    data don't produce empty commits.
    """
    if old is None:
        return True
    strip = lambda d: {k: v for k, v in d.items() if k != "generated"}  # noqa: E731
    return strip(old) != strip(new)


def write(res: dict, out: str | Path = DEFAULT_OUT) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(res), separators=(",", ":"), ensure_ascii=False))
    return out
