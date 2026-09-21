"""Measure each term's life cycle: rise, peak, decay, and what's left.

Every metric is read off a lightly smoothed curve (a centered 3-month average)
so one odd month can't move a peak or fake a collapse.

* **Peak**: the highest smoothed month.
* **Rise time**: months from the last month below 10% of peak to the peak. If
  the curve never sat below 10% before peaking (it was already popular when
  the data or the page began), the rise wasn't observed and is left blank.
* **Half-life**: months from the peak until the curve drops to half the peak
  *and stays there* for 3 straight months. A term that hasn't done that by the
  last month of data is **still alive**: it gets no half-life, only a record of
  how long it has been watched since its peak. Survival analysis (next step)
  uses those "survived at least this long" records instead of throwing them out.
* **Stickiness**: the last 12 months' average as a share of the peak. Near 0
  means the term faded away; high means it settled into the language.

A term is **measurable** only if its smoothed peak reached at least
``MIN_PEAK_VIEWS`` raw lookups a month. Below that, random month-to-month noise
is large relative to the halfway line (a count of ~100 swings by about ±10%
from chance alone). The threshold was set before looking at any results.

A peak within the first ``EDGE_MONTHS`` observed months may not be the real
peak (the term could have peaked before the data or the page began), so its
half-life isn't trusted either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SMOOTH_WINDOW = 3
RISE_FLOOR = 0.10
HALF = 0.5
SUSTAIN_MONTHS = 3
STICKY_MONTHS = 12
MIN_PEAK_VIEWS = 200
EDGE_MONTHS = 3


@dataclass
class Lifecycle:
    peak_month: pd.Period
    peak: float
    rise_months: float  # NaN when the rise wasn't observed
    decayed: bool  # did it fall to half its peak and stay there?
    half_life_months: float  # NaN when still alive
    followup_months: int  # months watched after the peak (the censoring time if alive)
    stickiness: float
    peak_observed: bool  # False if the peak sits at the start of the data


def smooth(s: pd.Series, window: int = SMOOTH_WINDOW) -> pd.Series:
    """Centered rolling mean that leaves months before the entry existed missing."""
    if window <= 1:
        return s.astype(float)
    out = s.rolling(window, center=True, min_periods=1).mean()
    return out.where(s.notna())


def measure(s: pd.Series, window: int = SMOOTH_WINDOW) -> Lifecycle:
    """Life-cycle metrics for one term's monthly series (NaN = page didn't exist yet)."""
    sm = smooth(s, window).dropna()
    if sm.empty:
        raise ValueError(f"{s.name!r} has no observed months")
    values = sm.to_numpy()
    i_peak = int(np.argmax(values))
    peak = float(values[i_peak])

    below_floor = np.flatnonzero(values[:i_peak] < RISE_FLOOR * peak)
    rise = float(i_peak - below_floor[-1]) if below_floor.size else np.nan

    at_or_below_half = values <= HALF * peak * (1 + 1e-9)
    half_life = np.nan
    for j in range(i_peak + 1, len(values) - SUSTAIN_MONTHS + 1):
        if at_or_below_half[j : j + SUSTAIN_MONTHS].all():
            half_life = float(j - i_peak)
            break

    return Lifecycle(
        peak_month=sm.index[i_peak],
        peak=peak,
        rise_months=rise,
        decayed=not np.isnan(half_life),
        half_life_months=half_life,
        followup_months=len(values) - 1 - i_peak,
        stickiness=float(values[-STICKY_MONTHS:].mean() / peak) if peak > 0 else np.nan,
        peak_observed=i_peak >= EDGE_MONTHS,
    )


def measure_all(prepared: pd.DataFrame, raw: pd.DataFrame, window: int = SMOOTH_WINDOW) -> pd.DataFrame:
    """One row of metrics per term.

    ``prepared`` is the normalized, creation-masked table from
    ``normalize.prepare``; ``raw`` holds the raw monthly counts, used only for
    the minimum-size rule. Terms whose page was created after the last month of
    data have nothing to measure and are left out.
    """
    rows = []
    for term in prepared.columns:
        if prepared[term].notna().sum() == 0:
            continue
        lc = measure(prepared[term], window)
        raw_peak = float(smooth(raw[term].where(prepared[term].notna()), window).max())
        rows.append({"term": term, **lc.__dict__, "peak_raw": raw_peak,
                     "measurable": raw_peak >= MIN_PEAK_VIEWS})
    out = pd.DataFrame(rows).set_index("term")
    out["peak_year"] = out["peak_month"].map(lambda p: p.year)
    # Only these terms give a trustworthy half-life (or censoring time).
    out["usable"] = out["measurable"] & out["peak_observed"]
    return out
