"""Is slang dying faster than it used to? Survival analysis of half-lives.

Treat each term's fall to half its peak as the "event". Terms that haven't
fallen yet are **censored**: we know only that they survived at least as long
as we've watched them. Dropping them would bias the answer, because they're
mostly recent terms (not enough time to fade) and slow faders (the ones with
the longest half-lives), so the newest eras would look faster than they are.

* **Kaplan-Meier** estimates the share of terms still above half their peak
  ``t`` months after peaking, using censored terms for as long as they were
  watched.
* **Log-rank test** asks whether the eras' curves differ by more than chance.
  It compares, at every month an event happens, how many events each era had
  with how many it would have had if all eras faded at the same rate.

Eras are assigned by each term's **measured** peak year, so the grouping comes
from the data, not from the hand-assigned takeoff years in the term list.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .terms import era_of

HORIZONS = (3, 6, 12, 24)


def peak_era(year: int) -> str:
    """Era label for a measured peak year (a 2015 peak joins the first era)."""
    return era_of(max(int(year), 2016))


def kaplan_meier(durations, events) -> pd.DataFrame:
    """Kaplan-Meier survival table.

    ``durations`` are months from peak to the event (or to censoring);
    ``events`` is True where the term fell to half its peak. At a tie, events
    come before censorings (the standard convention). Returns one row per
    distinct event or censoring time with ``at_risk``, ``events``, ``survival``.
    """
    d = np.asarray(durations, float)
    e = np.asarray(events, bool)
    if len(d) == 0:
        raise ValueError("no durations")
    rows, s = [], 1.0
    for t in np.unique(d):
        at_risk = int((d >= t).sum())
        n_events = int(((d == t) & e).sum())
        s *= 1 - n_events / at_risk
        rows.append({"time": t, "at_risk": at_risk, "events": n_events, "survival": s})
    return pd.DataFrame(rows)


def survival_at(km: pd.DataFrame, t: float) -> float:
    """S(t): the share of terms still above half their peak ``t`` months after it."""
    before = km[km["time"] <= t]
    return 1.0 if before.empty else float(before["survival"].iloc[-1])


def median_survival(km: pd.DataFrame) -> float:
    """The median half-life: first time S(t) drops to 0.5 or below (NaN if never)."""
    hit = km[km["survival"] <= 0.5]
    return float(hit["time"].iloc[0]) if not hit.empty else np.nan


@dataclass
class LogRank:
    statistic: float
    df: int
    p_value: float
    observed: dict
    expected: dict


def logrank(durations, events, groups) -> LogRank:
    """k-group log-rank test (chi-square with k-1 degrees of freedom)."""
    d = np.asarray(durations, float)
    e = np.asarray(events, bool)
    g = np.asarray(groups)
    labels = list(pd.unique(g))
    k = len(labels)
    if k < 2:
        raise ValueError("need at least two groups")
    O = np.zeros(k)
    E = np.zeros(k)
    V = np.zeros((k, k))
    for t in np.unique(d[e]):
        risk = np.array([((d >= t) & (g == lab)).sum() for lab in labels], float)
        died = np.array([((d == t) & e & (g == lab)).sum() for lab in labels], float)
        n, m = risk.sum(), died.sum()
        O += died
        E += m * risk / n
        if n > 1:
            c = m * (n - m) / (n * n * (n - 1))
            V += c * (np.diag(risk * n) - np.outer(risk, risk))
    diff = (O - E)[:-1]
    stat = float(diff @ np.linalg.pinv(V[:-1, :-1]) @ diff)
    return LogRank(stat, k - 1, float(stats.chi2.sf(stat, k - 1)),
                   dict(zip(labels, O)), dict(zip(labels, E.round(2))))


def sidak(p: float, n_tests: int) -> float:
    """Šidák correction: the chance of at least one p this small across n tests."""
    return 1 - (1 - p) ** n_tests


def compare_eras(metrics: pd.DataFrame, era_col: str = "peak_era") -> dict:
    """Headline comparison across eras, from ``lifecycle.measure_all`` output.

    Uses only usable terms. Each term's duration is its half-life if it
    decayed, otherwise its follow-up time (censored).
    """
    u = metrics[metrics["usable"]].copy()
    u["duration"] = np.where(u["decayed"], u["half_life_months"], u["followup_months"])
    order = sorted(u[era_col].unique())
    summary, curves = [], {}
    for era in order:
        grp = u[u[era_col] == era]
        km = kaplan_meier(grp["duration"], grp["decayed"])
        curves[era] = km
        summary.append({"era": era, "terms": len(grp), "decayed": int(grp["decayed"].sum()),
                        "still_alive": int((~grp["decayed"]).sum()),
                        "median_half_life": median_survival(km),
                        **{f"alive_at_{h}m": survival_at(km, h) for h in HORIZONS}})
    overall = logrank(u["duration"], u["decayed"], u[era_col])
    pairs = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            sub = u[u[era_col].isin([a, b])]
            lr = logrank(sub["duration"], sub["decayed"], sub[era_col])
            pairs.append({"pair": f"{a} vs {b}", "p_value": lr.p_value,
                          "p_sidak": sidak(lr.p_value, len(order) * (len(order) - 1) // 2)})
    return {"summary": pd.DataFrame(summary).set_index("era"), "curves": curves,
            "logrank": overall, "pairs": pd.DataFrame(pairs)}
