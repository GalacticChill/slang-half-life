"""The four charts in the README.

Styling follows one small system: a warm off-white surface, recessive grid and
axes, 2px lines, and a fixed three-color categorical order (blue, orange,
aqua) validated for color-vision deficiency. Colors are assigned to a thing
and stay with it (the same era is the same color in every panel), and every
colored line also carries a direct text label, so color is never the only cue.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .lifecycle import measure, smooth  # noqa: E402
from .survival import compare_eras  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua: validated order
ERA_COLORS = dict(zip(["2016-2018", "2019-2021", "2022+"], SERIES))
SHAPE_COLORS = dict(zip(["flash in the pan", "slow burn", "stuck around"], SERIES))


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "legend.frameon": False,
    })


def _title(fig, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.975, title, fontsize=14, fontweight="bold", va="top", color=INK)
    fig.text(0.02, 0.925, subtitle, fontsize=10, va="top", color=INK_2)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_hero(raw: pd.DataFrame, prepared: pd.DataFrame, term: str, out: Path) -> Path:
    """One term's whole life, annotated with the same peak and half-life the analysis measures.

    The thin line is the raw monthly count; the thick line is the 3-month
    average the metrics are read from, so the annotations match the tables.
    """
    _style()
    life = measure(prepared[term])
    s = raw[term]
    s = s[s.index >= s[s > 0].index[0] - 3]
    sm = smooth(s.astype(float))
    x = s.index.to_timestamp()
    peak_m = life.peak_month
    peak = sm[peak_m]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.8, bottom=0.12)
    ax.plot(x, s.to_numpy(), color=SERIES[0], linewidth=1, alpha=0.45)
    ax.plot(x, sm.to_numpy(), color=SERIES[0], linewidth=2.5)
    ax.axhline(peak / 2, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(x[-1], peak / 2, "half of peak", color=INK_2, fontsize=9, ha="right", va="bottom")

    def dot(month, text, offset):
        xm, ym = month.to_timestamp(), sm[month]
        ax.plot(xm, ym, "o", color=SERIES[0], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(text, (xm, ym), xytext=offset, textcoords="offset points", fontsize=9, color=INK)

    dot(peak_m, f"Peak: {peak:,.0f} lookups a month\n({peak_m.strftime('%B %Y')}, 3-month average)", (14, -4))
    if life.decayed:
        hl = int(life.half_life_months)
        dot(peak_m + hl, f"Half-life: {hl} month{'s' if hl != 1 else ''}\n"
            f"below half from {(peak_m + hl).strftime('%B %Y')} on", (14, 6))
    ax.set_ylim(0, s.max() * 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0"))
    ax.set_ylabel("Wiktionary lookups per month")
    ax.grid(axis="x", visible=False)
    top_m = s.idxmax()
    _title(fig, f"The life of \u201c{term}\u201d",
           f"Thin line: monthly lookups of the Wiktionary entry (highest: {s.max():,.0f} in "
           f"{top_m.strftime('%B %Y')}). Thick line: the 3-month average the analysis measures.")
    return _save(fig, out)


def _spread(ys: list[float], gap: float) -> list[float]:
    """Nudge label positions apart so none sit closer than ``gap``, keeping their order."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return list(out)


def _km_panel(ax, metrics: pd.DataFrame, heading: str, max_months: int = 60) -> None:
    res = compare_eras(metrics)
    ends, labels = [], []
    for era, km in res["curves"].items():
        color = ERA_COLORS[era]
        row = res["summary"].loc[era]
        t = np.r_[0, km["time"].to_numpy()]
        sv = np.r_[1.0, km["survival"].to_numpy()]
        keep = t <= max_months
        t, sv = np.r_[t[keep], max_months], np.r_[sv[keep], sv[keep][-1]]
        ax.step(t, sv, where="post", color=color)
        cens = km[(km["events"] < km["at_risk"] - km["at_risk"].shift(-1, fill_value=0)) & (km["time"] <= max_months)]
        ax.plot(cens["time"], cens["survival"], "|", color=color, markersize=7, markeredgewidth=1.5)
        med = row["median_half_life"]
        n = int(row["terms"])
        labels.append((f"{era}\nn={n}, median {med:.0f} mo" if not np.isnan(med) else f"{era}\nn={n}", color))
        ends.append(sv[-1])
    for (text, color), y in zip(labels, _spread(ends, 0.13)):
        ax.text(max_months + 2, y, text, color=INK, fontsize=8.5, va="center", linespacing=1.3)
        ax.plot([max_months + 0.6], [y], "s", color=color, markersize=5, clip_on=False)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xlim(0, max_months)
    ax.set_ylim(0, 1.03)
    ax.set_xticks(range(0, max_months + 1, 12))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Months after peak")
    p = res["logrank"].p_value
    ax.set_title(f"{heading}\nlog-rank p = {p:.2f}", fontsize=10, loc="left", color=INK)


def plot_survival(curated: pd.DataFrame, sample: pd.DataFrame, out: Path) -> Path:
    """Kaplan-Meier half-life curves by peak era: hand-picked list and random sample."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.87, top=0.74, bottom=0.12, wspace=0.42)
    _km_panel(axes[0], curated, f"Hand-picked slang ({int(curated['usable'].sum())} measurable terms)")
    _km_panel(axes[1], sample, f"Random sample of internet slang ({int(sample['usable'].sum())} measurable)")
    axes[0].set_ylabel("Still above half their peak")
    _title(fig, "Is slang dying faster?",
           "Share of terms still above half their peak lookups, by the era they peaked in. "
           "Ticks mark terms still alive when the data ends.")
    return _save(fig, out)


def plot_shapes(shapes_: pd.DataFrame, named: pd.Series, out: Path, examples: int = 4) -> Path:
    """Small multiples: every term's peak-aligned curve, and each shape type's average."""
    _style()
    order = ["flash in the pan", "slow burn", "stuck around"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.72, bottom=0.14, wspace=0.12)
    x = np.array(shapes_.columns, float)
    for ax, name in zip(axes, order):
        members = named[named == name].index
        for term in members:
            ax.plot(x, shapes_.loc[term], color=MUTED, linewidth=0.8, alpha=0.35)
        centroid = shapes_.loc[members].mean()
        ax.plot(x, centroid, color=SHAPE_COLORS[name], linewidth=2.5)
        ax.axvline(0, color=AXIS, linewidth=0.8)
        ax.set_xticks([-12, 0, 12, 24])
        ax.set_xlabel("Months from peak")
        # Name the most typical members: the curves closest to the type's average.
        dist = ((shapes_.loc[members] - centroid) ** 2).sum(axis=1)
        ex = ", ".join(dist.nsmallest(examples).index)
        ax.set_title(f"{name.capitalize()}  ({len(members)})\n{ex}", fontsize=10, loc="left", color=INK)
    axes[0].set_ylabel("Share of peak")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylim(0, 1.05)
    _title(fig, "Three ways a slang term lives",
           "Each gray line is one term, lined up at its peak; the colored line is the type's average")
    return _save(fig, out)


def plot_lag(lags: pd.DataFrame, out: Path) -> Path:
    """Dumbbells: Urban Dictionary's peak vs Wiktionary's peak for each term."""
    _style()
    d = lags.sort_values("wiktionary_peak")
    n = len(d)
    h = 0.26 * n + 2.0
    fig, ax = plt.subplots(figsize=(9, h))
    fig.subplots_adjust(left=0.2, right=0.97, top=1 - 1.45 / h, bottom=0.6 / h)
    y = np.arange(n)
    ud = d["ud_peak"].map(lambda p: p.to_timestamp())
    wk = d["wiktionary_peak"].map(lambda p: p.to_timestamp())
    ax.hlines(y, np.minimum(ud, wk), np.maximum(ud, wk), color=AXIS, linewidth=1.5)
    ax.plot(ud, y, "o", color=SERIES[1], markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label="Urban Dictionary: most definitions written")
    ax.plot(wk, y, "o", color=SERIES[0], markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label="Wiktionary: most lookups")
    ax.set_yticks(y, d.index, fontsize=8.5)
    ax.set_ylim(-0.8, n - 0.2)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", fontsize=9, bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.3)
    ahead = int((d["peak_lead"] > 0).sum())
    behind = int((d["peak_lead"] < 0).sum())
    same = int((d["peak_lead"] == 0).sum())
    fig.text(0.02, 1 - 0.2 / h, "Insiders first, then the mainstream", fontsize=14,
             fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.55 / h,
             f"Urban Dictionary peaked first for {ahead} of {n} terms, Wiktionary for {behind}, "
             f"same month for {same} (one dot).\nMedian lead {d['peak_lead'].median():.0f} months. "
             "Non-ambiguous terms only.",
             fontsize=10, color=INK_2, va="top")
    return _save(fig, out)
