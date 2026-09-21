"""Run every analysis from the cached data and gather the results in one place.

The command-line tool prints these results, and the dashboard (later) reads
the same numbers, so there is exactly one source for every figure in the README.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import pipeline, plots, robustness, shapes, survival, terms, urban


def compute(with_sample: bool = True) -> dict:
    """Every headline number, from the data in ``data/``."""
    m, prepared = pipeline.curated()
    raw = pipeline.collect.load()
    usable = m[m["usable"]]
    non_amb = m[~m["ambiguous"]]

    shp, named = shapes.classify(prepared, usable.index)
    shp_na, named_na = shapes.classify(prepared, non_amb.index[non_amb["usable"]])

    defs = urban.load()
    lags = urban.lags(defs, usable, prepared.index).join(m[["ambiguous"]])
    counts = urban.monthly_counts(defs, prepared.index)
    volume = counts.sum(axis=1).rolling(12, center=True, min_periods=1).mean()
    lead_adj = pd.Series({t: (m.loc[t, "peak_month"] - (counts[t] / volume).pipe(pipeline.lifecycle.smooth).idxmax()).n
                          for t in lags.index})

    out = {
        "metrics": m, "prepared": prepared, "raw": raw,
        "counts": {"terms": len(m), "measurable": int(m["measurable"].sum()),
                   "usable": int(m["usable"].sum()), "decayed": int(usable["decayed"].sum()),
                   "ambiguous": int(m["ambiguous"].sum())},
        "eras": survival.compare_eras(m, "peak_era"),
        "eras_hand": survival.compare_eras(m, "era"),
        "eras_non_ambiguous": survival.compare_eras(non_amb, "peak_era"),
        "eras_non_ambiguous_hand": survival.compare_eras(non_amb, "era"),
        "shapes": shp, "named": named,
        "shape_centroids": shapes.centroids(shp, named),
        "silhouette": shapes.silhouettes(shp, ks=range(2, 6)),
        "named_non_ambiguous": named_na,
        "lags": lags,
        "lag_all": urban.sign_test(lags["peak_lead"]),
        "lag_non_ambiguous": urban.sign_test(lags.loc[~lags["ambiguous"], "peak_lead"]),
        "lag_activity_adjusted": urban.sign_test(lead_adj),
        "definitions": len(defs),
    }
    if with_sample and robustness.SAMPLE_VIEWS.exists():
        sm, _ = robustness.sample_metrics()
        out.update({"sample_metrics": sm, "eras_sample": survival.compare_eras(sm, "peak_era")})
    return out


def draw(res: dict, assets: Path) -> list[Path]:
    """Regenerate the four README charts."""
    m = res["metrics"]
    lags = res["lags"]
    paths = [
        plots.plot_hero(res["raw"], res["prepared"], "rizz", assets / "rizz_lifecycle.png"),
        plots.plot_shapes(res["shapes"], res["named"], assets / "shape_types.png"),
        plots.plot_lag(lags[~lags["ambiguous"]], assets / "urban_dictionary_lead.png"),
    ]
    if "sample_metrics" in res:
        paths.append(plots.plot_survival(m, res["sample_metrics"], assets / "half_life_by_era.png"))
    return paths


def _eras_text(label: str, eras: dict) -> list[str]:
    s = eras["summary"]
    lines = [f"  {label}  (log-rank p = {eras['logrank'].p_value:.3f})"]
    for era, row in s.iterrows():
        med = "not reached" if pd.isna(row["median_half_life"]) else f"{row['median_half_life']:.0f} months"
        lines.append(f"    {era:<10} {int(row['terms']):>3} terms  median half-life {med:<12}"
                     f"  faded within 3 months {1 - row['alive_at_3m']:.0%}")
    return lines


def _p(p: float) -> str:
    return "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"


def _lag_text(label: str, r: dict) -> str:
    return (f"  {label:<26} Urban Dictionary first {r['ud_first']:>2}, Wiktionary first "
            f"{r['wiktionary_first']:>2}, same month {r['same_month']}  "
            f"median lead {r['median_lead']:.0f} mo  ({_p(r['p_value'])})")


def text(res: dict) -> str:
    """A plain-text report of every finding."""
    c = res["counts"]
    m = res["metrics"]
    rizz = m.loc["rizz"]
    lines = [
        f"Slang Half-Life: {c['terms']} hand-picked terms, "
        f"{res['prepared'].index[0]} to {res['prepared'].index[-1]}",
        f"  measurable {c['measurable']}, usable (peak observed) {c['usable']}, "
        f"of which {c['decayed']} have fallen to half their peak",
        f"  rizz: peak {rizz['peak_month']}, half-life {rizz['half_life_months']:.0f} months, "
        f"now at {rizz['stickiness']:.0%} of its peak",
        "",
        "1. Is slang dying faster?",
        *_eras_text("by measured peak year", res["eras"]),
        *_eras_text("by hand-assigned takeoff year", res["eras_hand"]),
        "",
        "2. Shape types (Ward clustering, K = 3 fixed in advance)",
    ]
    cen = res["shape_centroids"]
    for name, g in res["named"].groupby(res["named"]):
        amb = int(m.loc[g.index, "ambiguous"].sum())
        lines.append(f"  {name:<17} {len(g):>2} terms ({amb} ambiguous)  "
                     f"3 months after peak {cen.loc[name, 3]:.0%}, 24 months after {cen.loc[name, 24]:.0%}")
    sil = res["silhouette"]
    lines.append("  silhouette by K: " + ", ".join(f"{k}: {v:.2f}" for k, v in sil.items()))
    lines += [
        "",
        f"3. Insiders first? Urban Dictionary vs Wiktionary peaks ({res['definitions']:,} definitions)",
        _lag_text("all terms", res["lag_all"]),
        _lag_text("non-ambiguous terms", res["lag_non_ambiguous"]),
        _lag_text("adjusted for UD activity", res["lag_activity_adjusted"]),
        "",
        "4. Robustness",
        *_eras_text("no ambiguous words, by peak year", res["eras_non_ambiguous"]),
        *_eras_text("no ambiguous words, by takeoff year", res["eras_non_ambiguous_hand"]),
    ]
    stuck = sorted(res["named_non_ambiguous"][res["named_non_ambiguous"] == "stuck around"].index)
    lines.append(f"  'stuck around' without ambiguous words: {', '.join(stuck)}")
    if "eras_sample" in res:
        sm = res["sample_metrics"]
        lines.append(f"  random sample: {len(sm)} entries with data, {int(sm['usable'].sum())} usable")
        lines += _eras_text("random sample, by peak year", res["eras_sample"])
    return "\n".join(lines)


def check_terms() -> dict[str, str]:
    """Live check that every term and variant still has a Wiktionary entry."""
    t = terms.load_terms()
    titles = list(t["term"]) + [v for vs in t["variants"] for v in vs]
    return {k: v for k, v in terms.check_entries(titles).items() if v != "ok"}
