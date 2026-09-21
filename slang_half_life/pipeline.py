"""The shared path from cached data to per-term metrics."""

from __future__ import annotations

import pandas as pd

from . import collect, lifecycle, normalize, survival, terms


def load_prepared(views_path=collect.DEFAULT_PAGEVIEWS, created_path=normalize.DEFAULT_CREATED,
                  totals_path=normalize.DEFAULT_TOTALS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(prepared, raw): per-million lookups masked before creation, and raw counts."""
    raw = collect.load(views_path)
    prepared = normalize.prepare(raw, normalize.load_totals(totals_path), normalize.load_created(created_path))
    return prepared, raw


def metrics(prepared: pd.DataFrame, raw: pd.DataFrame, term_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Lifecycle metrics plus the measured peak era, joined to term metadata if given."""
    m = lifecycle.measure_all(prepared, raw)
    m["peak_era"] = m["peak_year"].map(survival.peak_era)
    if term_table is not None:
        m = m.join(term_table.set_index("term")[["era", "ambiguous"]])
    return m


def curated() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(metrics, prepared) for the hand-picked term list."""
    prepared, raw = load_prepared()
    return metrics(prepared, raw, terms.load_terms()), prepared
