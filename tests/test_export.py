import json

import numpy as np
import pandas as pd

from slang_half_life import export, report


def _table(cols, months=12):
    idx = pd.period_range("2025-09", periods=months, freq="M")
    return pd.DataFrame(cols, index=idx, dtype=float)


def test_rising_ranks_by_growth_and_ignores_tiny_terms():
    flat = [100.0] * 12
    doubling = [100.0] * 9 + [200.0] * 3          # months 7-9 at 100, last 3 at 200: x2
    tripling_tiny = [1.0] * 9 + [3.0] * 3          # x3, but only ~3 lookups a month
    fading = [100.0] * 9 + [50.0] * 3
    prepared = _table({"flat": flat, "doubling": doubling, "tiny": tripling_tiny, "fading": fading})
    raw = prepared * 1.0
    out = export.rising(prepared, raw)
    assert [r["term"] for r in out] == ["doubling"]
    assert out[0]["growth"] == 2.0 and out[0]["recent_views"] == 200


def test_rising_skips_terms_without_a_before_period():
    prepared = _table({"new": [np.nan] * 9 + [50.0, 80.0, 120.0]})
    assert export.rising(prepared, prepared * 10) == []


def test_build_from_shipped_data_is_valid_json(tmp_path):
    res = report.compute(with_sample=False)
    path = export.write(res, tmp_path / "latest.json")
    doc = json.loads(path.read_text())
    assert doc["last_month"] == doc["months"][-1]
    assert len(doc["terms"]) == 77
    rizz = next(t for t in doc["terms"] if t["term"] == "rizz")
    assert rizz["half_life"] == 3 and rizz["shape"] == "flash in the pan"
    assert len(rizz["views"]) == len(rizz["adjusted"]) == len(doc["months"]) - rizz["start"]
    ok_boomer = next(t for t in doc["terms"] if t["term"] == "OK boomer")
    assert not ok_boomer["measured"] and not ok_boomer["peak_observed"]
    assert set(doc["headline"]["shapes"]) == {"flash in the pan", "slow burn", "stuck around"}
    assert all(t["growth"] > 1 for t in doc["rising"])


def test_data_changed_ignores_the_generated_date():
    old = {"generated": "2026-09-21", "last_month": "2026-08", "terms": [{"term": "rizz", "views": [1, 2]}]}
    assert not export.data_changed(old, {**old, "generated": "2026-09-28"})
    assert export.data_changed(old, {**old, "last_month": "2026-09"})
    assert export.data_changed(old, {**old, "terms": [{"term": "rizz", "views": [1, 3]}]})
    assert export.data_changed(None, old)
