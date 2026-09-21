import numpy as np
import pandas as pd

from slang_half_life import plots, shapes as sh


def _curves(n_terms=9, months=60):
    idx = pd.period_range("2019-01", periods=months, freq="M")
    t = np.arange(months) - 20
    cols = {}
    for i in range(n_terms):
        hl = [1.5, 8, 30][i % 3]
        cols[f"t{i}"] = np.where(t <= 0, np.exp(t / 3), 0.5 ** (np.maximum(t, 0) / hl)) * 1000 + 1
    return pd.DataFrame(cols, index=idx)


def test_hero_and_shapes_render(tmp_path):
    prepared = _curves()
    assert plots.plot_hero(prepared, prepared, "t0", tmp_path / "hero.png").stat().st_size > 10_000
    shp, named = sh.classify(prepared, prepared.columns)
    assert plots.plot_shapes(shp, named, tmp_path / "shapes.png").stat().st_size > 10_000
