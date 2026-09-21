"""What kinds of life does a slang term have? Clustering curve shapes.

Each term's smoothed curve is lined up at its peak and scaled so the peak is
1, giving a shape from ``BEFORE`` months before the peak to ``AFTER`` months
after it. Months before the page existed count as 0: no dictionary entry, no
lookups. Terms watched for fewer than ``AFTER`` months after peaking are too
recent to classify and are left out, not guessed at.

Ward hierarchical clustering then groups the shapes into ``K`` types. K = 3
was fixed in advance, matching the three shapes the project set out to look
for; silhouette scores for other K are reported as a check, not used to pick.

Clusters are named from what they look like, by a fixed rule:

* **Stuck around**: the highest level at the end of the window (the term
  settled into the language).
* **Flash in the pan**: of the rest, the lowest level 3 months after the peak.
* **Slow burn**: whatever remains.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score

from .lifecycle import smooth

BEFORE = 12
AFTER = 24
K = 3
TAIL_MONTHS = 6
EARLY_MONTH = 3


def aligned_shapes(prepared: pd.DataFrame, terms, before: int = BEFORE, after: int = AFTER) -> pd.DataFrame:
    """Peak-aligned, peak-scaled curves: rows are terms, columns are months from peak.

    Terms with fewer than ``after`` months of data after their peak are dropped.
    """
    rows = {}
    for term in terms:
        sm = smooth(prepared[term])
        observed = sm.dropna()
        peak_month = observed.idxmax()
        i = sm.index.get_loc(peak_month)
        if i + after >= len(sm):
            continue
        idx = np.arange(i - before, i + after + 1)
        vals = np.array([sm.iloc[j] if j >= 0 else np.nan for j in idx], float)
        rows[term] = np.nan_to_num(vals, nan=0.0) / observed.max()
    return pd.DataFrame.from_dict(rows, orient="index", columns=range(-before, after + 1))


def cluster(shapes: pd.DataFrame, k: int = K) -> pd.Series:
    """Ward clustering of shapes into ``k`` groups (labels 1..k)."""
    z = linkage(shapes.to_numpy(), method="ward")
    return pd.Series(fcluster(z, k, criterion="maxclust"), index=shapes.index, name="cluster")


def silhouettes(shapes: pd.DataFrame, ks=range(2, 7)) -> dict[int, float]:
    """Silhouette score for each K (higher = better-separated clusters)."""
    return {k: float(silhouette_score(shapes, cluster(shapes, k))) for k in ks}


def name_clusters(shapes: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """Map each cluster label to its shape name using the fixed rule above."""
    tail_cols = shapes.columns[-TAIL_MONTHS:]
    tail = shapes[tail_cols].mean(axis=1).groupby(labels).mean()
    early = shapes[EARLY_MONTH].groupby(labels).mean()
    names = {tail.idxmax(): "stuck around"}
    rest = early.drop(tail.idxmax())
    names[rest.idxmin()] = "flash in the pan"
    for lab in rest.index:
        names.setdefault(lab, "slow burn")
    return labels.map(names).rename("shape")


def classify(prepared: pd.DataFrame, terms, k: int = K) -> tuple[pd.DataFrame, pd.Series]:
    """Shapes and their named type for every classifiable term."""
    shapes = aligned_shapes(prepared, terms)
    return shapes, name_clusters(shapes, cluster(shapes, k))


def centroids(shapes: pd.DataFrame, named: pd.Series) -> pd.DataFrame:
    """The average curve of each shape type."""
    return shapes.groupby(named).mean()
