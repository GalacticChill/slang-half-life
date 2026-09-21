"""Do the findings depend on my choices? Two checks, fixed before running them.

1. **Without ambiguous words.** 27 of the 77 hand-picked terms share their page
   with a common non-slang meaning (``cap``, ``lit``). Their lookups mix both
   meanings, so the headline tests are rerun on the other 50.
2. **A random sample instead of my list.** ``SAMPLE_SIZE`` entries drawn at
   random (seed ``SEED``) from Wiktionary's *English internet slang* category,
   excluding the hand-picked terms, go through exactly the same pipeline and
   thresholds. Unlike the hand-picked list, spelling variants are not merged
   (that step was manual), and there's no Urban Dictionary data (collecting it
   at a polite pace would take hours).

Whatever these show is reported, including if it disagrees with the headline.
"""

from __future__ import annotations

import random
import re
import urllib.parse
from pathlib import Path

import pandas as pd

from . import collect, normalize, pipeline, shapes, survival
from .http import get_json

CATEGORY = "English internet slang"
SAMPLE_SIZE = 400
SEED = 42

DATA = Path(__file__).resolve().parent.parent / "data"
SAMPLE_TERMS = DATA / "sample_terms.csv"
SAMPLE_VIEWS = DATA / "sample_pageviews.csv"
SAMPLE_CREATED = DATA / "sample_created.csv"


def category_members(category: str = CATEGORY, fetch=get_json) -> list[str]:
    """Every main-namespace page title in a Wiktionary category."""
    params = {"action": "query", "list": "categorymembers", "cmtitle": f"Category:{category}",
              "cmlimit": "max", "cmnamespace": "0", "format": "json", "formatversion": "2"}
    titles = []
    while True:
        r = fetch(f"{normalize.WIKTIONARY_API}?{urllib.parse.urlencode(params)}")
        titles += [m["title"] for m in r["query"]["categorymembers"]]
        if "continue" not in r:
            return titles
        params.update(r["continue"])


def eligible(title: str) -> bool:
    """A real word or phrase: not a placeholder page, has at least two letters."""
    return not title.startswith("Unsupported titles/") and len(re.findall(r"[A-Za-z]", title)) >= 2


def draw_sample(members: list[str], exclude, n: int = SAMPLE_SIZE, seed: int = SEED) -> list[str]:
    """A reproducible random sample of eligible titles, excluding ``exclude``."""
    exclude = set(exclude)
    pool = sorted(t for t in set(members) if eligible(t) and t not in exclude)
    return sorted(random.Random(seed).sample(pool, min(n, len(pool))))


def collect_sample(titles: list[str], end: str) -> None:
    """Download and cache pageviews and creation dates for the sample."""
    table = pd.DataFrame({"term": titles, "variants": [[] for _ in titles]})
    pd.DataFrame({"term": titles}).to_csv(SAMPLE_TERMS, index=False)
    collect.save(collect.collect(table, end=end), SAMPLE_VIEWS)
    normalize.collect_created(table).to_csv(SAMPLE_CREATED, index=False)


def sample_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared, raw = pipeline.load_prepared(SAMPLE_VIEWS, SAMPLE_CREATED)
    return pipeline.metrics(prepared, raw), prepared


def headline(m: pd.DataFrame, prepared: pd.DataFrame, era_col: str = "peak_era") -> dict:
    """The era comparison and shape types for one set of terms."""
    eras = survival.compare_eras(m, era_col)
    shp, named = shapes.classify(prepared, m.index[m["usable"]])
    return {"eras": eras, "shapes": shp, "named": named,
            "silhouette": shapes.silhouettes(shp, ks=[3])[3]}
