"""Command-line entry point for slang-half-life."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from . import __version__

DEFAULT_ASSETS = Path(__file__).resolve().parent.parent / "assets"


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="slang-half-life",
        description="How long does a slang term live? Measure the rise, peak, and "
        "decay of slang terms from their monthly Wiktionary lookups.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--collect", action="store_true",
                   help="Re-download Wiktionary lookups, site traffic, and entry dates "
                        "(hand-picked terms and the random sample) first. A few minutes.")
    p.add_argument("--collect-urban", action="store_true",
                   help="Re-download Urban Dictionary definition dates first. Slow: "
                        "about 30 minutes at one request a second.")
    p.add_argument("--end", help="Last month to collect, YYYY-MM (default: last full month).")
    p.add_argument("--check-terms", action="store_true",
                   help="Check that every term still has a Wiktionary entry, then exit.")
    p.add_argument("--no-sample", action="store_true", help="Skip the random-sample check.")
    p.add_argument("--no-plots", action="store_true", help="Don't redraw the charts.")
    p.add_argument("--assets-dir", default=str(DEFAULT_ASSETS), help="Where to write charts.")
    return p.parse_args(argv)


def _collect(end: str | None, urban_too: bool) -> None:
    from . import collect, normalize, robustness, terms, urban

    t = terms.load_terms()
    end = end or collect.last_full_month()
    print(f"Collecting Wiktionary lookups for {len(t)} terms through {end}...")
    collect.save(collect.collect(t, end=end))
    normalize.save_totals(normalize.fetch_totals(end=end))
    normalize.collect_created(t).to_csv(normalize.DEFAULT_CREATED, index=False)
    exclude = set(t["term"]) | {v for vs in t["variants"] for v in vs}
    sample = robustness.draw_sample(robustness.category_members(), exclude)
    print(f"Collecting the random sample ({len(sample)} entries)...")
    robustness.collect_sample(sample, end=end)
    if urban_too:
        print("Collecting Urban Dictionary definitions (slow)...")
        urban.save(urban.collect(t, progress=lambda n, total, term: print(f"  {n}/{total} {term}")))


def main(argv=None) -> None:
    args = _parse_args(argv)
    warnings.filterwarnings("ignore", category=UserWarning)
    from . import report

    if args.check_terms:
        bad = report.check_terms()
        print("All terms have Wiktionary entries." if not bad else f"Problems: {bad}")
        return
    if args.collect or args.collect_urban:
        _collect(args.end, args.collect_urban)

    res = report.compute(with_sample=not args.no_sample)
    print(report.text(res))
    if not args.no_plots:
        paths = report.draw(res, Path(args.assets_dir))
        print(f"\nCharts written to {Path(args.assets_dir)}/: " + ", ".join(p.name for p in paths))


if __name__ == "__main__":
    main()
