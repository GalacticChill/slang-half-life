"""Command-line entry point for slang-half-life."""

from __future__ import annotations

import argparse

from . import __version__


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="slang-half-life",
        description="How long does a slang term live? Measure the rise, peak, and "
        "decay of slang terms from their monthly Wiktionary lookups.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args(argv)


def main(argv=None) -> None:
    _parse_args(argv)
    print("slang-half-life: nothing to analyze yet (scaffold only).")


if __name__ == "__main__":
    main()
