"""Tiny HTTP helper shared by every collector.

Wikimedia asks every API client to identify itself with a descriptive
User-Agent that includes a way to contact the maintainer. We use the repo URL.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

USER_AGENT = "slang-half-life/0.1 (https://github.com/GalacticChill/slang-half-life)"


def get_json(url: str, retries: int = 3, backoff: float = 2.0) -> dict:
    """GET a URL and parse JSON, retrying politely on rate limits, server errors and timeouts."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(backoff * (2**attempt))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries:
                time.sleep(backoff * (2**attempt))
                continue
            raise
    raise RuntimeError("unreachable")
