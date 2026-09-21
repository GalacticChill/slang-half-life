"""The dashboard's installable-app files: manifest, icons and head tags."""

import json
import re
from pathlib import Path

from PIL import Image

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _manifest():
    return json.loads((DOCS / "manifest.webmanifest").read_text())


def test_manifest_has_what_install_needs():
    m = _manifest()
    for key in ("name", "short_name", "start_url", "scope", "display", "icons"):
        assert m[key]
    assert m["display"] == "standalone"
    assert len(m["short_name"]) <= 12  # longer names get cut off under home-screen icons


def test_manifest_paths_are_relative():
    # The site lives at /slang-half-life/, not the domain root: a leading "/"
    # would point outside it and break installing.
    m = _manifest()
    paths = [m["start_url"], m["scope"], m["id"]] + [i["src"] for i in m["icons"]]
    assert not any(p.startswith("/") or "://" in p for p in paths)


def test_every_icon_exists_at_its_declared_size():
    for icon in _manifest()["icons"]:
        with Image.open(DOCS / icon["src"]) as im:
            w, h = (int(v) for v in icon["sizes"].split("x"))
            assert im.size == (w, h)
    sizes = {i["sizes"] for i in _manifest()["icons"] if i.get("purpose", "any") == "any"}
    assert {"192x192", "512x512"} <= sizes
    assert any(i.get("purpose") == "maskable" for i in _manifest()["icons"])


def test_maskable_icon_fills_the_whole_square():
    # Maskable icons get cropped to a circle or squircle: no transparent corners.
    with Image.open(DOCS / "icons" / "icon-maskable-512.png") as im:
        assert im.convert("RGBA").getpixel((0, 0))[3] == 255


def test_page_links_manifest_and_every_head_asset():
    html = (DOCS / "index.html").read_text()
    assert '<link rel="manifest" href="manifest.webmanifest">' in html
    for href in re.findall(r'<link rel="(?:icon|apple-touch-icon|manifest)" href="([^"]+)"', html):
        assert not href.startswith("/")
        assert (DOCS / href).exists(), href
    with Image.open(DOCS / "icons" / "apple-touch-icon.png") as im:
        assert im.size == (180, 180) and im.mode == "RGB"  # iOS shows transparency as black
