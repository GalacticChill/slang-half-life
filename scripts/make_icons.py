"""Draw the dashboard's app icons into docs/icons/.

The mark is a slang term's life: a quick rise to a rounded peak and a long
fade, in white on the project's blue, with a dashed "half of peak" line (left
out of the tiny favicon). Everything is drawn at 4x and scaled down so edges
are smooth.

Run from the repo root:  python scripts/make_icons.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BLUE = (42, 120, 214, 255)
WHITE = (255, 255, 255, 255)
OUT = Path(__file__).resolve().parent.parent / "docs" / "icons"
SCALE = 4


def curve(n=600):
    """Rise-peak-fade in [0, 1] x [0, 1]: a gamma-shaped pulse, quick up and slow down."""
    x = np.linspace(0, 1, n)
    t = x / 0.24  # peak at x = 0.24
    y = t ** 2.2 * np.exp(2.2 * (1 - t))
    return x, y / y.max()


def _stroke(d: ImageDraw.ImageDraw, pts, width: float, fill) -> None:
    """A smooth polyline: stamp a circle every half-pixel along the path."""
    r = width / 2
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        steps = max(1, int(np.hypot(x1 - x0, y1 - y0) * 2))
        for k in range(steps + 1):
            x, y = x0 + (x1 - x0) * k / steps, y0 + (y1 - y0) * k / steps
            d.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def draw(size: int, *, content: float, rounded: bool, detail: bool = True) -> Image.Image:
    """``content`` is the share of the canvas the mark spans (smaller for maskable icons)."""
    big = size * SCALE
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    radius = int(big * 0.22) if rounded else 0
    d.rounded_rectangle([0, 0, big - 1, big - 1], radius=radius, fill=BLUE)

    pad = big * (1 - content) / 2
    w = h = big - 2 * pad
    x, y = curve()
    base = pad + h * 0.86
    top = pad + h * 0.12
    px = pad + x * w
    py = base - y * (base - top)
    pts = list(zip(px, py))

    # soft fill under the curve
    fill = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    ImageDraw.Draw(fill).polygon(pts + [(px[-1], base), (px[0], base)], fill=(255, 255, 255, 60))
    img.alpha_composite(fill)

    stroke = big * (0.055 if detail else 0.11)
    if detail:
        # dashed half-of-peak line
        half = base - 0.5 * (base - top)
        dash, gap, lw = w * 0.07, w * 0.05, max(2, int(big * 0.018))
        xx = pad
        while xx < pad + w:
            d.line([(xx, half), (min(xx + dash, pad + w), half)], fill=(255, 255, 255, 170), width=lw)
            xx += dash + gap
        # baseline
        d.line([(pad, base + stroke), (pad + w, base + stroke)], fill=(255, 255, 255, 200),
               width=max(2, int(big * 0.02)))
    _stroke(d, pts, stroke, WHITE)
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    draw(192, content=0.74, rounded=True).save(OUT / "icon-192.png")
    draw(512, content=0.74, rounded=True).save(OUT / "icon-512.png")
    # Maskable: full-bleed background, mark inside the central 80% safe zone.
    draw(512, content=0.58, rounded=False).save(OUT / "icon-maskable-512.png")
    # iOS rounds the corners itself and shows transparency as black: full-bleed square.
    draw(180, content=0.70, rounded=False).convert("RGB").save(OUT / "apple-touch-icon.png")
    draw(32, content=0.78, rounded=True, detail=False).save(OUT / "favicon-32.png")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
