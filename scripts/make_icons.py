#!/usr/bin/env python3
"""Generate the site icon set and social card from a single geometry definition.

The mark is a filled open book on a teal squircle: the same book as the
nav glyph, but solid rather than stroked, because a 1.5px stroke
disappears at 16x16.

Rasterising needs Pillow, which the webapp itself does not use, so this is
a build-time script rather than a runtime dependency — like the Tailwind
build, run it and commit the output.

Usage:
  python3 scripts/make_icons.py                 # writes into static/
  python3 scripts/make_icons.py --out-dir DIR
"""

import argparse
import io
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --- Palette (brand teal, matching .prob-bar and brand-600) ---
TEAL_DARK = (13, 148, 136)
TEAL_LIGHT = (20, 184, 166)
WHITE = (255, 255, 255)

# --- Geometry, in a 64x64 design grid ---
GRID = 64.0
CORNER_RADIUS = 14.0
SPINE_X = 32.0
SPINE_GAP = 1.6          # total gap at the spine
PAGE_OUTER_X = 8.5       # outer edge of each page
PAGE_TOP_Y = 17.0        # top edge at the spine
PAGE_BOTTOM_Y = 48.0     # bottom edge at the spine
PAGE_BULGE = 3.0         # how far the outer corners ride above the spine edge

SUPERSAMPLE = 8


def _page_edge(y_spine):
    """Control points for one page edge, spine outwards to the left.

    The edge bows upwards away from the spine, so the pages read as a
    stack that has been splayed open rather than as two flat rectangles.
    """
    x0 = SPINE_X - SPINE_GAP / 2
    x1 = PAGE_OUTER_X
    y1 = y_spine - PAGE_BULGE
    mid = (x0 + x1) / 2
    return [
        (x0, y_spine),
        (mid + (x0 - mid) * 0.15, y_spine - PAGE_BULGE * 0.75),
        (mid + (x1 - mid) * 0.35, y1),
        (x1, y1),
    ]


def _bezier(p, steps=48):
    """Sample a cubic Bezier through the four control points in `p`."""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = p
    out = []
    for i in range(steps + 1):
        t = i / steps
        u = 1 - t
        a, b, c, d = u ** 3, 3 * u * u * t, 3 * u * t * t, t ** 3
        out.append((a * x0 + b * x1 + c * x2 + d * x3,
                    a * y0 + b * y1 + c * y2 + d * y3))
    return out


def left_page_polygon():
    """Closed outline of the left page, in design-grid coordinates."""
    top = _bezier(_page_edge(PAGE_TOP_Y))
    bottom = _bezier(_page_edge(PAGE_BOTTOM_Y))
    # spine -> outer along the top, down the outer edge, outer -> spine back
    return top + list(reversed(bottom))


def mirror(points):
    return [(GRID - x, y) for x, y in points]


# ---------- SVG ----------

def _svg_path(points):
    head = f"M{points[0][0]:.2f} {points[0][1]:.2f}"
    rest = "".join(f"L{x:.2f} {y:.2f}" for x, y in points[1:])
    return head + rest + "Z"


def write_svg(path, rounded=True):
    left = left_page_polygon()
    bg = (f'<rect width="64" height="64" rx="{CORNER_RADIUS}" fill="url(#g)"/>'
          if rounded else '<rect width="64" height="64" fill="url(#g)"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64" role="img" aria-label="Preprint Match">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="rgb{TEAL_DARK}"/>
      <stop offset="1" stop-color="rgb{TEAL_LIGHT}"/>
    </linearGradient>
  </defs>
  {bg}
  <g fill="#fff">
    <path d="{_svg_path(left)}"/>
    <path d="{_svg_path(mirror(left))}"/>
  </g>
</svg>
'''
    path.write_text(svg)
    return path


# ---------- Raster ----------

def _rounded_mask(size, radius):
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=radius, fill=255)
    return mask


def _gradient(size):
    """Diagonal teal gradient, drawn as a horizontal ramp then rotated."""
    ramp = Image.new("RGB", (size, size))
    px = ramp.load()
    for x in range(size):
        for y in range(size):
            t = (x + y) / (2 * (size - 1)) if size > 1 else 0
            px[x, y] = tuple(
                round(a + (b - a) * t) for a, b in zip(TEAL_DARK, TEAL_LIGHT))
    return ramp


def render_icon(size, rounded=True, padding=0.0):
    """Render the mark at `size` px, supersampled then downscaled.

    `padding` insets the squircle as a fraction of the size, so the Apple
    touch icon can sit on a full-bleed tile without its corners being
    clipped by the OS mask.
    """
    s = size * SUPERSAMPLE
    scale = s / GRID

    tile = _gradient(max(s // SUPERSAMPLE, 2)).resize((s, s), Image.BICUBIC)
    if rounded:
        tile.putalpha(_rounded_mask(s, round(CORNER_RADIUS * scale)))
    else:
        tile.putalpha(255)

    draw = ImageDraw.Draw(tile)
    for poly in (left_page_polygon(), mirror(left_page_polygon())):
        draw.polygon([(x * scale, y * scale) for x, y in poly], fill=WHITE)

    if padding:
        inset = round(s * padding)
        inner = tile.resize((s - 2 * inset, s - 2 * inset), Image.LANCZOS)
        canvas = Image.new("RGBA", (s, s), TEAL_DARK + (255,))
        canvas.paste(inner, (inset, inset), inner)
        tile = canvas

    return tile.resize((size, size), Image.LANCZOS)


def write_ico(path, sizes=(16, 32, 48, 64)):
    """Write a multi-resolution .ico, one independent render per size.

    Built by hand rather than through ``Image.save(format="ICO")``: Pillow
    derives every entry by downscaling a single source image, which throws
    away the per-size antialiasing that makes a 16px favicon legible. Each
    frame is stored as PNG, which every browser since IE Vista reads.
    """
    frames = []
    for size in sizes:
        buf = io.BytesIO()
        render_icon(size).save(buf, format="PNG", optimize=True)
        frames.append(buf.getvalue())

    header = struct.pack("<HHH", 0, 1, len(frames))  # reserved, type=icon, count
    offset = len(header) + 16 * len(frames)
    directory, payload = b"", b""
    for size, data in zip(sizes, frames):
        # 0 in the width/height byte means 256; every size here is smaller.
        directory += struct.pack("<BBBBHHII", size, size, 0, 0, 1, 32,
                                 len(data), offset)
        offset += len(data)
        payload += data

    path.write_bytes(header + directory + payload)
    return path


# ---------- Social card ----------

def _font(names, size):
    for name in names:
        for base in ("/System/Library/Fonts/Supplemental/",
                     "/System/Library/Fonts/", "/Library/Fonts/"):
            p = Path(base) / name
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), size)
                except OSError:
                    pass
    return ImageFont.load_default(size)


def _fitted(draw, text, names, size, max_width):
    """Largest font <= `size` from `names` that renders `text` within max_width."""
    while size > 8:
        font = _font(names, size)
        if draw.textlength(text, font=font) <= max_width:
            return font
        size -= 2
    return _font(names, size)


SERIF_BOLD = ["Georgia Bold.ttf", "Times New Roman Bold.ttf"]
SERIF = ["Georgia.ttf", "Times New Roman.ttf"]
SANS = ["Helvetica.ttc", "Arial.ttf"]


def write_og_image(path, width=1200, height=630):
    """1200x630 social card.

    Deliberately carries no paper or journal counts: the card is a static
    asset served from /static, and baked-in numbers would go stale within
    a week of the daily refresh.
    """
    img = Image.new("RGB", (width, height), (250, 253, 252))
    draw = ImageDraw.Draw(img)

    # Soft teal wash fading down the card, echoing the site's hero gradient.
    for y in range(height):
        t = min(y / height * 1.6, 1.0)
        draw.line([(0, y), (width, y)],
                  fill=tuple(round(a + (b - a) * t)
                             for a, b in zip((240, 253, 250), (250, 250, 250))))

    # Teal rule along the bottom, matching the probability bars.
    for x in range(width):
        t = x / width
        draw.line([(x, height - 10), (x, height)],
                  fill=tuple(round(a + (b - a) * t)
                             for a, b in zip(TEAL_DARK, TEAL_LIGHT)))

    # Two columns: copy on the left, a stub ranked list on the right.
    left_x, gutter = 90, 60
    bar_x, bar_w = 772, 338
    text_w = bar_x - gutter - left_x

    icon = render_icon(132)
    img.paste(icon, (left_x, 92), icon)

    for i, frac in enumerate((1.0, 0.71, 0.54, 0.37, 0.25, 0.16)):
        y = 176 + i * 52
        draw.rounded_rectangle([bar_x, y, bar_x + bar_w, y + 28], radius=14,
                               fill=(232, 240, 238))
        shade = min(i * 0.16, 1.0)
        draw.rounded_rectangle(
            [bar_x, y, bar_x + max(int(bar_w * frac), 28), y + 28], radius=14,
            fill=tuple(round(a + (b - a) * shade)
                       for a, b in zip(TEAL_DARK, (170, 218, 212))))

    title = _fitted(draw, "Preprint Match", SERIF_BOLD, 76, text_w)
    tagline = "Which journal will publish this preprint?"
    sub = _fitted(draw, tagline, SERIF, 34, text_w)
    sans = _fitted(draw, "Ranked journal predictions for medRxiv", SANS, 27,
                   text_w)

    draw.text((left_x, 274), "Preprint Match", font=title, fill=(23, 23, 23))
    draw.text((left_x, 378), tagline, font=sub, fill=(64, 64, 64))
    draw.text((left_x, 444), "Ranked journal predictions for every",
              font=sans, fill=(115, 115, 115))
    draw.text((left_x, 480), "medRxiv and bioRxiv preprint, updated daily.",
              font=sans, fill=(115, 115, 115))

    img.save(path, format="PNG", optimize=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="static")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = [write_svg(out / "icon.svg"), write_ico(out / "favicon.ico")]

    # Apple crops to its own squircle, so pad and full-bleed the tile.
    render_icon(180, rounded=False, padding=0.10).convert("RGB").save(
        out / "apple-touch-icon.png", optimize=True)
    written.append(out / "apple-touch-icon.png")

    for size in (192, 512):
        render_icon(size).save(out / f"icon-{size}.png", optimize=True)
        written.append(out / f"icon-{size}.png")

    written.append(write_og_image(out / "og-image.png"))

    for p in written:
        print(f"{p}  {p.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
