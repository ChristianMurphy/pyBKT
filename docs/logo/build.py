"""Generate the pyBKT logo option SVGs in this directory.

Run with ``python docs/logo/build.py``, then ``docs/logo/render.sh`` to
refresh the PNGs. The relief and halftone marks are computed, so they are
generated rather than hand-drawn.
"""

import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE = "#002676"  # Berkeley Blue
GOLD = "#FDB515"  # California Gold
STEEL = "#3B7EA1"  # Founder's Rock
MIST = "#9FB6CE"
WHITE = "#FFFFFF"
PAPER = "#F6F2E9"
GITHUB_DARK = "#0D1117"

# Marks are drawn in a 200x200 box. "light" is for light backgrounds,
# "dark" for Berkeley Blue or dark-mode backgrounds.
INK = {"light": BLUE, "dark": PAPER}


def emission_mosaic(theme="light"):
    # BKT emission table: columns are unlearned | learned, rows are
    # incorrect | correct. Guess and slip are the small cells.
    ground, unlearned = (BLUE, WHITE) if theme == "light" else (PAPER, BLUE)
    return "\n".join(
        [
            f'<rect x="20" y="20" width="160" height="160" fill="{ground}"/>',
            f'<rect x="28" y="28" width="48" height="109" fill="{unlearned}"/>',
            f'<rect x="28" y="145" width="48" height="27" fill="{MIST}"/>',
            f'<rect x="84" y="28" width="88" height="16" fill="{STEEL}"/>',
            f'<rect x="84" y="52" width="88" height="120" fill="{GOLD}"/>',
        ]
    )


def halftone_latent(theme="light"):
    # Small observation dots resolving into a gold latent disc.
    dots = []
    step = 13
    for r in range(13):
        for c in range(13):
            x, y = 22 + c * step, 22 + r * step
            d = math.hypot(x - 100, y - 100)
            if d > 82:
                continue
            if d < 46:
                rad = min(6.3, 2.4 + (46 - d) / 7)
                dots.append(f'<circle cx="{x}" cy="{y}" r="{rad:.1f}" fill="{GOLD}"/>')
            else:
                dots.append(f'<circle cx="{x}" cy="{y}" r="2" fill="{INK[theme]}"/>')
    return "\n".join(dots)


def seg_dist(px, py, s):
    vx, vy = s[2] - s[0], s[3] - s[1]
    t = ((px - s[0]) * vx + (py - s[1]) * vy) / (vx * vx + vy * vy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (s[0] + t * vx), py - (s[1] + t * vy))


def relief_k(theme="light"):
    # Horizontal lines that lift wherever they pass over a hidden K. The lift
    # is a plateau with soft (smoothstep) edges, so the K reads as a shape.
    segs = [(58, 34, 58, 166), (64, 110, 148, 34), (92, 94, 150, 166)]
    lift, half_width, edge = 7, 13, 7
    parts = []
    y0 = 28
    while y0 <= 176:
        pts = []
        for x in range(16, 185, 2):
            m = min(seg_dist(x, y0, s) for s in segs)
            e = max(0.0, min(1.0, (half_width - m) / edge))
            e = e * e * (3 - 2 * e)
            pts.append(f"{x} {y0 - lift * e:.1f}")
        line = "M" + " L".join(pts)
        parts.append(f'<path d="{line} L184 {y0 + 1} L16 {y0 + 1} Z" fill="{GOLD}"/>')
        parts.append(
            f'<path d="{line}" fill="none" stroke="{INK[theme]}" stroke-width="2.4" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        y0 += 7
    return "\n".join(parts)


def bkt_ligature(theme="light"):
    # b's bowl closes on k's stem; k's arm runs at 45 degrees into t's crossbar.
    bowl = "M40 100 L66 100 A33 33 0 0 1 66 166 L40 166"
    strokes = (
        f"M40 30 L40 166 {bowl} M99 30 L99 166 M104 135 L139 100 L180 100 "
        "M115 124 L141 166 M160 30 L160 150 Q160 166 176 166"
    )
    return "\n".join(
        [
            '<g transform="translate(-10 0)">',
            f'<path d="{bowl} Z" fill="{GOLD}"/>',
            f'<path d="{strokes}" fill="none" stroke="{INK[theme]}" stroke-width="16" '
            'stroke-linecap="square" stroke-linejoin="miter"/>',
            "</g>",
        ]
    )


OPTIONS = {
    "option-a-emission-mosaic": ("A · Emission mosaic", emission_mosaic),
    "option-b-halftone-latent": ("B · Halftone latent", halftone_latent),
    "option-c-relief-k": ("C · Relief K", relief_k),
    "option-d-bkt-ligature": ("D · bkt ligature", bkt_ligature),
}

SANS = "Inter, Helvetica, Arial, sans-serif"
SERIF = "Caladea, Cambria, Georgia, serif"


def mark_svg(build, theme, size=512):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 200 200">\n{build(theme)}\n</svg>\n'
    )


def place(build, theme, x, y, size):
    return f'<g transform="translate({x} {y}) scale({size / 200})">{build(theme)}</g>'


def text(x, y, s, size, fill, weight=400, family=SANS, anchor="start", extra=""):
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {extra}>{s}</text>'
    )


def wordmark(x, y, size, ink, accent):
    return text(
        x,
        y,
        f'<tspan font-weight="300" fill="{accent}">py</tspan>BKT',
        size,
        ink,
        weight=650,
        extra='letter-spacing="-1"',
    )


def label(x, y, s):
    return text(x, y, s, 15, "#2E4A73", weight=600)


def light_dark(build):
    # The mark on white and on Berkeley Blue, side by side.
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="512" '
        'viewBox="0 0 1024 512">\n'
        f'<rect width="512" height="512" fill="{WHITE}"/>\n'
        f'<rect x="512" width="512" height="512" fill="{BLUE}"/>\n'
        f"{place(build, 'light', 76, 76, 360)}\n"
        f"{place(build, 'dark', 588, 76, 360)}\n"
        "</svg>\n"
    )


def bars(x, y, widths, h, fill, gap=10):
    out = []
    for i, w in enumerate(widths):
        out.append(f'<rect x="{x}" y="{y + i * (h + gap)}" width="{w}" height="{h}" rx="2" fill="{fill}"/>')
    return "\n".join(out)


def hexagon(cx, cy, r):
    pts = []
    for i in range(6):
        a = math.pi / 6 + i * math.pi / 3
        pts.append(f"{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f}")
    return " ".join(pts)


def contexts(name, build):
    # One sheet showing the mark where it would actually be used.
    p = []
    p.append(f'<rect width="1600" height="1000" fill="#E9EDF3"/>')
    p.append(text(40, 46, name, 26, BLUE, weight=700))

    # README header, light and dark mode.
    p.append(label(40, 92, "README / docs header · light mode"))
    p.append(f'<rect x="40" y="104" width="560" height="170" rx="12" fill="{WHITE}"/>')
    p.append(place(build, "light", 66, 129, 120))
    p.append(wordmark(206, 216, 72, BLUE, STEEL))
    p.append(label(40, 318, "README / docs header · dark mode"))
    p.append(f'<rect x="40" y="330" width="560" height="170" rx="12" fill="{GITHUB_DARK}"/>')
    p.append(place(build, "dark", 66, 355, 120))
    p.append(wordmark(206, 442, 72, PAPER, MIST))

    # GitHub organization avatar, light and dark.
    p.append(label(640, 92, "GitHub avatar"))
    p.append(f'<rect x="640" y="104" width="170" height="170" rx="12" fill="{WHITE}"/>')
    p.append(f'<rect x="665" y="129" width="120" height="120" rx="14" fill="{WHITE}" stroke="#D0D7DE" stroke-width="2"/>')
    p.append(place(build, "light", 671, 135, 108))
    p.append(f'<rect x="640" y="330" width="170" height="170" rx="12" fill="{GITHUB_DARK}"/>')
    p.append(f'<rect x="665" y="355" width="120" height="120" rx="14" fill="{BLUE}"/>')
    p.append(place(build, "dark", 671, 361, 108))

    # Paper title page.
    p.append(label(850, 92, "Paper title page"))
    p.append(f'<rect x="850" y="104" width="330" height="430" fill="{WHITE}" stroke="#C9D1DB"/>')
    p.append(place(build, "light", 878, 128, 46))
    p.append(text(934, 150, "pyBKT", 14, BLUE, weight=700))
    p.append(text(1015, 192, "pyBKT: An Accessible Python Library", 15, "#111", weight=700, family=SERIF, anchor="middle"))
    p.append(text(1015, 212, "of Bayesian Knowledge Tracing Models", 15, "#111", weight=700, family=SERIF, anchor="middle"))
    p.append(text(1015, 238, "A. Badrinath, F. Wang, Z. A. Pardos", 11, "#333", family=SERIF, anchor="middle"))
    p.append(text(1015, 254, "UC Berkeley", 10, "#555", family=SERIF, anchor="middle", extra='font-style="italic"'))
    p.append(bars(880, 280, [270, 270, 270, 270, 200], 6, "#D5D9DF", gap=8))
    p.append(bars(880, 362, [125] * 13, 5, "#E1E4E8", gap=8))
    p.append(bars(1025, 362, [125] * 13, 5, "#E1E4E8", gap=8))

    # Conference poster header.
    p.append(label(1220, 92, "Conference poster"))
    p.append(f'<rect x="1220" y="104" width="340" height="430" fill="{WHITE}" stroke="#C9D1DB"/>')
    p.append(f'<rect x="1220" y="104" width="340" height="96" fill="{BLUE}"/>')
    p.append(place(build, "dark", 1234, 116, 72))
    p.append(wordmark(1316, 160, 38, PAPER, MIST))
    p.append(text(1318, 184, "Bayesian Knowledge Tracing in Python", 11, MIST))
    for col in range(3):
        cx = 1236 + col * 108
        p.append(f'<rect x="{cx}" y="216" width="96" height="16" fill="{GOLD}"/>')
        p.append(bars(cx, 242, [96, 90, 96, 80, 96, 70], 5, "#DDE2E8", gap=7))
        p.append(f'<rect x="{cx}" y="320" width="96" height="70" fill="#EEF2F7"/>')
        p.append(bars(cx, 402, [96, 84, 96, 90, 60], 5, "#DDE2E8", gap=7))
    p.append(f'<rect x="1220" y="500" width="340" height="34" fill="#EEF2F7"/>')

    # Stickers on a laptop lid.
    p.append(label(40, 580, "Stickers"))
    p.append(f'<rect x="40" y="592" width="1520" height="370" rx="18" fill="#2B3038"/>')
    # Die-cut: the mark on a white backing.
    p.append(f'<rect x="96" y="652" width="250" height="250" rx="34" fill="{WHITE}"/>')
    p.append(place(build, "light", 111, 667, 220))
    # Hex sticker.
    p.append(f'<polygon points="{hexagon(560, 777, 150)}" fill="{BLUE}" stroke="{PAPER}" stroke-width="8"/>')
    p.append(place(build, "dark", 490, 668, 140))
    p.append(text(560, 850, "pyBKT", 34, PAPER, weight=700, anchor="middle"))
    # Round sticker on off-white.
    p.append(f'<circle cx="890" cy="777" r="135" fill="{PAPER}" stroke="{WHITE}" stroke-width="8"/>')
    p.append(place(build, "light", 800, 687, 180))
    # Small rectangular sticker with wordmark.
    p.append(f'<rect x="1080" y="712" width="430" height="130" rx="22" fill="{BLUE}"/>')
    p.append(place(build, "dark", 1100, 727, 100))
    p.append(wordmark(1218, 802, 62, PAPER, MIST))

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" '
        'viewBox="0 0 1600 1000">\n' + "\n".join(p) + "\n</svg>\n"
    )


def overview():
    cells = []
    for i, (_, build) in enumerate(OPTIONS.values()):
        cells.append(place(build, "light", i * 220 + 10, 10, 200))
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1760" height="440" '
        'viewBox="0 0 880 220">\n'
        f'<rect width="880" height="220" fill="{WHITE}"/>\n'
        + "\n".join(cells)
        + "\n</svg>\n"
    )


def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def main():
    for name, (title, build) in OPTIONS.items():
        write(os.path.join(HERE, f"{name}.svg"), mark_svg(build, "light"))
        write(os.path.join(HERE, f"{name}-dark.svg"), mark_svg(build, "dark"))
        write(os.path.join(HERE, "mockups", f"{name}-light-dark.svg"), light_dark(build))
        write(os.path.join(HERE, "mockups", f"{name}-contexts.svg"), contexts(title, build))
    write(os.path.join(HERE, "overview.svg"), overview())


if __name__ == "__main__":
    main()
