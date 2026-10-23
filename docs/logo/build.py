"""Generate the pyBKT logo option SVGs in this directory.

Run with ``python docs/logo/build.py``. The relief and halftone marks are
computed, so they are generated rather than hand-drawn.
"""

import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE = "#002676"  # Berkeley Blue
GOLD = "#FDB515"  # California Gold
STEEL = "#3B7EA1"  # Founder's Rock
MIST = "#9FB6CE"
WHITE = "#FFFFFF"


def svg(body, size=512):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 200 200">\n{body}\n</svg>\n'
    )


def emission_mosaic():
    # BKT emission table: columns are unlearned | learned, rows are
    # incorrect | correct. Guess and slip are the small cells.
    return "\n".join(
        [
            f'<rect x="20" y="20" width="160" height="160" fill="{BLUE}"/>',
            f'<rect x="28" y="28" width="48" height="109" fill="{WHITE}"/>',
            f'<rect x="28" y="145" width="48" height="27" fill="{MIST}"/>',
            f'<rect x="84" y="28" width="88" height="16" fill="{STEEL}"/>',
            f'<rect x="84" y="52" width="88" height="120" fill="{GOLD}"/>',
        ]
    )


def halftone_latent():
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
                dots.append(f'<circle cx="{x}" cy="{y}" r="2" fill="{BLUE}"/>')
    return "\n".join(dots)


def seg_dist(px, py, s):
    vx, vy = s[2] - s[0], s[3] - s[1]
    t = ((px - s[0]) * vx + (py - s[1]) * vy) / (vx * vx + vy * vy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (s[0] + t * vx), py - (s[1] + t * vy))


def relief_k():
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
            f'<path d="{line}" fill="none" stroke="{BLUE}" stroke-width="2.4" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        y0 += 7
    return "\n".join(parts)


def bkt_ligature():
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
            f'<path d="{strokes}" fill="none" stroke="{BLUE}" stroke-width="16" '
            'stroke-linecap="square" stroke-linejoin="miter"/>',
            "</g>",
        ]
    )


OPTIONS = {
    "option-a-emission-mosaic": emission_mosaic,
    "option-b-halftone-latent": halftone_latent,
    "option-c-relief-k": relief_k,
    "option-d-bkt-ligature": bkt_ligature,
}


def overview():
    cells = []
    for i, build in enumerate(OPTIONS.values()):
        x = i * 220
        cells.append(f'<g transform="translate({x + 10} 10)">{build()}</g>')
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1760" height="440" '
        'viewBox="0 0 880 220">\n'
        f'<rect width="880" height="220" fill="{WHITE}"/>\n'
        + "\n".join(cells)
        + "\n</svg>\n"
    )


def main():
    for name, build in OPTIONS.items():
        with open(os.path.join(HERE, f"{name}.svg"), "w") as f:
            f.write(svg(build()))
    with open(os.path.join(HERE, "overview.svg"), "w") as f:
        f.write(overview())


if __name__ == "__main__":
    main()
