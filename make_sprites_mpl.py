#!/usr/bin/env python3
"""
Sprite sheet generator without PyVista.

Same output naming as make_sprites.py (``<prefix>_<row>_<col>.png``, ``<prefix>.all.png``,
``<prefix>.green.png``) and the same camera sweep (roll along columns, elevation along rows),
but the input can be either a ball type name or an existing texture file:

    python make_sprites_mpl.py panel-6 4 -s 128 -o sprites/p6
    python make_sprites_mpl.py examples/b-and-w.png 4 -o sprites/ball
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import balls
import render_sphere as rs

GREEN = (0.0, 0.5, 0.0)


def main(source, count, size="128x128", output_prefix="sprite", ssaa=3, individual=True):
    if "x" in size:
        width, height = map(int, size.split("x"))
    else:
        width = height = int(size)

    kw = {}
    if os.path.isfile(source):
        kw["texture"] = rs.load_texture(source)
    else:
        try:
            kw["ball"] = balls.get_ball(source)
        except KeyError as e:
            sys.exit(f"'{source}' is neither a texture file nor a ball type.\n{e}")

    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    step = 360 / count
    sheet = np.zeros((height * count, width * count, 4), dtype=np.float32)
    green = np.zeros((height * count, width * count, 3), dtype=np.float32)

    for i in range(count):
        for j in range(count):
            roll, elev = j * step, i * step
            print(f"  sprite {i + 1:02d}_{j + 1:02d} (roll {roll:.1f}°, elevation {elev:.1f}°)")
            img = rs.render_view(width, height, elevation=elev, roll=roll, ssaa=ssaa, **kw)
            if individual:
                rs.to_pil(img).save(f"{output_prefix}_{i + 1:02d}_{j + 1:02d}.png")
            ys, xs = slice(i * height, (i + 1) * height), slice(j * width, (j + 1) * width)
            sheet[ys, xs] = img
            a = img[..., 3:4]
            green[ys, xs] = img[..., :3] * a + np.array(GREEN) * (1 - a)

    rs.to_pil(sheet).save(f"{output_prefix}.all.png")
    Image.fromarray(balls.to_uint8(green), "RGB").save(f"{output_prefix}.green.png")
    print(f"Created: {output_prefix}.all.png, {output_prefix}.green.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate sphere sprite sheets (numpy renderer)")
    ap.add_argument("source", help="ball type name (see gen_ball.py --list) or equirectangular texture file")
    ap.add_argument("count", type=int, help="NxN sprites")
    ap.add_argument("-s", "--size", default="128x128")
    ap.add_argument("-o", "--output-prefix", default="sprite")
    ap.add_argument("--ssaa", type=int, default=3, help="supersampling factor (default 3)")
    ap.add_argument("--no-individual", action="store_true", help="only write the composite sheets")
    a = ap.parse_args()
    main(a.source, a.count, a.size, a.output_prefix, a.ssaa, not a.no_individual)
