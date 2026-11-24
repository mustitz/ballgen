#!/usr/bin/env python3
"""
Generate an equirectangular texture for any registered ball type.

Drop-in companion to gen_texture.py (same texture convention, same colour flags)
but panel layouts come from balls.py, so there is no flood fill and no leaking.

    python gen_ball.py --list
    python gen_ball.py panel-6 -s 2048 -o panel6.png
    python gen_ball.py classic-32 --bg-color white --pentagon-color blue --edge-color black
    python gen_ball.py panel-20 --palette viridis --seam 2.0
"""

import argparse
import sys

from PIL import Image

import balls

DEFAULT_SIZE = "1024x512"


def parse_size(size: str):
    if "x" in size:
        w, h = map(int, size.split("x"))
        return w, h
    w = int(size)
    return w, w // 2


def main(ball_name, output, size=DEFAULT_SIZE, bg_color=None, pentagon_color=None, edge_color=None,
         seam_deg=None, palette=None, twist=None, lat_rotation=0.0, lon_rotation=0.0):
    ball = balls.get_ball(ball_name)
    if twist is not None:
        ball.twist_deg = twist
    look = ball.default_look
    if bg_color or pentagon_color or edge_color:
        look = look.with_colors(bg=bg_color, pentagon=pentagon_color, edge=edge_color)
    if palette:
        look = look.with_palette(palette, ball.n_panels)
    if seam_deg is not None:
        look.seam_deg = seam_deg

    width, height = parse_size(size)
    rgb = balls.render_texture(ball, look, width, height, lat_rotation, lon_rotation)
    Image.fromarray(balls.to_uint8(rgb)).save(output)
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate a texture for a registered soccer-ball type")
    ap.add_argument("ball", nargs="?", help="ball type (see --list)")
    ap.add_argument("--list", action="store_true", help="list available ball types and exit")
    ap.add_argument("-o", "--output", default=None, help="output file (default: <ball>.png)")
    ap.add_argument("-s", "--size", default=DEFAULT_SIZE, help="WIDTH or WIDTHxHEIGHT (default 1024x512)")
    ap.add_argument("--bg-color", default=None, help="colour of every panel kind except the first")
    ap.add_argument("--pentagon-color", default=None, help="colour of the first panel kind (pentagons / squares)")
    ap.add_argument("--edge-color", default=None, help="seam colour")
    ap.add_argument("--seam", type=float, default=None, help="seam width in degrees of arc")
    ap.add_argument("--palette", default=None, help="matplotlib colormap; colours each panel separately")
    ap.add_argument("--twist", type=float, default=None, help="override the swirl angle (degrees)")
    ap.add_argument("--lat-rotation", type=float, default=0.0)
    ap.add_argument("--lon-rotation", type=float, default=0.0)
    args = ap.parse_args()

    if args.list:
        for name in balls.list_balls():
            b = balls.get_ball(name)
            print(f"{name:18s} {b.n_panels:3d} panels  {b.description}")
        sys.exit(0)
    if not args.ball:
        ap.error("give a ball type, or use --list")
    out = args.output or f"{args.ball}.png"
    try:
        main(args.ball, out, args.size, args.bg_color, args.pentagon_color, args.edge_color,
             args.seam, args.palette, args.twist, args.lat_rotation, args.lon_rotation)
    except KeyError as e:
        sys.exit(str(e))
    print(f"Wrote {out}")
