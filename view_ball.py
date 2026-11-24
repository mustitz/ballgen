#!/usr/bin/env python3
"""
Interactive matplotlib viewer (replaces view_texture.py's PyVista window).

    python view_ball.py panel-8                 # sliders for elevation / azimuth / roll / seam / twist
    python view_ball.py examples/b-and-w.png    # works on texture files too
    python view_ball.py panel-6 --save view.png # no window: write a single frame
"""

import argparse
import os
import sys

import matplotlib

import balls
import render_sphere as rs


def main(source, size=360, elevation=20.0, azimuth=0.0, roll=0.0, save=None):
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    ball = None
    texture = None
    if os.path.isfile(source):
        texture = rs.load_texture(source)
    else:
        ball = balls.get_ball(source)

    def draw(el, az, ro):
        look = None
        if ball is not None and "seam" in sliders:
            look = ball.default_look
            look.seam_deg = sliders["seam"].val
        kw = {"ball": ball, "look": look} if ball is not None else {"texture": texture}
        return rs.render_view(size, elevation=el, azimuth=az, roll=ro, ssaa=2,
                              background=(0.12, 0.12, 0.14), **kw)

    fig, ax = plt.subplots(figsize=(6, 7))
    fig.patch.set_facecolor("#1e1e22")
    plt.subplots_adjust(bottom=0.30)
    sliders = {}
    ax.axis("off")
    shown = ax.imshow(draw(elevation, azimuth, roll))
    ax.set_title(source, color="w")

    if save:
        fig.savefig(save, dpi=120, facecolor=fig.get_facecolor())
        print(f"Wrote {save}")
        return

    def mk(name, y, lo, hi, init):
        s = Slider(plt.axes([0.15, y, 0.7, 0.03]), name, lo, hi, valinit=init)
        s.label.set_color("w"); s.valtext.set_color("w")
        sliders[name.lower()] = s
        return s

    mk("Elevation", 0.20, -90, 90, elevation)
    mk("Azimuth", 0.15, -180, 180, azimuth)
    mk("Roll", 0.10, -180, 180, roll)
    if ball is not None:
        mk("Seam", 0.05, 0.2, 6.0, ball.default_look.seam_deg)

    def update(_):
        shown.set_data(draw(sliders["elevation"].val, sliders["azimuth"].val, sliders["roll"].val))
        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Interactive ball viewer (matplotlib)")
    ap.add_argument("source", help="ball type name or equirectangular texture file")
    ap.add_argument("-s", "--size", type=int, default=360, help="render size in pixels")
    ap.add_argument("--elevation", type=float, default=20.0)
    ap.add_argument("--azimuth", type=float, default=0.0)
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--save", default=None, help="write one frame to this file instead of opening a window")
    a = ap.parse_args()
    try:
        main(a.source, a.size, a.elevation, a.azimuth, a.roll, a.save)
    except KeyError as e:
        sys.exit(str(e))
