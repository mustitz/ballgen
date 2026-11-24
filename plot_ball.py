#!/usr/bin/env python3
"""
Scientific matplotlib figures for ball layouts.

    python plot_ball.py geometry    panel-14 -o geo.png     # 3D panels + seams
    python plot_ball.py projections classic-32 -o proj.png  # equirect+Tissot, Mollweide, 2 orthographic views
    python plot_ball.py analysis    panel-6  -o ana.png     # panel areas, seam density vs latitude, metrics
    python plot_ball.py compare     -o cmp.png              # all ball types side by side
    python plot_ball.py gallery     -o gallery.png          # every ball type, rendered

Without -o a window opens.  Everything is on the unit sphere (see ball_stats.py).
"""

import argparse
import math
import sys

import matplotlib
import numpy as np

import ball_stats
import balls
import render_sphere as rs

# validated categorical slots (blue, orange, aqua) from the reference palette
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdad4"


def _plt(save):
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9})
    return plt


def _finish(plt, fig, save):
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Wrote {save}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 3D geometry
# ---------------------------------------------------------------------------

def _slerp_edge(a, b, m):
    om = math.acos(np.clip(a @ b, -1, 1))
    t = np.linspace(0, 1, m, endpoint=False)[:, None]
    return (np.sin((1 - t) * om) * a + np.sin(t * om) * b) / math.sin(om)


def plot_geometry(ball, look=None, elev=25, azim=35, ax=None):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    look = look or ball.default_look
    cam = np.array([math.cos(math.radians(elev)) * math.cos(math.radians(azim)),
                    math.cos(math.radians(elev)) * math.sin(math.radians(azim)),
                    math.sin(math.radians(elev))])
    light = rs.LIGHT[0] * np.array([-math.sin(math.radians(azim)), math.cos(math.radians(azim)), 0]) + cam * 0.8 + \
        np.array([0, 0, 0.5])
    light /= np.linalg.norm(light)
    kc = np.array(look.kind_colors)
    quads, colors = [], []
    for f, V in enumerate(ball.face_vertices()):
        k = len(V)
        bnd = np.vstack([_slerp_edge(V[i], V[(i + 1) % k], 14) for i in range(k)])
        c = balls.unit(V.mean(0))
        radii = np.array([0, .3, .55, .75, .9, .975, 1.0])[:, None, None]
        grid = balls.unit((1 - radii) * c + radii * bnd[None])           # (R, M, 3) in panel space
        shp = grid.shape
        real = ball.twist(grid.reshape(-1, 3), inverse=True).reshape(shp)
        base = look.panel_colors[f % len(look.panel_colors)] if look.panel_colors is not None \
            else kc[min(ball.kinds[f], len(kc) - 1)]
        for r in range(shp[0] - 1):
            seam_ring = r == shp[0] - 2
            for m in range(shp[1]):
                m2 = (m + 1) % shp[1]
                q = np.array([real[r, m], real[r, m2], real[r + 1, m2], real[r + 1, m]])
                n = balls.unit(q.mean(0))
                if n @ cam < 0:
                    continue
                quads.append(q)
                shade = 0.45 + 0.65 * max(n @ light, 0)
                colors.append(np.clip((look.edge_color if seam_ring else base) * (shade if not seam_ring else 1), 0, 1))
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.figure(figsize=(6, 6)).add_subplot(projection="3d")
    ax.add_collection3d(Poly3DCollection(quads, facecolors=colors, edgecolors=colors, linewidths=0.4))
    ax.set_box_aspect((1, 1, 1), zoom=1.45)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.view_init(elev, azim)
    ax.set_axis_off()
    ax.set_title(f"{ball.name}: {ball.n_panels} panels", fontsize=10)
    return ax


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------

def mollweide_forward(lon, lat):
    """lon/lat radians -> (x, y) with x in [-2sqrt2, 2sqrt2], y in [-sqrt2, sqrt2]."""
    th = lat.copy()
    for _ in range(40):
        d = (2 * th + np.sin(2 * th) - math.pi * np.sin(lat)) / np.maximum(2 + 2 * np.cos(2 * th), 1e-9)
        th = th - d
    return 2 * math.sqrt(2) / math.pi * lon * np.cos(th), math.sqrt(2) * np.sin(th)


def render_mollweide(ball, look=None, width=1000):
    height = width // 2
    x = (np.arange(width) + 0.5) / width * 4 * math.sqrt(2) - 2 * math.sqrt(2)
    y = math.sqrt(2) - (np.arange(height) + 0.5) / height * 2 * math.sqrt(2)
    X, Y = np.meshgrid(x, y)
    inside = X ** 2 / 8 + Y ** 2 / 2 <= 1
    th = np.arcsin(np.clip(Y / math.sqrt(2), -1, 1))
    lat = np.arcsin(np.clip((2 * th + np.sin(2 * th)) / math.pi, -1, 1))
    lon = math.pi * X / (2 * math.sqrt(2) * np.maximum(np.cos(th), 1e-9))
    p = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)[inside]
    col = ball.colorize(p, look, aa_rad=2 * math.pi / width)
    out = np.ones((height, width, 4), dtype=np.float32)
    out[..., 3] = 0
    out[inside, :3] = col
    out[inside, 3] = 1
    return out


def tissot_circles(radius_deg=9, lons=range(-150, 181, 60), lats=(-60, -30, 0, 30, 60)):
    """Small circles of constant angular radius: their distortion shows the projection's stretch."""
    out = []
    d = math.radians(radius_deg)
    b = np.linspace(0, 2 * math.pi, 73)
    for lat0 in lats:
        for lon0 in lons:
            p0, l0 = math.radians(lat0), math.radians(lon0)
            lat = np.arcsin(math.sin(p0) * math.cos(d) + math.cos(p0) * math.sin(d) * np.cos(b))
            lon = l0 + np.arctan2(np.sin(b) * math.sin(d) * math.cos(p0), math.cos(d) - math.sin(p0) * np.sin(lat))
            out.append((lon, lat))
    return out


def plot_projections(ball, look=None, fig=None):
    import matplotlib.pyplot as plt
    fig = fig or plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.38, wspace=0.05)

    ax = fig.add_subplot(gs[0, 0])
    tex = balls.render_texture(ball, look, 1000, 500)
    ax.imshow(tex, extent=[-180, 180, -90, 90])
    for lon, lat in tissot_circles():
        ax.plot(np.degrees(lon), np.degrees(lat), color=INK, lw=0.9)
    ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-90, 91, 30))
    ax.set_xlabel("longitude (°)"); ax.set_ylabel("latitude (°)")
    ax.set_title("Equirectangular with Tissot circles (equal 9° radius on the sphere)", fontsize=9, loc="left")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(render_mollweide(ball, look), extent=[-2 * math.sqrt(2), 2 * math.sqrt(2), -math.sqrt(2), math.sqrt(2)])
    for lon, lat in tissot_circles():
        lon = (lon + math.pi) % (2 * math.pi) - math.pi
        x, y = mollweide_forward(lon, lat)
        if np.ptp(x) < 1.5:   # skip circles split by the dateline
            ax.plot(x, y, color=INK, lw=0.9)
    ax.set_axis_off()
    ax.set_title("Mollweide (equal-area): circles keep their area, shapes shear at the edge", fontsize=9, loc="left")

    for col, (az, name) in enumerate([(0, "front"), (180, "back")]):
        ax = fig.add_subplot(gs[1, col])
        img = rs.render_view(480, ball=ball, look=look, elevation=20, azimuth=az, ssaa=2)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"Orthographic, {name} (azimuth {az}°)", fontsize=9, loc="left")
    fig.suptitle(f"{ball.name}: {ball.n_panels} panels", x=0.02, ha="left", fontsize=12)
    return fig


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def plot_analysis(ball, stats=None, fig=None):
    import matplotlib.pyplot as plt
    st = stats or ball_stats.analyze(ball)
    fig = fig or plt.figure(figsize=(12, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.2, 0.9], wspace=0.28)

    # (a) panel areas, grouped by kind
    ax = fig.add_subplot(gs[0])
    order = np.lexsort((st["panel_area"], st["kinds"]))
    pal = [BLUE, ORANGE, AQUA]
    for kind in sorted(set(st["kinds"])):
        sel = [i for i, o in enumerate(order) if st["kinds"][o] == kind]
        ax.bar(sel, st["panel_area"][order][sel], color=pal[kind % 3], width=0.72,
               label=ball.kind_names[min(kind, len(ball.kind_names) - 1)])
    ax.axhline(st["area_mean"], color=MUTED, lw=1, ls="--")
    ax.text(len(order) - 0.5, st["area_mean"], " mean", color=MUTED, va="bottom", ha="right", fontsize=8)
    ax.set_xlabel("panel (sorted by kind, then area)")
    ax.set_ylabel("area on unit sphere (sr)")
    ax.set_ylim(0, st["panel_area"].max() * 1.15)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncols=3, fontsize=8)
    ax.set_title("Panel area", loc="left", fontsize=10)

    # (b) seam density vs latitude
    ax = fig.add_subplot(gs[1])
    dens = st["seam_density_lat"] * 100
    ax.plot(st["lat_bins"], dens, color=BLUE, lw=2)
    ax.axhline(dens.mean(), color=MUTED, lw=1, ls="--")
    ax.text(88, dens.mean(), "mean ", color=MUTED, va="bottom", ha="right", fontsize=8)
    ax.set_xlabel("latitude (°)"); ax.set_ylabel("area covered by seams (%)")
    ax.set_xlim(-90, 90); ax.set_ylim(0, max(dens.max() * 1.25, 1))
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("Seam density by latitude (1° seams)", loc="left", fontsize=10)

    # (c) metrics
    ax = fig.add_subplot(gs[2])
    ax.set_axis_off()
    rows = [("panels", f"{st['n_panels']}"),
            ("area CV", f"{st['area_cv']:.3f}"),
            ("largest / smallest", f"{st['area_ratio']:.2f}"),
            ("seam length", f"{st['seam_length']:.2f} rad"),
            ("length per area", f"{st['seam_per_area']:.3f}"),
            ("seam isotropy", f"{st['seam_anisotropy']:.3f}"),
            ("seam-free cap", f"{st['max_cap_deg']:.1f}°" if st['max_cap_deg'] == st['max_cap_deg'] else "n/a (twist)")]
    for i, (k, v) in enumerate(rows):
        y = 0.92 - i * 0.12
        ax.text(0.0, y, k, color=MUTED, transform=ax.transAxes)
        ax.text(1.0, y, v, ha="right", fontweight="bold", transform=ax.transAxes)
        ax.plot([0, 1], [y - 0.04] * 2, color=GRID, lw=0.6, transform=ax.transAxes)
    ax.set_title("Metrics", loc="left", fontsize=10)
    fig.suptitle(f"{ball.name}: {ball.description}", x=0.07, ha="left", fontsize=11)
    return fig


def _unique_balls():
    seen, out = set(), []
    for n in balls.list_balls():
        b = balls.get_ball(n)
        key = (b.n_panels, b.twist_deg, b.warp is not None, round(float(b.heights.sum()), 6))
        if key not in seen:
            seen.add(key)
            out.append(b)
    return out


def plot_compare(fig=None):
    import matplotlib.pyplot as plt
    bl = _unique_balls()
    stats = [ball_stats.analyze(b, n=200_000) for b in bl]
    fig = fig or plt.figure(figsize=(15, 4.6))
    metrics = [("seam_per_area", "Seam length per unit area (lower = fewer seams)", "{:.2f}"),
               ("area_cv", "Panel area variation, CV (0 = identical panels)", "{:.2f}"),
               ("max_cap_deg", "Largest seam-free cap radius (°)", "{:.0f}")]
    for i, (key, title, fmt) in enumerate(metrics):
        ax = fig.add_subplot(1, 3, i + 1)
        order = np.argsort([s["n_panels"] for s in stats])[::-1]   # same order in every panel
        names = [f"{stats[o]['name']} ({stats[o]['n_panels']})" for o in order]
        vals = [stats[o][key] for o in order]
        ax.barh(range(len(vals)), np.nan_to_num(vals), color=BLUE, height=0.62)
        for y, v in enumerate(vals):
            ax.text(0 if v != v else v, y, " n/a (twisted)" if v != v else " " + fmt.format(v),
                    va="center", fontsize=8, color=MUTED if v != v else INK)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(0, np.nanmax(vals) * 1.2 + 1e-9)
        ax.invert_yaxis()
        ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=9)
    fig.suptitle("Ball layouts compared (unit sphere; panel count in brackets)", x=0.02, ha="left", fontsize=12)
    fig.tight_layout()
    return fig


def plot_gallery(fig=None):
    import matplotlib.pyplot as plt
    names = balls.list_balls()
    cols = 3
    rows = math.ceil(len(names) / cols)
    fig = fig or plt.figure(figsize=(3.6 * cols, 3.6 * rows))
    for i, n in enumerate(names):
        ax = fig.add_subplot(rows, cols, i + 1)
        b = balls.get_ball(n)
        ax.imshow(rs.render_view(360, ball=b, elevation=25, azimuth=20, ssaa=2))
        ax.set_axis_off()
        ax.set_title(f"{n} ({b.n_panels})", fontsize=9)
    fig.tight_layout()
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description="Scientific plots for soccer-ball layouts")
    ap.add_argument("kind", choices=["geometry", "projections", "analysis", "compare", "gallery"])
    ap.add_argument("ball", nargs="?", help="ball type (not needed for compare / gallery)")
    ap.add_argument("-o", "--output", default=None, help="save PNG instead of opening a window")
    ap.add_argument("--palette", default=None, help="matplotlib colormap to colour each panel")
    ap.add_argument("--elev", type=float, default=25)
    ap.add_argument("--azim", type=float, default=35)
    a = ap.parse_args(argv)
    plt = _plt(a.output)

    ball = look = None
    if a.kind not in ("compare", "gallery"):
        if not a.ball:
            ap.error(f"'{a.kind}' needs a ball type")
        try:
            ball = balls.get_ball(a.ball)
        except KeyError as e:
            sys.exit(str(e))
        look = ball.default_look
        if a.palette:
            look = look.with_palette(a.palette, ball.n_panels)

    if a.kind == "geometry":
        fig = plt.figure(figsize=(6.5, 6.5))
        plot_geometry(ball, look, a.elev, a.azim, ax=fig.add_subplot(projection="3d"))
    elif a.kind == "projections":
        fig = plot_projections(ball, look)
    elif a.kind == "analysis":
        fig = plot_analysis(ball)
    elif a.kind == "compare":
        fig = plot_compare()
    else:
        fig = plot_gallery()
    _finish(plt, fig, a.output)


if __name__ == "__main__":
    main()
