#!/usr/bin/env python3
"""Draw every figure of the paper from data/*.json and the ball framework.

    python make_figures.py            # all figures -> figures/*.pdf
    python make_figures.py fig2 fig5  # only some
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import ball_metrics as bm
import balls
import make_data as md
import plot_ball as pb
import render_sphere as rs
import trionda_barycentric as tb

FIG = os.path.join(HERE, "figures")
DATA = os.path.join(HERE, "data")
BLUE, ORANGE, AQUA, MUTED, GRID, INK = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#dcdad4", "#0b0b0b"

plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "pdf.fonttype": 42, "axes.linewidth": 0.8, "lines.linewidth": 1.6})


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("wrote", name)


def lonlat(p):
    return np.arctan2(p[:, 1], p[:, 0]), np.arcsin(np.clip(p[:, 2], -1, 1))


def plot_curve(ax, p, **kw):
    """Plot a spherical polyline on a lon/lat axes, breaking it at the dateline."""
    lon, lat = lonlat(p)
    cut = np.where(np.abs(np.diff(lon)) > math.pi)[0] + 1
    for lo, la in zip(np.split(lon, cut), np.split(lat, cut)):
        ax.plot(lo, la, **kw)


def fixed_points():
    T = balls.TETRA_VERTICES
    mids = np.vstack([np.eye(3), -np.eye(3)])
    return T, -T, mids


# --------------------------------------------------------------------------- fig1: Construction A audit
def fig1():
    audit = load("audit_A.json")
    fig = plt.figure(figsize=(7.2, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.5, 1.0], wspace=0.5)

    ax = fig.add_subplot(gs[0])
    V, bnd, _, _ = tb.build_flat_panel(amp=0.21, harmonic=0.035)
    ax.fill(bnd[:, 0], bnd[:, 1], color=BLUE, alpha=0.18)
    ax.plot(*np.vstack([bnd, bnd[:1]]).T, color=BLUE)
    tri = np.vstack([V, V[:1]])
    ax.plot(tri[:, 0], tri[:, 1], "--", color=MUTED, lw=1)
    ax.scatter(V[:, 0], V[:, 1], s=14, color=INK, zorder=5)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) flat panel, $A=0.21$", loc="left", fontsize=8.5)

    ax = fig.add_subplot(gs[1])
    W, H = 720, 360
    G = balls.lonlat_grid(W, H).reshape(-1, 3)
    cnt = md.coverage_counts(G, 0.21, 0.035).reshape(H, W)
    img = np.ones((H, W, 3)) * np.array([0.91, 0.91, 0.89])
    img[cnt == 0] = np.array([0.92, 0.41, 0.20])
    img[cnt >= 2] = np.array([0.16, 0.47, 0.84])
    ax.imshow(img, extent=[-180, 180, -90, 90], aspect="auto")
    ax.set_xticks(range(-180, 181, 90)); ax.set_yticks(range(-90, 91, 45))
    ax.set_xlabel("longitude (°)"); ax.set_ylabel("latitude (°)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=(0.92, 0.41, 0.20)), plt.Rectangle((0, 0), 1, 1, color=(0.91, 0.91, 0.89))],
              labels=["covered by no panel", "covered once"], loc="lower left", frameon=True, facecolor="white",
              edgecolor="none", fontsize=7.5)
    ax.set_title(f"(b) coverage: {audit['user_params']['gap']*100:.1f}% uncovered", loc="left", fontsize=8.5)
    ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)

    ax = fig.add_subplot(gs[2])
    rows = [r for r in audit["sweep"] if r["amp"] > 0]
    amp = np.array([r["amp"] for r in rows]); gap = np.array([r["gap"] for r in rows])
    ax.loglog(amp, gap * 100, "o-", color=BLUE, ms=3.5)
    ref = gap[3] * 100 * (amp / amp[3]) ** 2
    ax.loglog(amp, ref, "--", color=MUTED, lw=1)
    ax.text(amp[-4], ref[-4] * 0.45, "slope 2", color=MUTED, fontsize=8)
    ax.set_xlabel("edge amplitude $A$"); ax.set_ylabel("uncovered area (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title(f"(c) gap $\\propto A^{{{audit['loglog_slope']:.2f}}}$", loc="left", fontsize=8.5, pad=8)
    save(fig, "fig1_construction_A")


# --------------------------------------------------------------------------- fig2: the flow
def fig2():
    fig = plt.figure(figsize=(7.2, 3.1))
    gs = fig.add_gridspec(1, 2, wspace=0.12)
    T, F, M = fixed_points()
    trionda = balls.get_ball("trionda")
    tet = balls.get_ball("tetra-4")

    # (a) speed + a few flow lines
    ax = fig.add_subplot(gs[0], projection="mollweide")
    lon = np.linspace(-math.pi, math.pi, 361); lat = np.linspace(-math.pi / 2, math.pi / 2, 181)
    LO, LA = np.meshgrid(lon, lat)
    P = np.stack([np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], -1).reshape(-1, 3)
    sp = np.linalg.norm(balls.tetra_flow_field(P), axis=1).reshape(LO.shape)
    mesh = ax.pcolormesh(LO, LA, sp, cmap="Blues", shading="auto", rasterized=True)
    rng = np.random.default_rng(3)
    seeds = balls.unit(rng.normal(size=(26, 3)))
    for s in seeds:
        pts = [s]
        q = s[None]
        for _ in range(220):
            h = 0.02
            k1 = balls.tetra_flow_field(q); k2 = balls.tetra_flow_field(balls.unit(q + .5 * h * k1))
            k3 = balls.tetra_flow_field(balls.unit(q + .5 * h * k2)); k4 = balls.tetra_flow_field(balls.unit(q + h * k3))
            q = balls.unit(q + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)); pts.append(q[0])
        plot_curve(ax, np.array(pts), color=INK, lw=0.5, alpha=0.55)
    for pts, mk, c, lab in [(T, "^", ORANGE, "vertices"), (F, "v", AQUA, "face centres"), (M, "o", INK, "edge midpoints")]:
        lo, la = lonlat(pts); ax.scatter(lo, la, marker=mk, s=26, color=c, zorder=5, edgecolor="white", linewidth=0.6, label=lab)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False, fontsize=7, handletextpad=0.1, columnspacing=0.6)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.grid(True, color=GRID, lw=0.4)
    ax.set_title("(a) flow of $v=p\\times\\nabla(xyz)$; shade = speed", loc="left", fontsize=8.5)

    # (b) straight seams vs flowed seams
    ax = fig.add_subplot(gs[1], projection="mollweide")
    for pl in bm.seam_polylines(tet):
        plot_curve(ax, pl, color=MUTED, lw=1.0, ls="--")
    for pl in bm.seam_polylines(trionda):
        plot_curve(ax, pl, color=BLUE, lw=1.8)
    for pts, mk, c in [(T, "^", ORANGE), (F, "v", AQUA), (M, "o", INK)]:
        lo, la = lonlat(pts); ax.scatter(lo, la, marker=mk, s=26, color=c, zorder=5, edgecolor="white", linewidth=0.6)
    ax.legend(handles=[Line2D([], [], color=MUTED, ls="--", lw=1), Line2D([], [], color=BLUE, lw=1.8)],
              labels=["$\\tau=0$ (tetrahedron edges)", f"$\\tau={balls.TRIONDA_TAU}$ (Trionda model)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=1, frameon=False, fontsize=7)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.grid(True, color=GRID, lw=0.4)
    ax.set_title("(b) seams: pre-image of the tetrahedron edges", loc="left", fontsize=8.5)
    save(fig, "fig2_flow")


# --------------------------------------------------------------------------- fig3: tau sweep views
def fig3():
    taus = [0.0, 0.4, 0.9, 1.5]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9))
    for j, tau in enumerate(taus):
        b = balls.make_trionda(tau, f"tau{tau}")
        for i, az in enumerate([20, 200]):
            axes[i, j].imshow(rs.render_view(300, ball=b, elevation=20, azimuth=az, ssaa=2, background=(1, 1, 1)))
            axes[i, j].axis("off")
        axes[0, j].set_title(f"$\\tau={tau}$" + ("  (straight)" if tau == 0 else f"  ({math.degrees(math.atan(math.tanh(tau))):.0f}° tilt)"), fontsize=8.5)
    axes[0, 0].text(-0.02, 0.5, "front", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8, color=MUTED)
    axes[1, 0].text(-0.02, 0.5, "back", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8, color=MUTED)
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    save(fig, "fig3_tau_sweep")


# --------------------------------------------------------------------------- fig4: projections of Trionda
def fig4():
    b = balls.get_ball("trionda")
    fig = plt.figure(figsize=(7.2, 2.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.06)
    ax = fig.add_subplot(gs[0])
    # the ball is tilted 20 degrees so that no seam passes through a pole of the map
    tex = balls.render_texture(b, b.default_look, 1200, 600, lat_rotation=20)
    ax.imshow(tex, extent=[-180, 180, -90, 90])
    ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-90, 91, 30))
    ax.set_xlabel("longitude (°)"); ax.set_ylabel("latitude (°)")
    ax.set_title("(a) equirectangular texture", loc="left", fontsize=8.5)
    ax = fig.add_subplot(gs[1])
    ax.imshow(pb.render_mollweide(b, b.default_look, 1200), extent=[-2 * math.sqrt(2), 2 * math.sqrt(2), -math.sqrt(2), math.sqrt(2)])
    ax.set_axis_off()
    ax.set_title("(b) Mollweide (equal-area) view of the same ball", loc="left", fontsize=8.5)
    save(fig, "fig4_projections")


# --------------------------------------------------------------------------- fig5: tau dependence
def fig5():
    d = load("tau_sweep.json")["rows"]
    fam = {r["key"]: r for r in load("family.json")}
    tau = np.array([r["tau"] for r in d])
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    ax[0].plot(tau, [r["tilt_closed_form"] for r in d], color=MUTED, lw=1.2, label="$\\arctan\\tanh\\tau$")
    ax[0].plot(tau, [r["tilt_measured"] for r in d], "o", color=BLUE, ms=3.5, label="measured")
    ax[0].axhline(45, color=GRID, lw=0.8)
    ax[0].set_ylabel("seam tilt at edge midpoint (°)"); ax[0].set_ylim(0, 48)
    ax[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    ax[1].plot(tau, [r["ratio"] for r in d], color=BLUE)
    ax[1].set_ylabel("seam length / straight length")
    ax[2].plot(tau, [r["exposure_cv"] for r in d], color=BLUE)
    ax[2].axhline(fam["classic-32"]["exposure"]["cv"], color=ORANGE, ls="--", lw=1.2)
    ax[2].text(0.02, fam["classic-32"]["exposure"]["cv"] + 0.006, "classic-32", color=ORANGE, fontsize=7.5)
    ax[2].set_ylabel("exposure CV"); ax[2].set_ylim(0, 0.21)
    for a, t in zip(ax, ["(a)", "(b)", "(c)"]):
        a.set_xlabel("flow time $\\tau$"); a.set_title(t, loc="left", fontsize=8.5)
        a.yaxis.grid(True, color=GRID, lw=0.5); a.set_axisbelow(True)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig5_tau_dependence")


# --------------------------------------------------------------------------- fig6: multipole spectrum
CLASSES = {  # (rotation angle, number of elements) per conjugacy class of the rotation groups
    "T": [(0, 1), (2 * math.pi / 3, 8), (math.pi, 3)],
    "O": [(0, 1), (2 * math.pi / 3, 8), (math.pi, 3), (math.pi / 2, 6), (math.pi, 6)],
    "I": [(0, 1), (2 * math.pi / 5, 12), (4 * math.pi / 5, 12), (2 * math.pi / 3, 20), (math.pi, 15)],
}


def invariant_dims(group, lmax=12):
    """Number of G-invariant spherical harmonics of each degree l (character formula)."""
    cl = CLASSES[group]
    order = sum(n for _, n in cl)
    out = []
    for l in range(lmax + 1):
        tot = 0.0
        for th, n in cl:
            tot += n * ((2 * l + 1) if th == 0 else math.sin((2 * l + 1) * th / 2) / math.sin(th / 2))
        out.append(int(round(tot / order)))
    return out


def fig6():
    fam = {r["key"]: r for r in load("family.json")}
    rows = [("tetra-4", "T", "tetra-4 ($T_d$)", BLUE), ("trionda", "T", "Trionda ($T$)", BLUE),
            ("cube-6", "O", "cube-6 ($O_h$)", ORANGE), ("cube-6-chiral", "O", "cube-6-chiral ($O$)", ORANGE),
            ("classic-32", "I", "classic-32 ($I_h$)", AQUA), ("icosa-20-chiral", "I", "icosa-20-chiral ($I$)", AQUA),
            ("goldberg-42-chiral", "I", "goldberg-42-chiral ($I$)", AQUA)]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for r, (key, grp, label, col) in enumerate(rows):
        y = len(rows) - 1 - r
        dims = invariant_dims(grp)
        q = np.array(fam[key]["q"])
        for l in range(1, 13):
            if dims[l] > 0:
                ax.scatter(l, y, marker="s", s=150, facecolor="none", edgecolor=GRID if False else MUTED, linewidth=0.7)
            if q[l] > 1e-9:
                ax.scatter(l, y, s=22 + 26 * np.clip(np.log10(q[l]) + 3, 0, 3.5), color=col, zorder=3)
                if dims[l] == 0:
                    ax.scatter(l, y, marker="x", s=90, color="red", zorder=4)   # would contradict the theory
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[2] for r in rows][::-1], fontsize=8)
    ax.set_xticks(range(1, 13)); ax.set_xlim(0.4, 12.6); ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("multipole degree $\\ell$")
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(handles=[Line2D([], [], marker="s", mfc="none", mec=MUTED, ls="", ms=9, mew=0.7),
                       Line2D([], [], marker="o", color=MUTED, ls="", ms=5)],
              labels=["invariant harmonic of degree $\\ell$ exists", "seam density has $Q_\\ell>10^{-9}$ (size $\\propto\\log Q_\\ell$)"],
              loc="upper center", bbox_to_anchor=(0.45, 1.2), ncol=2, frameon=False, fontsize=7.5, columnspacing=1.2)
    save(fig, "fig6_multipoles")


# --------------------------------------------------------------------------- fig7: exposure maps
def fig7():
    names = [("classic-32", "classic-32 ($I_h$)"), ("cube-6", "cube-6 ($O_h$)"), ("tetra-4", "tetra-4 ($T_d$)"), ("trionda", "Trionda model ($T$)")]
    lon = np.linspace(-math.pi, math.pi, 181); lat = np.linspace(-math.pi / 2, math.pi / 2, 91)
    LO, LA = np.meshgrid(lon, lat)
    axes_n = np.stack([np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], -1).reshape(-1, 3)
    maps = []
    for key, _ in names:
        g = bm.exposure(balls.get_ball(key), axes_n, samples=120)
        maps.append(g.reshape(LO.shape) / g.mean())
        print("   map", key, maps[-1].min(), maps[-1].max(), flush=True)
    vmin, vmax = min(m.min() for m in maps), max(m.max() for m in maps)
    fig = plt.figure(figsize=(7.2, 4.4))
    for i, ((key, title), m) in enumerate(zip(names, maps)):
        ax = fig.add_subplot(2, 2, i + 1, projection="mollweide")
        mesh = ax.pcolormesh(LO, LA, m, cmap="Blues", vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.grid(True, color="white", lw=0.3, alpha=0.6)
        ax.set_title(title, fontsize=8.5, loc="left")
    cax = fig.add_axes([0.93, 0.2, 0.014, 0.6])
    cb = fig.colorbar(mesh, cax=cax); cb.set_label("$G(n)/\\bar G$", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.02, right=0.91, wspace=0.05, hspace=0.12)
    save(fig, "fig7_exposure_maps")


# --------------------------------------------------------------------------- fig8: gallery
def fig8():
    keys = ["tetra-4", "trionda", "cube-6", "cube-6-chiral", "octa-8", "octa-8-chiral",
            "panel-12", "dodeca-12-chiral", "icosa-20", "icosa-20-chiral", "classic-32", "goldberg-42", "goldberg-42-chiral"]
    fig, axes = plt.subplots(3, 5, figsize=(8.6, 5.3))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, k in zip(axes.ravel(), keys):
        b = balls.get_ball(k)
        ax.imshow(rs.render_view(280, ball=b, elevation=22, azimuth=25, ssaa=2, background=(1, 1, 1)))
        ax.set_title(f"{k} ({b.n_panels})", fontsize=8)
    fig.subplots_adjust(wspace=0.02, hspace=0.12)
    save(fig, "fig8_gallery")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    todo = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8"]
    for name in todo:
        globals()[name]()
