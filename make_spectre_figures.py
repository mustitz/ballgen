#!/usr/bin/env python3
"""
Figures for the Spectre folding work (written to spectre/figures/):

  joined_spheres.png   two folded spheres sewn together through a hole each (saddle neck at the seam)
  closed_balls.png     gallery of patches that fold into a closed ball with no hole at all
  closure_rate.png     fraction of random N-tile patches whose whole boundary can be glued shut (exact DP)

    python make_spectre_figures.py            # all three (survey ~1-2 min, cached in spectre/figures/survey.json)
    python make_spectre_figures.py --fresh    # redo the survey
"""
import argparse
import json
import os
import random
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from spectre.patch import SpectrePatch
from spectre.search import PatchPool
from spectre.dp import can_close, fold_with_holes
from spectre.embed import embed, embed_joined
from spectre.connect import best_join, join_spheres

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spectre", "figures")
INK, MUTED = "#222222", "#6b6b6b"
WARM = ["#e8a87c", "#f2c4a0", "#d98e73", "#f0b6a8", "#e6b980", "#d9a58f"]
COOL = ["#7fb3d5", "#a8cde0", "#6aa6a8", "#94b8d6", "#8fc1b5", "#7d9fcf"]
MIXED = ["#e8a87c", "#7fb3d5", "#a9c98d", "#d98e9e", "#c3a6d9", "#e6cf7a", "#8fc1b5"]


def _tile_faces(tris, tile_ids, palette, seed):
    rng = np.random.default_rng(seed)
    col = [to_rgb(palette[i % len(palette)]) for i in rng.permutation(len(tile_ids) * 2)[:len(tile_ids)]]
    return [col[t] for t in tile_ids]


def _outlines(P, em, ax, color="#ffffff", lw=0.7):
    g = em.result.graph
    segs = []
    for lt in range(g.n_tiles):
        cls = em.vertex_class[g.tile_vertex[lt]]
        pts = P[cls]
        for k in range(len(pts)):
            a, b = pts[k], pts[(k + 1) % len(pts)]
            if np.linalg.norm(a - b) > 1e-9:
                segs.append((a, b))
    ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=lw, alpha=0.9))


def _frame(ax, P, elev, azim):
    c = (P.max(0) + P.min(0)) / 2
    h = (P.max(0) - P.min(0)).max() / 2 * 1.02
    ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_zlim(c[2] - h, c[2] + h)
    ax.set_box_aspect((1, 1, 1), zoom=1.4)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


# ---------------------------------------------------------------------------------------------- joined
def find_joinable(pool, seed=2, seconds=60):
    sq = [240, 270, 240, 270]
    rng = random.Random(seed)
    folds, t0 = [], time.time()
    while time.time() - t0 < seconds and len(folds) < 12:
        g = pool.graph(pool.random_subset(rng.randint(8, 12), rng, 0.3))
        if g is None:
            continue
        r = fold_with_holes(g, [sq, sq])
        if r:
            folds.append(r)
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            if best_join(folds[i], folds[i].holes[0], folds[j], folds[j].holes[0], True):
                return folds[i], folds[j]
    raise RuntimeError("no joinable pair found")


def fig_joined(pool, path):
    a, b = find_joinable(pool)
    js = join_spheres(a, 0, b, 0, match_kinds=True)
    je = embed_joined(js)
    P = je.position
    fig = plt.figure(figsize=(15, 5.6))
    seam_c = P[je.rim].mean(0)
    for col, (elev, azim, zoom) in enumerate(((16, 20, None), (60, -65, None), (22, 35, 3.4))):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        cols = []
        for part, pal, seed in ((0, WARM, 1), (1, COOL, 2)):
            tc = _tile_faces(None, range(js.parts[part].graph.n_tiles), pal, seed)
            idx = np.where(je.triangle_part == part)[0]
            cols += [(i, tc[je.triangle_tile[i]]) for i in idx]
        cols = [c for _, c in sorted(cols, key=lambda t: t[0])]
        ax.add_collection3d(Poly3DCollection([P[t] for t in je.triangles], facecolors=cols, edgecolors="none"))
        # tile outlines (parts have their own embedding objects: draw from triangle tile edges)
        for part, em_res in enumerate(js.parts):
            pass
        rim = np.vstack([P[je.rim], P[je.rim][:1]])
        ax.plot(rim[:, 0], rim[:, 1], rim[:, 2], color=INK, lw=4.2, solid_capstyle="round", zorder=10)
        for (_pi, _hi, cls) in je.open_loops:
            lp = np.vstack([P[cls], P[cls][:1]])
            ax.plot(lp[:, 0], lp[:, 1], lp[:, 2], color="#b3261e", lw=3.6, zorder=10)
        _frame(ax, P, elev, azim)
        if zoom:
            h = zoom
            ax.set_xlim(seam_c[0] - h, seam_c[0] + h); ax.set_ylim(seam_c[1] - h, seam_c[1] + h)
            ax.set_zlim(seam_c[2] - h, seam_c[2] + h)
            ax.set_title("close-up of the seam", fontsize=10, color=INK, pad=-2)
    fig.suptitle("Two folded Spectre spheres sewn together through a hole each", fontsize=12, color=INK, y=0.97)
    fig.text(0.5, 0.045,
             "black seam = joined 4-gon holes (junction vertices are saddles, deficit "
             + ", ".join(str(d) for d in js.junction_deficits[:2]) + " deg);  red = the remaining open holes (K=60 deg each);  "
             f"ledger {js.ledger} deg",
             ha="center", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.93, bottom=0.08, wspace=0.0)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return js, je


# ---------------------------------------------------------------------------------------------- closed
def find_closed(pool, N, rng, tries=4000):
    for _ in range(tries):
        g = pool.graph(pool.random_subset(N, rng, rng.choice([0.0, 0.3, 0.7])))
        if g is not None and can_close([int(x) for x in g.loops[0][1]]):
            return g
    return None


def fig_closed(pool, path, Ns=(6, 8, 10, 12, 19, 21)):
    rng = random.Random(7)
    fig = plt.figure(figsize=(13, 8.4))
    for k, N in enumerate(Ns):
        g = find_closed(pool, N, rng)
        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        if g is None:
            ax.set_axis_off(); ax.set_title(f"N={N}: none found", fontsize=10)
            continue
        r = fold_with_holes(g, [])
        em = embed(r)
        P = em.position
        tc = _tile_faces(None, range(r.graph.n_tiles), MIXED, k)
        ax.add_collection3d(Poly3DCollection([P[t] for t in em.triangles],
                                             facecolors=[tc[i] for i in em.triangle_tile], edgecolors="none"))
        _outlines(P, em, ax, color="#ffffff", lw=0.9)
        cs = [c for c in r.cones if c.deficit != 0]
        if cs:
            pts = np.array([P[em.vertex_class[c.vertex]] for c in cs])
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=[10 + 0.55 * abs(c.deficit) for c in cs],
                       c=["#b3261e" if c.deficit > 0 else "#1f4e79" for c in cs], depthshade=False,
                       edgecolors="white", linewidths=0.6)
        _frame(ax, P, 22, 35 + 40 * k)
        ax.set_title(f"{r.graph.n_tiles} tiles, {len(cs)} cone points (of {len(r.cones)} vertices)",
                     fontsize=10.5, color=INK, pad=-4)
    fig.suptitle("Spectre patches that fold into a closed ball with no hole (exact gluing found by DP)",
                 fontsize=12.5, color=INK, y=0.985)
    fig.text(0.5, 0.012, "dots: cone points, area ~ angle deficit, sum of deficits = 720 deg (Gauss-Bonnet); "
             "white lines: tile edges; 3D shape is a relaxed estimate", ha="center", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.93, bottom=0.04, wspace=0.0, hspace=0.04)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- survey
def survey(pool, Ns, samples, seed=5):
    rng = random.Random(seed)
    out = {}
    for N in Ns:
        disc = closes = 0
        for _ in range(samples):
            g = pool.graph(pool.random_subset(N, rng, rng.choice([0.0, 0.3, 0.7])))
            if g is None:
                continue
            disc += 1
            closes += can_close([int(x) for x in g.loops[0][1]])
        out[N] = {"discs": disc, "closed": closes}
    return out


def fig_rate(stats, path):
    Ns = sorted(int(k) for k in stats)
    frac = [100 * stats[str(n)]["closed"] / max(1, stats[str(n)]["discs"]) for n in Ns]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.bar(Ns, frac, color="#4c78a8", width=0.72)
    for n, f in zip(Ns, frac):
        if f > 0:
            ax.text(n, f + 1.0, f"{f:.0f}%" if f >= 1 else "<1%", ha="center", fontsize=8, color=INK)
    ax.set_xlabel("tiles in the patch (random connected subsets of the level-3 Spectre patch)", color=INK)
    ax.set_ylabel("% that fold into a closed ball", color=INK)
    ax.set_xticks(Ns)
    ax.set_ylim(0, max(frac) * 1.15)
    ax.grid(axis="y", color="#e2e2e2", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("How often can a patch boundary be glued completely shut?", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--only", choices=["joined", "closed", "rate"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pool = PatchPool(SpectrePatch(levels=3))
    if a.only in (None, "joined"):
        js, je = fig_joined(pool, os.path.join(OUT, "joined_spheres.png"))
        print(js.summary()); print("embedding strain max %.2f rms %.3f" % (je.strain, je.rms_strain))
    if a.only in (None, "closed"):
        fig_closed(pool, os.path.join(OUT, "closed_balls.png"))
    if a.only in (None, "rate"):
        cache = os.path.join(OUT, "survey.json")
        if a.fresh or not os.path.exists(cache):
            stats = survey(pool, list(range(4, 25)), a.samples)
            json.dump(stats, open(cache, "w"), indent=1)
        stats = json.load(open(cache))
        fig_rate(stats, os.path.join(OUT, "closure_rate.png"))


if __name__ == "__main__":
    main()
