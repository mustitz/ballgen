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

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from spectre.patch import SpectrePatch
from spectre.search import PatchPool
from spectre.dp import can_close, fold_with_holes
from spectre.embed import embed, embed_assembly
from spectre.connect import best_join, join_many, join_spheres

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


# ---------------------------------------------------------------------------------------------- assemblies
PALETTES = [WARM, COOL, ["#a9c98d", "#c4dba8", "#8fb87a", "#b7d19b", "#9cc08a", "#cfe3b5"],
            ["#c3a6d9", "#d6c0e6", "#b08fcb", "#cdb5e0", "#bda0d6", "#dccbea"]]
LIGHT = np.array([-0.45, -0.55, 0.7]) / np.linalg.norm([-0.45, -0.55, 0.7])


def two_hole_folds(pool, count=14, seed=2, seconds=90, thetas=None):
    sq = [240, 270, 240, 270]
    rng = random.Random(seed)
    out, t0 = [], time.time()
    while time.time() - t0 < seconds and len(out) < count:
        g = pool.graph(pool.random_subset(rng.randint(8, 12), rng, 0.3))
        if g is None:
            continue
        r = fold_with_holes(g, thetas or [sq, sq])
        if r:
            out.append(r)
    return out


def three_hole_fold(pool, seconds=150):
    sq = [240, 270, 240, 270]
    rng = random.Random(3)
    t0 = time.time()
    while time.time() - t0 < seconds:
        g = pool.graph(pool.random_subset(rng.randint(9, 16), rng, rng.choice([0.0, 0.3, 0.7])))
        if g is None:
            continue
        r = fold_with_holes(g, [sq] * 3)
        if r:
            return r
    raise RuntimeError("no 3-hole fold found")


def pick_chain(folds, k, cyclic=False):
    """Order k distinct two-hole folds so that hole 1 of each joins hole 0 of the next with matching edge kinds."""
    def ok(a, b):
        return best_join(a, a.holes[1], b, b.holes[0], True) is not None

    def go(path):
        if len(path) == k:
            return path if (not cyclic or ok(folds[path[-1]], folds[path[0]])) else None
        for i in range(len(folds)):
            if i not in path and ok(folds[path[-1]], folds[i]):
                r = go(path + [i])
                if r:
                    return r
        return None

    for s in range(len(folds)):
        r = go([s])
        if r:
            return [folds[i] for i in r]
    return None


def draw_assembly(ax, ae, elev, azim, zoom=1.4, focus=None, half=None):
    js, P = ae.surface, ae.position
    cols = []
    ntile = [r.graph.n_tiles for r in js.parts]
    base = []
    for p in range(len(js.parts)):
        pal = PALETTES[p % len(PALETTES)]
        rng = np.random.default_rng(10 + p)
        base.append([to_rgb(pal[i % len(pal)]) for i in rng.permutation(ntile[p] * 2)[:ntile[p]]])
    faces = P[ae.triangles]
    nrm = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    cen = faces.mean(1)
    out_sign = np.sign(np.einsum("ij,ij->i", nrm, cen - ae.part_centre[ae.triangle_part]))
    out_sign[out_sign == 0] = 1
    nrm *= out_sign[:, None]
    camv = np.array([math.cos(math.radians(elev)) * math.cos(math.radians(azim)),
                     math.cos(math.radians(elev)) * math.sin(math.radians(azim)), math.sin(math.radians(elev))])
    right = np.cross([0, 0, 1.0], camv)
    right /= max(np.linalg.norm(right), 1e-9)
    light = camv - 0.7 * right + 0.8 * np.array([0, 0, 1.0])
    light /= np.linalg.norm(light)
    shade = 0.5 + 0.5 * np.clip(nrm @ light, 0, 1)
    for i in range(len(faces)):
        c = np.array(base[ae.triangle_part[i]][ae.triangle_tile[i]])
        cols.append(tuple(np.clip(c * shade[i], 0, 1)))
    ax.add_collection3d(Poly3DCollection(faces, facecolors=cols, edgecolors="none"))
    segs = []
    for (_p, cls) in ae.tile_polys:
        pts = P[cls]
        for k in range(len(pts)):
            a, b = pts[k], pts[(k + 1) % len(pts)]
            if np.linalg.norm(a - b) > 1e-9:
                segs.append((a, b))
    ax.add_collection3d(Line3DCollection(segs, colors="#ffffff", linewidths=0.5, alpha=0.55))
    for rim in ae.rims:
        r = np.vstack([P[rim], P[rim][:1]])
        ax.plot(r[:, 0], r[:, 1], r[:, 2], color=INK, lw=3.4, solid_capstyle="round", zorder=10)
    for (_pi, _hi, cls) in ae.open_loops:
        lp = np.vstack([P[cls], P[cls][:1]])
        ax.plot(lp[:, 0], lp[:, 1], lp[:, 2], color="#b3261e", lw=3.0, zorder=10)
    _frame(ax, P, elev, azim)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    if focus is not None:
        ax.set_xlim(focus[0] - half, focus[0] + half); ax.set_ylim(focus[1] - half, focus[1] + half)
        ax.set_zlim(focus[2] - half, focus[2] + half)


def side_views(P):
    """Camera angles perpendicular to the long axis of the assembly (so the parts do not hide each other)."""
    c = P - P.mean(0)
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    ax = vt[0]
    az0 = math.degrees(math.atan2(ax[1], ax[0]))
    el_ax = math.degrees(math.asin(np.clip(ax[2], -1, 1)))
    return ((14, az0 + 90), (14 + abs(el_ax) * 0 + 40, az0 + 90 + 55), (-10, az0 - 90 + 40))


def fig_assembly(js, path, title, note, views=None, seam_zoom=True, W=0.4, zoom=1.5):
    ae = embed_assembly(js, sphere_weight=W)
    views = views or side_views(ae.position)[:2]
    nv = len(views) + (1 if seam_zoom else 0)
    fig = plt.figure(figsize=(5.4 * nv, 5.4))
    for k, (el, az) in enumerate(views):
        ax = fig.add_subplot(1, nv, k + 1, projection="3d")
        draw_assembly(ax, ae, el, az, zoom=zoom)
    if seam_zoom:
        ax = fig.add_subplot(1, nv, nv, projection="3d")
        c = ae.position[ae.rims[0]].mean(0)
        draw_assembly(ax, ae, 22, 35, zoom=1.5, focus=c, half=3.0)
        ax.set_title("close-up of the first seam", fontsize=10, color=INK, pad=-2)
    fig.suptitle(title, fontsize=12.5, color=INK, y=0.975)
    fig.text(0.5, 0.03, note + f"  |  ledger {js.ledger} deg, 3D strain rms {100 * ae.rms_strain:.1f}%", ha="center",
             fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.93, bottom=0.07, wspace=0.0)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return ae


def make_assemblies(pool, only=None):
    folds = two_hole_folds(pool)
    print(len(folds), "two-hole folds")
    done = {}
    if only in (None, "joined"):
        pair = pick_chain(folds, 2)
        js = join_many(pair, [(0, 1, 1, 0)], True)
        ae = fig_assembly(js, os.path.join(OUT, "joined_spheres.png"),
                          "Two folded Spectre spheres sewn together through a hole each",
                          "black = seam (saddle neck), red = open holes (K=60 deg each), shading shows the creases")
        print(js.summary(), "\nstrain", ae.strain, ae.rms_strain)
    if only in (None, "chain"):
        ch = pick_chain(folds, 4)
        if ch is None:
            ch = pick_chain(folds, 3)
        js = join_many(ch, [(i, 1, i + 1, 0) for i in range(len(ch) - 1)], True)
        fig_assembly(js, os.path.join(OUT, "sphere_chain.png"),
                     f"A chain of {len(ch)} folded Spectre spheres",
                     "black = seams, red = the two open ends (each K=60 deg)", seam_zoom=False, zoom=1.15)
        print("chain", len(ch), js.ledger)
    if only in (None, "star"):
        hub = three_hole_fold(pool)
        sats, used = [], set()
        for f in folds:
            if len(sats) == 3:
                break
            k = len(sats)
            if best_join(hub, hub.holes[k], f, f.holes[0], True) is not None:
                sats.append(f)
        if len(sats) == 3:
            js = join_many([hub] + sats, [(0, k, k + 1, 0) for k in range(3)], True)
            fig_assembly(js, os.path.join(OUT, "sphere_star.png"),
                         "A hub sphere with three holes, each joined to a satellite sphere",
                         "black = seams, red = open holes left on the satellites",
                         seam_zoom=False, zoom=1.15)
            print("star", js.ledger)
        else:
            print("star: no kind-compatible satellites")
    if only == "ring":      # kept for experiments: the closing seam of a ring comes out crumpled in 3D
        rg = pick_chain(folds, 4, cyclic=True) or pick_chain(folds, 3, cyclic=True)
        if rg is None:
            print("ring: none")
        else:
            k = len(rg)
            js = join_many(rg, [(i, 1, (i + 1) % k, 0) for i in range(k)], True)
            fig_assembly(js, os.path.join(OUT, "sphere_ring.png"),
                         f"A ring of {k} folded Spectre spheres (a torus, total curvature {js.ledger} deg)",
                         "black = seams; no open holes left (closing seam is approximate)",
                         seam_zoom=False, zoom=1.15, W=0.12)
            print("ring", k, js.ledger)


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
    ap.add_argument("--only", choices=["joined", "chain", "star", "ring", "closed", "rate"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pool = PatchPool(SpectrePatch(levels=3))
    if a.only in (None, "joined", "chain", "star", "ring"):
        make_assemblies(pool, a.only)
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
