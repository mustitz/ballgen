#!/usr/bin/env python3
"""
Fold a patch of Spectre (Einstein monotile) tiles into a closed ball and record the holes.

    python fold_spectre.py                 # table for N = 5,9,15,25,40 + figures in spectre_out/
    python fold_spectre.py -n 15 -n 40     # chosen tile counts
    python fold_spectre.py -n 25 --beam 600

Per hole it writes: boundary loop (half-edge ids), side count and kind, angle deficit K (exact integer
degrees), spherical area / fraction and centre direction (estimates from the 3D relaxation).
Output: spectre_out/holes.json, spectre_out/holes.csv, spectre_out/ball_N*.png.
"""
import argparse
import csv
import json
import os

import numpy as np

from spectre.patch import SpectrePatch
from spectre.fold import PatchGraph, fold_search, grow_patch
from spectre.embed import embed
from spectre.connect import hole_kinds, signature
from spectre.dp import fold_with_holes
from spectre.search import PatchPool


def hole_records(em, N):
    out = []
    for k, h in enumerate(em.result.holes):
        geo = em.hole_geometry(h)
        out.append({
            "N": N, "hole": k, "sides": h.n, "kind": h.kind,
            "angle_deficit_K_deg": int(h.K), "closes_flat": bool(h.closes),
            "corner_angles_deg": [int(x) for x in h.corners],
            "boundary_half_edges": [int(e) for e in h.edges],
            "edge_kinds": hole_kinds(em.result, h),
            "signature": [list(map(int, signature(em.result, h)[0])), list(signature(em.result, h)[1])],
            "area_sr": round(geo["area_sr"], 4), "area_fraction": round(geo["area_fraction"], 4),
            "centre_xyz": [round(float(x), 4) for x in geo["centre"]],
            "approximate_geometry": True,
        })
    return out


def draw(em, N, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    r = em.result
    P = em.position
    fig = plt.figure(figsize=(11, 5))
    for col, elev in enumerate((20, -20)):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        rng = np.random.default_rng(3)
        tile_col = rng.uniform(0.55, 0.95, (r.graph.n_tiles, 3))
        faces = [P[t] for t in em.triangles]
        pc = Poly3DCollection(faces, facecolors=[tile_col[i] for i in em.triangle_tile],
                              edgecolors="none", linewidths=0)
        ax.add_collection3d(pc)
        for h in r.holes:
            v = P[em.hole_vertices(h)]
            v = np.vstack([v, v[:1]])
            ax.plot(v[:, 0], v[:, 1], v[:, 2], color="k", lw=2)
        lim = np.abs(P).max() * 1.05
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1), zoom=1.25); ax.view_init(elev=elev, azim=40 + 180 * col)
        ax.set_axis_off()
    hs = ", ".join(f"{h.n}-gon K={h.K}" for h in r.holes) or "none"
    fig.suptitle(f"{N} Spectre tiles folded; holes (black outline): {hs}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


NAMED = {"square-alt": [240, 270, 240, 270], "hexagon": [240] * 6, "pentagon": [240] * 5,
         "square": [270] * 4, "triangle": [300] * 3}


def parse_hole(s):
    return NAMED[s] if s in NAMED else [int(x) for x in s.split(",")]


def search_holes(patch, thetas, nmin, nmax, seconds, seed=0):
    """Random connected tile subsets until the exact DP finds a gluing with exactly these holes."""
    import random
    import time
    pool = PatchPool(patch)
    rng = random.Random(seed)
    t0 = time.time()
    tried = 0
    while time.time() - t0 < seconds:
        tiles = pool.random_subset(rng.randint(nmin, nmax), rng, compact=rng.choice([0.0, 0.3, 0.7]))
        g = pool.graph(tiles)
        if g is None:
            continue
        tried += 1
        r = fold_with_holes(g, thetas)
        if r is not None:
            print(f"found after {tried} patches: {g.n_tiles} tiles")
            return g, r
    print(f"nothing found in {tried} patches")
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hole", action="append", help="target hole: a name (%s) or comma-separated corner angles; "
                    "repeat for several holes; --hole none asks for a closed ball" % ", ".join(NAMED))
    ap.add_argument("--nmin", type=int, default=6)
    ap.add_argument("--nmax", type=int, default=14)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("-n", "--tiles", type=int, action="append")
    ap.add_argument("--beam", type=int, default=300)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--out", default="spectre_out")
    a = ap.parse_args()
    Ns = a.tiles or [5, 9, 15, 25, 40]
    os.makedirs(a.out, exist_ok=True)
    patch = SpectrePatch(levels=a.levels)
    recs = []
    if a.hole:
        thetas = [] if a.hole == ["none"] else [parse_hole(h) for h in a.hole]
        g, r = search_holes(patch, thetas, a.nmin, a.nmax, a.seconds)
        if r is None:
            return
        print(r.summary())
        em = embed(r)
        recs = hole_records(em, g.n_tiles)
        draw(em, g.n_tiles, os.path.join(a.out, f"ball_{g.n_tiles}tiles_{len(r.holes)}holes.png"))
        with open(os.path.join(a.out, "holes_targeted.json"), "w") as fh:
            json.dump(recs, fh, indent=1)
        return
    print(f"{'N':>4} {'cones':>5} {'holes':>5} {'sides':>5} {'K':>5} {'area%':>6} {'strain rms':>10}  kind")
    for N in Ns:
        g = PatchGraph(patch, tiles=grow_patch(patch, N))
        r = fold_search(g, beam=a.beam)
        em = embed(r)
        rs = hole_records(em, N)
        recs += rs
        for x in rs:
            print(f"{N:>4} {len(r.cones):>5} {len(r.holes):>5} {x['sides']:>5} {x['angle_deficit_K_deg']:>5} "
                  f"{100 * x['area_fraction']:>6.1f} {em.rms_strain:>10.3f}  {x['kind']}")
        draw(em, N, os.path.join(a.out, f"ball_N{N}.png"))
    with open(os.path.join(a.out, "holes.json"), "w") as f:
        json.dump(recs, f, indent=1)
    cols = ["N", "hole", "sides", "kind", "angle_deficit_K_deg", "closes_flat", "area_fraction", "area_sr"]
    with open(os.path.join(a.out, "holes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader(); w.writerows(recs)


if __name__ == "__main__":
    main()
