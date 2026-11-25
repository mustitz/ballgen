#!/usr/bin/env python3
"""
Trionda  versus  the Hat Trick: what they share, how to compare them, and two ways to mix them.

    python compare_trionda_hat.py                 # everything -> compare_out/*.png, compare_out/compare.json
    python compare_trionda_hat.py --only table    # one stage: table, views, spectra, exposure, assign, flow, ledger

Both balls are partitions of the sphere by rotation groups, and the tetrahedral rotation group T (order 12) is a
subgroup of the icosahedral group I (order 60).  So the Trionda (T, 4 panels) and the Hat Trick (I, 60 hats + 12
pentagons) can be put in the same frame, sharing their 3-fold and 2-fold axes, and measured with the same tools
(`ball_metrics.py`: seam multipoles, flow-axis exposure, seam bending).

Stages
  table     numbers for both balls (panels, group, seam length, lowest multipole, exposure, bending, panel shape)
  views     side-by-side pictures along the shared 3-fold and 2-fold axes
  spectra   seam multipole power Q_l: T-invariant harmonics versus I-invariant harmonics
  exposure  flow-axis exposure maps
  assign    "Hat-Trionda": each Trionda panel redrawn as 15 hats + 3 pentagons (T-equivariant assignment)
  flow      the Hat Trick pushed through the Trionda flow for flow time tau: symmetry I -> T
  ledger    the curvature ledger of both flat models (cone points)
"""
import argparse
import itertools
import json
import math
import os

import numpy as np

import ball_metrics as bm
import balls
import the_hat_trick_ball as H

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_out")
TRIONDA_RED, TRIONDA_GREEN, TRIONDA_BLUE, TRIONDA_WHITE = "#d52b1e", "#0a7d4b", "#1f55b5", "#f2f2f2"
PANEL_COLOURS = ["#d52b1e", "#0a7d4b", "#1f55b5", "#e8b923"]          # the four tetrahedral panels


# ---------------------------------------------------------------------------------------------
# the two balls in one frame
# ---------------------------------------------------------------------------------------------

class NetBall:
    """A seam network given as polygons on the sphere, with just enough of the ``balls.Ball`` interface for
    ``ball_metrics`` (face_vertices, twist).  An optional warp (e.g. the Trionda flow) bends the seams."""

    def __init__(self, polys, name, warp=None):
        self._polys = [np.asarray(p, float) for p in polys]
        self.name, self.warp = name, warp
        self.n_panels = len(self._polys)

    def face_vertices(self):
        return self._polys

    def twist(self, p, inverse=False):
        return p if self.warp is None else self.warp(p, inverse)


def hat_ball(variant=0):
    """The Hat Trick, relaxed, with its icosahedral frame."""
    ball = H.HatTrick(variant=variant)
    ids, X, a, info = H.relax(ball)
    polys = [X[[ids[c] for c in ball.hat_outline(i)[0]]] for i in range(60)]
    polys += [X[[ids[c] for c in ball.pentagon_loop(p)]] for p in ball.pent]
    return ball, ids, X, info, polys


def tetra_in_icosa():
    """A regular tetrahedron whose 4 vertices are 3-fold axes of the (built-in) icosahedron, and the proper
    rotation R taking the Trionda frame (balls.TETRA_VERTICES) onto it.  T is a subgroup of I."""
    V0, F0 = H.icosahedron()
    cen = [V0[list(f)].sum(0) for f in F0]
    cen = [c / np.linalg.norm(c) for c in cen]
    for comb in itertools.combinations(range(20), 4):
        P = np.array([cen[i] for i in comb])
        G = P @ P.T
        if np.allclose(G[np.triu_indices(4, 1)], -1 / 3, atol=1e-6):
            for perm in itertools.permutations(range(4)):
                A = balls.TETRA_VERTICES                       # rows are source directions
                B = P[list(perm)]
                U, _, Vt = np.linalg.svd(A.T @ B)
                R = (U @ Vt).T                                 # R @ a = b
                if np.linalg.det(R) > 0 and np.allclose(A @ R.T, B, atol=1e-6):
                    return R, P
    raise RuntimeError("no tetrahedron inside the icosahedron")


class Rotated:
    """A ``Ball`` seen in a rotated frame: real point p  <->  ball point R^T p."""

    def __init__(self, ball, R):
        self.ball, self.R = ball, R
        self.n_panels = ball.n_panels

    def panel_index(self, p):
        return self.ball.panel_index(p @ self.R)               # (R^T p)^T = p^T R

    def colorize(self, p, **kw):
        return self.ball.colorize(p @ self.R, **kw)


def view_basis(d):
    d = np.asarray(d, float) / np.linalg.norm(d)
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(up, d)
    x /= np.linalg.norm(x)
    return x, np.cross(d, x), d


def raster(ball, view, n=520):
    """Orthographic image (RGBA) of a ``Ball`` (or ``Rotated``) seen from direction ``view``."""
    ex, ey, ez = view_basis(view)
    xs = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xs, -xs)
    r2 = X ** 2 + Y ** 2
    inside = r2 < 1
    Z = np.sqrt(np.clip(1 - r2, 0, 1))
    P = X[..., None] * ex + Y[..., None] * ey + Z[..., None] * ez
    img = np.ones((n, n, 4))
    pts = P[inside]
    col = ball.colorize(pts, aa_rad=2.0 / n)
    L = np.array([0.35, 0.55, 1.0])
    L = L[0] * ex + L[1] * ey + L[2] * ez
    L /= np.linalg.norm(L)
    sh = 0.55 + 0.45 * np.clip(pts @ L, 0, 1)
    img[inside, :3] = np.clip(col * sh[:, None], 0, 1)
    img[~inside, 3] = 0
    return img


def shared_axes(R):
    """3-fold and 2-fold axes common to the Trionda and the Hat Trick (the axes of T)."""
    three = R @ balls.TETRA_VERTICES[0]
    two = R @ np.array([1.0, 0, 0])
    return {"3-fold": three, "2-fold": two}


# ---------------------------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------------------------

def panel_shape(polys, samples=60):
    """Spherical isoperimetric quotient P^2 / (A (4 pi - A)) of every polygon (1 for a cap, larger = less round)."""
    out = []
    for V in polys:
        k = len(V)
        L = sum(math.acos(np.clip(V[i] @ V[(i + 1) % k], -1, 1)) for i in range(k))
        A = H.spherical_area(V)
        out.append(L ** 2 / (A * (4 * math.pi - A)))
    return np.array(out)


def trionda_panel_shape(tau=balls.TRIONDA_TAU, samples=2000):
    """Same quotient for one Trionda panel (curved boundary): sample its three seams."""
    b = balls.make_trionda(tau)
    V = b.face_vertices()[0]
    k = len(V)
    L = 0.0
    for i in range(k):
        a, c = V[i], V[(i + 1) % k]
        om = math.acos(np.clip(a @ c, -1, 1))
        t = np.linspace(0, 1, samples)[:, None]
        arc = b.twist((np.sin((1 - t) * om) * a + np.sin(t * om) * c) / math.sin(om), inverse=True)
        L += np.linalg.norm(np.diff(arc, axis=0), axis=1).sum()
    A = math.pi
    return L ** 2 / (A * (4 * math.pi - A)), L


def metrics(ball, n_axes=2000, max_points=20000):
    """The paper's seam metrics.  The number of samples per edge is chosen so that no ball needs more than
    ``max_points`` seam samples (the exposure matrix is n_axes x points)."""
    edges = sum(len(V) for V in ball.face_vertices())
    s = max(8, max_points // edges)
    q = bm.multipole_spectrum(ball, lmax=14, samples=max(s, 100))
    ex = bm.exposure_stats(ball, n_axes=n_axes, samples=s, chunk=250)
    bd = bm.bending(ball, samples=max(s, 60))
    return {"panels": ball.n_panels, "seam_length": bm.seam_length(ball, samples=max(s, 100)),
            "lowest_multipole": bm.lowest_multipole(q), "exposure_cv": ex["cv"],
            "exposure_min_over_mean": ex["min_over_mean"], "exposure_max_over_mean": ex["max_over_mean"],
            "mean_abs_geodesic_curvature": bd["mean_abs_kg"], "Q": [float(x) for x in q]}


def table(ctx):
    rows = {}
    trio = balls.make_trionda()
    tet = balls.tetra_4()
    rows["trionda"] = metrics(trio)
    rows["trionda"]["group"] = "T (12)"
    rows["trionda"]["panel_shape"] = trionda_panel_shape()[0]
    rows["tetra-4"] = metrics(tet)
    rows["tetra-4"]["group"] = "T_d (24)"
    for v in (0, 1):
        ball, ids, X, info, polys = ctx["hat"][v]
        net = NetBall(polys, f"hat-trick-v{v}")
        m = metrics(net)
        m["group"] = "I (60)"
        m["panel_shape_hat"] = float(panel_shape(polys[:60]).mean())
        m["panel_shape_pentagon"] = float(panel_shape(polys[60:]).mean())
        rows[f"hat-trick-v{v}"] = m
    for name in ("classic-32", "goldberg-42", "icosa-20"):
        key = name.replace("-", "_")
        fn = getattr(balls, key, None)
        if fn:
            rows[name] = metrics(fn())
    return rows


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def draw_hat(ax, ball, ids, X, view, colour_of_hat=None, pentagon_colour="#2b2b2b", warp=None, lines=True,
             faded=None):
    """Vector view of the hat ball.  colour_of_hat: callable(hat index) -> colour; warp: panel space -> real."""
    from matplotlib.collections import LineCollection, PolyCollection
    import matplotlib.colors as mc
    ex, ey, ez = view_basis(view)
    L = np.array([0.35, 0.55, 1.0])
    L = L[0] * ex + L[1] * ey + L[2] * ez
    L /= np.linalg.norm(L)
    W = (lambda q: warp(q, True)) if warp is not None else (lambda q: q)
    polys, cols = [], []

    def add(q, base):
        q = W(q)
        nrm = q.mean(0)
        nrm /= np.linalg.norm(nrm)
        if nrm @ ez <= 0:
            return
        polys.append(np.c_[q @ ex, q @ ey])
        cols.append(np.clip(np.array(mc.to_rgb(base)) * (0.55 + 0.45 * max(0.0, nrm @ L)), 0, 1))

    for k in ball.S.kites:
        P = X[[ids[c] for c in ball.S.corners(k)]]
        h = ball.hat_of[k]
        base = colour_of_hat(h) if colour_of_hat else "#9fb6cd"
        for q in H._kite_grid(P, 3):
            add(q, base)
    for p in ball.pent:
        Q = X[[ids[c] for c in ball.pentagon_loop(p)]]
        c = Q.mean(0)
        c /= np.linalg.norm(c)
        for i in range(len(Q)):
            for q in H._kite_grid([c, Q[i], Q[(i + 1) % len(Q)], c], 2):
                add(q, pentagon_colour)
    ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors=cols, linewidths=0.3))
    if lines:
        segs = []
        loops = [X[[ids[c] for c in ball.hat_outline(i)[0]]] for i in range(60)]
        loops += [X[[ids[c] for c in ball.pentagon_loop(p)]] for p in ball.pent]
        for Q in loops:
            for j in range(len(Q)):
                arc = W(H._arc(Q[j], Q[(j + 1) % len(Q)], 8))
                if arc.mean(0) @ ez > 0.02:
                    segs.append(np.c_[arc @ ex, arc @ ey])
        ax.add_collection(LineCollection(segs, colors="#101010", linewidths=0.9))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="#101010", lw=1.0)
    ax.set_xlim(-1.03, 1.03); ax.set_ylim(-1.03, 1.03); ax.set_aspect("equal"); ax.axis("off")


def fig_views(ctx, path):
    plt = _plt()
    ball, ids, X, info, polys = ctx["hat"][0]
    R = ctx["R"]
    trio = Rotated(balls.make_trionda(), R)
    axes_ = shared_axes(R)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10.5))
    for j, (name, d) in enumerate(axes_.items()):
        axes[0, j].imshow(raster(trio, d), extent=(-1, 1, -1, 1))
        axes[0, j].set_title(f"Trionda  (T, 4 panels), along a {name} axis", fontsize=10)
        axes[0, j].axis("off")
        pc = H.patch_colours(ball)[0]
        draw_hat(axes[1, j], ball, ids, X, d, colour_of_hat=lambda h: H.PALETTE[pc[h]])
        axes[1, j].set_title(f"Hat Trick  (I, 60 hats + 12 pentagons), along a {name} axis", fontsize=10)
    fig.suptitle("Same frame, shared axes: T is a subgroup of I", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def chirality_index(ball, improper, n=4000):
    """How far a seam network is from its own mirror image: the mean angular distance (degrees) from a seam
    point to the seam network after the best improper isometry in ``improper`` (list of 3x3 matrices).
    0 means the network is mirror symmetric (achiral); larger means more chiral."""
    from scipy.spatial import cKDTree
    p, w = bm.seam_curve_samples(ball, samples=max(8, n // max(1, sum(len(V) for V in ball.face_vertices()))))
    tree = cKDTree(p)
    best = None
    for g in improper:
        d, _ = tree.query(p @ g.T)
        ang = 2 * np.arcsin(np.clip(d / 2, 0, 1))
        val = float((w * ang).sum() / w.sum())
        best = val if best is None else min(best, val)
    return math.degrees(best)


def improper_cube():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            g = np.zeros((3, 3))
            for i, j in enumerate(perm):
                g[i, j] = signs[i]
            if np.linalg.det(g) < 0:
                out.append(g)
    return out


def improper_icosa():
    return [-g for g in H.rotation_group()]


def chirality_table(ctx):
    rows = {}
    rows["tetra-4"] = chirality_index(balls.tetra_4(), improper_cube())
    for tau in (0.3, 0.6, 0.9):
        rows[f"trionda tau={tau}"] = chirality_index(balls.make_trionda(tau), improper_cube())
    for v in (0, 1):
        rows[f"hat-trick-v{v}"] = chirality_index(NetBall(ctx["hat"][v][4], "hat"), improper_icosa())
    rows["goldberg-42"] = chirality_index(balls.goldberg_42(), improper_icosa())
    rows["goldberg-42-chiral"] = chirality_index(balls.goldberg_42_chiral(), improper_icosa())
    return rows


def fig_spectra(ctx, tab, path):
    plt = _plt()
    names = [("trionda", "Trionda (T)"), ("tetra-4", "tetra-4 (T$_d$)"), ("hat-trick-v0", "Hat Trick (I)")]
    L = np.arange(1, 15)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    floor = 1e-14
    w = 0.27
    for k, (key, label) in enumerate(names):
        q = np.maximum(np.array(tab[key]["Q"])[1:15], floor)
        ax.bar(L + (k - 1) * w, q, width=w, label=label, color=[PANEL_COLOURS[2], "#888888", PANEL_COLOURS[0]][k])
    ax.set_yscale("log")
    ax.set_ylim(floor, 10)
    ax.set_xticks(L)
    ax.set_xlabel("spherical-harmonic degree $\\ell$")
    ax.set_ylabel("seam-density power $Q_\\ell$ (floor $10^{-14}$)")
    ax.axhline(1e-9, color="k", lw=0.5, ls=":")
    t_allowed = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    i_allowed = [6, 10, 12]
    ax.text(0.01, 0.97, "allowed for T: $\\ell$ = 3, 4, 6, 7, 8, ...      allowed for I: $\\ell$ = 6, 10, 12, 15, ...",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(loc="lower right")
    ax.set_title("A seam network can only contain the harmonics its rotation group allows")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def exposure_map(ball, n_lat=60, n_lon=120):
    lat = np.linspace(-90, 90, n_lat)
    lon = np.linspace(-180, 180, n_lon)
    LA, LO = np.meshgrid(np.radians(lat), np.radians(lon), indexing="ij")
    axes = np.stack([np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], -1).reshape(-1, 3)
    edges = sum(len(V) for V in ball.face_vertices())
    g = bm.exposure(ball, axes, samples=max(8, 12000 // edges), chunk=250)
    return lat, lon, g.reshape(n_lat, n_lon)


def fig_exposure(ctx, path):
    plt = _plt()
    R = ctx["R"]
    trio = balls.make_trionda()
    hat = NetBall(ctx["hat"][0][4], "hat")
    # put the Trionda in the Hat Trick's frame so the maps share their axes
    trio_net = NetBall([V for V in trio.face_vertices()], "trionda", warp=lambda p, inv: trio.twist(p @ R, inv) @ R.T)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1, 0.7]})
    gs = []
    for ax, (name, b) in zip(axes[:2], (("Trionda", trio_net), ("Hat Trick", hat))):
        lat, lon, g = exposure_map(b)
        gs.append(g)
        im = ax.imshow(g / g.mean(), extent=(-180, 180, -90, 90), origin="lower", cmap="RdBu_r", vmin=0.6, vmax=1.4,
                       aspect="auto")
        ax.set_title(f"{name}: seam length near the separation circle\nof a flow along each axis (CV {g.std() / g.mean():.3f})",
                     fontsize=10)
        ax.set_xlabel("flow axis longitude")
        ax.set_ylabel("latitude")
    fig.colorbar(im, ax=axes[:2], shrink=0.8, label="exposure / mean")
    for g, c, lab in zip(gs, (PANEL_COLOURS[2], PANEL_COLOURS[0]), ("Trionda", "Hat Trick")):
        axes[2].hist((g / g.mean()).ravel(), bins=40, alpha=0.7, color=c, label=lab)
    axes[2].set_xlabel("exposure / mean")
    axes[2].set_ylabel("axes (grid cells)")
    axes[2].legend()
    axes[2].set_title("how uniform?", fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---- Hat-Trionda: each tetrahedral panel as a union of hats --------------------------------

def hat_trionda(ctx, variant=0, n=5):
    """Assign every hat and pentagon to one of the four Trionda panels.  The Trionda (placed on a tetrahedron
    inside the icosahedron) is T-invariant and so is the set of hats, so the assignment by the panel that holds
    most of a hat's area is T-equivariant and every panel gets 15 hats and 3 pentagons."""
    ball, ids, X, info, polys = ctx["hat"][variant]
    trio = Rotated(balls.make_trionda(), ctx["R"])
    frac = np.zeros((60, 4))
    kite_panel, kite_area = {}, {}
    for k in ball.S.kites:
        P = X[[ids[c] for c in ball.S.corners(k)]]
        q = np.array([v.mean(0) for v in H._kite_grid(P, n)])
        q /= np.linalg.norm(q, axis=1)[:, None]
        lab = trio.panel_index(q)
        cnt = np.bincount(lab, minlength=4) / len(lab)
        frac[ball.hat_of[k]] += cnt
        kite_panel[k] = cnt
    frac /= frac.sum(1, keepdims=True)
    assign = frac.argmax(1)
    pent_panel = {p: int(trio.panel_index(X[[ids[("c", t)] for t in [t for t, f in enumerate(ball.F) if p in f][:1]]]
                                           .mean(0, keepdims=True) * 0 + ball.V[p][None])[0]) for p in ball.pent}
    return {"frac": frac, "assign": assign, "pent_panel": pent_panel, "kite_panel": kite_panel, "trio": trio}


def super_panel_stats(ctx, hr, variant=0):
    ball, ids, X, info, polys = ctx["hat"][variant]
    assign, pent_panel = hr["assign"], hr["pent_panel"]
    owner = {}
    for i in range(60):
        loop = ball.hat_outline(i)[0]
        for j in range(len(loop)):
            owner.setdefault(frozenset((loop[j], loop[(j + 1) % len(loop)])), []).append(int(assign[i]))
    for p in ball.pent:
        loop = ball.pentagon_loop(p)
        for j in range(len(loop)):
            owner.setdefault(frozenset((loop[j], loop[(j + 1) % len(loop)])), []).append(pent_panel[p])
    seam = 0.0
    for key, o in owner.items():
        if len(o) == 2 and o[0] != o[1]:
            a, b = tuple(key)
            seam += math.acos(np.clip(X[ids[a]] @ X[ids[b]], -1, 1))
    counts_h = np.bincount(assign, minlength=4)
    counts_p = np.bincount(list(pent_panel.values()), minlength=4)
    # area mismatch against the smooth Trionda panels
    mism = 0.0
    tot = 0.0
    areas = {}
    for k in ball.S.kites:
        a = H.spherical_area(X[[ids[c] for c in ball.S.corners(k)]])
        cnt = hr["kite_panel"][k]
        mism += a * (1 - cnt[assign[ball.hat_of[k]]])
        tot += a
    return {"hats_per_panel": counts_h.tolist(), "pentagons_per_panel": counts_p.tolist(),
            "super_seam_length": seam, "area_mismatch_fraction": mism / tot,
            "straddling_hats": int((hr["frac"].max(1) < 0.999).sum()),
            "min_majority_fraction": float(hr["frac"].max(1).min())}


def fig_assign(ctx, path, variant=0):
    plt = _plt()
    ball, ids, X, info, polys = ctx["hat"][variant]
    hr = hat_trionda(ctx, variant)
    st = super_panel_stats(ctx, hr, variant)
    # panel colours: Trionda panel k gets PANEL_COLOURS[k]
    colour = lambda h: PANEL_COLOURS[hr["assign"][h]]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10))
    for r, (name, d) in enumerate(shared_axes(ctx["R"]).items()):
        axes[r, 0].imshow(raster(hr["trio"], d), extent=(-1, 1, -1, 1))
        axes[r, 0].axis("off")
        axes[r, 0].set_title(f"Trionda, {name} axis", fontsize=10)
        draw_hat(axes[r, 1], ball, ids, X, d, colour_of_hat=colour, pentagon_colour="#202020")
        axes[r, 1].set_title("Hat-Trionda: hats coloured by the panel holding most of their area", fontsize=9)
        # mismatch: kites whose Trionda panel differs from their hat's panel
        from matplotlib.collections import PolyCollection
        ex, ey, ez = view_basis(d)
        polys_, cols_ = [], []
        for k in ball.S.kites:
            P = X[[ids[c] for c in ball.S.corners(k)]]
            if P.mean(0) @ ez <= 0:
                continue
            bad = 1 - hr["kite_panel"][k][hr["assign"][ball.hat_of[k]]]
            for q in H._kite_grid(P, 2):
                polys_.append(np.c_[q @ ex, q @ ey])
                cols_.append((1.0, 1 - 0.8 * bad, 1 - 0.8 * bad))
        axes[r, 2].add_collection(PolyCollection(polys_, facecolors=cols_, edgecolors="none"))
        draw_hat(axes[r, 2], ball, ids, X, d, colour_of_hat=lambda h: "#ffffff", pentagon_colour="#cccccc", lines=True) \
            if False else None
        th = np.linspace(0, 2 * np.pi, 200)
        axes[r, 2].plot(np.cos(th), np.sin(th), color="k", lw=1)
        axes[r, 2].set_xlim(-1.03, 1.03); axes[r, 2].set_ylim(-1.03, 1.03); axes[r, 2].set_aspect("equal")
        axes[r, 2].axis("off")
        axes[r, 2].set_title("where a hat sticks out of its Trionda panel (red)", fontsize=9)
    fig.suptitle(f"Hat-Trionda: {st['hats_per_panel']} hats + {st['pentagons_per_panel']} pentagons per panel, "
                 f"{100 * st['area_mismatch_fraction']:.1f}% of the area differs from the smooth Trionda", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return st


# ---- flow blend ---------------------------------------------------------------------------

def hat_warp(R, tau):
    w = balls.make_trionda(tau).warp
    if w is None:
        return None
    return lambda p, inv: w(p @ R, inv) @ R.T


def flow_sweep(ctx, taus):
    out = []
    polys = ctx["hat"][0][4]
    for tau in taus:
        net = NetBall(polys, f"hat-flow-{tau}", warp=hat_warp(ctx["R"], tau))
        m = metrics(net, n_axes=1500, max_points=15000)
        out.append({"tau": tau, "Q3": m["Q"][3], "Q4": m["Q"][4], "Q6": m["Q"][6], "lowest": m["lowest_multipole"],
                    "seam_length": m["seam_length"], "exposure_cv": m["exposure_cv"]})
        print("flow", tau, {k: round(v, 5) for k, v in out[-1].items()}, flush=True)
    return out


def fig_flow(ctx, sweep, path, taus_show=(-0.9, 0.0, 0.9)):
    plt = _plt()
    ball, ids, X, info, polys = ctx["hat"][0]
    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    d = shared_axes(ctx["R"])["3-fold"]
    pc = H.patch_colours(ball)[0]
    for j, tau in enumerate(taus_show):
        ax = fig.add_subplot(gs[0, j])
        draw_hat(ax, ball, ids, X, d, colour_of_hat=lambda h: H.PALETTE[pc[h]], warp=hat_warp(ctx["R"], tau))
        ax.set_title(f"Hat Trick through the Trionda flow, $\\tau$ = {tau:+.1f}" + ("  (the Hat Trick)" if tau == 0 else ""),
                     fontsize=10)
    t = np.array([r["tau"] for r in sweep])
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(t, np.maximum([r["Q3"] for r in sweep], 1e-14), "o-", color=PANEL_COLOURS[2])
    ax.set_xlabel("flow time $\\tau$"); ax.set_ylabel("$Q_3$ (the harmonic I forbids and T allows)")
    ax.set_title("symmetry I $\\to$ T: $\\ell$ = 3 switches on", fontsize=10)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, [r["seam_length"] for r in sweep], "o-", color=PANEL_COLOURS[0])
    ax.set_xlabel("flow time $\\tau$"); ax.set_ylabel("seam length (unit sphere)")
    ax.set_title("the flow lengthens the seams", fontsize=10)
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, [r["exposure_cv"] for r in sweep], "o-", color="#444444")
    ax.set_xlabel("flow time $\\tau$"); ax.set_ylabel("exposure CV")
    ax.set_title("and makes the exposure less uniform", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


# ---- curvature ledger -------------------------------------------------------------------------

def fig_ledger(path):
    plt = _plt()
    V0, F0 = H.icosahedron()
    P5 = V0
    P3 = np.array([v / np.linalg.norm(v) for v in (V0[list(f)].sum(0) for f in F0)])
    tet = balls.TETRA_VERTICES
    cases = [
        ("Trionda / tetrahedron\n4 cone points x 180$^\\circ$ = 720$^\\circ$", tet, 180, PANEL_COLOURS[2]),
        ("Hat Trick (flat hats)\n12 pentagon holes x 60$^\\circ$ = 720$^\\circ$", P5, 60, PANEL_COLOURS[0]),
        ("icosahedron faces: 20 vertices of\nthe dodecahedron x 36$^\\circ$ = 720$^\\circ$", P3, 36, "#999999"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    ex, ey, ez = view_basis(np.array([0.35, 0.5, 1.0]))
    for ax, (title, P, deg, c) in zip(axes, cases):
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), "k", lw=1)
        for p in P:
            front = p @ ez > 0
            ax.scatter([p @ ex], [p @ ey], s=deg * 3.2, color=c, alpha=1 if front else 0.25, edgecolor="k", lw=0.8)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=10)
    fig.suptitle("Gauss-Bonnet ledger: the flat model of each ball puts all 720$^\\circ$ of curvature at its symmetry axes", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------

def context():
    R, tet = tetra_in_icosa()
    return {"R": R, "tetra": tet, "hat": {v: hat_ball(v) for v in (0, 1)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["table", "views", "spectra", "exposure", "assign", "flow", "ledger", "chirality"])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "compare.json")
    result = json.load(open(path)) if os.path.exists(path) else {}
    want = lambda name: args.only in (None, name)
    ctx = context()
    if want("table"):
        result["table"] = table(ctx)
    if want("chirality"):
        result["chirality_deg"] = chirality_table(ctx)
        print(result["chirality_deg"])
    if want("views"):
        fig_views(ctx, os.path.join(args.out, "compare_views.png"))
    if want("spectra"):
        if "table" not in result:
            result["table"] = table(ctx)
        fig_spectra(ctx, result["table"], os.path.join(args.out, "compare_spectra.png"))
    if want("exposure"):
        fig_exposure(ctx, os.path.join(args.out, "compare_exposure.png"))
    if want("assign"):
        result["hat_trionda"] = fig_assign(ctx, os.path.join(args.out, "compare_hat_trionda.png"))
        print(result["hat_trionda"])
    if want("flow"):
        taus = [-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9]
        result["flow"] = flow_sweep(ctx, taus)
        fig_flow(ctx, result["flow"], os.path.join(args.out, "compare_flow.png"))
    if want("ledger"):
        fig_ledger(os.path.join(args.out, "compare_ledger.png"))
    with open(path, "w") as f:
        json.dump(result, f, indent=1, default=float)
    for name, row in result.get("table", {}).items():
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "Q"})


if __name__ == "__main__":
    main()
