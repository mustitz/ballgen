#!/usr/bin/env python3
"""
the_hat_trick_ball.py -- a 72-panel ball: 60 "hat" tiles and 12 pentagons.

Design (reconstructed here, see README): the hat is a polykite of 8 kites of the
deltoidal trihexagonal tiling.  Take the Goldberg polyhedron GP(3,0) (12 pentagons +
80 hexagons), cut every hexagon into 6 kites (480 kites), keep the 12 pentagons as
holes, and cover the 480 kites exactly by 60 hats.  The cover is searched (not typed
in) and is required to be invariant under the 60 rotations of the icosahedron, so the
12 pentagons each carry a patch of 5 hats related by the 5-fold rotation.

Pipeline
  1. geodesic icosahedron, frequency 3  ->  kite complex (kite = vertex x triangle)
  2. the hat as 8 kites (hard-coded adjacency, cross-checked against the hat outline)
  3. every orientation-preserving embedding of the hat into the kite surface
  4. exact cover by the icosahedral orbit of one embedding  -> hat tilings
  5. geometry: relax all kite corners on the unit sphere so that every kite keeps its
     flat shape as well as possible (edges 1 and sqrt(3), diagonals 2 and sqrt(3))
  6. figures, JSON, strain report

Usage
  python the_hat_trick_ball.py                # build, report, write figures + json
  python the_hat_trick_ball.py --variant 1    # the other icosahedral tiling
  python the_hat_trick_ball.py --no-figures
"""
import argparse, json, math, os, sys
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

R3 = math.sqrt(3.0)
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------------
# 1. geodesic sphere and its icosahedral rotation group
# ---------------------------------------------------------------------------------

def icosahedron():
    p = (1 + 5 ** 0.5) / 2
    V = np.array([(-1, p, 0), (1, p, 0), (-1, -p, 0), (1, -p, 0), (0, -1, p), (0, 1, p),
                  (0, -1, -p), (0, 1, -p), (p, 0, -1), (p, 0, 1), (-p, 0, -1), (-p, 0, 1)], float)
    V /= np.linalg.norm(V, axis=1)[:, None]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
         (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    out = []
    for f in F:                                   # outward, counter-clockwise seen from outside
        a, b, c = V[list(f)]
        out.append(f if np.dot(np.cross(b - a, c - a), a + b + c) > 0 else (f[0], f[2], f[1]))
    return V, out


def geodesic(freq=3):
    """Icosahedron with each face cut into freq^2 triangles, projected to the sphere."""
    V0, F0 = icosahedron()
    verts, index, F = [], {}, []

    def vid(p):
        k = tuple(np.round(p, 6))
        if k not in index:
            index[k] = len(verts)
            verts.append(p)
        return index[k]

    for f in F0:
        A, B, C = V0[list(f)]
        P = lambda i, j: (A * (freq - i - j) + B * i + C * j) / freq
        for i in range(freq):
            for j in range(freq - i):
                F.append((vid(P(i, j)), vid(P(i + 1, j)), vid(P(i, j + 1))))
                if i + j < freq - 1:
                    F.append((vid(P(i + 1, j)), vid(P(i + 1, j + 1)), vid(P(i, j + 1))))
    V = np.array(verts)
    return V / np.linalg.norm(V, axis=1)[:, None], F


def rotation_group():
    """The 60 proper rotations of the icosahedron (closure of a 5-fold and a 3-fold turn)."""
    V0, F0 = icosahedron()

    def turn(ax, ang):
        ax = ax / np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K

    gens = [turn(V0[0], 2 * math.pi / 5), turn(V0[list(F0[0])].sum(0), 2 * math.pi / 3)]
    G = [np.eye(3)]
    grew = True
    while grew:
        grew = False
        for g in list(G):
            for h in gens:
                m = h @ g
                if not any(np.allclose(m, x, atol=1e-6) for x in G):
                    G.append(m)
                    grew = True
    assert len(G) == 60
    return G


# ---------------------------------------------------------------------------------
# 2. kite complex
# ---------------------------------------------------------------------------------
# A triangle t = (v0, v1, v2) (counter-clockwise) is cut into three kites around its
# centroid, kite id 3t+i belonging to vertex v_i.  Corners, counter-clockwise:
#     v_i  ->  m(v_i, v_{i+1})  ->  centroid(t)  ->  m(v_{i+2}, v_i)
# Sides 0..3: 0 long (length sqrt3), 1 short (1), 2 short (1), 3 long (sqrt3).
# With the triangulation edge 2*sqrt3, hexagon side 2, the long side is half a
# triangulation edge, the short side half a hexagon edge: the hat's two edge lengths.
KITE_SIDE = (R3, 1.0, 1.0, R3)
KITE_DIAG = {(0, 2): 2.0, (1, 3): R3}          # v-to-centroid, midpoint-to-midpoint
KITE_ANGLE = {'v': 60, 'm': 90, 'c': 120}      # interior angle of a kite at each corner type


class KiteComplex:
    def __init__(self, V, F, skip=()):
        self.V = np.asarray(V, float)
        self.F = [tuple(f) for f in F]
        self.skip = set(skip)
        e2t = {}
        for t, f in enumerate(self.F):
            for i in range(3):
                e2t[(f[i], f[(i + 1) % 3])] = t
        self.e2t = e2t
        self.nb = [[None] * 4 for _ in range(3 * len(self.F))]
        for t, f in enumerate(self.F):
            for i in range(3):
                v = f[i]
                if v in self.skip:
                    continue
                nb = self.nb[3 * t + i]
                t1, t2 = e2t.get((f[(i + 1) % 3], v)), e2t.get((v, f[(i + 2) % 3]))
                if t1 is not None:
                    nb[0] = 3 * t1 + self.F[t1].index(v)
                if f[(i + 1) % 3] not in self.skip:
                    nb[1] = 3 * t + (i + 1) % 3
                if f[(i + 2) % 3] not in self.skip:
                    nb[2] = 3 * t + (i + 2) % 3
                if t2 is not None:
                    nb[3] = 3 * t2 + self.F[t2].index(v)
        self.kites = [k for k in range(3 * len(self.F)) if self.F[k // 3][k % 3] not in self.skip]

    def corners(self, k):
        """Combinatorial ids of the four corners of kite k."""
        t, i = divmod(k, 3)
        f = self.F[t]
        v, b, c = f[i], f[(i + 1) % 3], f[(i + 2) % 3]
        return [('v', v), ('m', frozenset((v, b))), ('c', t), ('m', frozenset((c, v)))]


def hat_adjacency():
    """The hat as 8 kites: (kite a, side s, kite b) for every internal adjacency.
    Kite 0 is the anchor; the list was read off the hat outline and is re-derived in
    the tests (tests/test_hat_trick_ball.py)."""
    return [(0, 3, 1), (1, 0, 0), (1, 3, 2), (2, 0, 1), (2, 1, 3), (2, 2, 4), (2, 3, 5), (3, 0, 6),
            (3, 1, 4), (3, 2, 2), (4, 0, 7), (4, 1, 2), (4, 2, 3), (5, 0, 2), (5, 1, 7), (6, 3, 3),
            (7, 2, 5), (7, 3, 4)]


def embed_hat(S, anchor):
    """Embed the hat with its kite 0 on kite `anchor`; returns the 8 surface kites in hat
    order, or None.  Only orientation-preserving embeddings exist (labelled sides)."""
    adj = hat_adjacency()
    out = {0: anchor}
    todo = [0]
    while todo:
        h = todo.pop()
        for a, s, b in adj:
            if a != h:
                continue
            nxt = S.nb[out[h]][s]
            if nxt is None:
                return None
            if b in out:
                if out[b] != nxt:
                    return None
            else:
                out[b] = nxt
                todo.append(b)
    if len(out) != 8 or len(set(out.values())) != 8:
        return None
    for a, s, b in adj:                          # every internal adjacency must hold
        if S.nb[out[a]][s] != out[b]:
            return None
    return tuple(out[i] for i in range(8))


# ---------------------------------------------------------------------------------
# 3. the ball as a combinatorial object
# ---------------------------------------------------------------------------------

class HatTrick:
    def __init__(self, variant=0, freq=3):
        self.V, self.F = geodesic(freq)
        deg = np.zeros(len(self.V), int)
        for f in self.F:
            for v in f:
                deg[v] += 1
        self.pent = [v for v in range(len(self.V)) if deg[v] == 5]
        assert len(self.pent) == 12 and set(deg) == {5, 6}
        self.S = KiteComplex(self.V, self.F, skip=self.pent)
        self.G = rotation_group()
        self.kperm = [self._kite_perm(g) for g in self.G]
        self.placements = sorted({frozenset(p) for p in (embed_hat(self.S, k) for k in self.S.kites) if p})
        self.tilings = self._symmetric_tilings()
        if not 0 <= variant < len(self.tilings):
            raise SystemExit(f"variant must be in 0..{len(self.tilings) - 1}")
        self.variant = variant
        self.hats = [sorted(h) for h in sorted(map(sorted, self.tilings[variant]))]
        self.hat_of = {k: i for i, h in enumerate(self.hats) for k in h}
        self._outlines = [self._outline(h) for h in self.hats]

    def _kite_perm(self, g):
        vid = {tuple(np.round(v, 5)): i for i, v in enumerate(self.V)}
        p = [vid[tuple(np.round(w, 5))] for w in self.V @ g.T]
        fidx = {frozenset(f): t for t, f in enumerate(self.F)}
        out = {}
        for k in range(3 * len(self.F)):
            t, i = divmod(k, 3)
            f = self.F[t]
            t2 = fidx[frozenset(p[x] for x in f)]
            out[k] = 3 * t2 + self.F[t2].index(p[f[i]])
        return out

    def _symmetric_tilings(self):
        """Hat placements whose 60 rotated images tile all 480 kites exactly."""
        seen, out = set(), []
        for P in self.placements:
            imgs = {frozenset(kp[k] for k in P) for kp in self.kperm}
            if len(imgs) == 60 and len(set().union(*imgs)) == 480:
                key = frozenset(imgs)
                if key not in seen:
                    seen.add(key)
                    out.append(key)
        out.sort(key=lambda T: sorted(map(sorted, T)))
        return out

    # ---- outlines ----
    def _outline(self, kites):
        """Boundary loop of a union of kites: list of corner ids (counter-clockwise),
        straight points included, and the indices that are true corners."""
        edges = {}
        for k in kites:
            c = self.S.corners(k)
            for s in range(4):
                a, b = c[s], c[(s + 1) % 4]
                if (b, a) in edges:
                    del edges[(b, a)]
                else:
                    edges[(a, b)] = True
        nxt = {a: b for (a, b) in edges}
        start = next(iter(nxt))
        loop, p = [start], nxt[start]
        while p != start:
            loop.append(p)
            p = nxt[p]
        assert len(loop) == len(nxt)
        ang = {}
        for k in kites:
            for cid in self.S.corners(k):
                ang[cid] = ang.get(cid, 0) + KITE_ANGLE[cid[0]]
        true = [i for i, cid in enumerate(loop) if ang[cid] != 180]
        return loop, true, [ang[c] for c in loop]

    def hat_outline(self, i):
        return self._outlines[i]

    def pentagon_loop(self, p):
        """Corner ids of the pentagon hole around vertex p: centroid, midpoint, ... (10)."""
        tris = [t for t, f in enumerate(self.F) if p in f]
        # order the triangles around p (counter-clockwise)
        order = [tris[0]]
        while len(order) < 5:
            f = self.F[order[-1]]
            i = f.index(p)
            nxt = self.e2t_get(p, f[(i + 2) % 3])
            order.append(nxt)
        loop = []
        for t in order:
            f = self.F[t]
            i = f.index(p)
            loop.append(('c', t))
            loop.append(('m', frozenset((p, f[(i + 2) % 3]))))
        # loop currently centroid(t0), m(p, w) for the edge (c,p) of t0 ... ensure the
        # midpoint is between the centroids of the two triangles that share that edge
        return loop

    def e2t_get(self, a, b):
        return self.S.e2t[(a, b)]

    def pentagon_hats(self, p):
        """Hats that touch the pentagon around vertex p (across a short kite side)."""
        out = set()
        for k in self.S.kites:
            t, i = divmod(k, 3)
            f = self.F[t]
            for s in (1, 2):
                if f[(i + s) % 3] == p:
                    out.add(self.hat_of[k])
        return sorted(out)

    # ---- positions ----
    def initial_positions(self):
        ids, pos = {}, []
        for k in self.S.kites:
            for cid in self.S.corners(k):
                if cid not in ids:
                    ids[cid] = len(pos)
                    pos.append(self.corner_position(cid))
        return ids, np.array(pos)

    def corner_position(self, cid):
        kind, x = cid
        if kind == 'v':
            return self.V[x].copy()
        if kind == 'm':
            a, b = tuple(x)
            q = self.V[a] + self.V[b]
        else:
            q = self.V[list(self.F[x])].sum(0)
        return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------------
# 4. geometry: relax all kite corners on the sphere
# ---------------------------------------------------------------------------------

def relax(ball, rounds=1, max_nfev=200, verbose=False):
    """Move the corners on the unit sphere so that kites stay as close as possible to the
    flat kite of sides 1, sqrt3 (diagonals 2, sqrt3) times a common scale a, and so that
    the straight hexagon edges (2) and triangulation edges (2 sqrt3) stay straight.
    Returns (ids, positions, a, residual-report)."""
    ids, pos0 = ball.initial_positions()
    n = len(pos0)
    pairs, target = [], []
    for k in ball.S.kites:
        c = [ids[x] for x in ball.S.corners(k)]
        for s in range(4):
            pairs.append((c[s], c[(s + 1) % 4])); target.append(KITE_SIDE[s])
        for (i, j), d in KITE_DIAG.items():
            pairs.append((c[i], c[j])); target.append(d)
    # straightness: both halves of a hexagon edge, both halves of a triangulation edge
    seen = set()
    for t, f in enumerate(ball.F):
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            e = frozenset((a, b))
            if e in seen:
                continue
            seen.add(e)
            t2 = ball.e2t_get(b, a)
            cc = (ids.get(('c', t)), ids.get(('c', t2)))
            if None not in cc:
                pairs.append(cc); target.append(2.0)            # hexagon edge: 2 short sides
            if a not in ball.S.skip and b not in ball.S.skip:
                pairs.append((ids[('v', a)], ids[('v', b)])); target.append(2 * R3)
    P = np.array(pairs)
    T = np.array(target)
    m = len(P)

    def unit(X):
        X = X.reshape(n, 3)
        return X / np.linalg.norm(X, axis=1)[:, None]

    def resid(z):
        a, X = z[0], unit(z[1:])
        d = np.linalg.norm(X[P[:, 0]] - X[P[:, 1]], axis=1)
        return d - a * T

    # sparsity pattern
    A = lil_matrix((m, 1 + 3 * n), dtype=int)
    A[:, 0] = 1
    for r, (i, j) in enumerate(P):
        for q in (i, j):
            A[r, 1 + 3 * q:4 + 3 * q] = 1
    a0 = np.linalg.norm(pos0[P[:, 0]] - pos0[P[:, 1]], axis=1).sum() / T.sum()
    sol = least_squares(resid, np.concatenate([[a0], pos0.ravel()]), jac_sparsity=A, method='trf',
                        x_scale='jac', max_nfev=max_nfev, verbose=2 if verbose else 0)
    a, X = sol.x[0], unit(sol.x[1:])
    rel = resid(sol.x) / (a * T)
    return ids, X, a, {'a': float(a), 'a_initial': float(a0), 'rms_strain': float(np.sqrt(np.mean(rel ** 2))),
                       'max_strain': float(np.max(np.abs(rel))), 'n_constraints': int(m)}


# ---------------------------------------------------------------------------------
# 5. reports
# ---------------------------------------------------------------------------------

def spherical_area(poly_pts):
    """Area on the unit sphere of the polygon through the given unit vectors (fan of triangles
    around their mean direction)."""
    c = poly_pts.mean(0)
    c /= np.linalg.norm(c)
    tot = 0.0
    for i in range(len(poly_pts)):
        a, b = poly_pts[i], poly_pts[(i + 1) % len(poly_pts)]
        num = abs(np.dot(c, np.cross(a, b)))
        den = 1 + c @ a + a @ b + b @ c
        tot += 2 * math.atan2(num, den)
    return tot


def panel_areas(ball, ids, X):
    hats = []
    for i in range(60):
        loop, _, _ = ball.hat_outline(i)
        hats.append(spherical_area(X[[ids[c] for c in loop]]))
    pents = []
    for p in ball.pent:
        loop = ball.pentagon_loop(p)
        pents.append(spherical_area(X[[ids[c] for c in loop]]))
    return np.array(hats), np.array(pents)


def report(ball, ids, X, info):
    hats, pents = panel_areas(ball, ids, X)
    pat = {p: ball.pentagon_hats(p) for p in ball.pent}
    deficits = []
    for i in range(60):
        loop, true, ang = ball.hat_outline(i)
        deficits.append(len(true))
    out = {
        'variant': ball.variant,
        'n_tilings_with_icosahedral_symmetry': len(ball.tilings),
        'n_hat_placements': len(ball.placements),
        'hats': 60, 'pentagons': 12, 'panels': 72,
        'hat_corners': sorted(set(deficits)),
        'hats_per_pentagon': sorted({len(h) for h in pat.values()}),
        'hat_area_over_sphere': float(hats.mean() / (4 * math.pi)),
        'pentagon_area_over_sphere': float(pents.mean() / (4 * math.pi)),
        'hat_to_pentagon_area': float(hats.mean() / pents.mean()),
        'hat_area_spread': float((hats.max() - hats.min()) / hats.mean()),
        'total_area_check': float((hats.sum() + pents.sum()) / (4 * math.pi)),
    }
    out.update(info)
    return out


def hat_edge_on_pentagon(ball):
    """How the hat's length-2 edge relates to the pentagons (for the variant description)."""
    hits = {}
    for p in ball.pent[:1]:
        for i in ball.pentagon_hats(p):
            loop, true, ang = ball.hat_outline(i)
            hits[i] = len(true)
    return hits


# ---------------------------------------------------------------------------------
# 6. figures
# ---------------------------------------------------------------------------------

def _basis(d):
    d = np.asarray(d, float) / np.linalg.norm(d)
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(up, d)
    x /= np.linalg.norm(x)
    y = np.cross(d, x)
    return x, y, d


def _kite_grid(P, n=4):
    """Points of a kite subdivided n x n by bilinear interpolation, pushed to the sphere."""
    p0, p1, p2, p3 = P
    quads = []
    for i in range(n):
        for j in range(n):
            u0, u1, w0, w1 = i / n, (i + 1) / n, j / n, (j + 1) / n
            q = []
            for u, w in ((u0, w0), (u1, w0), (u1, w1), (u0, w1)):
                v = (1 - u) * (1 - w) * p0 + u * (1 - w) * p1 + u * w * p2 + (1 - u) * w * p3
                q.append(v / np.linalg.norm(v))
            quads.append(np.array(q))
    return quads


def _arc(a, b, n=6):
    t = np.linspace(0, 1, n)[:, None]
    p = (1 - t) * a + t * b
    return p / np.linalg.norm(p, axis=1)[:, None]


PALETTE = ['#e07a5f', '#3d85c6', '#81b29a', '#f2cc8f']


def patch_colours(ball):
    """Colour the 12 patches with 4 colours so that touching patches differ (greedy
    colouring of the icosahedron's vertex graph, which is 4-colourable)."""
    V0, F0 = icosahedron()
    adj = {i: set() for i in range(12)}
    for f in F0:
        for i in range(3):
            adj[f[i]].add(f[(i + 1) % 3]); adj[f[(i + 1) % 3]].add(f[i])
    order = sorted(adj, key=lambda x: -len(adj[x]))
    col = {}

    def solve(i):
        if i == len(order):
            return True
        v = order[i]
        for c in range(4):
            if all(col.get(w) != c for w in adj[v]):
                col[v] = c
                if solve(i + 1):
                    return True
                del col[v]
        return False

    assert solve(0)
    pent_dir = {p: ball.V[p] for p in ball.pent}
    which = {}
    for p, d in pent_dir.items():
        which[p] = col[int(np.argmax(V0 @ d))]
    out = {}
    for p in ball.pent:
        for h in ball.pentagon_hats(p):
            out[h] = which[p]
    return out, which


def draw_ball(ax, ball, ids, X, view, light=(0.35, 0.55, 1.0), lines=True):
    ex, ey, ez = _basis(view)
    L = np.asarray(light, float)
    L = L[0] * ex + L[1] * ey + L[2] * ez
    L /= np.linalg.norm(L)
    colour_of, pent_col = patch_colours(ball)
    from matplotlib.collections import PolyCollection, LineCollection
    import matplotlib.colors as mc
    polys, cols = [], []

    def add(q, base):
        nrm = q.mean(0)
        nrm /= np.linalg.norm(nrm)
        if nrm @ ez <= 0:
            return
        sh = 0.45 + 0.55 * max(0.0, nrm @ L)
        c = np.array(mc.to_rgb(base)) * sh
        polys.append(np.c_[q @ ex, q @ ey]); cols.append(np.clip(c, 0, 1))

    for k in ball.S.kites:
        P = X[[ids[c] for c in ball.S.corners(k)]]
        base = PALETTE[colour_of[ball.hat_of[k]]]
        for q in _kite_grid(P, 3):
            add(q, base)
    for p in ball.pent:
        loop = ball.pentagon_loop(p)
        Q = X[[ids[c] for c in loop]]
        c = Q.mean(0)
        c /= np.linalg.norm(c)
        for i in range(len(Q)):
            tri = np.array([c, Q[i], Q[(i + 1) % len(Q)]])
            for q in _kite_grid([tri[0], tri[1], tri[2], tri[0]], 2):
                add(q, '#2b2b2b')
    ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors=cols, linewidths=0.3))
    if lines:
        segs = []
        for i in range(60):
            loop, true, _ = ball.hat_outline(i)
            Q = X[[ids[c] for c in loop]]
            for j in range(len(Q)):
                arc = _arc(Q[j], Q[(j + 1) % len(Q)])
                if arc.mean(0) @ ez > 0.02:
                    segs.append(np.c_[arc @ ex, arc @ ey])
        for p in ball.pent:
            loop = ball.pentagon_loop(p)
            Q = X[[ids[c] for c in loop]]
            for j in range(len(Q)):
                arc = _arc(Q[j], Q[(j + 1) % len(Q)])
                if arc.mean(0) @ ez > 0.02:
                    segs.append(np.c_[arc @ ex, arc @ ey])
        ax.add_collection(LineCollection(segs, colors='#101010', linewidths=1.1))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color='#101010', lw=1.0)
    ax.set_xlim(-1.03, 1.03); ax.set_ylim(-1.03, 1.03); ax.set_aspect('equal'); ax.axis('off')


def symmetry_axes(ball):
    """Directions of a 5-fold, a 3-fold and a 2-fold axis of the icosahedron."""
    V0, F0 = icosahedron()
    five = V0[0]
    three = V0[list(F0[0])].sum(0)
    two = (V0[0] + V0[list(F0[0])[1]])
    return {'5-fold (pentagon)': five, '3-fold': three, '2-fold': two}


def fig_ball(ball, ids, X, info, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ax_ = symmetry_axes(ball)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    for ax, (name, d) in zip(axes, ax_.items()):
        draw_ball(ax, ball, ids, X, d)
        ax.set_title(f'view along a {name} axis', fontsize=11)
    fig.suptitle(f"The Hat Trick ball: 60 hats + 12 pentagons (variant {ball.variant}); "
                 f"rms strain {100 * info['rms_strain']:.1f}%", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def flat_hat():
    """The hat laid out in the plane from its kite adjacency: (8 kite polygons, outline loop).
    Raises if the adjacency is not planar-consistent."""
    c = np.array([2.0, 0.0])
    m1 = R3 * np.array([math.cos(-math.pi / 6), math.sin(-math.pi / 6)])
    m2 = R3 * np.array([math.cos(math.pi / 6), math.sin(math.pi / 6)])
    base = np.array([[0, 0], m1, c, m2])
    adj = hat_adjacency()
    back = {(a, b): next(s for aa, s, bb in adj if aa == b and bb == a) for a, s, b in adj}
    pose = {0: (np.eye(2), np.zeros(2))}
    todo = [0]
    while todo:
        h = todo.pop()
        R, t = pose[h]
        for a, s, b in adj:
            if a != h or b in pose:
                continue
            Q = base @ R.T + t
            e0, e1 = Q[s], Q[(s + 1) % 4]
            s2 = back[(a, b)]
            vb, va = base[(s2 + 1) % 4] - base[s2], e0 - e1
            ang = math.atan2(va[1], va[0]) - math.atan2(vb[1], vb[0])
            Rb = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
            pose[b] = (Rb, e1 - Rb @ base[s2])
            todo.append(b)
    kites = [base @ pose[i][0].T + pose[i][1] for i in range(8)]
    for a, s, b in adj:                          # non-tree adjacencies must close up too
        d = (np.linalg.norm(kites[a][s] - kites[b][(back[(a, b)] + 1) % 4])
             + np.linalg.norm(kites[a][(s + 1) % 4] - kites[b][back[(a, b)]]))
        if d > 1e-9:
            raise ValueError('hat adjacency is not planar-consistent')
    key = lambda p: (round(float(p[0]), 3), round(float(p[1]), 3))
    edges = {}
    for Q in kites:
        for s in range(4):
            a, b = key(Q[s]), key(Q[(s + 1) % 4])
            if (b, a) in edges:
                del edges[(b, a)]
            else:
                edges[(a, b)] = True
    nxt = {a: b for a, b in edges}
    start = next(iter(nxt))
    outline = [start]
    while nxt[outline[-1]] != start:
        outline.append(nxt[outline[-1]])
    return kites, np.array(outline)


def hat_angles(outline):
    """Interior angles (degrees) at the true corners of a counter-clockwise loop."""
    n = len(outline)
    out = []
    for i in range(n):
        a, b = outline[i - 1] - outline[i], outline[(i + 1) % n] - outline[i]
        t = math.degrees(math.atan2(b[0] * a[1] - b[1] * a[0], a @ b)) % 360
        if abs(t - 180) > 0.5:
            out.append(int(round(t)))
    return out


def fig_hat(path):
    """The hat as 8 kites in the plane."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    kites, outline = flat_hat()
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for Q in kites:
        ax.fill(Q[:, 0], Q[:, 1], color=PALETTE[1], ec='#9fc3e6', lw=0.8, zorder=1)
    ax.fill(outline[:, 0], outline[:, 1], fill=False, ec='#101010', lw=2.0, zorder=3, joinstyle='round')
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('the hat: 8 kites (edges 1 and $\\sqrt{3}$; one straight edge of length 2)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', type=int, default=0, help='which icosahedrally symmetric hat tiling (0 or 1)')
    ap.add_argument('--out', default=os.path.join(HERE, 'hat_trick_out'), help='output folder')
    ap.add_argument('--mirror', action='store_true', help='mirror image of the ball (opposite hand)')
    ap.add_argument('--no-figures', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args(argv)

    ball = HatTrick(variant=args.variant)
    print(f"GP(3,0) kite surface: {len(ball.S.kites)} kites, {len(ball.pent)} pentagon holes")
    print(f"hat placements: {len(ball.placements)}; tilings invariant under the 60 rotations: {len(ball.tilings)}")
    print(f"variant {ball.variant}: {len(ball.hats)} hats, hats per pentagon "
          f"{sorted({len(ball.pentagon_hats(p)) for p in ball.pent})}")
    ids, X, a, info = relax(ball, verbose=args.verbose)
    if args.mirror:
        X = X * np.array([-1.0, 1.0, 1.0])
    rep = report(ball, ids, X, info)
    for k, v in rep.items():
        print(f"  {k}: {v}")
    os.makedirs(args.out, exist_ok=True)
    panels = []
    for i in range(60):
        loop, true, _ = ball.hat_outline(i)
        panels.append({'kind': 'hat', 'id': i, 'corners': [int(j) for j in true],
                       'outline': X[[ids[c] for c in loop]].round(6).tolist()})
    for p in ball.pent:
        loop = ball.pentagon_loop(p)
        panels.append({'kind': 'pentagon', 'id': int(p), 'corners': [0, 2, 4, 6, 8],
                       'outline': X[[ids[c] for c in loop]].round(6).tolist()})
    with open(os.path.join(args.out, f'hat_trick_variant{ball.variant}.json'), 'w') as f:
        json.dump({'report': rep, 'panels': panels}, f)
    if not args.no_figures:
        fig_ball(ball, ids, X, info, os.path.join(args.out, f'hat_trick_ball_v{ball.variant}.png'))
        fig_hat(os.path.join(args.out, 'hat_trick_hat.png'))
        print('wrote figures to', args.out)
    return ball, ids, X, rep


if __name__ == '__main__':
    main()
