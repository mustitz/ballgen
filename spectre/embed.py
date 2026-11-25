"""
3D positions for a folded patch (for pictures, hole areas and hole centres).

The fold in ``spectre.fold`` is intrinsic: it says which edges are glued and what angle every vertex
carries, but not where anything is in space.  Here each tile is triangulated (ear clipping, 12 triangles)
and the folded surface is embedded in R^3 by

1. classical multidimensional scaling of the graph distances of the glued triangle mesh (a good first
   guess for a roughly round surface), pushed radially onto a sphere, and
2. a sparse Gauss-Newton relaxation that keeps every triangle edge at its exact length (so tiles stay
   intrinsically flat and can only crease along the diagonals) with a weak pull towards the sphere.

``embed`` returns the strain (largest relative edge-length error): it is small when the cone metric is
realised well, and a large value means the fold does not sit nicely on a sphere (typically because a
hole is large and flat).  Nothing here changes the exact integer ledger.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import shortest_path

from .fold import FoldResult, Hole


def triangulate(poly: np.ndarray) -> List[Tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple polygon (any orientation); indices into ``poly``."""
    n = len(poly)
    area2 = sum(poly[i, 0] * poly[(i + 1) % n, 1] - poly[(i + 1) % n, 0] * poly[i, 1] for i in range(n))
    sign = 1.0 if area2 > 0 else -1.0
    idx = list(range(n))

    def cross(o, a, b):
        return (poly[a, 0] - poly[o, 0]) * (poly[b, 1] - poly[o, 1]) - (poly[a, 1] - poly[o, 1]) * (poly[b, 0] - poly[o, 0])

    def inside(p, a, b, c):
        return cross(a, b, p) * sign >= -1e-12 and cross(b, c, p) * sign >= -1e-12 and cross(c, a, p) * sign >= -1e-12

    tris = []
    while len(idx) > 3:
        for k in range(len(idx)):
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            if cross(a, b, c) * sign <= 1e-12:
                continue
            if any(inside(p, a, b, c) for p in idx if p not in (a, b, c)
                   and not np.allclose(poly[p], poly[a]) and not np.allclose(poly[p], poly[b])
                   and not np.allclose(poly[p], poly[c])):
                continue
            tris.append((a, b, c))
            idx.pop(k)
            break
        else:
            raise RuntimeError("ear clipping failed")
    tris.append(tuple(idx))
    return tris


@dataclass
class Embedding:
    result: FoldResult
    vertex_class: np.ndarray        # (V_patch,) -> compact class index
    position: np.ndarray            # (C, 3) positions of the vertex classes
    triangles: np.ndarray           # (T, 3) class indices
    triangle_tile: np.ndarray       # (T,) local tile index
    radius: float
    strain: float                   # max relative edge-length error
    rms_strain: float

    def hole_vertices(self, hole: Hole) -> np.ndarray:
        g = self.result.graph
        return np.array([self.vertex_class[self.result.find(g.start_vertex(h))] for h in hole.edges])

    def hole_geometry(self, hole: Hole) -> Dict:
        """Centre direction (unit vector) and spherical area (steradians, fraction of the sphere)."""
        v = self.position[self.hole_vertices(hole)]
        dirs = v / np.linalg.norm(v, axis=1, keepdims=True)
        c = dirs.mean(0)
        nc = np.linalg.norm(c)
        c = c / nc if nc > 1e-9 else np.array([0.0, 0.0, 1.0])
        area = 0.0
        for i in range(len(dirs)):
            a, b = dirs[i], dirs[(i + 1) % len(dirs)]
            num = abs(np.dot(c, np.cross(a, b)))
            den = 1 + np.dot(c, a) + np.dot(a, b) + np.dot(b, c)
            area += 2 * math.atan2(num, den)
        return {"centre": c, "area_sr": float(area), "area_fraction": float(area / (4 * math.pi))}


def embed(result: FoldResult, sphere_weight: float = 0.05, max_nfev: int = 400) -> Embedding:
    g = result.graph
    patch = g.patch
    roots = np.array([result.find(v) for v in range(len(g.xy))])
    uniq, cls = np.unique(roots, return_inverse=True)
    C = len(uniq)
    tris, tri_tile = [], []
    edges: Dict[Tuple[int, int], float] = {}
    base = None
    for lt, t in enumerate(g.tile_ids):
        poly = patch.straight_polygon(t)
        if base is None:
            base = triangulate(poly)
        for (a, b, c) in base:
            ids = [int(cls[g.tile_vertex[lt, x]]) for x in (a, b, c)]
            tris.append(ids)
            tri_tile.append(lt)
            for (p, q), (pa, pb) in (((0, 1), (a, b)), ((1, 2), (b, c)), ((2, 0), (c, a))):
                i, j = ids[p], ids[q]
                d = float(np.linalg.norm(poly[pa] - poly[pb]))
                if i == j:
                    continue
                key = (min(i, j), max(i, j))
                # two different intrinsic lengths between the same pair of vertices (a crease of a tile
                # seen across a glued vertex): keep the shorter and let the strain report it
                edges[key] = min(edges.get(key, d), d)
    E = np.array(list(edges.keys()))
    L = np.array([edges[tuple(e)] for e in E])
    W = csr_matrix((np.r_[L, L], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(C, C))
    D = shortest_path(W, directed=False)
    D[~np.isfinite(D)] = D[np.isfinite(D)].max()
    J = np.eye(C) - 1.0 / C
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    top = np.argsort(w)[::-1][:3]
    X = V[:, top] * np.sqrt(np.maximum(w[top], 1e-12))
    X -= X.mean(0)
    radius = float(np.mean(np.linalg.norm(X, axis=1)))
    X *= radius / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)

    def residual(flat):
        P = flat.reshape(C, 3)
        d = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
        s = sphere_weight * (np.linalg.norm(P, axis=1) - radius) / radius
        return np.r_[(d - L) / L, s]

    def jac(flat):
        P = flat.reshape(C, 3)
        diff = P[E[:, 0]] - P[E[:, 1]]
        d = np.maximum(np.linalg.norm(diff, axis=1), 1e-12)
        u = diff / d[:, None] / L[:, None]
        m = len(E)
        rows = np.repeat(np.arange(m), 3)
        cols_a = (E[:, 0][:, None] * 3 + np.arange(3)).ravel()
        cols_b = (E[:, 1][:, None] * 3 + np.arange(3)).ravel()
        vals = u.ravel()
        un = P / np.maximum(np.linalg.norm(P, axis=1), 1e-12)[:, None] * (sphere_weight / radius)
        rows2 = m + np.repeat(np.arange(C), 3)
        cols2 = np.arange(C * 3)
        return coo_matrix((np.r_[vals, -vals, un.ravel()],
                           (np.r_[rows, rows, rows2], np.r_[cols_a, cols_b, cols2])),
                          shape=(m + C, 3 * C)).tocsr()

    sol = least_squares(residual, X.ravel(), jac=jac, method="trf", x_scale="jac", max_nfev=max_nfev,
                        xtol=1e-10, ftol=1e-10)
    P = sol.x.reshape(C, 3)
    rel = np.abs(np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1) - L) / L
    return Embedding(result, cls, P, np.array(tris), np.array(tri_tile), float(np.mean(np.linalg.norm(P, axis=1))),
                     float(rel.max()), float(np.sqrt(np.mean(rel ** 2))))


# ---------------------------------------------------------------------------------------------------
# two spheres sewn together through a hole each
# ---------------------------------------------------------------------------------------------------
@dataclass
class JoinedEmbedding:
    surface: object                 # connect.JoinedSurface
    position: np.ndarray            # (C, 3)
    triangles: np.ndarray           # (T, 3)
    triangle_part: np.ndarray       # (T,) 0 or 1
    triangle_tile: np.ndarray       # (T,) local tile index inside its part
    rim: np.ndarray                 # class indices of the joined vertices (in order of hole 1)
    open_loops: list                # [(part, hole, class index array)]
    strain: float
    rms_strain: float
    reflected: bool                 # part 2 was mirrored to fit


def _rot_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation taking unit vector a to unit vector b."""
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(np.dot(a, b))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        p = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, [0, 1.0, 0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def _axis_rot(axis: np.ndarray, ang: float) -> np.ndarray:
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def embed_joined(js, max_nfev: int = 600) -> JoinedEmbedding:
    (pa, ha, pb, hb), r = js.joins[0], js.rotation[0]
    ra, rb = js.parts[pa], js.parts[pb]
    ea, eb = embed(ra), embed(rb)
    Ca, Cb = len(ea.position), len(eb.position)
    n = ra.holes[ha].n
    va = ea.hole_vertices(ra.holes[ha])
    vb = eb.hole_vertices(rb.holes[hb])
    partner = [(r - i + 1) % n for i in range(n)]       # vertex i of hole a <-> vertex partner[i] of hole b
    ga, gb = ea.hole_geometry(ra.holes[ha]), eb.hole_geometry(rb.holes[hb])
    ca, cb = ga["centre"], gb["centre"]
    Pa = ea.position.copy()
    best = None
    for refl in (False, True):
        Pb0 = eb.position.copy()
        cb0 = cb.copy()
        if refl:
            Pb0[:, 0] *= -1
            cb0 = cb.copy()
            cb0[0] *= -1
        R0 = _rot_to(cb0, -ca)
        for ang in np.linspace(0, 2 * math.pi, 180, endpoint=False):
            R = _axis_rot(ca, ang) @ R0
            Pb = Pb0 @ R.T + (ea.radius + eb.radius) * ca
            d = sum(np.sum((Pa[va[i]] - Pb[vb[partner[i]]]) ** 2) for i in range(n))
            if best is None or d < best[0]:
                best = (d, Pb, refl)
    _, Pb, refl = best
    # merge classes of the sewn rim
    mapb = np.arange(Cb) + Ca
    for i in range(n):
        mapb[vb[partner[i]]] = va[i]
    uniq = np.unique(np.r_[np.arange(Ca), mapb])
    remap = {int(c): k for k, c in enumerate(uniq)}
    C = len(uniq)
    X = np.zeros((C, 3))
    cnt = np.zeros(C)
    for c in range(Ca):
        X[remap[c]] += Pa[c]
        cnt[remap[c]] += 1
    for c in range(Cb):
        X[remap[int(mapb[c])]] += Pb[c]
        cnt[remap[int(mapb[c])]] += 1
    X /= cnt[:, None]
    tris, tpart, ttile, edges = [], [], [], {}
    for part, (em, off) in enumerate(((ea, None), (eb, mapb))):
        res = em.result
        g = res.graph
        base = None
        for lt, t in enumerate(g.tile_ids):
            poly = g.patch.straight_polygon(t)
            if base is None:
                base = triangulate(poly)
            for (a, b, c) in base:
                ids = [int(em.vertex_class[g.tile_vertex[lt, x]]) for x in (a, b, c)]
                if off is not None:
                    ids = [int(off[i]) for i in ids]
                ids = [remap[i] for i in ids]
                tris.append(ids)
                tpart.append(part)
                ttile.append(lt)
                for (p, q), (pa_, pb_) in (((0, 1), (a, b)), ((1, 2), (b, c)), ((2, 0), (c, a))):
                    i, j = ids[p], ids[q]
                    if i != j:
                        key = (min(i, j), max(i, j))
                        d = float(np.linalg.norm(poly[pa_] - poly[pb_]))
                        edges[key] = min(edges.get(key, d), d)
    E = np.array(list(edges.keys()))
    L = np.array([edges[tuple(e)] for e in E])

    # weak pull of each part towards its own sphere (keeps the two balls round; the seam is free)
    owner = np.full(C, -1)
    for c in range(Ca):
        owner[remap[c]] = 0
    for c in range(Cb):
        k = remap[int(mapb[c])]
        owner[k] = 1 if owner[k] == -1 else owner[k]
    for i in range(n):
        owner[remap[int(va[i])]] = -1                  # rim belongs to both
    cen = [Pa.mean(0), Pb.mean(0)]
    rad = [float(np.mean(np.linalg.norm(Pa - cen[0], axis=1))), float(np.mean(np.linalg.norm(Pb - cen[1], axis=1)))]
    pulled = np.where(owner >= 0)[0]
    W = 0.08

    def sph(P):
        cc = np.array([cen[owner[k]] for k in pulled])
        rr = np.array([rad[owner[k]] for k in pulled])
        d = P[pulled] - cc
        return d, np.maximum(np.linalg.norm(d, axis=1), 1e-12), rr

    def residual(flat):
        P = flat.reshape(C, 3)
        d, nd, rr = sph(P)
        return np.r_[(np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1) - L) / L, W * (nd - rr) / rr]

    def jac(flat):
        P = flat.reshape(C, 3)
        diff = P[E[:, 0]] - P[E[:, 1]]
        d = np.maximum(np.linalg.norm(diff, axis=1), 1e-12)
        u = diff / d[:, None] / L[:, None]
        m = len(E)
        rows = np.repeat(np.arange(m), 3)
        ca_ = (E[:, 0][:, None] * 3 + np.arange(3)).ravel()
        cb_ = (E[:, 1][:, None] * 3 + np.arange(3)).ravel()
        dd, nd, rr = sph(P)
        un = dd / nd[:, None] * (W / rr)[:, None]
        rows2 = m + np.repeat(np.arange(len(pulled)), 3)
        cols2 = (pulled[:, None] * 3 + np.arange(3)).ravel()
        return coo_matrix((np.r_[u.ravel(), -u.ravel(), un.ravel()],
                           (np.r_[rows, rows, rows2], np.r_[ca_, cb_, cols2])),
                          shape=(m + len(pulled), 3 * C)).tocsr()

    sol = least_squares(residual, X.ravel(), jac=jac, method="trf", x_scale="jac", max_nfev=max_nfev,
                        xtol=1e-10, ftol=1e-10)
    P = sol.x.reshape(C, 3)
    rel = np.abs(residual(sol.x)[:len(E)])
    rim = np.array([remap[int(va[i])] for i in range(n)])
    loops = []
    for (pi, hi, _K) in js.open_holes:
        em = ea if pi == pa else eb
        cls = em.hole_vertices(js.parts[pi].holes[hi])
        if pi == pb:
            cls = mapb[cls]
        loops.append((pi, hi, np.array([remap[int(c)] for c in cls])))
    return JoinedEmbedding(js, P, np.array(tris), np.array(tpart), np.array(ttile), rim, loops,
                           float(rel.max()), float(np.sqrt(np.mean(rel ** 2))), refl)
