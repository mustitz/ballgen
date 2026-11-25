"""
Folding a patch of the Spectre tiling into a closed surface with rigid tiles, and keeping track of the holes.

The model
---------
Every tile stays an exact flat Spectre (all 14 edges have length 1 for a = b = 1, and every corner
angle is a multiple of 30 degrees).  A fold glues boundary edges of the patch to each other:

* two boundary edges can be glued when they are different tiles that do not already share an edge
  (and, optionally, have the same kind a / b), run in opposite directions, and have equal length;
* gluing merges the corners at their ends, and the angle of a merged corner is the SUM of the angles
  (the tile corners that now meet at that vertex).  A corner may never exceed 360 degrees;
* a corner that closes up completely (its two boundary edges are glued) becomes a CONE point with
  angle deficit 360 - (angle sum);
* whatever boundary loops are left when no more gluing is allowed are the HOLES.

Because the only curvature a flat-tile surface can carry is at its vertices, the bookkeeping is exact
integer arithmetic (degrees).  Write n for the number of edges of a hole and theta_i for the angle the
tiles fill at its corners.  The hole is a polygon with n unit sides and interior angles
alpha_i = 360 - theta_i, and the curvature it must hold if it were closed up is

        K = sum_i alpha_i - (n - 2) * 180        ( = 360 + 180 n - sum_i theta_i ).

The discrete Gauss-Bonnet theorem for the folded sphere then reads

        sum over cone points (deficit)  +  sum over holes K  =  720 degrees,            (*)

which this module asserts for every fold.  A hole with K = 0 can be filled by a FLAT equilateral polygon
with exactly those angles if it closes up (a regular hexagon when all alpha_i = 120, the hole in the
photo of a hand-sewn ball); a hole with K > 0 is a cap that needs positive curvature, K < 0 a saddle.

    from spectre.patch import SpectrePatch
    from spectre.fold import PatchGraph, fold, grow_patch

    patch = SpectrePatch(levels=2)                       # 71 tiles
    graph = PatchGraph(patch, tiles=grow_patch(patch, 40))   # a connected patch of 40 tiles
    result = fold(graph, max_deficit=180)
    result.summary()                                     # cones, holes, the ledger (*)
    result.holes[0].corners, result.holes[0].K, result.holes[0].kind

Everything here is combinatorial and exact; ``spectre.embed`` computes 3D positions for pictures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import tile as so
from .patch import SpectrePatch

N_E = so.N_EDGES
FULL = 360


# --------------------------------------------------------------------------
# Patch as a graph with exact integer corner angles
# --------------------------------------------------------------------------
def corner_angles(poly: np.ndarray) -> np.ndarray:
    """Interior angles (integer degrees) at the corners of a placed tile polygon."""
    prev, nxt = np.roll(poly, 1, axis=0), np.roll(poly, -1, axis=0)
    a, b = prev - poly, nxt - poly
    ang = np.degrees(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1))) % 360
    r = np.round(ang / 30.0) * 30
    if not np.allclose(ang, r, atol=1e-6):
        raise AssertionError("tile corner angles are not multiples of 30 degrees")
    return r.astype(int)


def grow_patch(patch: SpectrePatch, n_tiles: int, seed: Optional[int] = None) -> List[int]:
    """A connected set of ``n_tiles`` tiles, grown breadth-first (by shared edges) from the tile nearest
    the patch centroid (or from tile ``seed``).  Ties are broken by distance to the centroid."""
    if not 1 <= n_tiles <= patch.n_tiles:
        raise ValueError("n_tiles must be between 1 and %d" % patch.n_tiles)
    cents = np.array([patch.straight_polygon(t).mean(0) for t in range(patch.n_tiles)])
    c0 = cents.mean(0)
    start = int(np.argmin(np.linalg.norm(cents - c0, axis=1))) if seed is None else int(seed)
    nbrs: Dict[int, set] = {t: set() for t in range(patch.n_tiles)}
    for t, _, u, _ in patch.edge_pairs():
        nbrs[int(t)].add(int(u))
        nbrs[int(u)].add(int(t))
    chosen, frontier = [start], {start}
    in_set = {start}
    while len(chosen) < n_tiles:
        cand = {u for t in frontier for u in nbrs[t]} - in_set
        if not cand:
            frontier = set(in_set)
            cand = {u for t in frontier for u in nbrs[t]} - in_set
            if not cand:
                break
        nxt = min(cand, key=lambda u: (np.linalg.norm(cents[u] - c0), u))
        chosen.append(nxt)
        in_set.add(nxt)
        frontier = {nxt} | {t for t in frontier if nbrs[t] - in_set}
    return chosen


class PatchGraph:
    """Vertices, half-edges, corner angles and boundary loops of (a subset of) a Spectre patch.

    Half-edge ``h = 14 * local_tile + k`` is edge k of the tile, from corner k to corner k + 1.
    """

    def __init__(self, patch: SpectrePatch, tiles: Optional[Sequence[int]] = None):
        self.patch = patch
        self.tile_ids = list(range(patch.n_tiles)) if tiles is None else [int(t) for t in tiles]
        T = len(self.tile_ids)
        self.n_tiles = T
        keys: Dict[Tuple[float, float], int] = {}
        xy: List[Tuple[float, float]] = []
        tv = np.zeros((T, N_E), dtype=int)
        ang = np.zeros((T, N_E), dtype=int)
        for lt, t in enumerate(self.tile_ids):
            poly = patch.straight_polygon(t)
            ang[lt] = corner_angles(poly)
            for k, p in enumerate(poly):
                key = (round(float(p[0]), 6) + 0.0, round(float(p[1]), 6) + 0.0)
                if key not in keys:
                    keys[key] = len(xy)
                    xy.append(key)
                tv[lt, k] = keys[key]
        self.xy = np.array(xy)
        self.tile_vertex = tv                      # (T, 14) vertex id of corner k
        self.angle = ang                           # (T, 14) integer degrees
        self.edge_kind = np.array([so.EDGE_KIND[k] for k in range(N_E)])
        start, end = tv, np.roll(tv, -1, axis=1)
        directed: Dict[Tuple[int, int], int] = {}
        for h in range(T * N_E):
            s, e = int(start.flat[h]), int(end.flat[h])
            if (s, e) in directed:
                raise AssertionError("two tiles run the same directed edge: they overlap")
            directed[(s, e)] = h
        self.partner: Dict[int, int] = {}
        for (s, e), h in directed.items():
            o = directed.get((e, s))
            if o is not None:
                self.partner[h] = o
        # vertex angle sums and which vertices are interior
        self.vertex_angle = np.zeros(len(xy), dtype=int)
        for lt in range(T):
            for k in range(N_E):
                self.vertex_angle[tv[lt, k]] += ang[lt, k]
        self.loops = self._boundary_loops()

    def tile_of(self, h: int) -> int:
        return h // N_E

    def edge_index(self, h: int) -> int:
        return h % N_E

    def start_vertex(self, h: int) -> int:
        return int(self.tile_vertex[h // N_E, h % N_E])

    def end_vertex(self, h: int) -> int:
        return int(self.tile_vertex[h // N_E, (h % N_E + 1) % N_E])

    def _boundary_loops(self):
        T = self.n_tiles
        bnd = {h for h in range(T * N_E) if h not in self.partner}
        seen, loops = set(), []
        for h0 in sorted(bnd):
            if h0 in seen:
                continue
            edges, ang_end = [], []
            h = h0
            while True:
                seen.add(h)
                edges.append(h)
                t, k = divmod(h, N_E)
                total = int(self.angle[t, (k + 1) % N_E])
                g = t * N_E + (k + 1) % N_E            # outgoing edge of the same tile at the end vertex
                while g not in bnd:
                    u, m = divmod(self.partner[g], N_E)
                    total += int(self.angle[u, (m + 1) % N_E])
                    g = u * N_E + (m + 1) % N_E
                ang_end.append(total)
                h = g
                if h == h0:
                    break
            # angle at the START of edge i is the angle at the end of edge i - 1
            ang_start = [ang_end[-1]] + ang_end[:-1]
            loops.append((edges, ang_start))
        return loops

    def boundary_turning(self) -> int:
        """Sum over all boundary loops of (180 n - sum of angles): +360 for a disc, 0 for an annulus."""
        return sum(180 * len(e) - sum(a) for e, a in self.loops)


# --------------------------------------------------------------------------
# Result objects
# --------------------------------------------------------------------------
@dataclass
class Cone:
    vertex: int            # a vertex id of the patch graph (representative)
    angle: int             # sum of the tile corners meeting there (degrees)

    @property
    def deficit(self) -> int:
        return FULL - self.angle


@dataclass
class Hole:
    edges: List[int]       # half-edges (14 * tile + edge) around the hole, in order
    corners: List[int]     # angle the tiles fill at the START of each edge
    vertices: List[int]    # representative vertex id at the start of each edge (after identification)

    @property
    def n(self) -> int:
        return len(self.edges)

    @property
    def alpha(self) -> List[int]:
        """Interior angle of the hole polygon at each corner."""
        return [FULL - c for c in self.corners]

    @property
    def K(self) -> int:
        """Curvature the hole must hold if closed up: sum(alpha) - (n - 2) 180."""
        return sum(self.alpha) - (self.n - 2) * 180

    def polygon(self) -> Optional[np.ndarray]:
        """Vertices of the flat equilateral polygon with these interior angles, or None if it does not
        close (or K != 0).  Unit sides, counterclockwise."""
        if self.K != 0 or any(a <= 0 or a >= 360 for a in self.alpha):
            if self.K != 0:
                return None
        pts, heading, p = [np.zeros(2)], 0.0, np.zeros(2)
        for a in self.alpha[1:] + self.alpha[:1]:
            p = p + np.array([math.cos(heading), math.sin(heading)])
            pts.append(p.copy())
            heading += math.radians(180 - a)
        pts = np.array(pts)
        if np.linalg.norm(pts[-1] - pts[0]) > 1e-6:
            return None
        return pts[:-1]

    @property
    def closes(self) -> bool:
        return self.polygon() is not None

    @property
    def kind(self) -> str:
        a = self.alpha
        n = self.n
        reflex = sum(1 for x in a if x > 180)
        if n == 2:
            return "slit"
        if all(x == a[0] for x in a):
            return f"regular {n}-gon" if self.K == 0 else f"equiangular {n}-gon (K={self.K})"
        if reflex >= 2 and n % 2 == 0 and all((a[i] > 180) != (a[i + 1] > 180) for i in range(n - 1)) \
                and (a[-1] > 180) != (a[0] > 180):
            return f"star {n // 2}-point"
        base = {3: "triangle", 4: "quadrilateral", 5: "pentagon", 6: "hexagon"}.get(n, f"{n}-gon")
        return base if reflex == 0 else f"non-convex {base}"


@dataclass
class FoldResult:
    graph: PatchGraph
    glued: List[Tuple[int, int]]                 # glued half-edge pairs
    cones: List[Cone]
    holes: List[Hole]
    parent: List[int]                            # union-find over vertex ids
    max_deficit: int
    match_kinds: bool

    def find(self, v: int) -> int:
        while self.parent[v] != v:
            self.parent[v] = self.parent[self.parent[v]]
            v = self.parent[v]
        return v

    @property
    def total_deficit(self) -> int:
        return sum(c.deficit for c in self.cones)

    @property
    def total_hole_curvature(self) -> int:
        return sum(h.K for h in self.holes)

    @property
    def ledger(self) -> int:
        """Cone deficits + hole curvatures; the discrete Gauss-Bonnet theorem says this is 720."""
        return self.total_deficit + self.total_hole_curvature

    def summary(self) -> str:
        g = self.graph
        lines = [f"{g.n_tiles} tiles, {len(self.glued)} glued edge pairs, {len(self.cones)} cone points, "
                 f"{len(self.holes)} holes; ledger {self.total_deficit} + {self.total_hole_curvature} = {self.ledger}"]
        by = {}
        for c in self.cones:
            by[c.deficit] = by.get(c.deficit, 0) + 1
        if by:
            lines.append("cone deficits: " + ", ".join(f"{d}deg x{n}" for d, n in sorted(by.items())))
        for i, h in enumerate(self.holes):
            lines.append(f"hole {i}: {h.n} edges, K={h.K}, {h.kind}, alpha={h.alpha}, closes={h.closes}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# The fold
# --------------------------------------------------------------------------
def fold(graph: PatchGraph, max_deficit: int = 180, match_kinds: bool = True, max_span: int = 1,
         rigid_flat: bool = False) -> FoldResult:
    """Greedily glue boundary edges of the patch until nothing more is allowed.

    At every step all admissible pairs of boundary edges (within one boundary loop) are scored by the
    angle left open at the two merged corners, ``(360 - merged_1) + (360 - merged_2)``; the tightest fit
    is glued (ties: the pair closest along the boundary).  A pair is admissible when

    * the two edges belong to different tiles, and (``match_kinds``) have the same kind; with
      ``rigid_flat`` the two tiles must also not already share an edge (two flat tiles cannot close a cone
      around a vertex, a convex vertex needs three faces); by default tiles may crease like paper, which
      is intrinsically flat and does not change any angle;
    * both merged corners stay at most 360 degrees;
    * a corner that closes into a cone point has deficit at most ``max_deficit``;
    * the two edges are at most ``max_span`` steps apart along the boundary loop (1: only neighbouring
      edges are zipped, like closing a bag; larger values allow longer seams and split the boundary).
    """
    T = graph.n_tiles
    parent = list(range(len(graph.xy)))

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    adj = np.zeros((T, T), dtype=bool)
    for h, o in graph.partner.items():
        adj[h // N_E, o // N_E] = True
    loops = [(list(e), list(a)) for e, a in graph.loops]
    cones: List[Cone] = []
    glued: List[Tuple[int, int]] = []

    def best_in_loop(edges, ang):
        m = len(edges)
        if m < 2:
            return None
        I, J = np.triu_indices(m, 1)
        E = np.array(edges)
        tile = E // N_E
        kind = np.array([graph.edge_kind[h % N_E] for h in edges])
        A = np.array(ang)
        ok = tile[I] != tile[J]
        if rigid_flat:
            ok &= ~adj[tile[I], tile[J]]
        if match_kinds:
            ok &= kind[I] == kind[J]
        inner_empty = J == I + 1
        outer_empty = (I == 0) & (J == m - 1)
        # corner angle that results from merging b with c (inner side) and a with d (outer side)
        inner = np.where(inner_empty, A[(I + 1) % m], A[(I + 1) % m] + A[J])
        outer = np.where(outer_empty, A[I], A[(J + 1) % m] + A[I])
        ok &= (inner <= FULL) & (outer <= FULL)
        # cone events
        ok &= ~(inner_empty & (FULL - inner > max_deficit))
        ok &= ~(outer_empty & (FULL - outer > max_deficit))
        ok &= np.minimum(J - I, m - (J - I)) <= max_span
        if not ok.any():
            return None
        cost = (FULL - inner) + (FULL - outer)
        dist = np.minimum(J - I, m - (J - I))
        score = np.where(ok, cost * 10000 + dist, np.iinfo(np.int64).max)
        k = int(np.argmin(score))
        return int(score[k]), int(I[k]), int(J[k])

    while True:
        best = None
        for li, (edges, ang) in enumerate(loops):
            r = best_in_loop(edges, ang)
            if r is not None and (best is None or r[0] < best[0]):
                best = (r[0], li, r[1], r[2])
        if best is None:
            break
        _, li, i, j = best
        edges, ang = loops.pop(li)
        m = len(edges)
        hi, hj = edges[i], edges[j]
        glued.append((hi, hj))
        t1, t2 = hi // N_E, hj // N_E
        adj[t1, t2] = adj[t2, t1] = True
        a, b = graph.start_vertex(hi), graph.end_vertex(hi)
        c, d = graph.start_vertex(hj), graph.end_vertex(hj)
        union(b, c)
        union(a, d)
        # inner loop: edges i+1 .. j-1, first corner = b + c
        if j == i + 1:
            cones.append(Cone(b, ang[(i + 1) % m]))
        else:
            e_in = edges[i + 1:j]
            a_in = [ang[i + 1] + ang[j]] + ang[i + 2:j]
            loops.append((e_in, a_in))
        # outer loop: edges j+1 .. m-1, 0 .. i-1, first corner = d + a
        n_out = m - (j - i + 1)
        if n_out == 0:
            cones.append(Cone(a, ang[i]))
        else:
            idx = [(j + 1 + s) % m for s in range(n_out)]
            e_out = [edges[x] for x in idx]
            a_out = [ang[idx[0]] + ang[i]] + [ang[x] for x in idx[1:]]
            loops.append((e_out, a_out))

    holes = []
    for edges, ang in loops:
        verts = [find(graph.start_vertex(h)) for h in edges]
        holes.append(Hole(edges, ang, verts))
    res = FoldResult(graph, glued, cones, holes, parent, max_deficit, match_kinds)
    res.max_span, res.rigid_flat = max_span, rigid_flat
    expected = 720
    if res.ledger != expected:
        raise AssertionError(f"Gauss-Bonnet ledger {res.ledger} != {expected} (bug)")
    return res


# --------------------------------------------------------------------------
# Search: zip a single boundary loop with a curvature budget
# --------------------------------------------------------------------------
def fold_search(graph: PatchGraph, max_deficit: int = 180, budget: int = FULL * 2, match_kinds: bool = False,
                beam: int = 40, loop: int = 0, objective: str = "small_hole") -> FoldResult:
    """Zip neighbouring boundary edges of one boundary loop, searching over the order with a beam.

    Each zip closes the corner between two neighbouring boundary edges into a cone point of deficit
    ``360 - angle`` and merges the corners at the far ends (their angles add, never above 360).  The
    total cone deficit is limited to ``budget`` (720 degrees is the whole sphere), so the hole that is left
    has K = 720 - (total deficit) >= 0: a cap, or exactly flat when the budget is used up.

    ``objective``: "small_hole" keeps the fewest unglued edges (ties: smallest K); "flat_hole" prefers
    a hole with K = 0 that closes into a flat equilateral polygon, then the fewest edges.
    """
    edges0, ang0 = graph.loops[loop]
    kinds = {h: graph.edge_kind[h % N_E] for h in edges0}
    start = (tuple(edges0), tuple(ang0))
    # a state is (edges, corners, cost, parent_state_id, move); keep a table to rebuild the zips
    table: List[Tuple[int, Optional[int], Optional[Tuple[int, int]], int]] = [(0, None, None, 0)]
    level = {start: 0}
    keys = {0: start}
    best_final = None

    def final_score(edges, corners, cost):
        n = len(edges)
        K = 720 - cost
        if objective == "flat_hole":
            hole = Hole(list(edges), list(corners), [0] * n)
            return (0 if (K == 0 and (n == 0 or hole.closes)) else 1, n, K)
        return (n, K)

    seen_global = set()
    while level:
        nxt_level: Dict[Tuple, int] = {}
        for state, sid in level.items():
            edges, corners = state
            cost = table[sid][0]
            m = len(edges)
            moved = False
            if m == 2:
                d0, d1 = FULL - corners[0], FULL - corners[1]
                if 0 <= d0 <= max_deficit and 0 <= d1 <= max_deficit and cost + d0 + d1 <= budget:
                    moved = True
                    table.append((cost + d0 + d1, sid, (edges[0], edges[1]), len(table)))
                    fin = ((), (), cost + d0 + d1)
                    sc = final_score((), (), cost + d0 + d1)
                    if best_final is None or sc < best_final[0]:
                        best_final = (sc, len(table) - 1, ((), ()))
            elif m > 2:
                for i in range(m):
                    j = (i + 1) % m
                    x = corners[j]
                    d = FULL - x
                    if d < 0 or d > max_deficit or cost + d > budget:
                        continue
                    if match_kinds and kinds[edges[i]] != kinds[edges[j]]:
                        continue
                    k = (i + 2) % m
                    y = corners[i] + corners[k]
                    if y > FULL:
                        continue
                    # new loop: remove edges i, j and corner j; corner i and k merge
                    if m == 3:
                        # one edge left with merged corner
                        keep = [e for t, e in enumerate(edges) if t not in (i, j)]
                        new_edges = tuple(keep)
                        new_corners = (y,)
                    else:
                        order = [(k + s) % m for s in range(m - 2)]
                        new_edges = tuple(edges[t] for t in order)
                        new_corners = tuple([y] + [corners[t] for t in order[1:]])
                    moved = True
                    key = (new_edges, new_corners)
                    if key in seen_global:
                        continue
                    if key not in nxt_level or cost + d < table[nxt_level[key]][0]:
                        table.append((cost + d, sid, (edges[i], edges[j]), len(table)))
                        nxt_level[key] = len(table) - 1
            if not moved or m < 2:
                sc = final_score(edges, corners, cost)
                if best_final is None or sc < best_final[0]:
                    best_final = (sc, sid, state)
            else:
                # also consider stopping here
                sc = final_score(edges, corners, cost)
                if best_final is None or sc < best_final[0]:
                    best_final = (sc, sid, state)
        # prune the beam: lowest cost first
        items = sorted(nxt_level.items(), key=lambda kv: (table[kv[1]][0], sum(c * c for c in kv[0][1])))[:beam]
        for st, _ in items:
            seen_global.add(st)
        level = dict(items)
        for st, sid in level.items():
            keys[sid] = st
    sc, sid, state = best_final
    # rebuild the zips by following parents
    zips = []
    cur = sid
    while table[cur][1] is not None:
        zips.append(table[cur][2])
        cur = table[cur][1]
    zips.reverse()
    return _replay(graph, zips, loop, max_deficit, match_kinds)


def _replay(graph: PatchGraph, zips, loop: int, max_deficit: int, match_kinds: bool) -> FoldResult:
    """Apply a list of (half-edge, half-edge) zips of one loop using the exact fold bookkeeping."""
    parent = list(range(len(graph.xy)))

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    loops = [(list(e), list(a)) for e, a in graph.loops]
    cones: List[Cone] = []
    glued: List[Tuple[int, int]] = []
    for h1, h2 in zips:
        li = next(n for n, (e, _) in enumerate(loops) if h1 in e and h2 in e)
        edges, ang = loops.pop(li)
        m = len(edges)
        i, j = edges.index(h1), edges.index(h2)
        if (j - i) % m != 1:
            i, j = j, i
        assert (j - i) % m == 1, "replay: edges are not neighbours"
        if i > j:                       # wrap-around neighbours: rotate so that i = 0, j = 1
            r = i
            edges = edges[r:] + edges[:r]
            ang = ang[r:] + ang[:r]
            i, j = 0, 1
        glued.append((edges[i], edges[j]))
        a, b = graph.start_vertex(edges[i]), graph.end_vertex(edges[i])
        c, d = graph.start_vertex(edges[j]), graph.end_vertex(edges[j])
        union(b, c)
        union(a, d)
        cones.append(Cone(b, ang[j % m]))
        n_out = m - 2
        if n_out > 0:
            idx = [(j + 1 + s) % m for s in range(n_out)]
            loops.append(([edges[x] for x in idx], [ang[idx[0]] + ang[i]] + [ang[x] for x in idx[1:]]))
        else:
            cones.append(Cone(a, ang[i]))
    holes = []
    for edges, ang in loops:
        holes.append(Hole(edges, ang, [find(graph.start_vertex(h)) for h in edges]))
    res = FoldResult(graph, glued, cones, holes, parent, max_deficit, match_kinds)
    res.max_span, res.rigid_flat = 1, False
    if res.ledger != 720:
        raise AssertionError(f"Gauss-Bonnet ledger {res.ledger} != 720 (bug)")
    return res


# ---------------------------------------------------------------------------------------------------
# folding toward a chosen hole
# ---------------------------------------------------------------------------------------------------
#: target holes as the tile angle (degrees) wanted at each corner.  alpha = 360 - theta is the angle
#: seen inside the hole; K = sum(alpha) - (n-2)*180.  Only multiples of 30 deg are reachable.
TARGETS = {
    "hexagon": [240] * 6,       # regular, K = 0: flat-fillable
    "pentagon": [240] * 5,      # alpha = 120 each, K = 60 (a Goldberg-style cap; 12 of them = 720)
    "square": [270] * 4,        # alpha = 90, K = 0
    "triangle": [300] * 3,      # alpha = 60, K = 0
    "dodecagon": [210] * 12,    # alpha = 150, K = 0
}


def target_curvature(theta: Sequence[int]) -> int:
    n = len(theta)
    return sum(FULL - t for t in theta) - (n - 2) * 180


def fold_to_target(graph: PatchGraph, theta: Sequence[int], loop: int = 0, beam: int = 150,
                   max_deficit: int = 180, tries: int = 400, seed: int = 0,
                   time_limit: float = 60.0) -> Optional[FoldResult]:
    """Fold the patch so that the leftover hole has exactly the corner angles ``theta``.

    ``len(theta)`` boundary edges are reserved (never zipped); the gaps between them hold an even
    number of other edges, which are zipped away by a beam search.  Zips may not push the corner at a
    reserved edge above its target.  Layouts of the reserved edges are drawn at random.  Returns None if
    no layout worked within ``tries`` / ``time_limit``; otherwise a ``FoldResult`` whose single hole has
    the target corners (checked) and whose ledger is 720.
    """
    import random
    import time
    k = len(theta)
    edges0, ang0 = graph.loops[loop]
    m = len(edges0)
    if (m - k) % 2 or m < k:
        return None
    rng = random.Random(seed)
    t0 = time.time()
    for _ in range(tries):
        if time.time() - t0 > time_limit:
            break
        # random even gaps summing to m - k
        pairs = (m - k) // 2
        cuts = sorted(rng.randint(0, pairs) for _ in range(k - 1))
        gaps = [2 * (b - a) for a, b in zip([0] + cuts, cuts + [pairs])]
        pos, p = [], rng.randrange(m)
        for g in gaps:
            pos.append(p % m)
            p += g + 1
        rot = rng.randrange(k)
        cap = {edges0[pos[j]]: theta[(j + rot) % k] for j in range(k)}
        if any(ang0[pos[j]] > cap[edges0[pos[j]]] for j in range(k)):
            continue
        # a reserved edge that follows another reserved edge directly keeps its own corner
        if any(gaps[(j - 1) % k] == 0 and ang0[pos[j]] != cap[edges0[pos[j]]] for j in range(k)):
            continue
        out = _beam_reserved(graph, loop, cap, beam, max_deficit)
        if out is not None:
            res = _replay(graph, out, loop, max_deficit, False)
            if len(res.holes) == 1 and res.holes[0].n == k and sorted(res.holes[0].corners) == sorted(theta):
                return res
    return None


def _beam_reserved(graph, loop, cap, beam, max_deficit):
    import random
    edges0, ang0 = graph.loops[loop]
    k = len(cap)
    cur = {(tuple(edges0), tuple(ang0)): ((), 0)}   # state -> (zip list, cost)
    while True:
        nxt = {}
        for (edges, corners), (zl, cost) in cur.items():
            m = len(edges)
            if m == k:
                if all(corners[t] == cap[edges[t]] for t in range(m)):
                    return list(zl)
                continue
            for i in range(m):
                j = (i + 1) % m
                if edges[i] in cap or edges[j] in cap:
                    continue
                d = FULL - corners[j]
                if d < 0 or d > max_deficit:
                    continue
                kk = (i + 2) % m
                y = corners[i] + corners[kk]
                if y > FULL or (edges[kk] in cap and y > cap[edges[kk]]):
                    continue
                order = [(kk + s) % m for s in range(m - 2)]
                ne = tuple(edges[t] for t in order)
                nc = tuple([y] + [corners[t] for t in order[1:]])
                key = (ne, nc)
                if key not in nxt or cost + d < nxt[key][1]:
                    nxt[key] = (zl + ((edges[i], edges[j]),), cost + d)
        if not nxt:
            return None
        def score(item):
            (ne, nc), (_, c) = item
            return (sum(cap[e] - nc[t] for t, e in enumerate(ne) if e in cap), c)
        ranked = sorted(nxt.items(), key=score)
        half = beam // 2
        rest = ranked[half:]
        random.Random(len(ranked)).shuffle(rest)
        cur = dict(ranked[:half] + rest[:beam - half])
