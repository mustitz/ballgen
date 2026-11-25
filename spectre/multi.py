"""
Multi-hole folding.

``fold.py`` only glues neighbouring boundary edges, which leaves one hole.  Gluing two NON-adjacent edges
i and j of a boundary loop splits it in two (edges i+1..j-1 and edges j+1..i-1).  Choosing j = i + n + 1
"carves" a loop of n edges out of the boundary: that loop is kept as a hole, the rest of the boundary is
zipped away with the usual adjacent-zip beam.  Several carves give several holes on one sphere; the
exact integer ledger (cone deficits + hole K = 720) still has to balance and is asserted.

Corner bookkeeping for gluing edges i < j of a loop with tile angles A[k] at the start of edge k:
edge i is glued to edge j reversed, so vertex i+1 meets vertex j and vertex j+1 meets vertex i.
  loop 1: edges i+1..j-1, corner at its first vertex A[i+1] + A[j]
  loop 2: edges j+1..i-1, corner at its first vertex A[j+1] + A[i]
A loop that would have no edges is a closed cone point (deficit 360 - its angle).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from .fold import (Cone, FoldResult, FULL, Hole, PatchGraph, N_E)


def split_loop(edges: Sequence[int], ang: Sequence[int], i: int, j: int):
    """Glue edge index i to edge index j (i != j).  Returns (loop1, loop2, cone_angles) where each loop is
    (edges, corners) or None if empty, and cone_angles are (which, angle) for closed cone points."""
    m = len(edges)
    edges, ang = list(edges), list(ang)
    r = i
    edges, ang = edges[r:] + edges[:r], ang[r:] + ang[:r]
    j = (j - r) % m
    assert j != 0
    n1 = j - 1
    n2 = m - j - 1
    cones = []
    l1 = l2 = None
    if n1 > 0:
        l1 = (edges[1:j], [ang[1] + ang[j]] + ang[2:j])
    else:
        cones.append(("b", ang[1]))
    if n2 > 0:
        l2 = (edges[j + 1:], [ang[j + 1] + ang[0]] + ang[j + 2:])
    else:
        cones.append(("a", ang[0]))
    return l1, l2, cones


def replay_ops(graph: PatchGraph, ops: Sequence[Tuple[int, int]], max_deficit: int = FULL) -> FoldResult:
    """Apply glue operations (pairs of half-edges lying in a common boundary loop, any positions)."""
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
    glued = []
    for h1, h2 in ops:
        li = next(n for n, (e, _) in enumerate(loops) if h1 in e and h2 in e)
        edges, ang = loops.pop(li)
        i, j = edges.index(h1), edges.index(h2)
        glued.append((h1, h2))
        a, b = graph.start_vertex(h1), graph.end_vertex(h1)
        c, d = graph.start_vertex(h2), graph.end_vertex(h2)
        union(b, c)
        union(a, d)
        l1, l2, cs = split_loop(edges, ang, i, j)
        for which, angle in cs:
            cones.append(Cone(b if which == "b" else a, angle))
        for l in (l1, l2):
            if l is not None:
                loops.append(l)
    holes = [Hole(e, a, [find(graph.start_vertex(h)) for h in e]) for e, a in loops]
    res = FoldResult(graph, glued, cones, holes, parent, max_deficit, False)
    res.max_span, res.rigid_flat = 0, False
    if res.ledger != 720:
        raise AssertionError(f"Gauss-Bonnet ledger {res.ledger} != 720 (bug)")
    return res


def _hole_K(corners: Sequence[int]) -> int:
    n = len(corners)
    return 360 + 180 * n - sum(corners)


def carve_candidates(edges, ang, n: int, min_rest: int = 0, max_K: int = 720, min_K: int = 0):
    """All (i, j, K, corners) that carve an n-edge hole loop out of the loop (j = i + n + 1)."""
    m = len(edges)
    out = []
    rest = m - n - 2
    if n < 3 or rest < 0 or rest == 1 or rest < min_rest or (m - n) % 2:
        return out
    for i in range(m):
        j = (i + n + 1) % m
        l1, l2, _ = split_loop(edges, ang, i, j)
        if l1 is None or len(l1[0]) != n:
            continue
        if max(l1[1]) >= FULL:
            continue
        if l2 is not None and max(l2[1]) > FULL:
            continue
        K = _hole_K(l1[1])
        if min_K <= K <= max_K:
            out.append((i, j, K, l1[1]))
    return out


def close_loop(edges, corners, beam: int = 150, max_deficit: int = 180, budget: int = 720):
    """Zip a loop completely away with adjacent zips (beam search).  Returns the zip list or None."""
    edges, corners = tuple(edges), tuple(corners)
    cur: Dict[Tuple, Tuple[tuple, int]] = {(edges, corners): ((), 0)}
    while cur:
        nxt = {}
        for (ed, co), (zl, cost) in cur.items():
            m = len(ed)
            if m == 0:
                return list(zl)
            if m == 2:
                d0, d1 = FULL - co[0], FULL - co[1]
                if 0 <= d0 <= max_deficit and 0 <= d1 <= max_deficit and cost + d0 + d1 <= budget:
                    return list(zl) + [(ed[0], ed[1])]
                continue
            for i in range(m):
                j = (i + 1) % m
                d = FULL - co[j]
                if d < 0 or d > max_deficit or cost + d > budget:
                    continue
                k = (i + 2) % m
                y = co[i] + co[k]
                if y > FULL:
                    continue
                if m == 3:
                    keep = [t for t in range(3) if t not in (i, j)]
                    ne, nc = (ed[keep[0]],), (y,)
                else:
                    order = [(k + s) % m for s in range(m - 2)]
                    ne = tuple(ed[t] for t in order)
                    nc = tuple([y] + [co[t] for t in order[1:]])
                key = (ne, nc)
                if key not in nxt or cost + d < nxt[key][1]:
                    nxt[key] = (zl + ((ed[i], ed[j]),), cost + d)
        cur = dict(sorted(nxt.items(), key=lambda kv: (kv[1][1], sum(c * c for c in kv[0][1])))[:beam])
    return None


def fold_multi(graph: PatchGraph, sizes: Sequence[int], tries: int = 200, beam: int = 150, seed: int = 0,
               loop: int = 0, flat_only: bool = False, time_limit: float = 60.0) -> Optional[FoldResult]:
    """Carve holes with the given side counts out of the boundary loop and zip the rest away.

    ``flat_only`` keeps only carves with K = 0 (holes that close up flat).  Returns a FoldResult with
    exactly ``len(sizes)`` holes, or None if no random choice of carve positions closed the rest.
    """
    import time
    rng = random.Random(seed)
    t0 = time.time()
    edges0, ang0 = graph.loops[loop]
    for _ in range(tries):
        if time.time() - t0 > time_limit:
            break
        edges, ang = list(edges0), list(ang0)
        ops: List[Tuple[int, int]] = []
        ok = True
        for n in sizes:
            cands = carve_candidates(edges, ang, n, max_K=0 if flat_only else 720)
            if flat_only:
                cands = [c for c in cands if c[2] == 0]
            if not cands:
                ok = False
                break
            i, j, K, cor = rng.choice(cands)
            ops.append((edges[i], edges[j]))
            _, l2, _ = split_loop(edges, ang, i, j)
            if l2 is None:
                ok = False
                break
            edges, ang = l2
        if not ok:
            continue
        zips = close_loop(edges, ang, beam=beam)
        if zips is None:
            continue
        try:
            res = replay_ops(graph, ops + zips)
        except AssertionError:
            continue
        if len(res.holes) == len(sizes):
            return res
    return None
