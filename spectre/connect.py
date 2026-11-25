"""
Hole signatures and compatibility, for joining folded spheres through their holes later.

A hole is described by the cyclic list of tile angles theta_i at its corners and the cyclic list of edge
kinds ('a'/'b') of its unit edges.  Two holes can be joined edge to edge when they have the same number of
sides, the edge kinds match when one boundary is read backwards, and at every joined vertex the two tile
angles add up to at most 360 degrees (equal to 360 means the joined vertex is flat, less means a cone point).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from .fold import FoldResult, Hole, N_E


def hole_kinds(res: FoldResult, hole: Hole) -> List[str]:
    return [str(res.graph.edge_kind[h % N_E]) for h in hole.edges]


def _canon(seq):
    n = len(seq)
    best = None
    for r in range(n):
        for s in (list(seq[r:] + seq[:r]), list(reversed(seq[r:] + seq[:r]))):
            if best is None or s < best:
                best = s
    return tuple(best)


def signature(res: FoldResult, hole: Hole) -> Tuple:
    """Canonical (corner angles, edge kinds) up to rotation and reflection."""
    th, kd = list(hole.corners), hole_kinds(res, hole)
    n = len(th)
    best = None
    for refl in (False, True):
        t = list(th)
        k = list(kd)
        if refl:
            # reading the boundary backwards: corner i sits between edge i-1 and edge i
            t = [th[(-i) % n] for i in range(n)]
            k = [kd[(-i - 1) % n] for i in range(n)]
        for r in range(n):
            cand = (tuple(t[r:] + t[:r]), tuple(k[r:] + k[:r]))
            if best is None or cand < best:
                best = cand
    return best


def join_vertices(res1, hole1, res2, hole2) -> Optional[List[int]]:
    """Angles at the joined vertices (tile angles of side 1 plus side 2) for the best matching, or None when
    the boundaries cannot be glued edge to edge.  A hole is glued to the other one traversed backwards."""
    n = hole1.n
    if hole2.n != n:
        return None
    k1 = hole_kinds(res1, hole1)
    k2 = hole_kinds(res2, hole2)
    t1, t2 = hole1.corners, hole2.corners
    best = None
    for r in range(n):
        # edge i of hole 1 meets edge (r - i) mod n of hole 2 (reversed direction)
        if all(k1[i] == k2[(r - i) % n] for i in range(n)):
            # vertex at the start of edge i of hole 1 meets the END of edge (r-i) of hole 2
            # = the start of edge (r - i + 1) of hole 2
            ang = [t1[i] + t2[(r - i + 1) % n] for i in range(n)]
            if best is None or max(ang) < max(best):
                best = ang
    if best is None or max(best) > 360:
        return None
    return best
