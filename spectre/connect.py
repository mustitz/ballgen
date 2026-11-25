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


# ---------------------------------------------------------------------------------------------------
# joining two folded spheres through a hole each
# ---------------------------------------------------------------------------------------------------
from dataclasses import dataclass, field


def best_join(res1: FoldResult, hole1: Hole, res2: FoldResult, hole2: Hole, match_kinds: bool = False):
    """Choose how to glue hole2 onto hole1 (boundaries run in opposite directions).

    Edge i of hole 1 meets edge (r - i) of hole 2 and vertex i (start of edge i of hole 1) meets vertex
    (r - i + 1) of hole 2.  Among the n rotations r the one is chosen that (1) agrees in edge kind most
    often (or exactly, when ``match_kinds``), then (2) keeps the joined vertex angles closest to 360.
    Returns (r, joined_angles, kinds_agree) or None when the side counts differ / kinds are required
    to match and cannot."""
    n = hole1.n
    if hole2.n != n:
        return None
    k1, k2 = hole_kinds(res1, hole1), hole_kinds(res2, hole2)
    best = None
    for r in range(n):
        agree = sum(k1[i] == k2[(r - i) % n] for i in range(n))
        if match_kinds and agree != n:
            continue
        ang = [hole1.corners[i] + hole2.corners[(r - i + 1) % n] for i in range(n)]
        key = (-agree, sum(abs(360 - a) for a in ang), r)
        if best is None or key < best[0]:
            best = (key, r, ang, agree == n)
    if best is None:
        return None
    return best[1], best[2], best[3]


@dataclass
class JoinedSurface:
    """Two folded patches sewn together along one hole each (the other holes stay open)."""
    parts: List[FoldResult]
    joins: List[Tuple[int, int, int, int]]     # (part a, hole a, part b, hole b)
    rotation: List[int]                        # r of every join
    junction_angles: List[List[int]]           # tile angles added up at every joined vertex
    kinds_agree: List[bool]
    cone_deficits: List[int] = field(default_factory=list)       # deficits of ALL vertices (flat ones are 0)
    open_holes: List[Tuple[int, int, int]] = field(default_factory=list)   # (part, hole, K)

    @property
    def junction_deficits(self) -> List[int]:
        return [360 - a for ang in self.junction_angles for a in ang]

    @property
    def ledger(self) -> int:
        """Cone deficits (including the junction ones, negative at saddle vertices) plus the curvature of the
        remaining holes: 720 for a sphere with holes."""
        return sum(p.total_deficit for p in self.parts) + sum(self.junction_deficits) + \
            sum(K for _, _, K in self.open_holes)

    def summary(self) -> str:
        jd = self.junction_deficits
        lines = [f"{len(self.parts)} spheres, {len(self.joins)} join(s); ledger {self.ledger}",
                 f"junction deficits {jd} (negative = saddle neck)"]
        for (pa, ha, K) in self.open_holes:
            lines.append(f"open hole: part {pa} hole {ha} K={K}")
        return "\n".join(lines)


def join_spheres(res1: FoldResult, h1: int, res2: FoldResult, h2: int, match_kinds: bool = False) -> JoinedSurface:
    """Join hole ``h1`` of ``res1`` to hole ``h2`` of ``res2``."""
    out = best_join(res1, res1.holes[h1], res2, res2.holes[h2], match_kinds)
    if out is None:
        raise ValueError("holes cannot be joined")
    r, ang, agree = out
    js = JoinedSurface([res1, res2], [(0, h1, 1, h2)], [r], [ang], [agree])
    for pi, res in enumerate(js.parts):
        for hi, h in enumerate(res.holes):
            if (pi, hi) in ((0, h1), (1, h2)):
                continue
            js.open_holes.append((pi, hi, h.K))
    if js.ledger != 720:
        raise AssertionError(f"joined ledger {js.ledger} != 720")
    return js
