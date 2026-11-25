"""
Exact interval DP over non-crossing gluings of a boundary loop.

Notation: a boundary loop has m edges; A[k] is the tile angle (degrees, multiple of 30) at the start vertex of
edge k, i.e. at the vertex between edge k-1 and edge k.  Gluing edge i to edge j (reversed) merges vertex
i+1 with vertex j and vertex j+1 with vertex i.

A *bubble* is a run of edges l..r (even length) that is matched completely among itself in a non-crossing
way.  Its top-level pairs (l1,j1), (j1+1,j2), ... leave the vertices v_l and v_{r+1} identified, and the class
containing them has angle  S + A[r+1]  with  S = sum of A at the top-level pair starts.  Every matched pair
(i,j) closes a cone point whose angle is  S_inner + A[j]  (S_inner of the bubble strictly inside the pair),
which must be <= 360.  Hence ``bubble_masks`` records, for every run, the set of achievable S as a bitmask in
units of 30 degrees; a hole corner at reserved edge p after a bubble with sum S is  S + A[p].
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

FULL = 360
U = 30


def bubble_masks(A: Sequence[int]):
    """ok[l][r] (r = l-1 means empty) -> bitmask of achievable top-level sums S/30 for the run of edges l..r.
    Also returns the choice table for reconstruction."""
    m = len(A)
    ok: Dict[Tuple[int, int], int] = {}
    for l in range(m + 2):
        ok[(l, l - 1)] = 1
    for length in range(2, m + 1, 2):
        for l in range(0, m - length + 1):
            r = l + length - 1
            mask = 0
            for j in range(l + 1, r + 1, 2):
                inner = ok[(l + 1, j - 1)]
                # need some S_in with S_in + A[j] <= 360
                lim = (FULL - A[j]) // U
                if lim < 0 or not (inner & ((1 << (lim + 1)) - 1)):
                    continue
                tail = ok[(j + 1, r)]
                mask |= (tail << (A[l] // U)) & ((1 << (FULL // U + 1)) - 1)
            ok[(l, r)] = mask
    return ok


def bubble_pairs(A: Sequence[int], ok, l: int, r: int, S: int) -> List[Tuple[int, int]]:
    """A non-crossing perfect matching of edges l..r (indices) whose top-level sum is S, outer pairs first."""
    if r < l:
        assert S == 0
        return []
    for j in range(l + 1, r + 1, 2):
        inner = ok[(l + 1, j - 1)]
        lim = (FULL - A[j]) // U
        if lim < 0 or not (inner & ((1 << (lim + 1)) - 1)):
            continue
        st = S - A[l]
        if st < 0 or not (ok[(j + 1, r)] >> (st // U)) & 1:
            continue
        # choose any S_in that keeps the cone legal
        s_in = next(s * U for s in range(lim + 1) if (inner >> s) & 1)
        return ([(l, j)] + bubble_pairs(A, ok, l + 1, j - 1, s_in) + bubble_pairs(A, ok, j + 1, r, st))
    raise AssertionError("no decomposition")


def can_close(A: Sequence[int]) -> bool:
    """Can the whole loop be glued shut (no hole)?  Cyclic: cut at edge 0; the class of v_0 is the sum of the
    top-level starts and must be <= 360."""
    ok = bubble_masks(A)
    mask = ok[(0, len(A) - 1)]
    return bool(mask & ((1 << (FULL // U + 1)) - 1))


def close_pairs(A: Sequence[int]) -> Optional[List[Tuple[int, int]]]:
    ok = bubble_masks(A)
    mask = ok[(0, len(A) - 1)]
    for s in range(FULL // U + 1):
        if (mask >> s) & 1:
            return bubble_pairs(A, ok, 0, len(A) - 1, s * U)
    return None


def find_single_hole(A: Sequence[int], theta: Sequence[int], rotations: bool = True):
    """Layout of a single hole whose corners (tile angles, in boundary order) are ``theta``.

    Returns (reserved indices in the loop, list of matched index pairs) or None.  All other edges are
    matched in non-crossing gaps between reserved edges (each gap has an even length)."""
    m, k = len(A), len(theta)
    if (m - k) % 2:
        return None
    rots = range(k) if rotations else [0]
    for p1 in range(m):
        B = list(A[p1:]) + list(A[:p1])          # rotate so the first reserved edge is index 0
        ok = bubble_masks(B)
        full_mask = (1 << (FULL // U + 1)) - 1
        for rot in rots:
            th = [theta[(t + rot) % k] for t in range(k)]
            if B[0] > th[0]:
                continue
            # dp[t][pos]: reserved edge number t placed at pos (t=0 at 0); store predecessor and gap sum
            prev: List[Dict[int, Tuple[int, int]]] = [dict() for _ in range(k)]
            prev[0][0] = (-1, 0)
            for t in range(1, k):
                for pos0 in prev[t - 1]:
                    for pos in range(pos0 + 1, m, 2):      # gap length pos - pos0 - 1 must be even
                        if pos in prev[t]:
                            continue
                        need = th[t] - B[pos]
                        if need < 0 or need % U:
                            continue
                        if (ok[(pos0 + 1, pos - 1)] >> (need // U)) & 1:
                            prev[t][pos] = (pos0, need)
            for pos_last in prev[k - 1]:
                need = th[0] - B[0]
                gap = (pos_last + 1, m - 1)
                if need >= 0 and need % U == 0 and (ok[gap] >> (need // U)) & 1:
                    # reconstruct
                    res, pos, t = [], pos_last, k - 1
                    pos_list = []
                    while t >= 0:
                        pos_list.append(pos)
                        pos, _ = prev[t][pos] if t > 0 else (-1, 0)
                        t -= 1
                    pos_list.reverse()
                    pairs = []
                    for t in range(k):
                        l = pos_list[t] + 1
                        r = (pos_list[t + 1] - 1) if t + 1 < k else m - 1
                        need_s = (th[t + 1] - B[pos_list[t + 1]]) if t + 1 < k else (th[0] - B[0])
                        pairs += bubble_pairs(B, ok, l, r, need_s)
                    # indices refer to the rotated list B; map back to original loop indices
                    back = lambda x: (x + p1) % m
                    return [back(p) for p in pos_list], [(back(a), back(b)) for a, b in pairs]
    return None


# ---------------------------------------------------------------------------------------------------
# several holes
# ---------------------------------------------------------------------------------------------------
FULLMASK = (1 << (FULL // U + 1)) - 1


def _variants(theta):
    k = len(theta)
    seen, out = set(), []
    for r in range(k):
        v = tuple(theta[(t + r) % k] for t in range(k))
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _hole_fit(A, ok, theta):
    """fit[(l1, j)] = variant when the run of edges l1..j-1, inside a glued pair (l1-1, j), can host a hole
    with corners ``theta`` (some rotation), otherwise absent.  The gaps between its reserved edges are plain
    bubbles.  The first corner is  s_first + s_last + A[r1] + A[j]."""
    m, k = len(A), len(theta)
    fit = {}
    for var in _variants(theta):
        for l1 in range(1, m):
            # Mc[t][pos] = bitmask of c = s_first + A[r1] values
            Mc = [dict() for _ in range(k)]
            for p1 in range(l1, m, 2):
                if (p1 - l1) % 2:
                    continue
                s0 = ok[(l1, p1 - 1)]
                if not s0:
                    continue
                c = (s0 << (A[p1] // U)) & FULLMASK
                if c:
                    Mc[0][p1] = c
            for t in range(1, k):
                for pos0, cm in Mc[t - 1].items():
                    for pos in range(pos0 + 1, m, 2):
                        need = var[t] - A[pos]
                        if need < 0 or need % U:
                            continue
                        if (ok[(pos0 + 1, pos - 1)] >> (need // U)) & 1:
                            Mc[t][pos] = Mc[t].get(pos, 0) | cm
            for pos, cm in Mc[k - 1].items():
                for j in range(pos + 1, m, 2):
                    if (l1, j) in fit:
                        continue
                    base = var[0] - A[j]
                    for cb in range(FULL // U + 1):
                        if not (cm >> cb) & 1:
                            continue
                        sk = base - cb * U
                        if sk >= 0 and sk % U == 0 and (ok[(pos + 1, j - 1)] >> (sk // U)) & 1:
                            fit[(l1, j)] = var
                            break
    return fit


def _layout(A, ok, var, l1, j):
    """Reserved positions and gap pairs for a hole with corners ``var`` in the run l1..j-1."""
    m, k = len(A), len(var)
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def go(t, pos, c):
        # reserved t at pos, accumulated c; return list of positions after
        if t == k - 1:
            for jj in [j]:
                if (jj - pos - 1) % 2:
                    return None
                sk = var[0] - A[j] - c * U
                if sk >= 0 and sk % U == 0 and (ok[(pos + 1, j - 1)] >> (sk // U)) & 1:
                    return ((pos,), sk)
            return None
        for pos2 in range(pos + 1, j, 2):
            need = var[t + 1] - A[pos2]
            if need < 0 or need % U:
                continue
            if (ok[(pos + 1, pos2 - 1)] >> (need // U)) & 1:
                r = go(t + 1, pos2, c)
                if r is not None:
                    return ((pos,) + r[0], r[1])
        return None

    for p1 in range(l1, j, 2):
        s0m = ok[(l1, p1 - 1)]
        for s0b in range(FULL // U + 1):
            if not (s0m >> s0b) & 1:
                continue
            c = s0b + A[p1] // U
            if c > FULL // U:
                continue
            r = go(0, p1, c)
            if r is not None:
                pos, sk = r
                pairs = bubble_pairs(A, ok, l1, pos[0] - 1, s0b * U)
                for t in range(k - 1):
                    pairs += bubble_pairs(A, ok, pos[t] + 1, pos[t + 1] - 1, var[t + 1] - A[pos[t + 1]])
                pairs += bubble_pairs(A, ok, pos[-1] + 1, j - 1, sk)
                return list(pos), pairs
    raise AssertionError("layout not found")


def find_holes(A: Sequence[int], thetas: Sequence[Sequence[int]]):
    """Exact search for a gluing of the whole loop that leaves holes with the given corner lists (one hole
    each, each hole inside its own glued pair, none nested in another hole).

    Returns (ops, holes) where ops are glued index pairs, outer pairs first, and holes lists, for each
    requested hole, its reserved loop indices; or None when no such gluing exists."""
    m, H = len(A), len(thetas)
    ok = bubble_masks(A)
    fits = [_hole_fit(A, ok, th) for th in thetas]
    ok2: Dict[Tuple[int, int], Dict[int, int]] = {}
    for l in range(m + 2):
        ok2[(l, l - 1)] = {0: 1}
    opts_cache = {}

    def opts(l, j):
        key = (l, j)
        if key in opts_cache:
            return opts_cache[key]
        out = []
        lim = (FULL - A[j]) // U
        if lim >= 0:
            low = (1 << (lim + 1)) - 1
            for u, mk in ok2[(l + 1, j - 1)].items():
                if mk & low:
                    out.append((u, "plain"))
        for t in range(H):
            if (l + 1, j) in fits[t]:
                out.append((1 << t, t))
        opts_cache[key] = out
        return out

    for length in range(2, m + 1, 2):
        for l in range(0, m - length + 1):
            r = l + length - 1
            res: Dict[int, int] = {}
            for j in range(l + 1, r + 1, 2):
                o = opts(l, j)
                if not o:
                    continue
                for ut, mt in ok2[(j + 1, r)].items():
                    sh = (mt << (A[l] // U)) & FULLMASK
                    if not sh:
                        continue
                    for ui, _kind in o:
                        if ui & ut:
                            continue
                        res[ui | ut] = res.get(ui | ut, 0) | sh
            ok2[(l, r)] = res
    full = (1 << H) - 1
    top = ok2[(0, m - 1)].get(full, 0)
    if not top:
        return None
    ops, holes = [], [None] * H

    def recon(l, r, u, S):
        if r < l:
            return True
        for j in range(l + 1, r + 1, 2):
            for ui, kind in opts(l, j):
                if ui & ~u:
                    continue
                ut = u & ~ui
                st = S - A[l]
                if st < 0 or not (ok2[(j + 1, r)].get(ut, 0) >> (st // U)) & 1:
                    continue
                ops.append((l, j))
                if kind == "plain":
                    # inner bubble: pick an S_in with cone condition
                    lim = (FULL - A[j]) // U
                    sin = next(s for s in range(lim + 1) if (ok2[(l + 1, j - 1)].get(ui, 0) >> s) & 1)
                    recon(l + 1, j - 1, ui, sin * U)
                else:
                    pos, pairs = _layout(A, ok, fits[kind][(l + 1, j)], l + 1, j)
                    holes[kind] = pos
                    ops.extend(pairs)
                recon(j + 1, r, ut, st)
                return True
        raise AssertionError("recon failed")

    S0 = next(s * U for s in range(FULL // U + 1) if (top >> s) & 1)
    recon(0, m - 1, full, S0)
    return ops, holes


def fold_with_holes(graph, thetas, loop: int = 0):
    """FoldResult for the exact DP gluing that leaves holes with the given corner lists (see ``find_holes``).
    Returns None if the patch's boundary admits no such gluing.  ``thetas=[]`` asks for a closed ball."""
    from .multi import replay_ops
    edges, ang = graph.loops[loop]
    A = [int(a) for a in ang]
    if not thetas:
        pairs = close_pairs(A)
        if pairs is None:
            return None
        return replay_ops(graph, [(edges[a], edges[b]) for a, b in pairs])
    out = find_holes(A, thetas)
    if out is None:
        return None
    ops, _ = out
    return replay_ops(graph, [(edges[a], edges[b]) for a, b in ops])
