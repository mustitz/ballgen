#!/usr/bin/env python3
"""
Numbers for the Spectre-folding section of the paper -> data/spectre.json  (and figures copied into figures/).

    python make_spectre_data.py                  # everything (about 5-8 minutes)
    python make_spectre_data.py --only closure   # closure | survey | hub | validate | figures

Every draw is seeded and has a fixed size, so the counts reproduce exactly.  The DP itself is exact; the
numbers below are about *random patches*, not about all patches.
"""
import argparse
import itertools
import json
import os
import random
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from spectre.patch import SpectrePatch
from spectre.search import PatchPool
from spectre.dp import can_close, find_holes, fold_with_holes, FULL
from spectre.connect import join_many, best_join

OUT = os.path.join(HERE, "data", "spectre.json")
SQ = [240, 270, 240, 270]


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save(key, val):
    d = load()
    d[key] = val
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(d, open(OUT, "w"), indent=1)
    print("wrote", key)


def angles(g):
    return [int(a) for a in g.loops[0][1]]


# ---------------------------------------------------------------- closure rate by patch size
def closure(pool, per_n=150, seed=5, Ns=range(4, 25)):
    rng = random.Random(seed)
    rows = {}
    for N in Ns:
        disc = closed = 0
        for _ in range(per_n):
            g = pool.graph(pool.random_subset(N, rng, rng.choice([0.0, 0.3, 0.7])))
            if g is None:
                continue
            assert len(g.loops[0][0]) % 2 == 0            # boundary edge count is even (parity lemma)
            disc += 1
            closed += can_close(angles(g))
        rows[str(N)] = {"discs": disc, "closed": closed}
    return {"per_n": per_n, "seed": seed, "rows": rows}


# ---------------------------------------------------------------- which holes occur in random patches
def hexagon_patterns():
    """Corner patterns (tile angles) of equilateral hexagons that close up flat (K=0), up to symmetry."""
    import math

    def closes(alpha):
        h = x = y = 0.0
        for a in alpha:
            x += math.cos(h); y += math.sin(h); h += math.radians(180 - a)
        return abs(x) < 1e-6 and abs(y) < 1e-6
    out = set()
    for al in itertools.product(range(30, 360, 30), repeat=6):
        if sum(al) != 720 or not closes(al):
            continue
        th = tuple(360 - a for a in al)
        c = [th[i:] + th[:i] for i in range(6)]
        c += [tuple(reversed(x)) for x in c]
        out.add(min(c))
    return sorted(out)


def survey(pool, draws=800, seed=5, nmin=6, nmax=24, flat_draws=200, flat_nmax=16):
    rng = random.Random(seed)
    hexes = hexagon_patterns()
    cnt = dict(discs=0, closed=0, square=0, square_pair=0, hexagon_regular=0, pentagon_pair=0, flat_hexagon_any=0,
               flat_draws=0)
    folds = 0
    for _ in range(draws):
        g = pool.graph(pool.random_subset(rng.randint(nmin, nmax), rng, rng.choice([0.0, 0.3, 0.7])))
        if g is None:
            continue
        A = angles(g)
        cnt["discs"] += 1
        cnt["closed"] += can_close(A)
        cnt["square"] += bool(find_holes(A, [SQ]))
        cnt["square_pair"] += bool(find_holes(A, [SQ, SQ]))
        cnt["hexagon_regular"] += bool(find_holes(A, [[240] * 6]))
        cnt["pentagon_pair"] += bool(find_holes(A, [[240] * 5, [240] * 5]))
    rng = random.Random(seed + 1)
    for _ in range(flat_draws):
        g = pool.graph(pool.random_subset(rng.randint(nmin, flat_nmax), rng, rng.choice([0.0, 0.3, 0.7])))
        if g is None:
            continue
        cnt["flat_draws"] += 1
        A = angles(g)
        cnt["flat_hexagon_any"] += any(find_holes(A, [list(h)]) for h in hexes)
    return dict(counts=cnt, draws=draws, seed=seed, n_range=[nmin, nmax], n_flat_hexagon_patterns=len(hexes),
                flat_n_range=[nmin, flat_nmax])


# ---------------------------------------------------------------- hubs: folds with several square holes
def hub(pool, k, draws, seed):
    rng = random.Random(seed)
    disc = found = 0
    first = None
    for _ in range(draws):
        g = pool.graph(pool.random_subset(rng.randint(9, 16), rng, rng.choice([0.0, 0.3, 0.7])))
        if g is None:
            continue
        disc += 1
        r = fold_with_holes(g, [SQ] * k)
        if r:
            found += 1
            first = first or {"tiles": g.n_tiles, "K": [h.K for h in r.holes], "ledger": r.ledger}
    return dict(k=k, discs=disc, found=found, seed=seed, first=first)


def joins(pool):
    """Two-hole folds, a joined pair and a chain: ledgers and junction deficits."""
    rng = random.Random(2)
    folds = []
    while len(folds) < 14:
        g = pool.graph(pool.random_subset(rng.randint(8, 12), rng, 0.3))
        r = fold_with_holes(g, [SQ, SQ]) if g is not None else None
        if r:
            folds.append(r)
    pair = chain = None
    for i, a in enumerate(folds):
        for j, b in enumerate(folds):
            if i != j and best_join(a, a.holes[1], b, b.holes[0], True):
                pair = (i, j)
                break
        if pair:
            break
    ja = join_many([folds[pair[0]], folds[pair[1]]], [(0, 1, 1, 0)], True)
    out = dict(pair_tiles=[f.graph.n_tiles for f in (folds[pair[0]], folds[pair[1]])], pair_ledger=ja.ledger,
               pair_junction_deficits=sorted(set(ja.junction_deficits)), pair_open_K=[K for *_, K in ja.open_holes],
               pair_cones=[len([c for c in f.cones if c.deficit]) for f in (folds[pair[0]], folds[pair[1]])])
    # three-fold ring ledger (a torus) from any three two-hole folds
    jr = join_many(folds[:3], [(0, 1, 1, 0), (1, 1, 2, 0), (2, 1, 0, 0)])
    jc = join_many(folds[:3], [(0, 1, 1, 0), (1, 1, 2, 0)])
    out.update(ring3_ledger=jr.ledger, chain3_ledger=jc.ledger)
    return out


# ---------------------------------------------------------------- DP against brute force (one tile)
def brute_quads(A):
    """All corner multisets of 4-edge holes of one boundary loop, by exhaustive enumeration of layouts and of
    the adjacent zips of the remaining edges (no pruning)."""
    m, k = len(A), 4
    found = set()
    for pos in itertools.combinations(range(m), k):
        gaps = [(pos[(i + 1) % k] - pos[i] - 1) % m for i in range(k)]
        if any(x % 2 for x in gaps):
            continue
        cap = set(pos)
        cur = {(tuple(range(m)), tuple(A))}
        while True:
            nxt = set()
            for ed, co in cur:
                mm = len(ed)
                if mm == k:
                    continue
                for i in range(mm):
                    j = (i + 1) % mm
                    if ed[i] in cap or ed[j] in cap:
                        continue
                    if 360 - co[j] < 0:
                        continue
                    kk = (i + 2) % mm
                    y = co[i] + co[kk]
                    if y > 360:
                        continue
                    o = [(kk + s) % mm for s in range(mm - 2)]
                    nxt.add((tuple(ed[t] for t in o), tuple([y] + [co[t] for t in o[1:]])))
            if not nxt:
                break
            cur = nxt
            if len(next(iter(cur))[0]) == k:
                break
        for ed, co in cur:
            if len(ed) == k and max(co) < 360:
                found.add(tuple(sorted(co)))
    return found


def orderings(ms):
    k = len(ms)
    return {min(tuple(o[i:] + o[:i]) for i in range(k)) for o in set(itertools.permutations(ms))}


def validate(pool):
    """DP against exhaustive enumeration (reserved layouts x all zip orders; no DP, no pruning)."""
    from spectre.fold import PatchGraph, grow_patch
    out = {}
    # one tile, four-sided holes: the two sets of reachable corner multisets must be equal
    A = angles(PatchGraph(pool.patch, tiles=grow_patch(pool.patch, 1)))
    brute = brute_quads(A)
    dp, seen = set(), set()
    for th in itertools.product(range(30, 360, 30), repeat=4):
        c = min(th[i:] + th[:i] for i in range(4))
        if c in seen:
            continue
        seen.add(c)
        if find_holes(A, [list(c)]):
            dp.add(tuple(sorted(c)))
    out["one_tile_k4"] = dict(boundary_edges=len(A), brute=len(brute), dp=len(dp), equal=brute == dp)
    # completeness only (every brute-force shape is found, in some order of its corners)
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_spectre_fold as T
    for key, N, k in (("one_tile_k6", 1, 6), ("two_tiles_k4", 2, 4)):
        A = angles(PatchGraph(pool.patch, tiles=grow_patch(pool.patch, N)))
        b = T._brute_holes(A, k)
        miss = sum(1 for ms in b if not any(find_holes(A, [list(o)]) for o in orderings(ms)))
        out[key] = dict(boundary_edges=len(A), brute=len(b), dp_missing=miss)
    out["all_agree"] = out["one_tile_k4"]["equal"] and all(out[k_]["dp_missing"] == 0 for k_ in ("one_tile_k6", "two_tiles_k4"))
    return out


def figures():
    src = os.path.join(ROOT, "spectre", "figures")
    for f in ("closed_balls.png", "sphere_chain.png", "joined_spheres.png", "sphere_star.png"):
        shutil.copy(os.path.join(src, f), os.path.join(HERE, "figures", f))
        print("copied", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["closure", "survey", "hub", "joins", "validate", "figures"])
    a = ap.parse_args()
    pool = PatchPool(SpectrePatch(levels=3))
    if a.only in (None, "closure"):
        save("closure", closure(pool))
    if a.only in (None, "survey"):
        save("survey", survey(pool))
    if a.only in (None, "hub"):
        save("hub3", hub(pool, 3, 600, 3))
        save("hub4", hub(pool, 4, 600, 4))
    if a.only in (None, "joins"):
        save("joins", joins(pool))
    if a.only in (None, "validate"):
        save("validate", validate(pool))
    if a.only in (None, "figures"):
        figures()


if __name__ == "__main__":
    main()
