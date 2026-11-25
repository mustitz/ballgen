"""Search over tile subsets: the DP in ``dp.py`` is exact for a given patch, so the way to get the holes you
want is to look for patches whose boundary admits them."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import numpy as np

from .fold import PatchGraph
from .patch import SpectrePatch


class PatchPool:
    """Adjacency of a big patch, so many random sub-patches can be drawn cheaply."""

    def __init__(self, patch: SpectrePatch):
        self.patch = patch
        self.nbrs: Dict[int, set] = {t: set() for t in range(patch.n_tiles)}
        for t, _, u, _ in patch.edge_pairs():
            self.nbrs[int(t)].add(int(u))
            self.nbrs[int(u)].add(int(t))
        self.cents = np.array([patch.straight_polygon(t).mean(0) for t in range(patch.n_tiles)])
        self.c0 = self.cents.mean(0)

    def random_subset(self, n: int, rng: random.Random, compact: float = 0.5) -> List[int]:
        """Random connected subset: start near the centre, add a random frontier tile; with probability
        ``compact`` take the frontier tile nearest the start instead (rounder blobs)."""
        d0 = np.linalg.norm(self.cents - self.c0, axis=1)
        pool = [int(t) for t in np.argsort(d0)[:40]]
        start = rng.choice(pool)
        chosen, in_set = [start], {start}
        frontier = set(self.nbrs[start])
        cs = self.cents[start]
        while len(chosen) < n and frontier:
            if rng.random() < compact:
                nxt = min(frontier, key=lambda u: (np.linalg.norm(self.cents[u] - cs), u))
            else:
                nxt = rng.choice(sorted(frontier))
            chosen.append(nxt)
            in_set.add(nxt)
            frontier.discard(nxt)
            frontier |= self.nbrs[nxt] - in_set
        return chosen

    def graph(self, tiles: Sequence[int]) -> Optional[PatchGraph]:
        """PatchGraph if the subset is a topological disc (one boundary loop), else None."""
        g = PatchGraph(self.patch, tiles=tiles)
        if len(g.loops) != 1 or g.boundary_turning() != 360:
            return None
        return g
