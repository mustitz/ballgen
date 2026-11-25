#!/usr/bin/env python3
"""Numbers for the Trionda / Hat Trick sections of the paper -> data/hat_trick.json.

Everything is computed by the repository code (the_hat_trick_ball.py, compare_trionda_hat.py, ball_metrics.py,
balls.py); nothing is typed in.  The frequency scan (data/hat_frequencies.json) is produced by
examples/hat_trick_frequencies.py in the hexasphere repository and copied here unchanged."""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
import compare_trionda_hat as C          # noqa: E402
import the_hat_trick_ball as H           # noqa: E402


def kite_angle_ledger(ball):
    """Total kite angle at every corner of the kite complex, counted by (corner type, total in degrees).
    A total of 360 is a flat interior point; smaller totals are the corners of the 12 pentagonal holes."""
    tot = defaultdict(float)
    for k in ball.S.kites:
        for cid in ball.S.corners(k):
            tot[cid] += H.KITE_ANGLE[cid[0]]
    count = defaultdict(int)
    for cid, a in tot.items():
        count[f"{cid[0]}:{int(round(a))}"] += 1
    return dict(count)


def main():
    ctx = C.context()
    ball, ids, X, info, polys = ctx["hat"][0]
    out = {}
    out["table"] = C.table(ctx)
    out["chirality_deg"] = C.chirality_table(ctx)
    hr = C.hat_trionda(ctx, 0)
    out["hat_trionda"] = C.super_panel_stats(ctx, hr, 0)
    out["flow"] = C.flow_sweep(ctx, [-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9])
    # structure of the design
    out["design"] = {
        "tiles": int(len(ball.V)),
        "triangles": len(ball.F), "kites": len(ball.S.kites), "hats": len(ball.hats),
        "pentagons": len(ball.pent), "placements": len(ball.placements), "symmetric_tilings": len(ball.tilings),
        "kites_per_hat": sorted({len(h) for h in ball.hats}),
        "kite_angle_totals": kite_angle_ledger(ball),
        "rotations": len(ball.G),
        "relax": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in info.items()
                  if isinstance(v, (int, float, np.floating))},
        "report": {k: float(v) for k, v in H.report(ball, ids, X, info).items() if isinstance(v, (int, float, np.floating))},
    }
    # the tetrahedral subgroup T of I that fixes the 4 shared 3-fold axes, and its orbits
    R, P = ctx["R"], ctx["tetra"]
    def keeps(g):
        Q = P @ g.T
        return all(np.isclose(Q @ p, 1, atol=1e-6).any() for p in P)
    T = [i for i, g in enumerate(ball.G) if keeps(g)]
    hat_perm = lambda i: [ball.hat_of[ball.kperm[i][min(h)]] for h in ball.hats]
    def orbits(perms, n):
        seen, orb = set(), []
        for x in range(n):
            if x in seen:
                continue
            o = {p[x] for p in perms}
            seen |= o
            orb.append(len(o))
        return orb
    Vp = ball.V[ball.pent]
    pent_perm = lambda g: [int(np.argmax(Vp @ (g @ v))) for v in Vp]
    def stab(perms, n):
        return max(sum(1 for p in perms if p[x] == x) for x in range(n))
    hp = [hat_perm(i) for i in T]
    pp = [pent_perm(ball.G[i]) for i in T]
    out["tetra_subgroup"] = {"order": len(T), "hat_orbit_sizes": orbits(hp, 60), "pentagon_orbit_sizes": orbits(pp, 12),
                             "max_hat_stabiliser": stab(hp, 60), "max_pentagon_stabiliser": stab(pp, 12)}
    full = [hat_perm(i) for i in range(len(ball.G))]
    out["icosa_group"] = {"hat_orbit_sizes": orbits(full, 60), "max_hat_stabiliser": stab(full, 60)}
    out["frequency_scan"] = json.load(open(os.path.join(HERE, "data", "hat_frequencies.json")))
    json.dump(out, open(os.path.join(HERE, "data", "hat_trick.json"), "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("design", "tetra_subgroup", "icosa_group", "hat_trionda")}, indent=1, default=float))


if __name__ == "__main__":
    main()
