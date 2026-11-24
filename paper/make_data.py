#!/usr/bin/env python3
"""
Compute every number used in the paper and store it in data/*.json.

    python make_data.py            # everything (about 5-10 minutes)
    python make_data.py --quick    # coarser sampling, for a smoke test

The LaTeX tables are generated from these files by make_tables.py, so no number in the paper is typed by hand.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import matplotlib
matplotlib.use("Agg")
from matplotlib.path import Path

import ball_metrics as bm
import balls
import trionda_barycentric as tb          # the original barycentric script ("Construction A")

OUT = os.path.join(HERE, "data")
TAU_TRIONDA = balls.TRIONDA_TAU
TILT = balls.FAMILY_TILT_DEG
TETRA = balls.TETRA_VERTICES
FACES = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]


def jdump(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))
    print("wrote", name)


# ---------------------------------------------------------------- Construction A audit
def coverage_counts(P, amp, harm):
    """For each unit vector in P: how many of the four barycentric panels contain it."""
    V2, bnd, _, _ = tb.build_flat_panel(amp=amp, harmonic=harm, grid_res=15)
    path = Path(bnd)
    count = np.zeros(len(P), int)
    for f in FACES:
        w = (np.linalg.inv(TETRA[f].T) @ P.T).T
        s = w.sum(1, keepdims=True)
        w = w / np.where(s == 0, 1, s)
        count += path.contains_points(w @ V2) & (s[:, 0] > 0)
    return count


def coverage_A(amp, harm, n):
    """Fraction of the sphere covered 0, 1, 2+ times by the four barycentric panels."""
    count = coverage_counts(balls.fibonacci_sphere(n), amp, harm)
    c = np.bincount(count, minlength=4)[:4] / n
    return {"gap": float(c[0]), "once": float(c[1]), "overlap": float(c[2:].sum())}


def audit_A(n):
    amps = np.concatenate([[0.0], np.geomspace(0.01, 0.25, 12)])
    rows = []
    for a in amps:
        r = coverage_A(a, a / 6.0, n)       # user's ratio: harmonic = amp * 0.035/0.21
        r["amp"] = float(a)
        rows.append(r)
    user = coverage_A(0.21, 0.035, n)
    # log-log slope of gap vs amplitude (small-amplitude regime)
    sel = [(r["amp"], r["gap"]) for r in rows if 0.01 <= r["amp"] <= 0.12 and r["gap"] > 0]
    slope = float(np.polyfit(np.log([s[0] for s in sel]), np.log([s[1] for s in sel]), 1)[0])
    return {"sweep": rows, "user_params": {"amp": 0.21, "harmonic": 0.035, **user}, "loglog_slope": slope}


# ---------------------------------------------------------------- tau sweep for the Trionda model
def tilt_measured(ball):
    V = ball.face_vertices()[0]
    m = balls.unit(V[0] + V[1]); e = balls.unit(V[1] - V[0]); s = 1e-4
    a = ball.twist(balls.unit(np.array([m + s * e, m - s * e])), inverse=True)
    d = a[0] - a[1]; d /= np.linalg.norm(d)
    return math.degrees(math.acos(min(1.0, abs(d @ e))))


def tau_sweep(taus, n_axes):
    straight = bm.seam_length(balls.make_trionda(0.0, "t0"))
    rows = []
    for tau in taus:
        b = balls.make_trionda(float(tau), f"t{tau:.2f}")
        L = bm.seam_length(b, 600)
        ex = bm.exposure_stats(b, n_axes=n_axes)
        bd = bm.bending(b)
        rows.append({"tau": float(tau), "seam_length": L, "ratio": L / straight,
                     "tilt_measured": tilt_measured(b) if tau > 0 else 0.0,
                     "tilt_closed_form": math.degrees(math.atan(math.tanh(tau))),
                     "exposure_cv": ex["cv"], "mean_abs_kg": bd["mean_abs_kg"], "max_abs_kg": bd["max_abs_kg"]})
    return {"straight_length": straight, "rows": rows}


# ---------------------------------------------------------------- family comparison
def family(quick):
    names = [
        ("tetra-4", balls.get_ball("tetra-4"), "T_d"),
        ("trionda", balls.get_ball("trionda"), "T"),
        ("trionda-20", balls.make_trionda(math.atanh(math.tan(math.radians(TILT))), "trionda-20"), "T"),
        ("cube-6", balls.get_ball("cube-6"), "O_h"),
        ("cube-6-chiral", balls.get_ball("cube-6-chiral"), "O"),
        ("octa-8", balls.get_ball("octa-8"), "O_h"),
        ("octa-8-chiral", balls.get_ball("octa-8-chiral"), "O"),
        ("dodeca-12", balls.get_ball("panel-12"), "I_h"),
        ("dodeca-12-chiral", balls.get_ball("dodeca-12-chiral"), "I"),
        ("icosa-20", balls.get_ball("icosa-20"), "I_h"),
        ("icosa-20-chiral", balls.get_ball("icosa-20-chiral"), "I"),
        ("goldberg-42", balls.get_ball("goldberg-42"), "I_h"),
        ("goldberg-42-chiral", balls.get_ball("goldberg-42-chiral"), "I"),
        ("classic-32", balls.get_ball("classic-32"), "I_h"),
    ]
    out = []
    for key, b, expected in names:
        t0 = time.time()
        q = bm.multipole_spectrum(b, lmax=12)
        g = bm.symmetry_group(b)
        ex = bm.exposure_stats(b, n_axes=1000 if quick else 3000)
        bd = bm.bending(b)
        L = bm.seam_length(b)
        st = {"key": key, "panels": b.n_panels, "expected_group": expected,
              "rotations": g["proper"], "improper": g["improper"], "chiral": not g["achiral"],
              "seam_length": L, "seam_per_area": L / (4 * math.pi),
              "q": q, "lowest_multipole": bm.lowest_multipole(q),
              "exposure": ex, "mean_abs_kg": bd["mean_abs_kg"],
              "tau": getattr(b, "tau", 0.0), "edge_tilt": getattr(b, "edge_tilt", 0.0)}
        out.append(st)
        print(f"  {key:18s} rot={g['proper']:3d} imp={g['improper']:3d} l*={st['lowest_multipole']} "
              f"L={L:.2f} expCV={ex['cv']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------- sensitivity of the exposure proxy
def sensitivity(n_axes):
    """Exposure CV of selected balls as the separation angle and band width are varied."""
    balls_ = {"classic-32": balls.get_ball("classic-32"), "cube-6": balls.get_ball("cube-6"),
              "tetra-4": balls.get_ball("tetra-4"), "trionda": balls.get_ball("trionda"),
              "trionda-tau2": balls.make_trionda(2.0, "trionda-tau2")}
    seps = [70, 80, 90, 100, 110, 120]
    sigs = [5, 10, 15]
    out = {"seps": seps, "sigmas": sigs, "cv": {}}
    for k, b in balls_.items():
        out["cv"][k] = [[bm.exposure_stats(b, n_axes=n_axes, sep_deg=sp, sigma_deg=sg)["cv"] for sp in seps] for sg in sigs]
        print("  ", k, np.round(out["cv"][k][1], 3).tolist(), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["audit", "sweep", "family", "sens"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.only in (None, "audit"):
        print("Construction A audit"); jdump("audit_A.json", audit_A(100_000 if a.quick else 400_000))
    if a.only in (None, "sweep"):
        print("tau sweep"); jdump("tau_sweep.json", tau_sweep(np.linspace(0, 2.0, 9 if a.quick else 21), 500 if a.quick else 1500))
    if a.only in (None, "sens"):
        print("sensitivity"); jdump("sensitivity.json", sensitivity(500 if a.quick else 1500))
    if a.only in (None, "family"):
        print("family comparison"); jdump("family.json", family(a.quick))


if __name__ == "__main__":
    main()
