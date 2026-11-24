#!/usr/bin/env python3
"""
Quantities used in the Trionda paper, all computed from the seam network on the unit sphere.

* ``seam_curve_samples``  exact arc-length-weighted samples of the seam network
* ``multipole_spectrum``  rotation-invariant spherical-harmonic power of the seam density
* ``exposure``            seam length inside a band around a "separation circle" for every flow axis
* ``symmetry_group``      numerically detected rotation (and improper) symmetries of the partition
* ``bending``             geodesic curvature of the seams

Nothing here is an aerodynamic prediction; ``exposure`` is a purely geometric proxy.
"""

import math
from typing import Dict, Tuple

import numpy as np
from scipy.special import sph_harm_y

import balls


# --------------------------------------------------------------------------
# seam network samples
# --------------------------------------------------------------------------

def seam_curve_samples(ball: balls.Ball, samples: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """Midpoints of short seam segments (unit vectors) and their arc lengths.

    Each polygon edge is a great-circle arc in panel space, pulled back through the warp.
    Every seam is the boundary of two panels, so each curve is visited twice and gets weight 1/2.
    """
    pts, wts = [], []
    for V in ball.face_vertices():
        k = len(V)
        for i in range(k):
            a, b = V[i], V[(i + 1) % k]
            om = math.acos(np.clip(a @ b, -1, 1))
            t = np.linspace(0, 1, samples + 1)[:, None]
            arc = (np.sin((1 - t) * om) * a + np.sin(t * om) * b) / math.sin(om)
            arc = ball.twist(arc, inverse=True)
            seg = np.linalg.norm(np.diff(arc, axis=0), axis=1)
            mid = balls.unit(0.5 * (arc[1:] + arc[:-1]))
            pts.append(mid)
            wts.append(0.5 * seg)
    return np.vstack(pts), np.concatenate(wts)


def seam_polylines(ball: balls.Ball, samples: int = 400):
    """One ordered polyline (unit vectors) per distinct seam curve (each shared edge once)."""
    seen, out = set(), []
    for V in ball.face_vertices():
        k = len(V)
        for i in range(k):
            a, b = V[i], V[(i + 1) % k]
            key = tuple(sorted([tuple(np.round(a, 6)), tuple(np.round(b, 6))]))
            if key in seen:
                continue
            seen.add(key)
            om = math.acos(np.clip(a @ b, -1, 1))
            t = np.linspace(0, 1, samples + 1)[:, None]
            out.append(ball.twist((np.sin((1 - t) * om) * a + np.sin(t * om) * b) / math.sin(om), inverse=True))
    return out


def seam_length(ball: balls.Ball, samples: int = 400) -> float:
    return float(seam_curve_samples(ball, samples)[1].sum())


# --------------------------------------------------------------------------
# multipole spectrum
# --------------------------------------------------------------------------

def multipole_spectrum(ball: balls.Ball, lmax: int = 12, samples: int = 400) -> np.ndarray:
    """Q_l = sum_m |a_lm|^2 / |a_00|^2 for the seam measure rho = sum_j w_j delta(p - p_j).

    Q_l is invariant under rotations of the ball.  If the ball's rotation group G has no
    non-constant invariant harmonic of degree l, then Q_l = 0 exactly (up to quadrature error).
    """
    p, w = seam_curve_samples(ball, samples)
    theta = np.arccos(np.clip(p[:, 2], -1, 1))
    phi = np.arctan2(p[:, 1], p[:, 0])
    a00 = (w * sph_harm_y(0, 0, theta, phi)).sum()
    q = np.zeros(lmax + 1)
    for l in range(lmax + 1):
        acc = 0.0
        for m in range(-l, l + 1):
            acc += abs((w * np.conj(sph_harm_y(l, m, theta, phi))).sum()) ** 2
        q[l] = acc / abs(a00) ** 2
    return q


def lowest_multipole(q: np.ndarray, tol: float = 1e-9) -> int:
    """Smallest l >= 1 with Q_l > tol (or -1 if none up to lmax)."""
    for l in range(1, len(q)):
        if q[l] > tol:
            return l
    return -1


# --------------------------------------------------------------------------
# flow-axis exposure
# --------------------------------------------------------------------------

def exposure(ball: balls.Ball, axes: np.ndarray, sep_deg: float = 90.0, sigma_deg: float = 10.0,
             samples: int = 200, chunk: int = 2000) -> np.ndarray:
    """Seam length near the separation circle, for each flow axis n in ``axes`` (M, 3).

    The ball moves along n; the separation circle is the set of points at angle ``sep_deg`` from n.
    The result is  G(n) = sum_j w_j exp(-(angle(p_j, n) - sep)^2 / (2 sigma^2)) / (sigma sqrt(2 pi)),
    i.e. seam length per radian of polar angle near the circle.  It is a geometric proxy for how
    much seam a flow meets where the boundary layer separates; it does not model the flow.
    """
    p, w = seam_curve_samples(ball, samples)
    sep, sig = math.radians(sep_deg), math.radians(sigma_deg)
    out = np.empty(len(axes))
    for a in range(0, len(axes), chunk):
        ang = np.arccos(np.clip(axes[a:a + chunk] @ p.T, -1, 1))
        out[a:a + chunk] = (np.exp(-0.5 * ((ang - sep) / sig) ** 2) @ w) / (sig * math.sqrt(2 * math.pi))
    return out


def exposure_stats(ball: balls.Ball, n_axes: int = 3000, **kw) -> Dict[str, float]:
    g = exposure(ball, balls.fibonacci_sphere(n_axes), **kw)
    return {"mean": float(g.mean()), "cv": float(g.std() / g.mean()), "min_over_mean": float(g.min() / g.mean()),
            "max_over_mean": float(g.max() / g.mean())}


# --------------------------------------------------------------------------
# bending
# --------------------------------------------------------------------------

def bending(ball: balls.Ball, samples: int = 600) -> Dict[str, float]:
    """Geodesic curvature statistics of the seams (0 for great-circle arcs)."""
    ks, ws = [], []
    for V in ball.face_vertices():
        k = len(V)
        for i in range(k):
            a, b = V[i], V[(i + 1) % k]
            om = math.acos(np.clip(a @ b, -1, 1))
            t = np.linspace(0, 1, samples)[:, None]
            c = ball.twist((np.sin((1 - t) * om) * a + np.sin(t * om) * b) / math.sin(om), inverse=True)
            ds = np.linalg.norm(np.diff(c, axis=0), axis=1)
            tan = balls.unit(np.gradient(c, axis=0))
            s_cum = np.concatenate([[0], np.cumsum(ds)])
            dt = np.gradient(tan, s_cum, axis=0)
            kg = np.einsum("ij,ij->i", dt, np.cross(c, tan))
            ks.append(np.abs(kg)); ws.append(np.gradient(s_cum) * 0.5)
    k = np.concatenate(ks); w = np.concatenate(ws)
    return {"mean_abs_kg": float((k * w).sum() / w.sum()), "max_abs_kg": float(k.max()),
            "rms_kg": float(math.sqrt((k ** 2 * w).sum() / w.sum()))}


# --------------------------------------------------------------------------
# symmetry detection
# --------------------------------------------------------------------------

def _frame(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    e1 = a
    e2 = balls.unit(b - (b @ a) * a)
    return np.stack([e1, e2, np.cross(e1, e2)], axis=1)


def symmetry_group(ball: balls.Ball, n_pts: int = 5000, tol: float = 0.995) -> Dict:
    """Count proper rotations and improper isometries that map the panel partition onto itself.

    Colours are ignored.  A candidate isometry g is found from the images of two panel centroids
    and accepted if panel labels of g p depend only on the label of p (for >= tol of sample points)
    and the induced map of panels is a bijection.
    """
    P = balls.fibonacci_sphere(n_pts)
    lab = ball.panel_index(P)
    F = ball.n_panels
    # panel centroids from a dense sample: their accuracy sets the accuracy of every candidate map
    Pc = balls.fibonacci_sphere(300_000)
    labc = ball.panel_index(Pc)
    cen = np.array([balls.unit(Pc[labc == f].sum(0)) if np.linalg.norm(Pc[labc == f].sum(0)) > 1e-3 * 300_000 / F
                    else np.zeros(3) for f in range(F)])
    c0 = cen[0]
    # second reference panel: the one whose centroid is most perpendicular to the first
    j = int(np.argmax([np.linalg.norm(np.cross(c0, c)) for c in cen]))
    cj = cen[j]
    dref = float(c0 @ cj)
    src = _frame(c0, cj)
    found = {"proper": 0, "improper": 0}
    for k in range(F):
        for l in range(F):
            if abs(cen[k] @ cen[l] - dref) > 1e-2 or k == l:
                continue
            tgt = _frame(cen[k], cen[l])
            for kind, flip in (("proper", 1.0), ("improper", -1.0)):
                R = tgt @ np.diag([1.0, 1.0, flip]) @ src.T
                lab2 = ball.panel_index(P @ R.T)
                table = -np.ones(F, dtype=int)
                good = 0
                for f in range(F):
                    vals = lab2[lab == f]
                    if len(vals):
                        m = np.bincount(vals, minlength=F)
                        table[f] = int(m.argmax())
                        good += int(m.max())
                if good / n_pts >= tol and len(set(table.tolist())) == F:
                    found[kind] += 1
    found["order_rotations"] = found["proper"]
    found["achiral"] = found["improper"] > 0
    return found


GROUP_NAMES = {60: "I", 24: "O", 12: "T", 6: "D3 or C6", 4: "D2 or C4", 3: "C3", 2: "C2", 1: "C1"}


if __name__ == "__main__":
    import sys
    for name in (sys.argv[1:] or ["classic-32", "tetra-4", "trionda"]):
        b = balls.get_ball(name)
        q = multipole_spectrum(b)
        g = symmetry_group(b)
        print(f"{name:18s} rotations={g['proper']:3d} improper={g['improper']:3d} "
              f"lowest multipole l*={lowest_multipole(q)}  seam={seam_length(b):.3f}  exposure CV={exposure_stats(b)['cv']:.4f}")
