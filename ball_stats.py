#!/usr/bin/env python3
"""
Numerical statistics of a ball's panel layout, estimated by sampling the sphere.

All areas are on the unit sphere (total 4*pi) and all lengths are arc lengths on it,
so numbers are comparable between ball types.  Multiply lengths by the real radius
(0.11 m for a size-5 ball) and areas by radius squared to get physical values.

    python ball_stats.py                # table for every registered ball
    python ball_stats.py panel-6        # one ball
"""

import math
import sys
from typing import Dict

import numpy as np

import balls

SEAM_PROBE_DEG = 1.0  # width of the probe band used to measure seam length


def exact_seam_length(ball: balls.Ball, samples: int = 400) -> float:
    """Total seam arc length (radians) by walking every panel edge, including any twist."""
    total = 0.0
    for V in ball.face_vertices():
        k = len(V)
        for i in range(k):
            a, b = V[i], V[(i + 1) % k]
            om = math.acos(np.clip(a @ b, -1, 1))
            t = np.linspace(0, 1, samples)[:, None]
            pts = (np.sin((1 - t) * om) * a + np.sin(t * om) * b) / math.sin(om)
            pts = ball.twist(pts, inverse=True)
            total += np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    return total / 2          # every edge is shared by two panels


def analyze(ball: balls.Ball, n: int = 400_000, probe_deg: float = SEAM_PROBE_DEG) -> Dict:
    """Sample the sphere and return panel / seam statistics."""
    pts = balls.fibonacci_sphere(n)
    idx, dist = ball.panel_and_seam(pts)

    area = np.bincount(idx, minlength=ball.n_panels) / n * 4 * math.pi
    half = math.radians(probe_deg) / 2
    on_seam = dist < half
    # exact length along the real (twisted) edge curves; the band estimate is only a cross-check
    seam_len = exact_seam_length(ball)
    seam_len_band = on_seam.mean() * 4 * math.pi / math.radians(probe_deg)

    # second-moment tensor of the seam network: isotropic networks have equal eigenvalues
    sp = pts[on_seam]
    tensor = sp.T @ sp / max(len(sp), 1)
    eig = np.sort(np.linalg.eigvalsh(tensor))[::-1]          # sums to 1
    anisotropy = float(eig[0] / eig[-1])

    # radius of the largest seam-free cap (distance estimate at the deepest point of any panel)
    # (panel-space distance, so only meaningful without twist -> NaN for twisted balls)
    max_cap_deg = float(np.degrees(dist.max())) if not ball.is_warped else float("nan")

    lat = np.degrees(np.arcsin(pts[:, 2]))
    bins = np.linspace(-90, 90, 37)
    h_seam, _ = np.histogram(lat[on_seam], bins)
    h_all, _ = np.histogram(lat, bins)
    seam_density = np.divide(h_seam, h_all, out=np.zeros(len(h_seam)), where=h_all > 0)

    return {
        "name": ball.name,
        "n_panels": ball.n_panels,
        "panel_area": area,
        "area_mean": float(area.mean()),
        "area_cv": float(area.std() / area.mean()),          # 0 for equal panels
        "area_ratio": float(area.max() / area.min()),
        "seam_length": float(seam_len),
        "seam_length_band": float(seam_len_band),
        "seam_per_area": float(seam_len / (4 * math.pi)),
        "seam_eigenvalues": eig,
        "max_cap_deg": max_cap_deg,                         # largest seam-free cap radius
        "seam_anisotropy": anisotropy,                      # 1 = isotropic
        "seam_density_lat": seam_density,                   # fraction of area within the seam band
        "lat_bins": 0.5 * (bins[:-1] + bins[1:]),
        "kinds": ball.kinds,
    }


def format_table(rows) -> str:
    head = f"{'ball':18s}{'panels':>7s}{'area CV':>9s}{'max/min':>9s}{'seam len':>10s}{'len/area':>10s}{'anisotropy':>12s}{'free cap':>10s}"
    lines = [head, "-" * len(head)]
    for r in rows:
        cap = r["max_cap_deg"]
        cap_s = f"{cap:9.1f}°" if cap == cap else f"{'n/a':>10s}"
        lines.append(f"{r['name']:18s}{r['n_panels']:7d}{r['area_cv']:9.3f}{r['area_ratio']:9.2f}"
                     f"{r['seam_length']:10.2f}{r['seam_per_area']:10.3f}{r['seam_anisotropy']:12.3f}{cap_s}")
    return "\n".join(lines)


if __name__ == "__main__":
    names = sys.argv[1:] or balls.list_balls()
    print(format_table([analyze(balls.get_ball(n)) for n in names]))
    print("\nunit sphere: area total 4*pi = 12.57, lengths in radians of arc")
