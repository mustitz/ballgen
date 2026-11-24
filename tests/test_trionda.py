import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "paper"))

import ball_metrics as bm
import ball_stats
import balls

T = balls.TETRA_VERTICES
_CYC = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
T_GROUP = [np.diag(d) @ np.linalg.matrix_power(_CYC, k)
           for d in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]) for k in range(3)]


def sample(n=2000, seed=1):
    return balls.unit(np.random.default_rng(seed).normal(size=(n, 3)))


def test_tetra_group_is_the_rotation_group_T():
    assert len(T_GROUP) == 12 and all(abs(np.linalg.det(g) - 1) < 1e-12 for g in T_GROUP)
    # each element permutes the four tetrahedron vertices
    for g in T_GROUP:
        img = T @ g.T
        assert all(min(np.linalg.norm(img[i] - T[j]) for j in range(4)) < 1e-12 for i in range(4))


def test_flow_is_T_equivariant_and_chiral():
    w, P = balls.make_flow_warp(0.9), sample()
    assert max(np.abs(w(P @ g.T) - w(P) @ g.T).max() for g in T_GROUP) < 1e-12
    refl = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.0]])           # a mirror of T_d
    assert np.abs(w(P @ refl.T) - w(P) @ refl.T).max() > 0.1


def test_flow_inverse_and_fixed_points():
    w, P = balls.make_flow_warp(1.3), sample()
    assert np.abs(w(w(P), inverse=True) - P).max() < 1e-7
    fixed = np.vstack([T, -T, np.eye(3), -np.eye(3)])
    assert np.abs(w(fixed) - fixed).max() < 1e-14


@pytest.mark.parametrize("tau", [0.3, 0.9, 1.5])
def test_tilt_at_edge_midpoint_matches_closed_form(tau):
    b = balls.make_trionda(tau, "t")
    V = b.face_vertices()[0]
    m, e = balls.unit(V[0] + V[1]), balls.unit(V[1] - V[0])
    a = b.twist(balls.unit(np.array([m + 1e-4 * e, m - 1e-4 * e])), inverse=True)
    d = (a[0] - a[1]) / np.linalg.norm(a[0] - a[1])
    assert math.degrees(math.acos(min(1, abs(d @ e)))) == pytest.approx(math.degrees(math.atan(math.tanh(tau))), abs=1e-3)


def test_vertex_swirl_rate_is_two_over_sqrt3():
    m = T[0]
    e1 = balls.unit(np.cross(m, [0.267, 0.534, 0.802])); e2 = np.cross(m, e1)
    J = np.zeros((2, 2)); eps = 1e-5
    for j, e in enumerate((e1, e2)):
        dv = (balls.tetra_flow_field(balls.unit((m + eps * e)[None])) - balls.tetra_flow_field(balls.unit((m - eps * e)[None])))[0] / (2 * eps)
        J[:, j] = [dv @ e1, dv @ e2]
    assert abs(J[1, 0] - J[0, 1]) / 2 == pytest.approx(2 / math.sqrt(3), abs=1e-6)   # rotation rate
    assert np.linalg.norm((J + J.T) / 2) < 1e-6                                          # no strain


def test_trionda_panels_have_equal_area_pi_and_seam_length_grows():
    st = ball_stats.analyze(balls.get_ball("trionda"), n=200000)
    assert np.allclose(st["panel_area"], math.pi, atol=0.01)
    assert ball_stats.exact_seam_length(balls.get_ball("tetra-4")) == pytest.approx(6 * math.acos(-1 / 3), rel=1e-6)
    assert st["seam_length"] > 6 * math.acos(-1 / 3)


def test_symmetry_groups_trionda_vs_straight():
    g = bm.symmetry_group(balls.get_ball("trionda"))
    assert (g["proper"], g["improper"]) == (12, 0)
    g0 = bm.symmetry_group(balls.get_ball("tetra-4"))
    assert (g0["proper"], g0["improper"]) == (12, 12)


def test_multipole_selection_rule():
    q = bm.multipole_spectrum(balls.get_ball("trionda"), lmax=8)
    assert q[1] < 1e-9 and q[2] < 1e-9 and q[3] > 1e-3
    qc = bm.multipole_spectrum(balls.get_ball("classic-32"), lmax=8)
    assert max(qc[1:6]) < 1e-9 and qc[6] > 1e-3


def test_invariant_dimension_formula():
    import make_figures as mf
    assert mf.invariant_dims("T")[:5] == [1, 0, 0, 1, 1]
    assert mf.invariant_dims("O")[:5] == [1, 0, 0, 0, 1]
    assert mf.invariant_dims("I")[:7] == [1, 0, 0, 0, 0, 0, 1]


@pytest.mark.parametrize("name", ["cube-6-chiral", "octa-8-chiral", "dodeca-12-chiral", "icosa-20-chiral"])
def test_chiral_families_are_calibrated_and_keep_group_order(name):
    b = balls.get_ball(name)
    assert b.edge_tilt == pytest.approx(balls.FAMILY_TILT_DEG, abs=1e-6)
    g = bm.symmetry_group(b)
    assert g["improper"] == 0 and g["proper"] == {"cube-6-chiral": 24, "octa-8-chiral": 24,
                                                   "dodeca-12-chiral": 60, "icosa-20-chiral": 60}[name]


def test_unreachable_tilt_raises():
    with pytest.raises(ValueError):
        balls.make_chiral(balls.poly_cube_k(), balls.grad_cube, "x", "", tilt_deg=40.0)


def test_construction_A_leaves_a_gap():
    import make_data as md
    gap = md.coverage_A(0.21, 0.035, 100_000)["gap"]
    assert 0.10 < gap < 0.13
    assert md.coverage_A(0.0, 0.0, 50_000)["gap"] == 0.0
