import numpy as np
import pytest

import ball_metrics as bm
import balls


@pytest.mark.parametrize("name", ["goldberg-42", "goldberg-42-chiral"])
def test_goldberg_panels_are_equal_area(name):
    b = balls.get_ball(name)
    idx = b.panel_index(balls.fibonacci_sphere(300000))
    frac = np.bincount(idx, minlength=42) / 300000 * 42
    assert len(b.normals) == 42 and (b.kinds == 0).sum() == 12
    assert frac.min() > 0.995 and frac.max() < 1.005


def test_goldberg_faces_are_pentagons_and_hexagons():
    b = balls.get_ball("goldberg-42")
    sides = np.array([len(v) for v in b.face_vertices()])
    assert np.all(sides[b.kinds == 0] == 5) and np.all(sides[b.kinds == 1] == 6)


def test_goldberg_symmetry_groups():
    s = bm.symmetry_group(balls.get_ball("goldberg-42"))
    c = bm.symmetry_group(balls.get_ball("goldberg-42-chiral"))
    assert (s["proper"], s["improper"]) == (60, 60)
    assert (c["proper"], c["improper"]) == (60, 0)


def test_goldberg_chiral_multipole_selection_rule():
    # I-invariant harmonics first appear at l = 6
    q = bm.multipole_spectrum(balls.get_ball("goldberg-42-chiral"), lmax=8)
    assert q[1:6].max() < 1e-9 and q[6] > 1e-9
