"""Tests for the_hat_trick_ball.py (60 hats + 12 pentagons on the icosahedral Goldberg GP(3,0) frame)."""
import math, os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import the_hat_trick_ball as H


@pytest.fixture(scope='module')
def balls():
    return [H.HatTrick(variant=v) for v in (0, 1)]


@pytest.fixture(scope='module')
def relaxed(balls):
    return H.relax(balls[0])


def test_hat_is_the_hat():
    kites, outline = H.flat_hat()
    assert len(kites) == 8
    ang = sorted(H.hat_angles(outline))
    assert ang == [90] * 4 + [120] * 5 + [240] * 2 + [270] * 2          # the hat's 13 corners
    assert sum(ang) == 11 * 180
    # area of 8 kites; kite area sqrt(3)
    area = 0.5 * sum(outline[i, 0] * outline[(i + 1) % len(outline), 1]
                     - outline[(i + 1) % len(outline), 0] * outline[i, 1] for i in range(len(outline)))
    assert area == pytest.approx(8 * math.sqrt(3), abs=1e-2)
    # edge lengths: six 1s, six sqrt(3)s and one 2 (after merging the straight point)
    n = len(outline)
    L = np.array([np.linalg.norm(outline[(i + 1) % n] - outline[i]) for i in range(n)])
    assert np.sum(abs(L - 1) < 1e-2) == 8 and np.sum(abs(L - math.sqrt(3)) < 1e-2) == 6   # 2-edge = two 1-segments


def test_goldberg_frame(balls):
    b = balls[0]
    assert len(b.V) == 92 and len(b.F) == 180 and len(b.pent) == 12
    assert len(b.S.kites) == 480                                       # 80 hexagons x 6 kites
    assert len(H.rotation_group()) == 60


def test_hat_placements_and_symmetric_tilings(balls):
    b = balls[0]
    assert len(b.placements) == 360
    assert len(b.tilings) == 2
    for t in b.tilings:
        assert len(t) == 60 and len(set().union(*t)) == 480


@pytest.mark.parametrize('variant', [0, 1])
def test_72_panels_and_patches(balls, variant):
    b = balls[variant]
    assert len(b.hats) == 60 and all(len(h) == 8 for h in b.hats)
    assert sorted(k for h in b.hats for k in h) == sorted(b.S.kites)
    for p in b.pent:
        assert len(b.pentagon_hats(p)) == 5                            # five hats round each pentagon
    for i in range(60):
        loop, true, ang = b.hat_outline(i)
        assert len(true) == 13 and len(loop) == 14                     # a hat has 13 corners
        assert sorted(ang[j] for j in true) == [90] * 4 + [120] * 5 + [240] * 2 + [270] * 2


def test_hats_are_one_orbit_of_the_rotation_group(balls):
    b = balls[0]
    h0 = frozenset(b.hats[0])
    assert {frozenset(kp[k] for k in h0) for kp in b.kperm} == {frozenset(h) for h in b.hats}


def test_pentagon_corner_angles_are_120(balls):
    """Hats put 240 degrees at every pentagon corner, so the pentagon is a 120-degree pentagon
    (K = 60 degrees each; 12 x 60 = 720)."""
    b = balls[0]
    total = 0
    # corner centroids of a pentagon carry exactly two kites (2 x 120 = 240)
    for p in b.pent:
        for t, f in enumerate(b.F):
            if p in f:
                kites = [3 * t + j for j in range(3) if b.F[t][j] not in b.S.skip]
                assert len(kites) == 2
                total += 1
    assert total == 60
    assert 12 * (5 * 120 - 540) == 720


def test_relaxation_is_nearly_flat_kites(relaxed):
    ids, X, a, info = relaxed
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)
    assert info['rms_strain'] < 0.03 and info['max_strain'] < 0.07


def test_panel_areas(balls, relaxed):
    ids, X, a, info = relaxed
    hats, pents = H.panel_areas(balls[0], ids, X)
    assert (hats.sum() + pents.sum()) == pytest.approx(4 * math.pi, rel=1e-6)
    assert (hats.max() - hats.min()) / hats.mean() < 1e-4              # all 60 hats equal
    assert 2.0 < hats.mean() / pents.mean() < 2.6
