import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ball_stats
import balls
import render_sphere as rs

ALL = balls.list_balls()


@pytest.mark.parametrize("name", ALL)
def test_every_panel_is_used_and_areas_sum_to_sphere(name):
    b = balls.get_ball(name)
    st = ball_stats.analyze(b, n=100_000)
    assert np.all(st["panel_area"] > 0)
    assert st["panel_area"].sum() == pytest.approx(4 * math.pi, rel=1e-6)


@pytest.mark.parametrize("name", [n for n in ALL if "twist" not in n])
def test_polygon_vertex_counts(name):
    b = balls.get_ball(name)
    sides = sorted(len(f) for f in b.face_vertices())
    expected = {"classic": [5] * 12 + [6] * 20, "panel-12": [5] * 12, "panel-14": [4] * 6 + [6] * 8,
                "panel-20": [3] * 20, "panel-6": [4] * 6, "panel-8": [3] * 8,
                "tetra-4": [3] * 4, "cube-6": [4] * 6, "octa-8": [3] * 8, "dodeca-12": [5] * 12, "icosa-20": [3] * 20, "goldberg-42": [5] * 12 + [6] * 30}
    key = "classic" if name.startswith("classic") else name.replace("-chiral", "")
    key = "tetra-4" if key == "trionda" else key
    assert sides == expected[key]


def test_classic_matches_truncated_icosahedron():
    b = balls.get_ball("classic-32")
    assert b.n_panels == 32 and sorted(set(b.kinds)) == [0, 1]
    # 90 edges, unit circumradius -> edge arc 0.4062 rad
    assert ball_stats.exact_seam_length(b) == pytest.approx(90 * 0.4062, rel=2e-3)


def test_cube_seam_length_and_cap():
    b = balls.get_ball("panel-6")
    b.twist_deg = 0
    st = ball_stats.analyze(b, n=300_000)
    assert st["seam_length"] == pytest.approx(12 * math.acos(1 / 3), rel=1e-6)
    assert st["max_cap_deg"] == pytest.approx(45.0, abs=0.5)
    assert st["seam_length_band"] == pytest.approx(st["seam_length"], rel=0.03)


def test_twist_is_area_preserving_and_inverts():
    b = balls.get_ball("panel-6")
    p = balls.fibonacci_sphere(1000)
    assert np.allclose(b.twist(b.twist(p), inverse=True), p)
    st = ball_stats.analyze(b, n=300_000)
    assert st["area_cv"] < 0.01
    untwisted = balls.get_ball("panel-6"); untwisted.twist_deg = 0
    assert st["seam_length"] > ball_stats.exact_seam_length(untwisted)  # swirled seams are longer


def test_seam_distance_nonnegative_and_small_at_boundary():
    b = balls.get_ball("panel-12")
    idx, d = b.panel_and_seam(balls.fibonacci_sphere(20000))
    assert d.min() >= 0 and d.max() < math.radians(32)


def test_texture_shape_and_wrap_continuity():
    rgb = balls.render_texture(balls.get_ball("classic-32"), width=256, height=128)
    assert rgb.shape == (128, 256, 3)
    # no seam at the dateline: left and right edge columns differ no more than neighbours do
    assert np.abs(rgb[:, 0] - rgb[:, -1]).mean() < 0.15


def test_texture_roundtrip_through_sprite_renderer():
    b = balls.get_ball("panel-8")
    tex = balls.render_texture(b, width=512, height=256)
    a = rs.render_view(64, ball=b, shade=False, ssaa=1)
    c = rs.render_view(64, texture=tex, shade=False, ssaa=1)
    inside = a[..., 3] > 0.99
    assert np.abs(a[inside, :3] - c[inside, :3]).mean() < 0.06


def test_sprite_has_transparent_corners_and_opaque_centre():
    img = rs.render_view(64, ball=balls.get_ball("classic-32"))
    assert img[0, 0, 3] == 0 and img[32, 32, 3] == 1


def test_unknown_ball_raises():
    with pytest.raises(KeyError):
        balls.get_ball("nope")
