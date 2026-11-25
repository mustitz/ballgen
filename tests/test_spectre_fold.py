import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pytest
from spectre.patch import SpectrePatch
from spectre.fold import PatchGraph, fold_search, grow_patch, corner_angles

@pytest.fixture(scope="module")
def patch():
    return SpectrePatch(levels=2)

def test_tile_angles_are_integer_and_sum(patch):
    a = corner_angles(patch.straight_polygon(0))
    assert len(a) == 14 and sum(a) == 2160 and all(x % 30 == 0 for x in a)

def test_patch_sizes():
    assert SpectrePatch(levels=2).placements.__len__() == 71

@pytest.mark.parametrize("N", [5, 9, 15])
def test_gauss_bonnet_ledger(patch, N):
    r = fold_search(PatchGraph(patch, tiles=grow_patch(patch, N)), beam=40)
    assert r.ledger == 720
    assert r.total_deficit + r.total_hole_curvature == 720
    assert all(c.deficit >= 0 for c in r.cones)

@pytest.mark.parametrize("N", [5, 9])
def test_hole_curvature_formula(patch, N):
    r = fold_search(PatchGraph(patch, tiles=grow_patch(patch, N)), beam=40)
    for h in r.holes:
        assert h.K == 360 + 180 * h.n - sum(h.corners) or h.K == sum(h.alpha) - (h.n - 2) * 180
        assert len(h.edges) == h.n


def test_holes_roundtrip(patch, tmp_path):
    import json
    from spectre.fold import PatchGraph, fold_search, grow_patch
    from spectre.embed import embed
    from spectre.holes_io import load_holes
    import fold_spectre
    r = fold_search(PatchGraph(patch, tiles=grow_patch(patch, 9)), beam=40)
    rows = fold_spectre.hole_records(embed(r), 9)
    p = tmp_path / "h.json"
    p.write_text(json.dumps(rows))
    got = load_holes(str(p))
    assert len(got) == len(r.holes)
    for g, h in zip(got, r.holes):
        assert g.sides == h.n and g.K == h.K and g.alpha == list(h.alpha)


# ---- exact DP / multi-hole ------------------------------------------------------------------------
@pytest.fixture(scope="module")
def p3():
    return SpectrePatch(levels=3)


def _graph(pt, N):
    return PatchGraph(pt, tiles=grow_patch(pt, N))


@pytest.mark.parametrize("N", [1, 2, 3, 6])
def test_dp_closes_ball(p3, N):
    from spectre.dp import fold_with_holes
    r = fold_with_holes(_graph(p3, N), [])
    assert r is not None and r.holes == [] and r.ledger == 720
    assert all(c.angle <= 360 for c in r.cones)


def test_dp_agrees_with_beam_on_infeasible(p3):
    from spectre.dp import can_close
    g = _graph(p3, 4)
    assert not can_close([int(a) for a in g.loops[0][1]])


def test_dp_single_tile_hole_matches_exhaustive(p3):
    from spectre.dp import fold_with_holes
    g = _graph(p3, 1)
    r = fold_with_holes(g, [[240, 270, 240, 270]])
    assert r is not None and r.ledger == 720
    h = r.holes[0]
    assert h.n == 4 and h.K == 60 and sorted(h.corners) == [240, 240, 270, 270]
    # the exhaustive enumeration found no K=0 hexagon for one tile
    assert fold_with_holes(g, [[240] * 6]) is None


def test_two_holes_ledger():
    import random
    from spectre.patch import SpectrePatch
    from spectre.search import PatchPool
    from spectre.dp import fold_with_holes
    pool = PatchPool(SpectrePatch(levels=3))
    rng = random.Random(2)
    sq = [240, 270, 240, 270]
    for _ in range(300):
        g = pool.graph(pool.random_subset(rng.randint(8, 12), rng, 0.3))
        if g is None:
            continue
        r = fold_with_holes(g, [sq, sq])
        if r:
            assert len(r.holes) == 2 and r.ledger == 720 == r.total_deficit + 120
            assert all(h.K == 60 for h in r.holes)
            return
    pytest.fail("no two-hole fold found")


def test_signature_is_rotation_invariant(p3):
    from spectre.dp import fold_with_holes
    from spectre.connect import signature
    from spectre.fold import Hole
    r = fold_with_holes(_graph(p3, 1), [[240, 270, 240, 270]])
    h = r.holes[0]
    rot = Hole(h.edges[1:] + h.edges[:1], h.corners[1:] + h.corners[:1], h.vertices[1:] + h.vertices[:1])
    assert signature(r, h) == signature(r, rot)


# ---- joining spheres ------------------------------------------------------------------------------
def _two_hole_folds(k=3):
    import random
    from spectre.patch import SpectrePatch
    from spectre.search import PatchPool
    from spectre.dp import fold_with_holes
    pool = PatchPool(SpectrePatch(levels=3))
    rng = random.Random(2)
    out = []
    for _ in range(400):
        g = pool.graph(pool.random_subset(rng.randint(8, 12), rng, 0.3))
        if g is None:
            continue
        r = fold_with_holes(g, [[240, 270, 240, 270]] * 2)
        if r:
            out.append(r)
        if len(out) == k:
            break
    return out


def test_join_ledger_and_saddles():
    from spectre.connect import join_spheres
    a, b, *_ = _two_hole_folds()
    js = join_spheres(a, 0, b, 0)
    assert js.ledger == 720
    assert len(js.open_holes) == 2 and all(K == 60 for _, _, K in js.open_holes)
    assert all(d < 0 for d in js.junction_deficits)       # 240/270 + 240/270 > 360: saddle neck


def test_join_requires_equal_sides(p3):
    from spectre.connect import join_spheres
    from spectre.dp import fold_with_holes
    a = fold_with_holes(_graph(p3, 1), [[240, 270, 240, 270]])
    b = fold_with_holes(_graph(p3, 1), [[120, 240, 240, 270, 270, 270]])
    if b is not None:
        with pytest.raises(ValueError):
            join_spheres(a, 0, b, 0)


def test_embed_joined_seam_is_closed():
    from spectre.connect import join_spheres
    from spectre.embed import embed_joined
    a, b, *_ = _two_hole_folds()
    je = embed_joined(join_spheres(a, 0, b, 0))
    assert len(set(je.rim.tolist())) == 4                  # the 4 seam vertices are single points
    assert np.isfinite(je.position).all() and je.rms_strain < 0.1
