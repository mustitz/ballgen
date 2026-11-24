#!/usr/bin/env python3
"""
Ball geometry core: a registry of soccer-ball panel layouts.

Every ball is described by a convex polyhedron given as half-spaces
``n_f . x <= h_f`` (unit normal ``n_f``, plane distance ``h_f``).  Radially
projecting that polyhedron onto the unit sphere gives the panel layout, and the
panel that owns a point ``p`` on the sphere is simply

    argmax_f  (p . n_f) / h_f

Seams are the places where the best and second-best scores meet, and their
distance from ``p`` can be computed analytically.  This replaces the flood
fill used by gen_texture.py: there are no leaks, no polar distortion hack and
seam width is uniform on the sphere.

Modern "curved panel" balls apply a smooth *twist* (rotation about an axis by
an angle proportional to height along that axis) to the points before the
lookup, which turns straight great-circle seams into swirling ones.

Texture convention is identical to gen_texture.py:
    x = cos(lat) cos(lon), y = cos(lat) sin(lon), z = sin(lat)
    u = (lon + pi) / 2pi,   v = (pi/2 - lat) / pi   (v = 0 is the top row)
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import ImageColor

PHI = (1 + math.sqrt(5)) / 2
CHUNK = 200_000  # points processed per numpy batch (bounds memory use)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def parse_color(c) -> np.ndarray:
    """Color name / hex / 'rgb(r,g,b)' / tuple -> float RGB in [0, 1]."""
    if isinstance(c, (tuple, list, np.ndarray)):
        a = np.asarray(c, dtype=float)
        return a / 255.0 if a.max() > 1.0 else a
    return np.asarray(ImageColor.getrgb(c), dtype=float) / 255.0


def rotation_matrix(lat_deg: float = 0.0, lon_deg: float = 0.0) -> np.ndarray:
    """Same two rotations gen_texture.py applies (X axis, then Y axis)."""
    a, b = math.radians(lat_deg), math.radians(lon_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    return ry @ rx


def lonlat_grid(width: int, height: int) -> np.ndarray:
    """Unit vectors for every pixel centre of an equirectangular image, (H, W, 3)."""
    lon = (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi
    lon, lat = np.meshgrid(lon, lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def fibonacci_sphere(n: int) -> np.ndarray:
    """Near-uniform points on the sphere (equal-area sampling)."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    t = math.pi * (1 + math.sqrt(5)) * i
    return np.stack([r * np.cos(t), r * np.sin(t), z], axis=1)


def rotate_about_axis(p: np.ndarray, axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues rotation of points p (N,3) about a unit axis by per-point angles (N,)."""
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    dot = (p @ axis)[:, None]
    return p * c + np.cross(axis, p) * s + axis[None, :] * dot * (1 - c)


# --------------------------------------------------------------------------
# Look: colours and seam appearance (kept apart from geometry)
# --------------------------------------------------------------------------

@dataclass
class Look:
    kind_colors: List[np.ndarray]          # one RGB per panel "kind"
    edge_color: np.ndarray                 # seam colour
    seam_deg: float = 1.6                  # full seam width in degrees of arc
    panel_colors: Optional[np.ndarray] = None  # optional (F, 3) overrides kind colours

    def with_colors(self, bg=None, pentagon=None, edge=None) -> "Look":
        """Mimic gen_texture.py flags: ``pentagon`` = kind 0, ``bg`` = every other kind."""
        kc = [c.copy() for c in self.kind_colors]
        if pentagon is not None:
            kc[0] = parse_color(pentagon)
        if bg is not None:
            kc = [kc[0]] + [parse_color(bg)] * (len(kc) - 1)
        return replace(self, kind_colors=kc, panel_colors=None,
                       edge_color=parse_color(edge) if edge is not None else self.edge_color)

    def with_palette(self, cmap_name: str, n_panels: int) -> "Look":
        """Colour every panel individually from a matplotlib colormap."""
        from matplotlib import colormaps
        cmap = colormaps[cmap_name]
        cols = np.array([cmap(i / max(1, n_panels - 1))[:3] for i in range(n_panels)])
        return replace(self, panel_colors=cols)


# --------------------------------------------------------------------------
# Ball
# --------------------------------------------------------------------------

@dataclass
class Ball:
    name: str
    description: str
    normals: np.ndarray            # (F, 3) unit face normals
    heights: np.ndarray            # (F,)   plane distances (scale is arbitrary)
    kinds: np.ndarray              # (F,)   int panel kind (e.g. 0 = pentagon, 1 = hexagon)
    default_look: Look
    twist_deg: float = 0.0         # swirl angle at the twist axis poles
    twist_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    kind_names: Tuple[str, ...] = ("panel",)
    warp: Optional[Callable] = None   # general equivariant warp  warp(p, inverse) -> p  (replaces twist)
    _verts: Optional[List[np.ndarray]] = field(default=None, repr=False)

    # -- basic properties --------------------------------------------------
    @property
    def n_panels(self) -> int:
        return len(self.normals)

    @property
    def planes(self) -> np.ndarray:
        """n_f / h_f, so that score = p @ planes.T."""
        return self.normals / self.heights[:, None]

    @property
    def is_warped(self) -> bool:
        """True if seams are curved by a twist or warp (panel space != real space)."""
        return self.warp is not None or self.twist_deg != 0.0

    # -- twist -------------------------------------------------------------
    def twist(self, p: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Apply (or undo) the swirl/warp.  Maps real surface points <-> panel-space points."""
        if self.warp is not None:
            return self.warp(p, inverse)
        if self.twist_deg == 0.0:
            return p
        ang = math.radians(self.twist_deg) * (p @ self.twist_axis)
        return rotate_about_axis(p, self.twist_axis, -ang if inverse else ang)

    # -- panel lookup ------------------------------------------------------
    def panel_index(self, p: np.ndarray) -> np.ndarray:
        """Panel id for each unit vector in p (N, 3)."""
        out = np.empty(len(p), dtype=np.int32)
        for a in range(0, len(p), CHUNK):
            q = self.twist(p[a:a + CHUNK])
            out[a:a + CHUNK] = np.argmax(q @ self.planes.T, axis=1)
        return out

    def panel_and_seam(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Panel id and approximate angular distance (radians) to the nearest seam."""
        idx = np.empty(len(p), dtype=np.int32)
        dist = np.empty(len(p), dtype=np.float32)
        planes = self.planes
        for a in range(0, len(p), CHUNK):
            raw = p[a:a + CHUNK]
            q = self.twist(raw)
            b, sec, sb, ss = self._top2_from(q)
            g = planes[b] - planes[sec]
            idx[a:a + CHUNK] = b
            if not self.is_warped:
                # seam between two panels lies on the great circle p . g = 0, so the exact
                # angular distance to it is asin(p.g / |g|)
                gn = np.maximum(np.linalg.norm(g, axis=1), 1e-12)
                dist[a:a + CHUNK] = np.arcsin(np.clip((sb - ss) / gn, 0.0, 1.0))
            else:
                # curved seams: first-order real-space distance |F| / |grad F| with
                # F(p) = g . Phi(p) and the gradient taken by finite differences in the tangent plane
                eps = 2e-3
                t1 = np.cross(raw, np.array([0.267, 0.534, 0.802]))
                t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-9)
                t2 = np.cross(raw, t1)
                f0 = sb - ss
                d1 = ((g * self.twist(unit(raw + eps * t1))).sum(1) - (g * self.twist(unit(raw - eps * t1))).sum(1)) / (2 * eps)
                d2 = ((g * self.twist(unit(raw + eps * t2))).sum(1) - (g * self.twist(unit(raw - eps * t2))).sum(1)) / (2 * eps)
                gradn = np.maximum(np.hypot(d1, d2), 1e-12)
                dist[a:a + CHUNK] = np.arctan(f0 / gradn)
        return idx, dist

    def _top2_from(self, q: np.ndarray):
        s = q @ self.planes.T
        part = np.argpartition(s, self.n_panels - 2, axis=1)[:, -2:]
        rows = np.arange(len(q))[:, None]
        sv = s[rows, part]
        order = np.argsort(sv, axis=1)
        part, sv = part[rows, order], sv[rows, order]
        return part[:, 1], part[:, 0], sv[:, 1], sv[:, 0]

    # -- colouring ---------------------------------------------------------
    def colorize(self, p: np.ndarray, look: Optional[Look] = None, aa_rad: float = 0.0) -> np.ndarray:
        """RGB (N, 3) in [0, 1] for unit vectors p.  ``aa_rad`` = angular size of one pixel."""
        look = look or self.default_look
        idx, dist = self.panel_and_seam(p)
        if look.panel_colors is not None:
            base = look.panel_colors[idx % len(look.panel_colors)]
        else:
            kc = np.array(look.kind_colors)
            base = kc[np.minimum(self.kinds[idx], len(kc) - 1)]
        half = math.radians(look.seam_deg) / 2
        soft = max(aa_rad, 1e-6)
        m = np.clip((half - dist) / soft + 0.5, 0.0, 1.0)[:, None]
        return base * (1 - m) + look.edge_color[None, :] * m

    # -- geometry of the panels (for plotting / statistics) ----------------
    def face_vertices(self) -> List[np.ndarray]:
        """Ordered polygon corners (unit sphere, panel space) for every face."""
        if self._verts is None:
            self._verts = _polyhedron_faces(self.normals, self.heights)
        return self._verts


def _polyhedron_faces(normals: np.ndarray, heights: np.ndarray, eps: float = 1e-7) -> List[np.ndarray]:
    """Vertices of the polyhedron {x : n.x <= h}, grouped and ordered per face, on the unit sphere."""
    F = len(normals)
    corners: List[np.ndarray] = []
    for i, j, k in itertools.combinations(range(F), 3):
        m = normals[[i, j, k]]
        if abs(np.linalg.det(m)) < 1e-9:
            continue
        x = np.linalg.solve(m, heights[[i, j, k]])
        if np.all(normals @ x <= heights + eps):
            corners.append(x)
    corners = np.array(corners)
    # de-duplicate
    uniq: List[np.ndarray] = []
    for c in corners:
        if not any(np.linalg.norm(c - u) < 1e-6 for u in uniq):
            uniq.append(c)
    verts = np.array(uniq)
    faces = []
    for n, h in zip(normals, heights):
        on = verts[np.abs(verts @ n - h) < 1e-6]
        c = on.mean(0)
        a = unit(on[0] - c)
        b = np.cross(n, a)
        ang = np.arctan2((on - c) @ b, (on - c) @ a)
        faces.append(unit(on[np.argsort(ang)]))
    return faces


# --------------------------------------------------------------------------
# Polyhedra (half-space descriptions)
# --------------------------------------------------------------------------

def _sign_triples() -> np.ndarray:
    return np.array([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)], dtype=float)


def _axes() -> np.ndarray:
    return np.array([s * np.eye(3)[i] for i in range(3) for s in (1, -1)], dtype=float)


def _cyclic(v: Sequence[float]) -> List[np.ndarray]:
    a, b, c = v
    return [np.array([a, b, c]), np.array([c, a, b]), np.array([b, c, a])]


def _icosahedron_vertices() -> np.ndarray:
    out = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            out += _cyclic((0, s1, s2 * PHI))
    return unit(np.array(out))


def _dodecahedron_vertices() -> np.ndarray:
    out = [t for t in _sign_triples()]
    for s1 in (1, -1):
        for s2 in (1, -1):
            out += _cyclic((0, s1 / PHI, s2 * PHI))
    return unit(np.array(out))


def poly_cube():
    return _axes(), np.ones(6), np.zeros(6, dtype=int)


def poly_octahedron():
    n = unit(_sign_triples())
    return n, np.full(8, 1 / math.sqrt(3)), np.zeros(8, dtype=int)


def poly_dodecahedron():
    n = _icosahedron_vertices()
    return n, np.ones(12), np.zeros(12, dtype=int)


def poly_icosahedron():
    n = _dodecahedron_vertices()
    return n, np.ones(20), np.zeros(20, dtype=int)


def poly_truncated_octahedron():
    sq, hx = _axes(), unit(_sign_triples())
    n = np.vstack([sq, hx])
    h = np.concatenate([np.full(6, 2.0), np.full(8, math.sqrt(3))])
    kinds = np.array([0] * 6 + [1] * 8)
    return n, h, kinds


def poly_truncated_icosahedron():
    """Classic 32-panel ball, built from the vertex/face tables in gen_texture.py."""
    import gen_texture as gt
    verts = np.array(gt.VERTICES, dtype=float)
    seen, normals, heights, kinds = set(), [], [], []
    for face in gt.FACES:
        key = frozenset(face)
        if key in seen:        # gen_texture lists one hexagon twice
            continue
        seen.add(key)
        pts = verts[list(face)]
        n = unit(pts.mean(0))
        normals.append(n)
        heights.append(float(pts[0] @ n))
        kinds.append(0 if len(face) == 5 else 1)
    return np.array(normals), np.array(heights), np.array(kinds)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

REGISTRY: Dict[str, Callable[[], Ball]] = {}


def register(fn: Callable[[], Ball]) -> Callable[[], Ball]:
    REGISTRY[fn.__name__.replace("_", "-")] = fn
    return fn


def _look(kind_colors, edge="#1b1b1b", seam_deg=1.6) -> Look:
    return Look([parse_color(c) for c in kind_colors], parse_color(edge), seam_deg)


def _ball(name, desc, poly, look, twist=0.0, kind_names=("panel",), axis=(0, 0, 1)) -> Ball:
    n, h, k = poly
    return Ball(name=name, description=desc, normals=n, heights=h, kinds=k, default_look=look,
                twist_deg=twist, twist_axis=unit(np.array(axis, dtype=float)), kind_names=kind_names)


@register
def classic_32() -> Ball:
    return _ball("classic-32", "Truncated icosahedron: 12 pentagons + 20 hexagons (the 'Telstar' layout)",
                 poly_truncated_icosahedron(), _look(["#111111", "#f4f4f4"], "#111111", 1.2),
                 kind_names=("pentagon", "hexagon"))


@register
def classic_32_red() -> Ball:
    return _ball("classic-32-red", "Truncated icosahedron, red pentagons / white hexagons, grey seams",
                 poly_truncated_icosahedron(), _look(["#c8102e", "#f7f7f7"], "#444444", 1.2),
                 kind_names=("pentagon", "hexagon"))


@register
def classic_32_retro() -> Ball:
    return _ball("classic-32-retro", "Truncated icosahedron, cream hexagons with brown leather pentagons",
                 poly_truncated_icosahedron(), _look(["#5a3a22", "#e9dcc0"], "#3a2616", 1.8),
                 kind_names=("pentagon", "hexagon"))


@register
def panel_6() -> Ball:
    return _ball("panel-6", "6 swirled panels (spherical cube, twisted seams); opposite panels share a colour",
                 _kinds_by_axis(poly_cube()), _look(["#f2f2f2", "#1e5bc6", "#f29a00"], "#202020", 1.6),
                 twist=50, kind_names=("x panel", "y panel", "z panel"), axis=(1, 1, 1))


@register
def panel_8() -> Ball:
    n, h, _ = poly_octahedron()
    kinds = (np.prod(np.sign(n), axis=1) > 0).astype(int)
    return _ball("panel-8", "8 twisted triangular-ish panels (octahedral) in two alternating colours",
                 (n, h, kinds), _look(["#f2f2f2", "#16335e"], "#202020", 1.6),
                 twist=40, kind_names=("light", "dark"), axis=(0, 0, 1))


@register
def panel_12() -> Ball:
    return _ball("panel-12", "12 pentagonal panels (dodecahedral), no twist",
                 poly_dodecahedron(), _look(["#f2f2f2"], "#202020", 1.5))


@register
def panel_14() -> Ball:
    return _ball("panel-14", "14 panels: 6 squares + 8 hexagons (truncated octahedron)",
                 poly_truncated_octahedron(), _look(["#101010", "#f2f2f2"], "#101010", 1.4),
                 kind_names=("square", "hexagon"))


@register
def panel_14_twist() -> Ball:
    return _ball("panel-14-twist", "Truncated-octahedron topology with a gentle swirl",
                 poly_truncated_octahedron(), _look(["#e4572e", "#f2f2f2"], "#222222", 1.4),
                 twist=25, kind_names=("square", "hexagon"), axis=(1, 1, 1))


@register
def panel_20() -> Ball:
    return _ball("panel-20", "20 swirled triangular-ish panels (icosahedral)",
                 poly_icosahedron(), _look(["#f2f2f2"], "#c0392b", 1.6), twist=30)



# --------------------------------------------------------------------------
# Chiral equivariant balls (Trionda = the tetrahedral case)
# --------------------------------------------------------------------------
#
# For a finite rotation group G and a G-invariant smooth function f on the sphere, the field
#     v(p) = p x grad f(p)
# is tangent, divergence-free (its flow preserves area), G-equivariant (v(gp) = g v(p) for
# rotations g) and odd under reflections.  It vanishes at every point with a non-trivial
# stabiliser, i.e. at all vertices, edge midpoints and face centres of the polyhedron.
# The flow Phi_tau therefore maps a G-invariant partition to a chiral partition whose panels
# are still G-congruent, with exactly shared seams.

TETRA_VERTICES = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) / math.sqrt(3)


def poly_tetrahedron():
    """Four faces; the face opposite vertex T_k has outward normal -T_k and plane distance 1/3."""
    return -TETRA_VERTICES.copy(), np.full(4, 1 / 3), np.arange(4)


def grad_tetra(p):
    """grad of f = xyz  (the degree-3 invariant of the tetrahedral group)."""
    return np.stack([p[:, 1] * p[:, 2], p[:, 0] * p[:, 2], p[:, 0] * p[:, 1]], axis=1)


def grad_cube(p):
    """grad of f = x^4 + y^4 + z^4  (octahedral group invariant)."""
    return 4 * p ** 3


def _one_per_pair(vs: np.ndarray) -> np.ndarray:
    axes: List[np.ndarray] = []
    for x in vs:
        if not any(abs(abs(x @ y) - 1) < 1e-9 for y in axes):
            axes.append(x)
    return np.array(axes)


def make_grad_icosa(axes: np.ndarray):
    """grad of f = sum over the 6 five-fold axes a of (p.a)^6  (icosahedral group invariant).

    The axes must be taken from the polyhedron being twisted so that the frames agree.
    """
    def grad(p):
        d = p @ axes.T
        return 6 * (d ** 5) @ axes
    return grad


def icosa_axes_of(poly) -> np.ndarray:
    """Five-fold axes of an icosahedron (its vertices) or of a dodecahedron (its face normals)."""
    n, h, _ = poly
    if len(n) == 12:
        return _one_per_pair(n)
    return _one_per_pair(np.vstack(_polyhedron_faces(n, h)))


def tetra_flow_field(p: np.ndarray, grad=grad_tetra, scale: float = 1.0) -> np.ndarray:
    """v(p) = scale * p x grad f(p)."""
    return scale * np.cross(p, grad(p))


def make_flow_warp(tau: float, grad=grad_tetra, scale: float = 1.0, steps: Optional[int] = None) -> Callable:
    """Time-tau flow of the rotated-gradient field (RK4 on the sphere); ``inverse`` flows backwards."""
    n = steps or max(8, int(math.ceil(16 * abs(tau))))

    def field(q):
        return tetra_flow_field(q, grad, scale)

    def warp(p: np.ndarray, inverse: bool = False) -> np.ndarray:
        h = (-tau if inverse else tau) / n
        q = unit(p)
        for _ in range(n):
            k1 = field(q)
            k2 = field(unit(q + 0.5 * h * k1))
            k3 = field(unit(q + 0.5 * h * k2))
            k4 = field(unit(q + h * k3))
            q = unit(q + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        return q

    return warp


def _edge_jacobian(grad, mid: np.ndarray, edge_dir: np.ndarray) -> np.ndarray:
    """2x2 Jacobian of the (unscaled) rotated-gradient field at an edge midpoint, in the
    orthonormal tangent basis (e, t): e along the straight edge, t = mid x e."""
    mid = unit(mid)
    e = unit(edge_dir - (edge_dir @ mid) * mid)
    t = np.cross(mid, e)
    eps = 1e-5
    J = np.zeros((2, 2))
    for j, d in enumerate((e, t)):
        dv = (tetra_flow_field(unit((mid + eps * d)[None]), grad) - tetra_flow_field(unit((mid - eps * d)[None]), grad))[0] / (2 * eps)
        J[:, j] = [dv @ e, dv @ t]
    return J


def edge_tilt_deg(J: np.ndarray, tau: float) -> float:
    """Angle (degrees) between a seam and the straight edge at the edge midpoint after the flow.

    J is traceless (divergence-free flow), so J^2 = lambda^2 I and exp(-tau J) = cosh(l tau) I -
    sinh(l tau)/l J; the seam is the pre-image of the edge, whose tangent is exp(-tau J) e.
    """
    lam = math.sqrt(max(-np.linalg.det(J), 1e-18))
    d = math.cosh(lam * tau) * np.array([1.0, 0.0]) - math.sinh(lam * tau) / lam * J[:, 0]
    return math.degrees(math.atan2(abs(d[1]), d[0]))


def solve_tau_for_tilt(J: np.ndarray, tilt_deg: float) -> float:
    lo, hi = 0.0, 40.0
    if edge_tilt_deg(J, hi) < tilt_deg:
        raise ValueError(f"tilt {tilt_deg} deg is not reachable (limit {edge_tilt_deg(J, hi):.2f} deg)")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if edge_tilt_deg(J, mid) < tilt_deg:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


TRIONDA_TAU = 0.9
_PALETTE = ["#f2f2f2", "#d52b1e", "#0a7d4b", "#1f55b5", "#f2a900", "#7b3fa0"]


FAMILY_TILT_DEG = 20.0   # common calibration for comparing families (icosahedral limit is 25.1 deg)


def make_chiral(base_poly, grad, name: str, description: str, tau: Optional[float] = None,
                tilt_deg: Optional[float] = None, kind_colors=None, edge="#1b1b1b", seam_deg=1.6) -> Ball:
    """Chiral (tau != 0) or straight (tau = 0) ball from a G-invariant polyhedron and invariant f.

    The flow speed is first normalised so the hyperbolic rate at an edge midpoint is 1.  Give
    either ``tau`` (flow time) or ``tilt_deg`` (the angle between a seam and the straight edge
    where it crosses the edge midpoint); the latter makes different families comparable.
    """
    n, h, k = base_poly
    ball = _ball(name, description, (n, h, k),
                 _look(kind_colors or _PALETTE[: max(1, int(k.max()) + 1)], edge, seam_deg),
                 kind_names=tuple(f"panel {c}" for c in "ABCDEFGH"[: int(k.max()) + 1]))
    ball.tau = 0.0
    ball.edge_tilt = 0.0
    if tau == 0 or tilt_deg == 0:
        return ball
    V = ball.face_vertices()[0]
    J = _edge_jacobian(grad, V[0] + V[1], V[1] - V[0])
    lam = math.sqrt(max(-np.linalg.det(J), 1e-18))
    if tau is None:
        tau = solve_tau_for_tilt(J / lam, tilt_deg)
    ball.warp = make_flow_warp(tau, grad, 1.0 / lam)
    ball.tau = tau
    ball.edge_tilt = edge_tilt_deg(J / lam, tau)
    return ball


def make_trionda(tau: float = TRIONDA_TAU, name: Optional[str] = None) -> Ball:
    """The Trionda model: four congruent chiral boomerang panels with tetrahedral rotation symmetry."""
    return make_chiral(poly_tetrahedron(), grad_tetra,
                       name or ("trionda" if tau == TRIONDA_TAU else f"trionda-tau{tau:g}"),
                       "4 congruent chiral boomerang panels (tetrahedral T symmetry, equivariant flow model)",
                       tau=tau, kind_colors=["#f2f2f2", "#d52b1e", "#0a7d4b", "#1f55b5"])


@register
def tetra_4() -> Ball:
    return make_chiral(poly_tetrahedron(), grad_tetra, "tetra-4",
                       "4 straight tetrahedral panels (T_d), the achiral limit of the Trionda model",
                       tau=0.0, kind_colors=["#f2f2f2", "#d52b1e", "#0a7d4b", "#1f55b5"])


@register
def trionda() -> Ball:
    return make_trionda(TRIONDA_TAU)


@register
def cube_6() -> Ball:
    return make_chiral(poly_cube_k(), grad_cube, "cube-6", "6 straight panels (spherical cube, O_h)", tau=0.0)


@register
def cube_6_chiral() -> Ball:
    return make_chiral(poly_cube_k(), grad_cube, "cube-6-chiral",
                       "6 chiral panels (cube topology, octahedral rotation group O)", tilt_deg=FAMILY_TILT_DEG)


@register
def octa_8() -> Ball:
    return make_chiral(poly_octa_k(), grad_cube, "octa-8", "8 straight panels (spherical octahedron, O_h)", tau=0.0)


@register
def octa_8_chiral() -> Ball:
    return make_chiral(poly_octa_k(), grad_cube, "octa-8-chiral",
                       "8 chiral panels (octahedron topology, rotation group O)", tilt_deg=FAMILY_TILT_DEG)


@register
def dodeca_12_chiral() -> Ball:
    return make_chiral(poly_dodecahedron_k(), make_grad_icosa(icosa_axes_of(poly_dodecahedron())), "dodeca-12-chiral",
                       "12 chiral panels (dodecahedron topology, icosahedral rotation group I)", tilt_deg=FAMILY_TILT_DEG)


@register
def icosa_20() -> Ball:
    return make_chiral(poly_icosahedron_k(), make_grad_icosa(icosa_axes_of(poly_icosahedron())), "icosa-20", "20 straight triangular panels (I_h)", tau=0.0)


@register
def icosa_20_chiral() -> Ball:
    return make_chiral(poly_icosahedron_k(), make_grad_icosa(icosa_axes_of(poly_icosahedron())), "icosa-20-chiral",
                       "20 chiral panels (icosahedron topology, rotation group I)", tilt_deg=FAMILY_TILT_DEG)


def _kinds_distinct(poly):
    n, h, _ = poly
    return n, h, np.arange(len(n)) % 6


def poly_cube_k():
    return _kinds_distinct(poly_cube())


def poly_octa_k():
    return _kinds_distinct(poly_octahedron())


def poly_dodecahedron_k():
    return _kinds_distinct(poly_dodecahedron())


def poly_icosahedron_k():
    return _kinds_distinct(poly_icosahedron())


def _kinds_by_axis(poly):
    n, h, _ = poly
    return n, h, np.argmax(np.abs(n), axis=1)


def get_ball(name: str) -> Ball:
    key = name.replace("_", "-")
    if key not in REGISTRY:
        raise KeyError(f"unknown ball '{name}'. Available: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[key]()


def list_balls() -> List[str]:
    return sorted(REGISTRY)


# --------------------------------------------------------------------------
# Rasterising
# --------------------------------------------------------------------------

def render_texture(ball: Ball, look: Optional[Look] = None, width: int = 1024, height: int = 512,
                   lat_rotation: float = 0.0, lon_rotation: float = 0.0) -> np.ndarray:
    """Equirectangular texture as float RGB (H, W, 3).

    ``lat_rotation`` / ``lon_rotation`` rotate the ball (same convention as gen_texture.py).
    """
    pts = lonlat_grid(width, height).reshape(-1, 3)
    rot = rotation_matrix(lat_rotation, lon_rotation)
    pts = pts @ rot          # rotating the sampling frame the opposite way == rotating the ball
    out = np.empty((len(pts), 3), dtype=np.float32)
    aa = 2 * math.pi / width
    for a in range(0, len(pts), CHUNK):
        out[a:a + CHUNK] = ball.colorize(pts[a:a + CHUNK], look, aa_rad=aa)
    return out.reshape(height, width, 3)


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
