#!/usr/bin/env python3
"""
Orthographic sphere renderer in pure numpy (no PyVista / OpenGL needed).

Draws a lit, anti-aliased, transparent-background view of a ball.  The surface
colour comes either from a ``balls.Ball`` (analytic, crisp at any size) or from an
equirectangular texture image (same convention as gen_texture.py).
"""

import math
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import balls

LIGHT = np.array([-0.35, 0.45, 0.82])  # in view space: (right, up, toward camera)


def camera_basis(elevation_deg: float = 20.0, azimuth_deg: float = 0.0, roll_deg: float = 0.0):
    """Return (right, up, toward_camera) unit vectors in world space.

    Camera sits at azimuth/elevation on the sphere looking at the origin (z is up);
    ``roll`` spins the image about the view axis.
    """
    el, az, ro = map(math.radians, (elevation_deg, azimuth_deg, roll_deg))
    c = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    right = np.array([-math.sin(az), math.cos(az), 0.0])
    up = np.cross(c, right)
    r2 = right * math.cos(ro) + up * math.sin(ro)
    u2 = -right * math.sin(ro) + up * math.cos(ro)
    return r2, u2, c


def sample_texture(tex: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Bilinear lookup in an equirectangular texture (H, W, 3 float) at unit vectors p (N, 3)."""
    h, w = tex.shape[:2]
    lon = np.arctan2(p[:, 1], p[:, 0])
    lat = np.arcsin(np.clip(p[:, 2], -1, 1))
    x = (lon + np.pi) / (2 * np.pi) * w - 0.5
    y = (np.pi / 2 - lat) / np.pi * h - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    xa, xb = x0 % w, (x0 + 1) % w
    ya, yb = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    top = tex[ya, xa] * (1 - fx) + tex[ya, xb] * fx
    bot = tex[yb, xa] * (1 - fx) + tex[yb, xb] * fx
    return top * (1 - fy) + bot * fy


def render_view(width: int = 256, height: Optional[int] = None, *, ball: Optional[balls.Ball] = None,
                look: Optional[balls.Look] = None, texture: Optional[np.ndarray] = None,
                elevation: float = 20.0, azimuth: float = 0.0, roll: float = 0.0,
                zoom: float = 1.0, ssaa: int = 3, shade: bool = True,
                background: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
    """Render one view.  Returns float RGBA (H, W, 4); alpha = 0 outside the ball.

    Give either ``ball`` (analytic) or ``texture`` (float RGB array in [0, 1]).
    ``background`` fills the transparent area instead (returns RGB then... still RGBA, alpha 1).
    """
    if (ball is None) == (texture is None):
        raise ValueError("pass exactly one of ball= or texture=")
    height = height or width
    W, H = width * ssaa, height * ssaa
    scale = min(W, H) / 2 * zoom / 1.05  # ball radius in pixels, small margin like the original zoom(1.2)
    xs = (np.arange(W) + 0.5 - W / 2) / scale
    ys = (H / 2 - (np.arange(H) + 0.5)) / scale
    sx, sy = np.meshgrid(xs, ys)
    r2 = sx * sx + sy * sy
    inside = r2 <= 1.0
    sz = np.sqrt(np.clip(1 - r2, 0, None))

    right, up, cam = camera_basis(elevation, azimuth, roll)
    view = np.stack([sx[inside], sy[inside], sz[inside]], axis=1)       # (N, 3) in view space
    world = view[:, :1] * right + view[:, 1:2] * up + view[:, 2:3] * cam  # (N, 3) on the sphere

    aa = 1.0 / scale  # angular size of one (super-sampled) pixel, in radians near the centre
    if ball is not None:
        col = np.empty((len(world), 3), dtype=np.float32)
        for a in range(0, len(world), balls.CHUNK):
            col[a:a + balls.CHUNK] = ball.colorize(world[a:a + balls.CHUNK], look, aa_rad=aa)
    else:
        col = sample_texture(texture, world)

    if shade:
        light = LIGHT / np.linalg.norm(LIGHT)
        diff = np.clip(view @ light, 0, 1)
        half = light + np.array([0, 0, 1.0])
        half /= np.linalg.norm(half)
        spec = np.clip(view @ half, 0, 1) ** 40
        col = col * (0.42 + 0.68 * diff)[:, None] + 0.12 * spec[:, None]

    img = np.zeros((H, W, 4), dtype=np.float32)
    img[inside, :3] = col
    img[inside, 3] = 1.0
    # box-filter down with premultiplied alpha so edges stay clean
    img[..., :3] *= img[..., 3:4]
    img = img.reshape(height, ssaa, width, ssaa, 4).mean(axis=(1, 3))
    a = img[..., 3:4]
    rgb = np.where(a > 1e-6, img[..., :3] / np.maximum(a, 1e-6), 0.0)
    out = np.concatenate([np.clip(rgb, 0, 1), a], axis=-1)
    if background is not None:
        bg = np.asarray(background, dtype=np.float32)
        out = np.concatenate([out[..., :3] * a + bg * (1 - a), np.ones_like(a)], axis=-1)
    return out


def load_texture(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def to_pil(rgba: np.ndarray) -> Image.Image:
    return Image.fromarray(balls.to_uint8(rgba), "RGBA")
