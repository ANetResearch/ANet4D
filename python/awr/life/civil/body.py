"""Pose-dependent footprints of animated people.

Both renderers (native raster, ``Mesh.vertices``, and the web player) draw a person in the rest pose at or below
``REST_BELOW_MPS`` and blend every vertex linearly towards the walk cycle, reaching the full walk pose at
``FULL_WALK_MPS``; the speed is the recorded horizontal speed of the tick. A blended vertex is a convex combination of
a rest vertex and a walk-cycle vertex, so the footprint at blend ``b`` is bounded, side by side, by
``(1-b)*rest + b*walk``. The walk extents include the rest extents (the walk cycle starts from the rest pose and its
interpolated frames stay within the per-frame extremes), so footprints grow monotonically with speed.

Extents are ``[x_back, x_front, y_right, y_left]`` in the body frame (x forward), scaled with the rendered height and
widened by ``MARGIN_M`` on every side. Vehicles and street objects keep their centred catalogue boxes.
"""

from __future__ import annotations

from functools import cache

import numpy as np

REST_BELOW_MPS = 0.03
FULL_WALK_MPS = 0.5
MARGIN_M = 0.02


@cache
def person_extents(asset: str, height_m: float) -> np.ndarray:
    """(2, 4) array: rest and walk extents of ``asset`` drawn ``height_m`` tall."""
    from awr.life.appearance.assets import load

    spec, mesh, _ = load(asset)
    scale = float(height_m) / float(spec["dimensions_m"][2])
    p = np.asarray(mesh["p"], float)[:, :2]
    rest = np.array([p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()])
    walk = rest.copy()
    if "poses" in mesh:
        poses = np.asarray(mesh["poses"], float)[..., :2]
        walk = np.array(
            [
                min(rest[0], poses[..., 0].min()),
                max(rest[1], poses[..., 0].max()),
                min(rest[2], poses[..., 1].min()),
                max(rest[3], poses[..., 1].max()),
            ]
        )
    pad = np.array([-MARGIN_M, MARGIN_M, -MARGIN_M, MARGIN_M])
    return np.stack([rest * scale + pad, walk * scale + pad])


def centred(size) -> np.ndarray:
    """(2, 4) extents of a rigid centred box (vehicles, street objects)."""
    half_l, half_w = float(size[0]) / 2, float(size[1]) / 2
    e = np.array([-half_l, half_l, -half_w, half_w])
    return np.stack([e, e])


def actor_extents(actors, sizes) -> np.ndarray:
    """(n, 2, 4) extents for an actor table (``appearance.asset`` and ``size_mm`` as in the tape)."""
    out = np.zeros((len(actors), 2, 4))
    for i, a in enumerate(actors):
        asset = (a.get("appearance") or {}).get("asset", "").split("@")[0]
        if a["class"] == "person" and asset.startswith("human-"):
            out[i] = person_extents(asset, round(float(a["size_mm"][2]) / 1000, 3))
        else:
            out[i] = centred(sizes[i])
    return out


def blend(speed) -> np.ndarray:
    speed = np.asarray(speed, float)
    return np.where(speed > REST_BELOW_MPS, np.minimum(speed / FULL_WALK_MPS, 1.0), 0.0)


def extents_at(ext, speed) -> np.ndarray:
    b = blend(speed)[..., None]
    return ext[..., 0, :] * (1 - b) + ext[..., 1, :] * b


def boxes(xy, yaw, speed, ext, heights=None):
    """Oriented boxes (centres, sizes) of the bodies at the given speeds.

    ``sizes`` has a third column (height, zero when not given) so it can be passed to the collision helpers.
    """
    e = extents_at(ext, speed)
    off = np.stack([(e[..., 0] + e[..., 1]) / 2, (e[..., 2] + e[..., 3]) / 2], -1)
    c, s = np.cos(yaw), np.sin(yaw)
    centres = np.asarray(xy, float) + np.stack(
        [off[..., 0] * c - off[..., 1] * s, off[..., 0] * s + off[..., 1] * c], -1
    )
    h = np.zeros(e.shape[:-1]) if heights is None else np.broadcast_to(heights, e.shape[:-1])
    sizes = np.stack([e[..., 1] - e[..., 0], e[..., 3] - e[..., 2], h], -1)
    return centres, sizes


def turn_radius(ext_rest) -> np.ndarray:
    """Radius of the disk swept by rotating the rest pose about the body origin."""
    e = np.asarray(ext_rest)
    return np.hypot(np.maximum(np.abs(e[..., 0]), np.abs(e[..., 1])), np.maximum(np.abs(e[..., 2]), np.abs(e[..., 3])))


def symmetric(ext) -> np.ndarray:
    """(length, width) of the smallest centred box containing the extents."""
    e = np.asarray(ext)
    return np.stack(
        [2 * np.maximum(np.abs(e[..., 0]), np.abs(e[..., 1])), 2 * np.maximum(np.abs(e[..., 2]), np.abs(e[..., 3]))], -1
    )
