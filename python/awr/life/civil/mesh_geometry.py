"""Conservative source-triangle/body geometry shared by diagnostics and admission.

Touching counts as intersection. Body orientation matches the renderer.
"""

import math

import numpy as np


def triangle_box_intersections(triangles, lo, hi):
    """Closed triangle/AABB overlap using all separating axes (touch counts).

    This also handles point/line degeneracy without a normalisation division.
    Input coordinates have already been transformed into the vehicle frame.
    """
    tri = np.asarray(triangles, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if (
        tri.ndim != 3
        or tri.shape[1:] != (3, 3)
        or lo.shape != (3,)
        or hi.shape != (3,)
        or not np.isfinite(tri).all()
        or not np.isfinite([lo, hi]).all()
        or np.any(lo > hi)
    ):
        raise ValueError("Finite triangles and ordered 3D box required")
    half = (hi - lo) / 2
    v = tri - (lo + hi) / 2
    edge = np.roll(v, -1, axis=1) - v
    result = np.all((v.min(1) <= half + 1e-9) & (v.max(1) >= -half - 1e-9), axis=1)
    axes = [np.cross(edge[:, 0], edge[:, 1])]
    axes.extend(np.cross(edge[:, j], axis) for j in range(3) for axis in np.eye(3))
    for axis in axes:
        p = np.einsum("nvi,ni->nv", v, axis)
        r = np.abs(axis) @ half
        eps = 1e-9 * np.linalg.norm(axis, axis=1)
        result &= (p.min(1) <= r + eps) & (p.max(1) >= -r - eps)
    return result


def rendered_pose(surface, xy, yaw, size):
    """Same ground/probe orientation and ZYX rotation as assembly/rendering."""
    f = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-f[1], f[0]])
    hl, hw = size[0] * 0.32, size[1] * 0.37
    z = surface.sample([xy], "vehicle")[0]
    front, back, zl, zr = surface.sample([xy + f * hl, xy - f * hl, xy + left * hw, xy - left * hw], "vehicle")
    pitch, roll = -math.atan2(front - back, 2 * hl), math.atan2(zl - zr, 2 * hw)
    c, s, cp, sp, cr, sr = (
        math.cos(yaw),
        math.sin(yaw),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(roll),
        math.sin(roll),
    )
    rotation = (
        np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )
    return np.r_[xy, z], rotation, pitch, roll
