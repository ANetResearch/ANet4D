"""Geometry checks shared by motion planning and independent tape validation."""

import numpy as np
from scipy.spatial import cKDTree


def rectangle(center, yaw, size, margin=0.0):
    l, w = np.asarray(size[:2]) / 2 + margin
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[-l, -w], [l, -w], [l, w], [-l, w]]) @ np.array([[c, s], [-s, c]]) + center


def overlapping(a, b, eps=0.005):
    """Separating-axis test; touching boundaries are permitted."""
    for p in (a, b):
        axes = np.roll(p, -1, axis=0) - p
        for edge in axes[:2]:
            axis = np.array([-edge[1], edge[0]])
            axis /= max(np.linalg.norm(axis), 1e-12)
            A = a @ axis
            B = b @ axis
            if min(A.max(), B.max()) - max(A.min(), B.min()) <= eps:
                return False
    return True


def collision_pairs(xy, yaw, sizes):
    if len(xy) < 2:
        return []
    radii = np.linalg.norm(sizes[:, :2], axis=1) / 2
    pairs = cKDTree(xy).query_pairs(2 * radii.max(), output_type="ndarray")
    if not len(pairs):
        return []
    i, j = pairs.T
    near = np.linalg.norm(xy[i] - xy[j], axis=1) <= radii[i] + radii[j]
    i, j = i[near], j[near]
    hit = obb_overlap(xy[i], yaw[i], sizes[i], xy[j], yaw[j], sizes[j])
    return sorted(zip(i[hit].tolist(), j[hit].tolist(), strict=False))


def obb_overlap(a, ay, az, b, by, bz, eps=0.005):
    """Vectorized exact rectangle SAT, with the same contact tolerance as overlapping."""
    delta = np.asarray(b) - np.asarray(a)
    ay, by = np.asarray(ay), np.asarray(by)
    u = np.stack([np.cos(ay), np.sin(ay)], -1)
    v = np.stack([-u[..., 1], u[..., 0]], -1)
    bu = np.stack([np.cos(by), np.sin(by)], -1)
    bv = np.stack([-bu[..., 1], bu[..., 0]], -1)
    ah = np.asarray(az)[..., :2] / 2
    bh = np.asarray(bz)[..., :2] / 2
    c = np.abs(np.cos(by - ay))
    s = np.abs(np.sin(by - ay))
    return (
        (np.abs((delta * u).sum(-1)) < ah[..., 0] + bh[..., 0] * c + bh[..., 1] * s - eps)
        & (np.abs((delta * v).sum(-1)) < ah[..., 1] + bh[..., 0] * s + bh[..., 1] * c - eps)
        & (np.abs((delta * bu).sum(-1)) < bh[..., 0] + ah[..., 0] * c + ah[..., 1] * s - eps)
        & (np.abs((delta * bv).sum(-1)) < bh[..., 1] + ah[..., 0] * s + ah[..., 1] * c - eps)
    )


def actor_blocked(i, position, angle, xy, yaw, sizes, tree, radii, body_size=None):
    """Only query neighbors of this proposed body; never rerun all-pair SAT per actor."""
    body = sizes[i] if body_size is None else np.asarray(body_size)
    ids = np.asarray(tree.query_ball_point(position, np.linalg.norm(body[:2]) / 2 + radii.max()), dtype=int)
    ids = ids[ids != i]
    return bool(len(ids) and obb_overlap(position, angle, body, xy[ids], yaw[ids], sizes[ids]).any())
