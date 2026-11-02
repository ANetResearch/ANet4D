"""Spatial exposure to coarse quality flags; a diagnostic, not terrain admission."""

import numpy as np
from scipy.spatial import cKDTree


class QualityExposure:
    """Intersect full oriented body rectangles with flagged raster cells.

    Nearest-centre distance is only a conservative broad phase. A four-axis
    separating-axis test includes cells inside a body even if its corners and
    centre all miss them. Touching a flagged cell counts as exposure. Unknown
    space outside the raster is reported separately.
    """

    def __init__(self, data, origin, cell_m, mask=35):
        data = np.asarray(data)
        self.origin = np.asarray(origin, float)
        self.cell = float(cell_m)
        if (
            data.ndim != 2
            or not data.size
            or self.origin.shape != (2,)
            or not np.isfinite(self.origin).all()
            or not np.isfinite(self.cell)
            or self.cell <= 0
            or not np.isfinite(data).all()
            or np.any(data < 0)
            or np.any(data != np.floor(data))
        ):
            raise ValueError("Invalid quality raster geometry or bits")
        self.upper = self.origin + np.array(data.shape[::-1]) * self.cell
        y, x = np.nonzero(data.astype(np.uint64) & mask)
        self.centres = self.origin + (np.column_stack((x, y)) + 0.5) * self.cell
        self.bits = data[y, x].astype(np.uint64) & mask
        self.tree = cKDTree(self.centres) if len(x) else None

    def intersect(self, centres, yaw, sizes):
        centres, yaw, sizes = np.asarray(centres, float), np.asarray(yaw, float), np.asarray(sizes, float)
        if (
            centres.ndim != 2
            or centres.shape[1] != 2
            or yaw.shape != (len(centres),)
            or sizes.shape != centres.shape
            or not np.isfinite(centres).all()
            or not np.isfinite(yaw).all()
            or not np.isfinite(sizes).all()
            or np.any(sizes <= 0)
        ):
            raise ValueError("Invalid body rectangles")
        c, s = np.cos(yaw), np.sin(yaw)
        half = sizes / 2
        aabb = np.column_stack(
            (np.abs(c) * half[:, 0] + np.abs(s) * half[:, 1], np.abs(s) * half[:, 0] + np.abs(c) * half[:, 1])
        )
        outside = np.any((centres - aabb < self.origin) | (centres + aabb > self.upper), axis=1)
        result = np.zeros(len(centres), np.uint64)
        if not len(centres) or self.tree is None:
            return result, outside
        radius = np.linalg.norm(half, axis=1) + self.cell / np.sqrt(2) + 1e-9
        near = np.flatnonzero(self.tree.query(centres)[0] <= radius)
        for i, candidates in zip(near, self.tree.query_ball_point(centres[near], radius[near]), strict=False):
            ids = np.asarray(candidates, int)
            delta = self.centres[ids] - centres[i]
            world_axes = np.all(np.abs(delta) <= aabb[i] + self.cell / 2 + 1e-9, axis=1)
            local = np.column_stack((delta[:, 0] * c[i] + delta[:, 1] * s[i], -delta[:, 0] * s[i] + delta[:, 1] * c[i]))
            body_axes = np.all(np.abs(local) <= half[i] + self.cell / 2 * (abs(c[i]) + abs(s[i])) + 1e-9, axis=1)
            if np.any(world_axes & body_axes):
                result[i] = np.bitwise_or.reduce(self.bits[ids[world_axes & body_axes]])
        return result, outside
