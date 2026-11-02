"""Cached offline full-body mesh rejection using the actual collision shape.

The surface supplies a nominal terrain pose for planning only. No physical
state is written, and a surviving sample is NOT a native/dynamic certificate.
Every original triangle intersecting the loaded patch is retained.
"""

import math
from collections import Counter, OrderedDict

import numpy as np
import shapely

from awr.life.civil.mesh_geometry import rendered_pose, triangle_box_intersections
from awr.life.civil.surface import SurfaceError
from awr.life.civil.vehicle_terrain import TrianglePatch


class IndexedPatch(TrianglePatch):
    def __init__(self, triangles, bounds):
        super().__init__(triangles, bounds)
        self.tree = shapely.STRtree(shapely.box(self.low[:, 0], self.low[:, 1], self.high[:, 0], self.high[:, 1]))

    def candidates(self, centre, rotation, lo, hi, padding):
        low, high = lo - padding, hi + padding
        corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
        vertices = corners @ rotation.T + centre
        lower = vertices.min(0)
        upper = vertices.max(0)
        if np.any(lower[:2] < self.bounds[:2]) or np.any(upper[:2] > self.bounds[2:]):
            return None
        ids = self.tree.query(shapely.box(*lower[:2], *upper[:2]))
        ids = np.sort(ids[np.all((self.low[ids] <= upper) & (self.high[ids] >= lower), axis=1)])
        return (self.triangles[ids] - centre) @ rotation


class HeightProbes:
    """Finite measured height for nominal tilt; no semantic driving permission.

    Road permission and centre support remain separate mandatory search gates.
    A side height probe is not a tyre/contact certificate or a walkability test.
    """

    def __init__(self, surface):
        self.surface = surface

    def sample(self, points, kind):
        return self.surface.sample(points, kind, require_walkable=False)


class MeshBodyClearance:
    candidate_only = True

    def __init__(self, profile, surface, triangles, bounds, *, clearance_m=0.02, cache_size=4096):
        if (
            not math.isfinite(clearance_m)
            or not 0 <= clearance_m <= 0.2
            or type(cache_size) is not int
            or not 1 <= cache_size <= 65536
        ):
            raise ValueError("Bounded physical clearance and exact-pose cache required")
        self.profile = profile
        self.surface = surface
        self.heights = HeightProbes(surface)
        self.patch = IndexedPatch(triangles, bounds)
        self.clearance = clearance_m
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.stats = Counter()
        self.size = profile.body_hi - profile.body_lo

    def at_pose(self, position, rotation):
        """Test a supplied rigid pose, including a recorded native pose."""
        p = np.asarray(position, float)
        R = np.asarray(rotation, float)
        if (
            p.shape != (3,)
            or R.shape != (3, 3)
            or not np.isfinite(p).all()
            or not np.isfinite(R).all()
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(R), 1)
        ):
            raise ValueError("Finite rigid body pose required")
        self.stats["mesh_queries"] += 1
        profile = self.profile
        pad = self.clearance
        triangles = self.patch.candidates(p, R, profile.body_lo, profile.body_hi, pad)
        if triangles is None:
            return False, "mesh_patch_boundary"
        self.stats["candidate_triangles"] += len(triangles)
        overlaps = triangle_box_intersections(triangles, profile.body_lo - pad, profile.body_hi + pad)
        if not overlaps.any():
            return True, None
        hull = profile.body_hull
        if hull is not None and hull.separated_triangles(triangles[overlaps], pad).all():
            return True, None
        self.stats["body_unclear"] += 1
        return False, "mesh_body_clearance"

    def pose(self, body_pose):
        p = np.asarray(body_pose, float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("Finite body reference pose required")
        key = p.tobytes()
        self.stats["pose_queries"] += 1
        if key in self.cache:
            self.stats["cache_hits"] += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        try:
            position, R, _, _ = rendered_pose(self.heights, p[:2], p[2], self.size)
            result = self.at_pose(position, R)
        except SurfaceError:
            self.stats["terrain_pose_unsupported"] += 1
            result = (False, "terrain_pose_unsupported")
        self.cache[key] = result
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result

    def route(self, body_points, yaw):
        p = np.asarray(body_points, float)
        yaw = np.unwrap(np.asarray(yaw, float))
        if (
            p.ndim != 2
            or p.shape[1] != 2
            or yaw.shape != (len(p),)
            or len(p) < 2
            or not np.isfinite(p).all()
            or not np.isfinite(yaw).all()
        ):
            raise ValueError("Finite body route required")
        counts = np.maximum(
            1, np.ceil(np.maximum(np.linalg.norm(np.diff(p, axis=0), axis=1) / 0.25, np.abs(np.diff(yaw)) / 0.05))
        ).astype(int)
        if counts.sum() > 100000:
            raise ValueError("Mesh route sample budget exceeded")
        for i, n in enumerate(counts):
            for t in np.arange(n) / n:
                result = self.pose(np.r_[p[i] * (1 - t) + p[i + 1] * t, yaw[i] * (1 - t) + yaw[i + 1] * t])
                if not result[0]:
                    return result
        return self.pose(np.r_[p[-1], yaw[-1]])
