"""Continuous non-tyre mesh checks with one immutable indexed world patch.

The certificate is the existing per-interval midpoint enclosure: linear
translation and quaternion slerp between every consecutive native frame.
Nothing is subsampled. This owns no simulation state or nominal terrain pose.
"""

from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation

from .body_clearance import IndexedPatch
from .triangle_sat import triangle_box_intersections


class MeshMotion:
    def __init__(self, profile, triangles, bounds):
        self.profile = profile
        self.patch = IndexedPatch(triangles, bounds)
        self.radius = float(np.linalg.norm(np.maximum(abs(profile.body_lo), abs(profile.body_hi))))
        self.stats = Counter()
        self.last_key = None
        self.last_failed = False

    def check(self, times, position, rotation):
        t = np.asarray(times, float)
        p = np.asarray(position, float)
        R = np.asarray(rotation, float)
        if (
            t.ndim != 1
            or len(t) < 2
            or p.shape != (len(t), 3)
            or R.shape != (len(t), 3, 3)
            or not np.isfinite(t).all()
            or not np.isfinite(p).all()
            or not np.isfinite(R).all()
            or (np.diff(t) <= 0).any()
            or not np.allclose(R.transpose(0, 2, 1) @ R, np.eye(3), atol=1e-8, rtol=0)
            or not np.allclose(np.linalg.det(R), 1.0, atol=1e-8, rtol=0)
        ):
            raise ValueError("Consecutive finite rigid poses required")
        vector = Rotation.from_matrix(R[:-1].transpose(0, 2, 1) @ R[1:]).as_rotvec()
        angle = np.linalg.norm(vector, axis=1)
        middle = R[:-1] @ Rotation.from_rotvec(vector * 0.5).as_matrix()
        pad = np.linalg.norm(np.diff(p, axis=0), axis=1) * 0.5 + 2 * self.radius * np.sin(angle / 4) + 0.001
        lo, hi = self.profile.body_lo, self.profile.body_hi
        hull = self.profile.body_hull
        bad = []
        self.stats["intervals"] += len(t) - 1
        for i, (centre, rot, margin) in enumerate(zip((p[:-1] + p[1:]) * 0.5, middle, pad, strict=False)):
            key = centre.tobytes() + rot.tobytes() + np.float64(margin).tobytes()
            if key == self.last_key:
                self.stats["exact_cache_hits"] += 1
            else:
                self.stats["mesh_queries"] += 1
                candidates = self.patch.candidates(centre, rot, lo, hi, margin)
                failed = candidates is None
                if candidates is not None:
                    self.stats["candidate_triangles"] += len(candidates)
                    overlap = triangle_box_intersections(candidates, lo - margin, hi + margin)
                    failed = bool(overlap.any())
                    if failed and hull is not None:
                        failed = not hull.separated_triangles(candidates[overlap], margin).all()
                self.last_key = key
                self.last_failed = failed
            if self.last_failed:
                bad.append(float(t[i]))
        return bad
