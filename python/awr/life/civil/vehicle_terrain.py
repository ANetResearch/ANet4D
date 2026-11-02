"""Candidate tyre-support / source-body clearance contract for measured terrain.

The legacy planar body-corner mask remains the production default. This
explicit wrapper proves a different, physical condition: existing tyre and
terrain-pose probes stay on supported ground, while the complete non-tyre
body box clears original world triangles. Bounds cover continuous XY/yaw
interpolation and its terrain-derived Z/pitch/roll, not just discrete poses.
It does not admit traffic, source semantics, parked bodies or a published tape.
"""

import math
from collections import Counter

import numpy as np

from .mesh_geometry import rendered_pose, triangle_box_intersections


class TrianglePatch:
    """Complete source triangles intersecting an explicitly bounded XY patch."""

    def __init__(self, triangles, bounds):
        self.triangles = np.asarray(triangles, float)
        self.bounds = np.asarray(bounds, float)
        if (
            self.triangles.ndim != 3
            or self.triangles.shape[1:] != (3, 3)
            or self.bounds.shape != (4,)
            or not np.isfinite(self.triangles).all()
            or not np.isfinite(self.bounds).all()
            or np.any(self.bounds[:2] >= self.bounds[2:])
        ):
            raise ValueError("Finite original triangles and ordered patch bounds required")
        self.low = self.triangles.min(1)
        self.high = self.triangles.max(1)

    def candidates(self, centre, rotation, lo, hi, padding):
        # Filter the same expanded local box tested below. Rotating a local
        # padding cube can extend farther than padding on a world axis.
        lo, hi = lo - padding, hi + padding
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        vertices = corners @ rotation.T + centre
        lower = vertices.min(0)
        upper = vertices.max(0)
        if np.any(lower[:2] < self.bounds[:2]) or np.any(upper[:2] > self.bounds[2:]):
            return None  # Unloaded geometry is not evidence of free space.
        select = np.all((self.low <= upper) & (self.high >= lower), axis=1)
        return (self.triangles[select] - centre) @ rotation


class VehicleTerrain:
    """Explicit offline candidate; caller supplies verified source-body bounds.

    `contacts` retains the existing vehicle_contacts probes plus the centre.
    `body_lo/hi` must enclose every non-contact source vertex, tied to the exact
    rendered mesh. A tyre animation mask alone is not that certificate.
    Adaptive subdivision only tightens conservative bounds; exhaustion rejects.
    """

    release_ready = False
    contract = "tyre_support_and_source_body_clearance.candidate.v1"

    def __init__(self, surface, patch, size, contacts, body_lo, body_hi, *, clearance_m=0.01, max_depth=12):
        self.surface = surface
        self.patch = patch
        self.size = np.asarray(size, float)
        self.contacts = np.asarray(contacts, float)
        self.lo = np.asarray(body_lo, float)
        self.hi = np.asarray(body_hi, float)
        if (
            self.size.shape != (3,)
            or np.any(self.size <= 0)
            or self.contacts.ndim != 2
            or self.contacts.shape[1] != 2
            or not len(self.contacts)
            or self.lo.shape != (3,)
            or self.hi.shape != (3,)
            or np.any(self.lo >= self.hi)
            or not np.isfinite(self.size).all()
            or not np.isfinite(self.contacts).all()
            or not np.isfinite([self.lo, self.hi]).all()
            or not math.isfinite(clearance_m)
            or clearance_m < 0
            or not isinstance(max_depth, int)
            or max_depth < 1
        ):
            raise ValueError("Finite authored body/contact geometry required")
        # The terrain-derived pose depends on these probes even when the
        # caller supplies only authored tyre centres.
        hl, hw = self.size[0] * 0.32, self.size[1] * 0.37
        self.contacts = np.unique(np.vstack([self.contacts, [[0, 0], [hl, 0], [-hl, 0], [0, hw], [0, -hw]]]), axis=0)
        self.clearance = clearance_m
        self.max_depth = max_depth
        self.radius = float(np.linalg.norm(np.maximum(abs(self.lo), abs(self.hi))))
        self.probe_radius = np.linalg.norm(self.contacts, axis=1)
        self.stats = Counter()

    def __getattr__(self, name):
        return getattr(self.surface, name)

    def _probe_bound(self, point, radius):
        """Bilinear height Lipschitz bound and continuous support in an XY box.

        Enumerate every nearest-cell support value and bilinear control vertex
        touched by the envelope, including both sides of a grid boundary.
        No interpolated mask or filled missing height is accepted.
        """
        surface = self.surface
        c = surface.cell_m
        origin = np.array([surface.x0, surface.y0])
        low = (point - radius - origin) / c
        high = (point + radius - origin) / c
        first = np.floor(low).astype(int)
        last = np.floor(high).astype(int)
        if np.any(first < 0) or np.any(last >= [surface.nx, surface.ny]):
            return None
        if not surface.walk["vehicle"][first[1] : last[1] + 1, first[0] : last[0] + 1].all():
            return None
        first = np.floor(low - 0.5).astype(int)
        last = np.floor(high - 0.5).astype(int)
        if np.any(first < 0) or np.any(last + 1 >= [surface.nx, surface.ny]):
            return None
        z = surface.z[first[1] : last[1] + 2, first[0] : last[0] + 2]
        if not np.isfinite(z).all():
            return None
        # Each bilinear derivative is a convex combination of edge slopes.
        gx = float(np.max(np.abs(np.diff(z, axis=1)))) / c
        gy = float(np.max(np.abs(np.diff(z, axis=0)))) / c
        return math.hypot(gx, gy)

    def _interval(self, a, b, ya, yb, depth):
        self.stats["intervals"] += 1
        mid = (a + b) / 2
        ym = (ya + yb) / 2
        h = float(np.linalg.norm(b - a)) / 2
        angle = abs(yb - ya) / 2
        c, s = math.cos(ym), math.sin(ym)
        probes = mid + self.contacts @ np.array([[c, -s], [s, c]]).T
        gradients = [
            self._probe_bound(p, h + radius * angle + 1e-10)
            for p, radius in zip(probes, self.probe_radius, strict=False)
        ]
        if all(g is not None for g in gradients):
            G = max(gradients)
            pos, R, _, _ = rendered_pose(self.surface, mid, ym, self.size)
            hl, hw = self.size[0] * 0.32, self.size[1] * 0.37
            # atan is 1-Lipschitz. The two terrain differences used by each
            # pitch/roll estimate change by <=2G(h+lever*angle). Rotation
            # matrix displacement is bounded by the sum of Euler changes.
            pitch = G * (h / hl + angle)
            roll = G * (h / hw + angle)
            shift = math.hypot(h, G * h) + self.radius * min(2.0, angle + pitch + roll)
            padding = shift + self.clearance + 1e-9
            triangles = self.patch.candidates(pos, R, self.lo, self.hi, padding)
            if triangles is not None:
                if not triangle_box_intersections(triangles, self.lo - padding, self.hi + padding).any():
                    self.stats["certified_intervals"] += 1
                    return True
                # A collision at the actual midpoint cannot be fixed by making
                # the sampling denser. Touching is still a collision.
                if triangle_box_intersections(triangles, self.lo - self.clearance, self.hi + self.clearance).any():
                    self.stats["body_collision"] += 1
                    return False
        if depth >= self.max_depth or (h < 1e-10 and angle < 1e-10):
            self.stats["uncertified_interval"] += 1
            return False
        return self._interval(a, mid, ya, ym, depth + 1) and self._interval(mid, b, ym, yb, depth + 1)

    def _route_inputs(self, points, yaw, size):
        points = np.asarray(points, float)
        yaw = np.unwrap(np.asarray(yaw, float))
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or yaw.shape != (len(points),)
            or not len(points)
            or not np.isfinite(points).all()
            or not np.isfinite(yaw).all()
            or not np.allclose(size, self.size, rtol=0, atol=1e-8)
        ):
            raise ValueError("Candidate terrain must match the actual admitted vehicle")
        return points, yaw

    def vehicle_route_supported(self, points, yaw, size):
        """Same all-route predicate, with cheap rejection before mesh work.

        Sampling here can only reject. Success still requires every continuous
        interval certificate; no failed intermediate interval is skipped.
        """
        points, yaw = self._route_inputs(points, yaw, size)
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        dx, dy = self.contacts.T
        probes = points[:, None] + np.stack([c * dx - s * dy, s * dx + c * dy], axis=-1)
        if not self.surface.supported(probes.reshape(-1, 2), "vehicle").all():
            self.stats["early_unsupported_route"] += 1
            return False
        for p, a in zip(points, yaw, strict=False):
            if not self._interval(p, p, a, a, 0):
                return False
        return all(
            self._interval(a, b, ya, yb, 0)
            for a, b, ya, yb in zip(points[:-1], points[1:], yaw[:-1], yaw[1:], strict=False)
        )

    def vehicle_support_mask(self, points, yaw, size):
        points, yaw = self._route_inputs(points, yaw, size)
        point_valid = np.array([self._interval(p, p, a, a, 0) for p, a in zip(points, yaw, strict=False)])
        valid = point_valid.copy()
        for i in range(len(points) - 1):
            if not point_valid[i] or not point_valid[i + 1]:
                valid[i : i + 2] = False
                continue
            if not self._interval(points[i], points[i + 1], yaw[i], yaw[i + 1], 0):
                valid[i : i + 2] = False
        return valid
