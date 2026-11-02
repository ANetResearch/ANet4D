"""Optional direction-separated walking routes within measured walking space.

Shapely/GEOS constructs a right offset, with connectors to the unchanged
activity endpoints. Full-body support and legal-space checks admit it once
per new route. Narrow/invalid corridors keep the original route, or use explicitly enabled
local separation on the independently admitted wider portions. No new A*,
per-person schedule, position change, or map widening is involved.
"""

import math

import numpy as np
from shapely.geometry import LineString
from shapely.ops import substring

from . import kernels as K
from .crowd import LEGAL_INSET_M


class KeepRightRoutes:
    def __init__(self, crowd, *, offset_m=0.8, merge_m=3.0, segment_clear=None, local_fallback=False):
        if not all(math.isfinite(v) and v > 0 for v in (offset_m, merge_m)):
            raise ValueError("Walking separation and merge distances must be positive")
        self.crowd, self.offset_m, self.merge_m = crowd, float(offset_m), float(merge_m)
        self.segment_clear = segment_clear
        self.local_fallback = bool(local_fallback)
        self.area = None if crowd.area is None else crowd.area.buffer(-LEGAL_INSET_M)
        self.stats = dict(requested=0, adjusted=0, partially_adjusted=0, unchanged=0)

    def __call__(self, i, points):
        p = np.asarray(points, float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 1 or not np.isfinite(p).all():
            raise ValueError("Walking route must contain finite XY points")
        self.stats["requested"] += 1
        if len(p) >= 2:
            line = LineString(p)
            if line.is_simple and line.length > 2 * self.merge_m:
                for offset in (self.offset_m, self.offset_m / 2):
                    shifted = line.offset_curve(-offset, join_style="round")
                    if shifted.geom_type != "LineString" or shifted.length <= 2 * self.merge_m:
                        continue
                    middle = substring(shifted, self.merge_m, shifted.length - self.merge_m)
                    q = np.vstack([p[0], np.asarray(middle.coords), p[-1]])
                    candidate = LineString(q)
                    if not candidate.is_simple or (self.area is not None and not self.area.covers(candidate)):
                        continue
                    if self.segment_clear is not None and not all(
                        self.segment_clear(a, b) for a, b in zip(q, q[1:], strict=False)
                    ):
                        continue
                    c = self.crowd
                    if K.route_supported(
                        q, c.ext[i], c.z_ok, c.walk, c.sx0, c.sy0, c.scell, c.free, c.fx0, c.fy0, c.fstep
                    ):
                        self.stats["adjusted"] += 1
                        return q
                if self.local_fallback:
                    local = self._local(i, p)
                    if local is not None:
                        self.stats["adjusted"] += 1
                        self.stats["partially_adjusted"] += 1
                        return local
        self.stats["unchanged"] += 1
        return p.copy()

    def _admitted(self, i, p):
        line = LineString(p)
        if not line.is_simple or (self.area is not None and not self.area.covers(line)):
            return False
        if self.segment_clear is not None and not all(self.segment_clear(a, b) for a, b in zip(p, p[1:], strict=False)):
            return False
        c = self.crowd
        return K.route_supported(p, c.ext[i], c.z_ok, c.walk, c.sx0, c.sy0, c.scell, c.free, c.fx0, c.fy0, c.fstep)

    def _local(self, i, p):
        """One bounded, forward pass over the unchanged route's short segments.

        Retain every original corner, and certify both the proposed segment and
        a return to the next original station. A local obstruction can therefore
        remove a lane offset without disabling separation throughout a long trip.
        No search, geometry expansion or new destination is introduced. The full
        proposed polyline is checked again; failure preserves the original route.
        """
        stations = [p[0]]
        for a, b in zip(p, p[1:], strict=False):
            distance = float(np.linalg.norm(b - a))
            if distance < 1e-8:
                continue
            count = max(1, int(math.ceil(distance / self.merge_m)))
            stations.extend(a + (b - a) * u / count for u in range(1, count + 1))
        base = np.asarray(stations)
        if len(base) < 3:
            return None
        delta = np.diff(base, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        tangents = delta / lengths[:, None]
        arc = np.r_[0.0, np.cumsum(lengths)]
        q = [base[0]]
        changed = False
        for k in range(1, len(base) - 1):
            tangent = tangents[k - 1] + tangents[k]
            norm = float(np.linalg.norm(tangent))
            right = np.zeros(2) if norm < 1e-8 else np.array([tangent[1], -tangent[0]]) / norm
            taper = min(1.0, arc[k] / self.merge_m, (arc[-1] - arc[k]) / self.merge_m)
            found = False
            for offset in (self.offset_m, self.offset_m / 2, 0.0):
                candidate = base[k] + offset * taper * right
                probe = np.array([q[-1], candidate, base[k + 1]])
                if self._admitted(i, probe):
                    q.append(candidate)
                    changed |= bool(np.linalg.norm(candidate - base[k]) > 1e-8)
                    found = True
                    break
            if not found:
                return None
        q = np.asarray(q + [base[-1]])
        return q if changed and self._admitted(i, q) else None
