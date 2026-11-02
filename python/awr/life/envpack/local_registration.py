"""Local registration of mapped roads to the measured carriageway.

The global registration (awr.life.envpack.registration) removes the systematic offset of the OpenStreetMap layers. The
road centrelines still deviate locally from the reconstruction: on a town's streets the measured carriageway lies
by up to several metres to one side of its centreline, which then runs along a kerb or over a pavement. Lanes, kerb
parking and junction areas built on such a centreline do not match the scene.

This module measures the deviation from the reconstruction alone and builds a smooth displacement field for the
road layers:

- Cross-sections (`cross_sections`): at stations along each road, the measured vehicle ground across the road
  (awr.life.civil.surface.flat_vehicle_ground, without the mapped-road mask of a town support, which would cut the
  carriageway at a buffer around the centreline being measured). The supported run at (or nearest to) the
  centreline gives the carriageway's centre and width there. A run that reaches
  the end of the scan (an open area) or whose width is far from the mapped width (a junction, a square, a parking
  lot beside the road) gives no measurement.
- Smoothing (`smooth`): per road, a measurement more than `outlier_m` from the running median of its neighbours is
  dropped.
- Field (`DisplacementField`): only the component of the displacement normal to each road is measured. At a point,
  the displacement D minimises sum_i w_i (n_i . D - o_i)^2 + prior |D|^2 with Gaussian weights of the measurements
  within 3 sigma: along one road D follows that road's normal offset, at a junction the offsets of the roads meeting
  there are met together, and away from any measurement D falls to zero. |D| is bounded by `max_m`.

The field is applied to the road layers only (the navigation graph, the lane network and the road features of the
region semantics), with every line densified first so that a varying offset is followed between the source
vertices. Topology, identifiers and attributes are unchanged; the source layers are never modified.
"""

from __future__ import annotations

import math

import numpy as np


def densify(points, step_m=2.0):
    """A polyline with points inserted so that no segment is longer than `step_m` (the original vertices kept)."""
    p = [list(map(float, q)) for q in points]
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:], strict=False):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(int(math.ceil(d / step_m - 1e-9)), 1)
        for k in range(1, n):
            u = k / n
            out.append(
                [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u]
                + [a[j] + (b[j] - a[j]) * u for j in range(2, min(len(a), len(b)))]
            )
        out.append(b)
    return out


def _runs(mask):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    return np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)


def cross_sections(
    surface,
    roads,
    *,
    step_m=2.0,
    end_m=6.0,
    margin_m=4.0,
    lateral_step_m=0.1,
    width_bounds=(0.5, 1.6),
    profile="vehicle",
):
    """Measured carriageway centre and width across each road (see the module docstring).

    `roads`: [{'id', 'line', 'width_m'}]. Returns (measurements, counts); a measurement is a dict with road, station
    s_m, xy, normal (left of the road direction), offset_m (positive to the left) and width_m."""
    out = []
    counts = dict(stations=0, no_support=0, unbounded=0, width_out_of_bounds=0, measured=0)
    for road in roads:
        line = np.asarray(road["line"], float)[:, :2]
        seg = np.diff(line, axis=0)
        length = np.linalg.norm(seg, axis=1)
        keep = length > 1e-9
        if not keep.any():
            continue
        line = np.vstack([line[:1], line[1:][keep]])
        seg = np.diff(line, axis=0)
        length = np.linalg.norm(seg, axis=1)
        arc = np.r_[0.0, np.cumsum(length)]
        width = float(road["width_m"])
        lateral = np.arange(-(width / 2 + margin_m), width / 2 + margin_m + 1e-9, lateral_step_m)
        for s in np.arange(end_m, arc[-1] - end_m + 1e-9, step_m):
            counts["stations"] += 1
            k = min(int(np.searchsorted(arc, s, side="right")) - 1, len(seg) - 1)
            t = seg[k] / length[k]
            normal = np.array([-t[1], t[0]])
            p = line[k] + t * (s - arc[k])
            ok = np.asarray(surface.supported(p + lateral[:, None] * normal, profile), bool)
            runs = _runs(ok)
            if not runs:
                counts["no_support"] += 1
                continue
            run = min(
                runs,
                key=lambda r: (
                    0.0 if lateral[r[0]] <= 0.0 <= lateral[r[-1]] else min(abs(lateral[r[0]]), abs(lateral[r[-1]]))
                ),
            )
            if run[0] == 0 or run[-1] == len(lateral) - 1:
                counts["unbounded"] += 1
                continue
            lo, hi = lateral[run[0]], lateral[run[-1]]
            w = hi - lo + lateral_step_m
            if not width_bounds[0] * width <= w <= width_bounds[1] * width:
                counts["width_out_of_bounds"] += 1
                continue
            out.append(
                dict(
                    road=road["id"],
                    s_m=float(s),
                    xy=p.tolist(),
                    normal=normal.tolist(),
                    offset_m=float((lo + hi) / 2),
                    width_m=float(w),
                )
            )
            counts["measured"] += 1
    return out, counts


def smooth(measurements, *, window=11, outlier_m=0.75):
    """Drop the measurements more than `outlier_m` from the running median of the same road (`window` stations)."""
    by_road = {}
    for m in measurements:
        by_road.setdefault(m["road"], []).append(m)
    kept, dropped = [], 0
    for rows in by_road.values():
        rows = sorted(rows, key=lambda m: m["s_m"])
        o = np.array([m["offset_m"] for m in rows])
        h = window // 2
        for i, m in enumerate(rows):
            med = float(np.median(o[max(0, i - h) : i + h + 1]))
            if abs(m["offset_m"] - med) <= outlier_m:
                kept.append(m)
            else:
                dropped += 1
    return kept, dropped


class DisplacementField:
    """Smooth horizontal displacement from normal offset measurements (see the module docstring)."""

    def __init__(self, measurements, *, sigma_m=6.0, prior=0.5, max_m=5.0):
        if not measurements:
            raise ValueError("A displacement field needs measurements")
        from scipy.spatial import cKDTree

        self.sigma, self.prior, self.max_m = float(sigma_m), float(prior), float(max_m)
        self.xy = np.array([m["xy"] for m in measurements], float)
        self.normal = np.array([m["normal"] for m in measurements], float)
        self.offset = np.array([m["offset_m"] for m in measurements], float)
        self.tree = cKDTree(self.xy)

    def displace(self, points):
        """Displacements (k, 2) at points (k, 2)."""
        P = np.asarray(points, float).reshape(-1, 2)
        D = np.zeros_like(P)
        for i, near in enumerate(self.tree.query_ball_point(P, 3 * self.sigma)):
            if not near:
                continue
            near = np.asarray(near)
            w = np.exp(-np.sum((self.xy[near] - P[i]) ** 2, axis=1) / (2 * self.sigma**2))
            n = self.normal[near]
            A = (n.T * w) @ n + self.prior * np.eye(2)
            b = (n.T * w) @ self.offset[near]
            d = np.linalg.solve(A, b)
            norm = float(np.hypot(*d))
            D[i] = d if norm <= self.max_m else d * (self.max_m / norm)
        return D

    def apply(self, x, y):
        d = self.displace([[x, y]])[0]
        return float(x + d[0]), float(y + d[1])

    def record(self):
        return dict(
            model="p' = p + D(p); D minimises sum w_i (n_i . D - o_i)^2 + prior |D|^2, "
            "w_i = exp(-|p - x_i|^2 / (2 sigma^2)) over measurements within 3 sigma",
            sigma_m=self.sigma,
            prior=self.prior,
            max_m=self.max_m,
            measurements=int(len(self.xy)),
        )
