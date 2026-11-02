"""Allocate independent curb trips without a front-car-completed dependency.

The allocator reserves physical berths and checks *all* routes against every
other car's possible parked poses. Failure is explicit, never a static fallback.
Temporal priority at a merge remains the collision/braking controller's job.
"""

import numpy as np
from shapely.geometry import Polygon

from awr.life.civil.collision import rectangle
from awr.life.civil.curb import manoeuvre


def departure_times(travel_s, duration, warmup, rng):
    """Seeded independent departures spanning the recording, not a convoy.

    Only a short first trip prefix is assigned to warm-up. Drawing uniformly
    from a ten-minute warm-up otherwise spends many finite trips before t=0.
    Schedules never wait for another vehicle to finish its entire journey.
    """
    travel = np.asarray(travel_s, float)
    if not len(travel):
        return np.empty(0)
    horizon = max(0.0, duration - float(travel.max()) - 10.0)
    base = np.linspace(0.0, horizon, len(travel)) if len(travel) > 1 else np.zeros(1)
    jitter = min(10.0, horizon / max(len(travel), 1) * 0.2)
    times = np.clip(base + rng.uniform(-jitter, jitter, len(travel)), 0.0, horizon)
    times[0] = -min(float(warmup), float(travel[0]) * 0.5)
    return times


def footprint(slot, size, *, margin_m=0.4):
    if not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError("Invalid footprint clearance")
    return Polygon(rectangle(np.asarray(slot["xy"]), slot["yaw"], size, margin=margin_m))


def swept(route, size, *, margin_m=0.4):
    """Conservative continuous envelope, with .4 m clearance by default.

    A zero margin exposes the physical body envelope for road containment.
    Collision and occupied-parking checks retain the default clearance.
    Neither changes the independent ground support or steering checks.

    Runtime interpolates position and unwrapped yaw linearly between samples.
    For a corner at radius r, the error between its rotated position and the
    chord joining its endpoint rotations is <= r * delta_yaw**2 / 8 (linear
    interpolation error bounded by its second derivative). The convex hull of
    the two endpoint rectangles, expanded by that bound, therefore contains
    every intermediate body. Translation is exactly linear and needs no
    fixed isotropic padding. This matters on real narrow streets: the old
    additional .375 m buffer incorrectly consumed carriageway width even for
    perfectly straight motion. Terrain/steering admission remains separate.
    """
    import shapely

    points = np.asarray(route["points"], float)
    yaw = np.asarray(route["yaw"], float)
    size = np.asarray(size, float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 2
        or yaw.shape != (len(points),)
        or size.shape != (3,)
        or not np.isfinite(np.r_[points.ravel(), yaw, size]).all()
        or np.any(size <= 0)
        or not np.isfinite(margin_m)
        or margin_m < 0
    ):
        raise ValueError("Invalid swept vehicle geometry")
    # Do not unwrap here: the exact supplied yaw interpolation is the contract.
    half = size[:2] / 2 + margin_m
    corners = np.array([[-half[0], -half[1]], [half[0], -half[1]], [half[0], half[1]], [-half[0], half[1]]])
    c, s = np.cos(yaw), np.sin(yaw)
    boxes = np.empty((len(points), 4, 2))
    boxes[:, :, 0] = points[:, 0, None] + c[:, None] * corners[:, 0] - s[:, None] * corners[:, 1]
    boxes[:, :, 1] = points[:, 1, None] + s[:, None] * corners[:, 0] + c[:, None] * corners[:, 1]
    hulls = shapely.convex_hull(shapely.multipoints(np.concatenate([boxes[:-1], boxes[1:]], axis=1)))
    error = np.linalg.norm(half) * np.diff(yaw) ** 2 / 8
    curved = error > 0
    if curved.any():
        # GEOS approximates a circular buffer by INSCRIBED polygons. Inflate
        # the radius so even their sides enclose the required error disk.
        quad_segs = 8
        radius = (error[curved] + 1e-10) / np.cos(np.pi / (4 * quad_segs))
        hulls[curved] = shapely.buffer(hulls[curved], radius, quad_segs=quad_segs)
    return shapely.union_all(hulls)


def opposing_conflict(route, size, other, other_size):
    """Reject unregulated head-on/crossing traffic at the actual local overlap.

    Looking only at start-to-end vectors misses an opposing leg on a bent road.
    Properly separated opposite lanes do not overlap and remain admissible.
    """
    from scipy.spatial import cKDTree

    distance, index = cKDTree(other["points"]).query(route["points"])
    near = distance < (float(size[1]) + float(other_size[1])) / 2 + 0.5
    return bool(np.any(near & (np.cos(route["yaw"] - other["yaw"][index]) <= 0.3)))


class Allocator:
    def __init__(self, surface, keeps_passage, obstacles=(), roads=None):
        """`roads`: optional awr.life.civil.activity.connected_parking.ConnectedRoads; a parked car must then leave the
        driving map's roads passable."""
        self.surface = surface
        self.keeps_passage = keeps_passage
        self.roads = roads
        self.parks = [Polygon(rectangle(np.asarray(p), a, s, margin=0.4)) for p, a, s in obstacles]
        self.paths = []
        self.used = []
        self.used_routes = []
        self.failed = set()

    def free(self, slots, size, path=None):
        boxes = [footprint(slot, size) for slot in slots]
        for box in boxes:
            if any(box.intersects(other) for other in self.parks + self.paths):
                return False
        if any(a.intersects(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :]):
            return False
        if path is not None and any(path.intersects(other) for other in self.parks):
            return False
        if not all(self.keeps_passage(np.asarray(s["xy"]), s["yaw"], size) for s in slots):
            return False
        return self.roads is None or all(self.roads.allows(box) for box in boxes)

    def reserve(self, slots, size, route=None):
        self.parks.extend(footprint(s, size) for s in slots)
        if self.roads is not None:
            for s in slots:
                self.roads.reserve(footprint(s, size))
        if route is not None:
            self.paths.append(swept(route, size))
            self.used_routes.append(dict(route=route, size=size))

    def trip(self, roads, size, *, min_trip=60.0, merge_m=18.0):
        revision = (len(self.parks), len(self.paths))
        failure_key = ("trip", id(roads), tuple(size), min_trip, merge_m, revision)
        if failure_key in self.failed:
            return None
        # Spread across roads, then prefer longer trips. Other destinations on
        # the same fragment are tried if a farthest endpoint is already taken.
        for road in sorted(roads, key=lambda r: (self.used.count(r["road"]), -r["geometry"][0].length)):
            for candidate in sorted(road["trips"], key=lambda t: -t["route"]["arc"][-1]):
                start = candidate["start"]
                for end in reversed(road["berths"]):
                    if end["s"] - start["s"] < min_trip:
                        break
                    if not self.free([start, end], size):
                        continue
                    route = (
                        candidate["route"]
                        if end == candidate["end"]
                        else manoeuvre(
                            road["geometry"],
                            start["s"],
                            end["s"],
                            road["layout"],
                            size,
                            self.surface,
                            road["road"],
                            merge_m=merge_m,
                        )
                    )
                    if route is None:
                        continue
                    path = swept(route, size)
                    if not self.free([start, end], size, path):
                        continue
                    # Intersecting travel paths are legal only when they point
                    # in the same direction; head-on conflicts in a narrow
                    # corridor require a reservation controller, not a timeout.
                    for other in self.used_routes:
                        if opposing_conflict(route, size, other["route"], other["size"]):
                            break
                    else:
                        self.parks.extend([footprint(start, size), footprint(end, size)])
                        self.paths.append(path)
                        self.used.append(road["road"])
                        self.used_routes.append(dict(route=route, size=size))
                        return dict(route=route, start=start, end=end, layout=road["layout"])
        self.failed.add(failure_key)
        return None

    def stationary(self, roads, size, *, region=None):
        failure_key = (
            "parked",
            id(roads),
            tuple(size),
            len(self.parks),
            len(self.paths),
            None if region is None else region.wkb,
        )
        if failure_key in self.failed:
            return None
        for road in sorted(roads, key=lambda r: self.used.count(r["road"])):
            for slot in road["berths"]:
                if region is not None and not region.covers(footprint(slot, size)):
                    continue
                if not self.free([slot], size):
                    continue
                self.reserve([slot], size)
                self.used.append(road["road"])
                return dict(slot=slot, road=road["road"], layout=road["layout"])
        self.failed.add(failure_key)
        return None
