"""Walking space around the cars that stand parked now.

A parked car is static geometry for walkers while it stands there, and none once it has left. The life engine reports
the resident cars that stand parked (outside an active trip) whenever that set changes. This module then rebuilds the
static walking grid from the scene's own static grid (street furniture and the cars that never move, unchanged) and
the oriented body of each standing car (awr.life.civil.walking.WalkNetwork.obstacle_mask with the builder's walking
clearance, the same test as for every other parked car), and replaces the shared static router. Of the routes
measured on the previous grid (the activity runtime's route cache) those that cross a cell the change closed are
dropped, and when cells opened the refusals are dropped as well; every other cached route stays feasible on the new
grid (it crosses no closed cell), though where cells opened a shorter route may now exist. A walker whose remaining
path intersects a newly placed body (its exact body test, `blocks_path`) plans again. Nothing is moved and no check
is relaxed.
"""

from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np

from ..civil.walking import ObstacleGrid, merge_obstacles
from .route import LifeRouter

BODY_FIELDS = ("positions", "sizes", "yaws", "circle_radii")
ROUND_FACTOR = 1e4  # cost factor of the cells a prospective parked body closes (ParkedSpace._round)


def _copy(grid):
    out = np.array(grid, copy=True).view(ObstacleGrid)
    for name in BODY_FIELDS:
        setattr(out, name, np.array(getattr(grid, name), copy=True))
    out.radius = grid.radius
    out.static_free = np.array(grid.static_free, copy=True)
    return out


class ParkedSpace:
    def __init__(self, activity, walking, radius):
        """`activity`: the activity runtime whose `static_free` grid (an ObstacleGrid) and static router the engine
        and the Director share; `walking`: its WalkNetwork; `radius`: the walking clearance of parked bodies."""
        if not isinstance(activity.static_free, ObstacleGrid):
            raise ValueError("Parked walking space requires the exact-obstacle static grid")
        self.activity, self.walking, self.radius = activity, walking, float(radius)
        self.base = _copy(activity.static_free)
        self.landmark_base = None  # router on `base`, whose landmark distances every router on a subset reuses
        self.masks = {}  # car index -> (body, ObstacleGrid)
        self.parked = None
        self.changed_bodies = ()
        self.stats = dict(updates=0, wall_s=0.0, max_wall_s=0.0)
        # Routes proven with one car moved to a candidate berth (plan_after_parking): at most two routers and 256
        # results, dropped when the parked cars change.
        self.future_routers = OrderedDict()
        self.future_routes = OrderedDict()
        self.future_stats = dict(
            queries=0, cache_hits=0, reused_routes=0, round_routes=0, router_builds=0, rejected=0, wall_s=0.0
        )
        self._static = None  # static cells of every future grid (_exact)

    def plan_after_parking(self, car, body, start, goal):
        """The walking route from start to goal with car `car` parked at `body` (x, y, yaw, length, width, height)
        instead of where it stands, every other parked car where it stands (adapted from codex 454fd00). Read-only:
        the present route is kept when every segment clears that geometry, else a route is planned on it; None when
        there is none."""
        if self.parked is None or car not in self.parked:
            raise ValueError("Prospective parking requires an observed parked car")
        body = tuple(float(x) for x in body)
        if (
            len(body) != 6
            or not np.isfinite(body).all()
            or min(body[3:]) <= 0
            or body[3:] != tuple(self.parked[car][3:])
        ):
            raise ValueError("Prospective parking requires the car body at a finite pose")
        began = time.perf_counter()
        self.future_stats["queries"] += 1
        geometry = (car, body)
        key = (geometry, tuple(start), tuple(goal))
        try:
            if key in self.future_routes:
                self.future_stats["cache_hits"] += 1
                self.future_routes.move_to_end(key)
                return self.future_routes[key]
            current = self.activity.measured_route(start, goal)
            # The present route is checked against the exact geometry of the future grid (its static cells and
            # bodies, all clear_segment reads) without building that grid: copying the town grid and merging every
            # parked car's mask for each candidate berth made this the largest decision cost (cruise r11f v9).
            light = self._exact(car, body)
            if self._clears(current, start, goal, light):
                route = current
                self.future_stats["reused_routes"] += 1
            elif self._clears(route := self._round(body, start, goal, light), start, goal, light):
                self.future_stats["round_routes"] += 1
            else:
                router = self.future_routers.get(geometry)
                if router is None:
                    future = _copy(self.base)
                    for i in sorted(self.masks):
                        if i != car:
                            merge_obstacles(future, self.masks[i][1])
                    merge_obstacles(
                        future,
                        self.walking.obstacle_mask(
                            np.array([body[:2]], float),
                            np.array([body[3:6]], float),
                            self.radius,
                            np.array([body[2]], float),
                        ),
                    )
                    router = self.future_routers[geometry] = LifeRouter(
                        self.activity.network, future, landmark_source=self._landmarks()
                    )
                    self.future_stats["router_builds"] += 1
                    if len(self.future_routers) > 2:
                        self.future_routers.popitem(last=False)
                self.future_routers.move_to_end(geometry)
                route = router.plan(np.asarray(start, float), np.asarray(goal, float))
            self.future_routes[key] = route
            if len(self.future_routes) > 256:
                self.future_routes.popitem(last=False)
            self.future_stats["rejected"] += int(route is None)
            return route
        finally:
            self.future_stats["wall_s"] += time.perf_counter() - began

    def _clears(self, route, start, goal, light):
        """Whether `route` (from start to goal) clears the exact geometry `light` (_exact) at every segment."""
        return (
            route is not None
            and self.walking.clear_segment(start, start, light)
            and self.walking.clear_segment(goal, goal, light)
            and all(
                self.walking.clear_segment(p, q, light)
                for p, q in zip(route["points"][:-1], route["points"][1:], strict=False)
            )
        )

    def _round(self, body, start, goal, light):
        """A route on the present static router that keeps off the cells the body at `body` closes (their cost raised
        ROUND_FACTOR times), its segments drawn on the exact future geometry `light`; or None. The present grid has
        every parked car in place, the car's own too, so such a route is a route of the future grid once its segments
        clear that grid's exact geometry (plan_after_parking checks); it may miss a way through the space the car
        leaves, which the future grid then finds. Building that grid and its router for each candidate berth made
        plan_after_parking the largest decision cost (cruise5 r11f v6)."""
        router = self.activity.static_router
        mask = self.walking.obstacle_mask(
            np.array([body[:2]], float), np.array([body[3:6]], float), self.radius, np.array([body[2]], float)
        )
        cells = router.index[np.asarray(router.free, bool) & ~np.asarray(mask, bool)]
        return router.plan(
            np.asarray(start, float), np.asarray(goal, float), avoid=(cells[cells >= 0], ROUND_FACTOR), geometry=light
        )

    def _exact(self, car, body):
        """The exact geometry plan_after_parking's future grid gives clear_segment, built without the grid: its static
        cells (the scene's static cells and the walking network's, as merge_obstacles leaves them) and the exact
        bodies of the scene, of every parked car but `car`, and of `car` at `body`. The Boolean node mask is not
        built; clear_segment does not read it."""
        if self._static is None:
            self._static = np.asarray(self.base.static_free, bool) & np.asarray(self.walking.free, bool)
        grid = self._static.view(ObstacleGrid)
        parts = [self.base] + [self.masks[i][1] for i in sorted(self.masks) if i != car]
        grid.positions = np.concatenate([p.positions for p in parts] + [np.array([body[:2]], float)])
        grid.sizes = np.concatenate([p.sizes for p in parts] + [np.array([body[3:6]], float)])
        grid.yaws = np.concatenate([p.yaws for p in parts] + [np.array([body[2]], float)])
        grid.circle_radii = np.concatenate([p.circle_radii for p in parts] + [np.full(1, np.nan)])
        grid.radius = max([p.radius for p in parts] + [self.radius])
        grid.static_free = self._static
        return grid

    def blocks_path(self, points):
        """Whether newly placed bodies obstruct a remaining walking path.

        Departures only open space. Do not revalidate the unchanged terrain
        inset here: an observed, supported pose after local avoidance need not
        be a valid source cell for a fresh global route. Motion safety remains
        the controller's responsibility on every tick.
        """
        return any(
            not self.walking._bodies_clear(a, b, self.masks[i][1])
            for i in self.changed_bodies
            for a, b in zip(points[:-1], points[1:], strict=False)
        )

    def update(self, parked):
        """`parked`: {car index: (x, y, yaw, length, width, height)} of the cars standing parked now.

        Returns None when nothing changed, else (router, closed): the new static router and the Boolean grid of the
        cells this change closed."""
        if parked == self.parked:
            return None
        self.future_routes.clear()
        self.future_routers.clear()
        began = time.perf_counter()
        changed = []
        for i, body in parked.items():
            if i not in self.masks or self.masks[i][0] != body:
                changed.append(i)
                self.masks[i] = (
                    body,
                    self.walking.obstacle_mask(
                        np.array([body[:2]], float),
                        np.array([body[3:6]], float),
                        self.radius,
                        np.array([body[2]], float),
                    ),
                )
        for i in [i for i in self.masks if i not in parked]:
            del self.masks[i]
        free = _copy(self.base)
        for i in sorted(self.masks):
            merge_obstacles(free, self.masks[i][1])
        target = self.activity.static_free
        closed = np.asarray(target, bool) & ~np.asarray(free, bool)
        opened = np.asarray(free, bool) & ~np.asarray(target, bool)
        target[...] = np.asarray(free)
        for name in BODY_FIELDS:
            setattr(target, name, getattr(free, name))
        target.radius, target.static_free = free.radius, free.static_free
        self._invalidate(closed, opened)
        router = self.activity.static_router = LifeRouter(
            self.activity.network, target, stats=self.activity.route_stats, landmark_source=self._landmarks()
        )
        self.parked = dict(parked)
        self.changed_bodies = tuple(sorted(changed))
        wall = time.perf_counter() - began
        self.stats["updates"] += 1
        self.stats["wall_s"] += wall
        self.stats["max_wall_s"] = max(self.stats["max_wall_s"], wall)
        return router, closed

    def _landmarks(self):
        """The router on the grid without parked cars (built once): parked cars only remove cells from it."""
        if self.landmark_base is None:
            self.landmark_base = LifeRouter(self.activity.network, self.base, stats=self.activity.route_stats)
        return self.landmark_base

    def _invalidate(self, closed, opened):
        """Drop the cached routes the change makes invalid (see the module docstring) and the grid's derived indices."""
        activity = self.activity
        cache = activity.route_cache
        dropped = 0
        if opened.any():
            for key in [k for k, v in cache.items() if v is None]:
                del cache[key]
                dropped += 1
        if closed.any():
            rows, cols = np.nonzero(closed)
            network = self.walking
            lo = network.origin + (np.array([cols.min(), rows.min()]) - 1) * network.step
            hi = network.origin + (np.array([cols.max(), rows.max()]) + 1) * network.step
            for key in list(cache):
                rr = cache[key]
                if rr is None:
                    continue
                p = np.asarray(rr["points"], float)
                if (p.max(0) < lo).any() or (p.min(0) > hi).any():
                    continue
                if self.crosses(p, closed):
                    del cache[key]
                    dropped += 1
        activity.static_router = None
        activity.static_labels = None
        activity.source_components.clear()
        activity.home_access_cache.clear()
        activity.route_stats["invalidations"] = activity.route_stats.get("invalidations", 0) + 1
        self.stats["routes_dropped"] = self.stats.get("routes_dropped", 0) + dropped
        self.stats["routes_kept"] = self.stats.get("routes_kept", 0) + len(cache)

    def crosses(self, points, closed, step=None):
        """Whether the polyline `points` passes a closed cell (samples at the grid step)."""
        network = self.walking
        step = float(step or network.step)
        p = np.asarray(points, float)
        if len(p) < 2:
            samples = p
        else:
            seg = np.diff(p, axis=0)
            length = np.linalg.norm(seg, axis=1)
            samples = [p[:1]]
            for a, d, l in zip(p[:-1], seg, length, strict=False):
                n = max(int(np.ceil(l / step)), 1)
                samples.append(a + np.outer(np.arange(1, n + 1) / n, d))
            samples = np.vstack(samples)
        rc = np.floor((samples - network.origin) / network.step + 0.5).astype(int)[:, ::-1]
        ok = (rc >= 0).all(1) & (rc[:, 0] < closed.shape[0]) & (rc[:, 1] < closed.shape[1])
        return bool(closed[rc[ok, 0], rc[ok, 1]].any())
