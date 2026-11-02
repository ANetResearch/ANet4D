"""Walkable ground of the stage for JuPedSim, and the routes people take on it.

The walkable area is the measured person support inside the stage (its largest connected part). Streets of this town
mostly have no continuous sidewalk, so people share the carriageway with cars, as the earlier walking policy did
(awr.life.civil outdoor places, 'shared_streets'): a route is the least-cost path on the 0.5 m support grid where a
cell under a car lane of the stage network costs CARRIAGEWAY_COST per metre, so routes keep to the sidewalks and road
edges and cross the roads directly. JuPedSim moves the people along the route's waypoints and keeps them apart; the
coupling with the cars is awr.life.ambient.coupling.
"""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely import contains_xy
from shapely.geometry import LineString, MultiPolygon, Polygon

MIN_PART_M2 = 20.0  # smaller walkable islands are dropped
CARRIAGEWAY_COST = 3.0  # per metre under a car lane (a sidewalk or road edge costs 1)
LANE_BAND_M = 2.2  # half-width of the ground counted as carriageway around a car lane: a car's half width
# (1.12), a person's largest half extent (0.69) and room to pass
WAYPOINT_M = 4.0  # spacing of the waypoints handed to JuPedSim
SIMPLIFY_M = 0.15
OBSTACLE_PAD_M = 0.3  # walkway cells this close to a standing obstacle (a parked car) are removed
SNAP_CELLS = 3  # a route from (or to) a point off the reachable cells starts at the nearest one within
# this many cells


@dataclass
class Walkway:
    x0: float
    y0: float
    cell_m: float
    walk: np.ndarray  # [row = y, col = x] walkable cells (largest part)
    cost: np.ndarray  # cost per metre of each walkable cell
    polygon: Polygon  # walkable area for JuPedSim

    @classmethod
    def build(cls, support, bbox, stage_dir=None, obstacles=()):
        """The walkway of `support` inside `bbox`, with the car lanes of the compiled stage network in `stage_dir`,
        around the standing `obstacles` (shapely polygons)."""
        from scipy import ndimage

        sup = support.window(bbox)
        walk = np.array(sup.walk_person, bool)
        c = sup.cell_m
        ys, xs = np.mgrid[0 : walk.shape[0], 0 : walk.shape[1]]
        X, Y = sup.x0 + (xs + 0.5) * c, sup.y0 + (ys + 0.5) * c
        walk &= (bbox[0] <= X) & (bbox[2] >= X) & (bbox[1] <= Y) & (bbox[3] >= Y)
        if len(obstacles):
            from shapely.ops import unary_union

            walk &= ~contains_xy(unary_union([o.buffer(OBSTACLE_PAD_M + c / 2) for o in obstacles]), X, Y)
        label, n = ndimage.label(walk)
        sizes = ndimage.sum(walk, label, range(1, n + 1)) * c * c
        walk = label == (1 + int(np.argmax(sizes)))
        walk = ndimage.binary_opening(walk, iterations=1) & walk
        label, n = ndimage.label(walk)
        sizes = ndimage.sum(walk, label, range(1, n + 1)) * c * c
        walk = label == (1 + int(np.argmax(sizes)))
        cost = np.where(walk, 1.0, np.inf)
        if stage_dir is not None:
            geo = json.loads((Path(stage_dir) / "stage-geometry.json").read_text())
            lines = [LineString(p) for p in geo["edges"].values() if len(p) > 1]
            lines += [LineString(c_["points"]) for c_ in geo["connections"] if len(c_["points"]) > 1]
            from shapely.ops import unary_union

            band = unary_union([l.buffer(LANE_BAND_M) for l in lines])
            under = contains_xy(band, X, Y)
            cost[walk & under] = CARRIAGEWAY_COST
        polygon = _polygon(walk, sup.x0, sup.y0, c)
        return cls(sup.x0, sup.y0, c, walk, cost, polygon)

    def cell(self, xy):
        return int((xy[1] - self.y0) // self.cell_m), int((xy[0] - self.x0) // self.cell_m)

    def centre(self, rc):
        return np.array([self.x0 + (rc[1] + 0.5) * self.cell_m, self.y0 + (rc[0] + 0.5) * self.cell_m])

    def inside(self, xy, margin=0.4):
        """Whether `xy` lies in the JuPedSim area at least `margin` from its boundary."""
        from shapely.geometry import Point

        p = Point(xy)
        return (
            self.polygon.contains(p)
            and self.polygon.exterior.distance(p) >= margin
            and all(h.distance(p) >= margin for h in self.polygon.interiors)
        )

    def route(self, a, b):
        """Waypoints (world xy) of the least-cost route from `a` to `b`, None without one."""
        pts = self.path(a, b)
        if pts is None:
            return None
        if len(pts) < 2:
            return [np.asarray(b, float)]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        keep = np.unique(np.searchsorted(arc, np.arange(WAYPOINT_M, arc[-1], WAYPOINT_M)))
        out = [pts[k] for k in keep if self.inside(pts[k])]
        return out + [np.asarray(b, float)]

    def path(self, a, b, cost=None):
        """Cell centres (world xy, (n, 2)) of the least-cost route from `a` to `b` on `cost` (default: the walkway's),
        None without one."""
        cost = self.cost if cost is None else cost
        path = _astar(cost, _finite(cost, self.cell(a)), _finite(cost, self.cell(b)))
        if path is None:
            return None
        return np.array([self.centre(rc) for rc in path])


def _polygon(mask, x0, y0, c):
    """The walkable cells as one polygon (with holes), in world coordinates, simplified by SIMPLIFY_M."""
    import contourpy

    z = np.pad(mask.astype(float), 1)
    gen = contourpy.contour_generator(z=z, fill_type=contourpy.FillType.OuterOffset)
    polys = []
    for pts, offsets in zip(*gen.filled(0.5, 1.5), strict=False):
        rings = [pts[offsets[k] : offsets[k + 1]] for k in range(len(offsets) - 1)]
        world = [np.c_[x0 + (r[:, 0] - 1 + 0.5) * c, y0 + (r[:, 1] - 1 + 0.5) * c] for r in rings]
        polys.append(Polygon(world[0], world[1:]).buffer(0))
    geom = max((p for p in polys if not p.is_empty), key=lambda p: p.area)
    geom = geom.simplify(SIMPLIFY_M).buffer(0)
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda p: p.area)
    return geom


def _finite(cost, rc, reach=SNAP_CELLS):
    """`rc` if walkable, else the nearest walkable cell within `reach` cells (a walker pushed to the edge of its area
    by others stands on a cell outside the walkway's reachable part)."""
    H, W = cost.shape
    if 0 <= rc[0] < H and 0 <= rc[1] < W and np.isfinite(cost[rc]):
        return rc
    r0, c0 = max(rc[0] - reach, 0), max(rc[1] - reach, 0)
    win = cost[r0 : rc[0] + reach + 1, c0 : rc[1] + reach + 1]
    ok = np.argwhere(np.isfinite(win))
    if not len(ok):
        return rc
    k = ((ok + [r0, c0] - np.asarray(rc)) ** 2).sum(1).argmin()
    return int(ok[k, 0] + r0), int(ok[k, 1] + c0)


def _astar(cost, a, b):
    """8-connected least-cost path on the grid (cost per metre per cell, inf = blocked); list of (row, col)."""
    H, W = cost.shape
    if not (0 <= a[0] < H and 0 <= a[1] < W and 0 <= b[0] < H and 0 <= b[1] < W):
        return None
    if not np.isfinite(cost[a]) or not np.isfinite(cost[b]):
        return None
    return _astar_nb(cost, a[0], a[1], b[0], b[1])


try:
    from numba import njit
except ImportError:  # pragma: no cover

    def njit(f):
        return f


@njit(cache=True)
def _astar_core(cost, ar, ac, br, bc):
    H, W = cost.shape
    INF = 1e30
    g = np.full(H * W, INF)
    parent = np.full(H * W, -1, np.int64)
    closed = np.zeros(H * W, np.bool_)
    start, goal = ar * W + ac, br * W + bc
    g[start] = 0.0
    heap = [(0.0, start)]
    dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    while len(heap):
        f, u = heapq.heappop(heap)
        if closed[u]:
            continue
        if u == goal:
            break
        closed[u] = True
        r, c = u // W, u % W
        for k in range(8):
            rr, cc = r + dr[k], c + dc[k]
            if rr < 0 or cc < 0 or rr >= H or cc >= W:
                continue
            cv = cost[rr, cc]
            if cv == np.inf:
                continue
            step = 1.4142135623730951 if dr[k] != 0 and dc[k] != 0 else 1.0
            if step > 1.0 and (cost[r, cc] == np.inf or cost[rr, c] == np.inf):
                continue  # no corner cutting
            v = rr * W + cc
            ng = g[u] + step * 0.5 * (cost[r, c] + cv) / 2.0
            if ng < g[v]:
                g[v] = ng
                parent[v] = u
                dy, dx = abs(rr - br), abs(cc - bc)
                h = (max(dy, dx) + 0.41421356 * min(dy, dx)) * 0.5
                heapq.heappush(heap, (ng + h, v))
    return parent, g[goal] < INF


def _astar_nb(cost, ar, ac, br, bc):
    parent, found = _astar_core(cost, ar, ac, br, bc)
    if not found:
        return None
    W = cost.shape[1]
    u, path = br * W + bc, []
    while u != -1:
        path.append((u // W, u % W))
        u = parent[u]
    return path[::-1]
