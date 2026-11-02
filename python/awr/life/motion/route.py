"""Point-to-point walking routes for the life runtime: compiled A* with landmark (ALT) lower bounds.

StaticWalkRouter answers a query with a full Dijkstra tree over the static walking snapshot (about 0.5 s on
the town's 1.4 M cells), which made route queries the largest tick spikes. Here a query expands only cells
that can lie on a shortest path: A* on the same four-neighbour grid and the same costs, with the heuristic
max(Manhattan distance, landmark triangle bounds) (Goldberg & Harrelson's ALT). Both bounds are consistent, so
the returned cost is the shortest-path cost; among equal-cost paths the choice may differ from the Dijkstra
tree's. Every result passes the same connector, body and turning checks (WalkNetwork.finish_grid_path).

Landmark distances are computed once per static snapshot (one SciPy Dijkstra per landmark, farthest-point
selection in each large component) and stored as float32; the bound is lowered by LM_SLACK to absorb rounding.

A query may add a non-negative extra cost to cells (`plan(..., avoid=...)`: the congestion of people standing in
the way, awr.life.motion.reroute). Static-cost bounds remain lower bounds of the raised costs, so the search stays exact.
String pulling of such a query keeps the detour: a shortcut may not run through the avoided cells for longer (by
more than AVOID_TOLERANCE_M) than the grid path it replaces, measured at AVOID_SAMPLE_M.

A walker may stand off the snapshot: its exact body is supported where the snapshot's cells, which keep half a
walking width from unsupported ground at their centres, are not (at the edge of a support gap that it was pushed to
while giving way). A route from there starts with a connector of at most CONNECT_M to the nearest snapshot node
reached over supported ground (centre line sampled at CONNECTOR_SAMPLE_M) clear of the static bodies; the rest of
the route passes every check as usual, and the crowd walks the connector with its exact support and body tests.
Likewise a goal whose cell only a body's clearance closes ends with a clear segment from the nearest free cell
(awr.life.civil.walking.WalkNetwork.goal_cell).
"""

import math
import time

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from awr.life.civil.walking import polyline

LM_SLACK = 1e-3
AVOID_SAMPLE_M = 0.1
AVOID_TOLERANCE_M = 0.2
CONNECT_M = 1.5
CONNECTOR_SAMPLE_M = 0.1


@njit(cache=True)
def _sift_up(hf, hv, i):
    while i > 0:
        p = (i - 1) >> 1
        if hf[i] < hf[p] or (hf[i] == hf[p] and hv[i] < hv[p]):
            hf[i], hf[p] = hf[p], hf[i]
            hv[i], hv[p] = hv[p], hv[i]
            i = p
        else:
            break


@njit(cache=True)
def _sift_down(hf, hv, size):
    i = 0
    while True:
        a, b, m = 2 * i + 1, 2 * i + 2, i
        if a < size and (hf[a] < hf[m] or (hf[a] == hf[m] and hv[a] < hv[m])):
            m = a
        if b < size and (hf[b] < hf[m] or (hf[b] == hf[m] and hv[b] < hv[m])):
            m = b
        if m == i:
            return
        hf[i], hf[m] = hf[m], hf[i]
        hv[i], hv[m] = hv[m], hv[i]
        i = m


@njit(cache=True)
def astar(nbr, cost, extra, yx, lm, src, dst, g, parent, seen, closed, stamp):
    """A* from src to dst on cell costs cost+extra; returns the number of expanded cells (negative when dst is
    unreachable).

    `seen`/`closed` hold the query stamp of cells whose g/parent are valid or final, so the arrays are reused
    across queries without clearing. Heap ties are broken by cell index (deterministic)."""
    L = lm.shape[1]
    ty, tx = yx[dst, 0], yx[dst, 1]
    cap = 4096
    hf = np.empty(cap)
    hv = np.empty(cap, np.int64)
    size = 0
    g[src] = 0.0
    parent[src] = src
    seen[src] = stamp
    hf[0] = 0.0
    hv[0] = src
    size = 1
    expanded = 0
    while size > 0:
        u = hv[0]
        size -= 1
        hf[0] = hf[size]
        hv[0] = hv[size]
        _sift_down(hf, hv, size)
        if closed[u] == stamp:
            continue
        closed[u] = stamp
        expanded += 1
        if u == dst:
            return expanded
        gu = g[u]
        for k in range(4):
            v = nbr[u, k]
            if v < 0 or closed[v] == stamp:
                continue
            nd = gu + (cost[u] + extra[u] + cost[v] + extra[v]) * 0.5
            if seen[v] != stamp or nd < g[v]:
                seen[v] = stamp
                g[v] = nd
                parent[v] = u
                h = float(abs(yx[v, 0] - ty) + abs(yx[v, 1] - tx))
                for j in range(L):
                    b = abs(np.float64(lm[dst, j]) - np.float64(lm[v, j])) - LM_SLACK
                    if b > h:
                        h = b
                if size == cap:
                    cap *= 2
                    nf = np.empty(cap)
                    nv = np.empty(cap, np.int64)
                    nf[:size] = hf[:size]
                    nv[:size] = hv[:size]
                    hf = nf
                    hv = nv
                hf[size] = nd + h
                hv[size] = v
                _sift_up(hf, hv, size)
                size += 1
    return -expanded


class LifeRouter:
    """Drop-in replacement for StaticWalkRouter.plan on one static snapshot (subset of the walking network)."""

    def __init__(self, network, free, *, landmarks=16, stats=None, landmark_source=None):
        """`landmark_source`: a router on a superset of `free` in the same network whose landmark distances this one
        reuses (adapted from codex 232db5e): removing cells cannot shorten a route, so they remain lower bounds here."""
        if np.shape(free) != network.free.shape or np.any(np.asarray(free, bool) & ~network.free):
            raise ValueError("Static planning snapshot must be a subset of the walking network")
        began = time.perf_counter()
        self.network = network
        self.free = free.copy()
        for name in ("positions", "sizes", "yaws", "radius", "circle_radii", "static_free", "turn_only"):
            if hasattr(free, name):
                value = getattr(free, name)
                setattr(self.free, name, value.copy() if isinstance(value, np.ndarray) else value)
        self.free.flags.writeable = False
        self.stats = stats if stats is not None else {}
        mask = np.asarray(self.free, bool)
        self.yx = np.argwhere(mask).astype(np.int32)
        n = len(self.yx)
        self.index = np.full(mask.shape, -1, np.int32)
        self.index[mask] = np.arange(n, dtype=np.int32)
        weights = np.ones(mask.shape) if network.weight is None else np.asarray(network.weight, np.float64)
        if not np.isfinite(weights[mask]).all() or np.any(weights[mask] < 1.0):
            raise ValueError("Finite walking costs of at least 1 required")
        self.cost = weights[mask].astype(np.float64)
        pad = np.pad(self.index, 1, constant_values=-1)
        y, x = self.yx[:, 0] + 1, self.yx[:, 1] + 1
        self.nbr = np.stack([pad[y, x + 1], pad[y + 1, x], pad[y, x - 1], pad[y - 1, x]], 1).astype(np.int64)
        rows = np.repeat(np.arange(n), 2)
        right_down = self.nbr[:, :2].ravel()
        keep = right_down >= 0
        rows, cols = rows[keep], right_down[keep]
        graph = csr_matrix(((self.cost[rows] + self.cost[cols]) / 2, (rows, cols)), shape=(n, n))
        _, self.components = connected_components(graph, directed=False)
        if landmark_source is not None:
            if not isinstance(landmark_source, LifeRouter) or landmark_source.network is not network:
                raise ValueError("Landmark source must use the same walking network")
            parent = landmark_source.index[mask]
            if np.any(parent < 0) or np.any(self.cost < landmark_source.cost[parent]):
                raise ValueError("Landmark reuse requires a subset with no reduced cell costs")
            lm = landmark_source.lm[parent]
            col = lm.shape[1]
            self.stats["life_router_landmark_reuses"] = self.stats.get("life_router_landmark_reuses", 0) + 1
        else:
            lm, col = self._landmarks(graph, landmarks, n)
        self.lm = np.ascontiguousarray(lm)
        self.extra = np.zeros(n)
        self.g = np.zeros(n)
        self.parent = np.zeros(n, np.int64)
        self.seen = np.zeros(n, np.int64)
        self.closed = np.zeros(n, np.int64)
        self.stamp = 0
        self.stats["life_router_builds"] = self.stats.get("life_router_builds", 0) + 1
        self.stats["life_router_build_wall_s"] = (
            self.stats.get("life_router_build_wall_s", 0.0) + time.perf_counter() - began
        )
        self.stats["life_router_landmarks"] = col

    def _landmarks(self, graph, landmarks, n):
        """Farthest-point landmarks, apportioned to components by size (components under 2 % get none: the Manhattan
        bound alone is exact enough for them); the distance table and its column count."""
        sizes = np.bincount(self.components)
        share = np.where(sizes >= max(0.02 * n, 1), np.maximum(1, np.round(landmarks * sizes / n)), 0).astype(int)
        lm = np.full((n, max(int(share.sum()), 1)), np.inf, np.float32)
        col = 0
        for c in np.flatnonzero(share):
            members = np.flatnonzero(self.components == c)
            far = dijkstra(graph, directed=False, indices=int(members[0]))
            pick = int(members[np.argmax(far[members])])
            best = np.full(len(members), np.inf)
            for _ in range(share[c]):
                d = dijkstra(graph, directed=False, indices=pick)
                lm[:, col] = d
                col += 1
                best = np.minimum(best, d[members])
                pick = int(members[np.argmax(best)])
        return lm, col

    def cells(self, source, target):
        """Target-to-source grid cells of a shortest path between two snapshot cell indices, or None."""
        self.stamp += 1
        began = time.perf_counter()
        expanded = astar(
            self.nbr,
            self.cost,
            self.extra,
            self.yx,
            self.lm,
            source,
            target,
            self.g,
            self.parent,
            self.seen,
            self.closed,
            self.stamp,
        )
        self.stats["life_router_searches"] = self.stats.get("life_router_searches", 0) + 1
        self.stats["life_router_expanded"] = self.stats.get("life_router_expanded", 0) + abs(expanded)
        self.stats["life_router_wall_s"] = self.stats.get("life_router_wall_s", 0.0) + time.perf_counter() - began
        if expanded < 0:
            return None
        out = [target]
        cursor = target
        while cursor != source:
            cursor = int(self.parent[cursor])
            out.append(cursor)
        return [tuple(self.yx[c]) for c in out]

    def components_near(self, xy, reach_m=1.5):
        """Static components of the network nodes within reach_m of xy (the planner's source convention)."""
        network = self.network
        dist, ids = network.tree.query(np.asarray(xy, float), k=min(24, len(network.xy)))
        out = set()
        for d, i in zip(np.atleast_1d(dist), np.atleast_1d(ids), strict=False):
            if d <= reach_m:
                k = self.index[tuple(network.nodes[int(i)])]
                if k >= 0:
                    out.add(int(self.components[k]))
        return out

    def connected(self, a, b):
        """Necessary condition for a walking route from a to b: a shared static component."""
        return bool(self.components_near(a) & self.components_near(b))

    def avoid_cells(self, points, radius_m):
        """Snapshot cell indices within radius_m of any of `points` (unique)."""
        network = self.network
        r = int(math.ceil(radius_m / network.step))
        dy, dx = np.mgrid[-r : r + 1, -r : r + 1]
        keep = (dy * dy + dx * dx) * network.step**2 <= radius_m**2
        dy, dx = dy[keep], dx[keep]
        centre = np.array([network.cell(p) for p in np.asarray(points, float).reshape(-1, 2)], int).reshape(-1, 2)
        rows = (centre[:, :1] + dy[None]).ravel()
        cols = (centre[:, 1:] + dx[None]).ravel()
        inside = (rows >= 0) & (cols >= 0) & (rows < self.index.shape[0]) & (cols < self.index.shape[1])
        cells = self.index[rows[inside], cols[inside]]
        return np.unique(cells[cells >= 0])

    def avoid_bodies(self, points, yaw, speed, ext, clearance_m=0.45):
        """Snapshot cell indices within `clearance_m` of the observed footprints (awr.life.civil.body.boxes) of bodies
        at `points` with headings `yaw`: a vehicle's whole rotated body, not a disc round its centre (port of codex
        8742f47). Soft congestion costs only (plan's `avoid`), never a collision exemption."""
        from awr.life.civil.body import boxes

        points, yaw, speed, ext = (np.asarray(a, float) for a in (points, yaw, speed, ext))
        n = len(points)
        if (
            points.shape != (n, 2)
            or yaw.shape != (n,)
            or speed.shape != (n,)
            or ext.shape != (n, 2, 4)
            or not all(np.isfinite(a).all() for a in (points, yaw, speed, ext))
            or not math.isfinite(clearance_m)
            or clearance_m < 0
        ):
            raise ValueError("Finite observed body geometry and nonnegative clearance required")
        if not n:
            return np.empty(0, np.int64)
        centres, sizes = boxes(points, yaw, np.abs(speed), ext)
        network, index = self.network, self.index
        result = []
        for centre, size, heading in zip(centres, sizes, yaw, strict=False):
            hl, hw = size[0] / 2, size[1] / 2
            c, sn = math.cos(heading), math.sin(heading)
            half = np.array([abs(c) * hl + abs(sn) * hw, abs(sn) * hl + abs(c) * hw]) + clearance_m
            low = np.floor((centre[:2] - half - network.origin) / network.step).astype(int)
            high = np.ceil((centre[:2] + half - network.origin) / network.step).astype(int)
            cols = np.arange(max(0, low[0]), min(index.shape[1] - 1, high[0]) + 1)
            rows = np.arange(max(0, low[1]), min(index.shape[0] - 1, high[1]) + 1)
            xx, yy = np.meshgrid(cols, rows)
            dx = network.origin[0] + xx * network.step - centre[0]
            dy = network.origin[1] + yy * network.step - centre[1]
            a = np.maximum(np.abs(dx * c + dy * sn) - hl, 0.0)
            b = np.maximum(np.abs(-dx * sn + dy * c) - hw, 0.0)
            cells = index[yy, xx][a * a + b * b <= clearance_m * clearance_m + 1e-12]
            result.extend(cells[cells >= 0].tolist())
        return np.unique(np.asarray(result, np.int64))

    def plan(self, start, goal, avoid=None, *, geometry=None):
        """Route from start to goal, or None. `avoid` = (cell indices, factor): those cells cost (1+factor) times
        their static cost in this query. `geometry`: the exact-obstacle grid whose static cells and bodies every
        segment of the route must clear (WalkNetwork.clear_segment), instead of this snapshot's; the cells searched
        stay this snapshot's (ParkedSpace._round)."""
        if avoid is None:
            return self._plan(start, goal, geometry=geometry)
        cells, factor = avoid
        self.extra[cells] = factor * self.cost[cells]
        try:
            return self._plan(start, goal, self._avoid_rule(cells), geometry=geometry)
        finally:
            self.extra[cells] = 0.0

    def avoided_m(self, a, b, avoided):
        """Length (m) of the segment a-b within the cells flagged in `avoided` (midpoint samples AVOID_SAMPLE_M
        apart, so that the lengths of consecutive segments add up)."""
        network = self.network
        a, b = np.asarray(a, float), np.asarray(b, float)
        length = float(np.linalg.norm(b - a))
        n = max(int(math.ceil(length / AVOID_SAMPLE_M)), 1)
        pts = a + ((np.arange(n) + 0.5) / n)[:, None] * (b - a)
        rc = np.floor((pts - network.origin) / network.step + 0.5).astype(int)[:, ::-1]
        ok = (rc >= 0).all(1) & (rc[:, 0] < self.index.shape[0]) & (rc[:, 1] < self.index.shape[1])
        k = self.index[rc[ok, 0], rc[ok, 1]]
        return length / n * float(avoided[k[k >= 0]].sum())

    def _avoid_rule(self, cells):
        avoided = np.zeros(len(self.cost), bool)
        avoided[cells] = True

        def rule(raw):
            along = np.concatenate(
                [[0.0], np.cumsum([self.avoided_m(raw[m], raw[m + 1], avoided) for m in range(len(raw) - 1)])]
            )
            return lambda k, j: self.avoided_m(raw[k], raw[j], avoided) <= along[j] - along[k] + AVOID_TOLERANCE_M

        return rule

    def _plan(self, start, goal, shortcut_rule=None, geometry=None):
        network, free = self.network, self.free
        check = free if geometry is None else geometry
        target = network.goal_cell(goal, free)
        if target is None:
            return None
        target_index = int(self.index[target])
        distances, indices = network.tree.query(start, k=min(24, len(network.xy)))
        near = [
            (d, int(i))
            for d, i in zip(np.atleast_1d(distances), np.atleast_1d(indices), strict=False)
            if d <= CONNECT_M
            and free[tuple(network.nodes[i])]
            and self.components[self.index[tuple(network.nodes[i])]] == self.components[target_index]
        ]
        candidates = [(d, i) for d, i in near if network.clear_segment(start, network.xy[i], check)]
        # A start off the snapshot, or on a snapshot cell that the 4-connected graph does not join to the target (an
        # island the walker's local avoidance reached, whose cells close round it diagonally), joins the network by a
        # connector on supported ground clear of the static bodies.
        off = not candidates and not self.joined(start, target_index)
        if off:
            candidates = [(d, i) for d, i in near if self.connector_clear(start, network.xy[i])]
        if not candidates:
            return None
        source = tuple(network.nodes[candidates[0][1]])
        if network.labels[source] != network.labels[target]:
            return None
        cells = self.cells(int(self.index[source]), target_index)
        if cells is None:
            return None
        if not off:
            return network.finish_grid_path(start, goal, cells, check, shortcut_rule)
        route = network.finish_grid_path(network.xy[candidates[0][1]], goal, cells, check, shortcut_rule)
        return None if route is None else polyline(np.vstack([start, route["points"]]), route["road"])

    def on_snapshot(self, xy):
        r, c = self.network.cell(xy)
        return 0 <= r < self.free.shape[0] and 0 <= c < self.free.shape[1] and bool(self.free[r, c])

    def joined(self, xy, target_index):
        """Whether xy stands on a snapshot cell of the component of the node with index target_index."""
        if not self.on_snapshot(xy):
            return False
        return bool(self.components[self.index[tuple(self.network.cell(xy))]] == self.components[target_index])

    def probe(self, start, goal):
        """Why a route from start to goal is found or not (a failure report's diagnostic): the start's cell and
        component, the goal's, and each network node the planner could join within CONNECT_M with its static-grid
        segment and connector checks."""
        network, free = self.network, self.free
        target = network.goal_cell(goal, free)
        target_component = None if target is None else int(self.components[self.index[target]])
        r, c = network.cell(start)
        on = self.on_snapshot(start)
        row = dict(
            on_snapshot=on,
            start_component=int(self.components[self.index[r, c]]) if on else None,
            target_component=target_component,
            nodes=[],
        )
        distances, indices = network.tree.query(start, k=min(24, len(network.xy)))
        for d, i in zip(np.atleast_1d(distances), np.atleast_1d(indices), strict=False):
            node = tuple(network.nodes[int(i)])
            if d > CONNECT_M:
                continue
            row["nodes"].append(
                dict(
                    xy=[round(float(v), 2) for v in network.xy[int(i)]],
                    distance_m=round(float(d), 2),
                    free=bool(free[node]),
                    component=int(self.components[self.index[node]]) if free[node] else None,
                    segment_clear=bool(network.clear_segment(start, network.xy[int(i)], free)),
                    connector_clear=bool(self.connector_clear(start, network.xy[int(i)])),
                )
            )
        return row

    def connector_clear(self, a, b):
        """Connector of an off-snapshot start: centre line on supported ground, clear of the static bodies."""
        surface = getattr(self.network, "surface", None)
        if surface is None:
            return False
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(int(math.ceil(float(np.linalg.norm(b - a)) / CONNECTOR_SAMPLE_M)), 1)
        pts = a + np.linspace(0.0, 1.0, n + 1)[:, None] * (b - a)
        return bool(surface.supported(pts, "person").all()) and self.network._bodies_clear(a, b, self.free)
