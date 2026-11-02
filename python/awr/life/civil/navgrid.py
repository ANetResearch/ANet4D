"""Pedestrian navigation on the 1 m person grid: cost field (sidewalks preferred, carriageway crossed at crossings),
multi-source shortest paths (scipy Dijkstra on the walkable-cell graph), string-pulled waypoints and an obstacle
distance field for the social-force model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString

OFFS = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (-1, -1, math.sqrt(2)),
]


class PedGrid:
    def __init__(self, nav_dir: Path, sem: dict, *, road_cost: float = 8.0, crossing_cost: float = 1.3) -> None:
        m = json.loads((nav_dir / "meta.json").read_text())
        self.x0, self.y0, self.nx, self.ny = float(m["x0"]), float(m["y0"]), int(m["nx"]), int(m["ny"])
        self.walk = np.load(nav_dir / "walk_person.npy").astype(bool)
        self.walk_v = np.load(nav_dir / "walk_vehicle.npy").astype(bool)
        self.z = np.load(nav_dir / "z_person.npy").astype(np.float64)
        self.walk &= np.isfinite(self.z)
        ys, xs = np.mgrid[0 : self.ny, 0 : self.nx]
        pts = shapely.points((xs + self.x0 + 0.5).ravel(), (ys + self.y0 + 0.5).ravel())
        self.road = np.zeros_like(self.walk)  # carriageway cells
        self.road_id = np.full(self.walk.shape, -1, np.int32)
        for i, r in enumerate(sem["roads"]):
            m_ = shapely.contains(LineString(r["line"]).buffer(r["width_m"] / 2.0 + 0.3), pts).reshape(self.ny, self.nx)
            self.road |= m_
            self.road_id[m_] = i
        self.roads = sem["roads"]
        xing = np.zeros_like(self.walk)
        for c in sem["crossings"]:
            m_ = shapely.contains(LineString(c["ends"]).buffer(3.0), pts).reshape(self.ny, self.nx)
            xing |= m_
        self.xing = xing
        self.cost = np.ones(self.walk.shape, np.float64)
        self.cost[self.road & ~xing] = road_cost
        self.cost[self.road & xing] = crossing_cost
        # keep pedestrians off the carriageway edge: cells adjacent to road but not road get a mild penalty
        near = ndimage.binary_dilation(self.road, iterations=1) & ~self.road
        self.cost[near] *= 1.15
        self.cost_jay = np.ones_like(self.cost)
        self.cost_jay[self.road] = 2.5
        # obstacle distance field (m) and gradient, from unwalkable cells
        self.dist = ndimage.distance_transform_edt(self.walk).astype(np.float64)
        gy, gx = np.gradient(self.dist)
        self.gx, self.gy = gx, gy
        rd = ndimage.distance_transform_edt(~(self.road & ~xing)).astype(
            np.float64
        )  # distance to carriageway cells (crossings excluded)
        rgy, rgx = np.gradient(rd)
        self.rdist, self.rgx, self.rgy = rd, rgx, rgy
        self.idx = -np.ones(self.walk.shape, np.int64)
        self.cells = np.argwhere(self.walk)  # (n, 2) [iy, ix]
        self.idx[self.cells[:, 0], self.cells[:, 1]] = np.arange(len(self.cells))
        self._graph_cache: dict[str, object] = {}
        from scipy.spatial import cKDTree

        self._tree = cKDTree(np.stack([self.cells[:, 1] + self.x0 + 0.5, self.cells[:, 0] + self.y0 + 0.5], 1))

    # ------------------------------------------------------------------ coordinates
    def cell(self, x: float, y: float) -> tuple[int, int] | None:
        ix, iy = int(math.floor(x - self.x0)), int(math.floor(y - self.y0))
        return (ix, iy) if 0 <= ix < self.nx and 0 <= iy < self.ny else None

    def walkable(self, x: float, y: float) -> bool:
        c = self.cell(x, y)
        return c is not None and bool(self.walk[c[1], c[0]])

    def z_at(self, x: float, y: float) -> float:
        c = self.cell(x, y)
        return float(self.z[c[1], c[0]]) if c else 0.0

    def z_batch(self, xy: np.ndarray) -> np.ndarray:
        ix = np.clip(np.floor(xy[:, 0] - self.x0).astype(int), 0, self.nx - 1)
        iy = np.clip(np.floor(xy[:, 1] - self.y0).astype(int), 0, self.ny - 1)
        return self.z[iy, ix]

    def snap(self, x: float, y: float, r: float = 0.75) -> tuple[float, float]:
        """Nearest walkable cell centre (main island); the point itself if already walkable."""
        if self.walkable(x, y):
            return x, y
        d, k = self._tree.query((x, y))
        if not np.isfinite(d) or d > r:
            raise ValueError(f"No walkable surface within {r:g} m of {(x, y)}")
        c = self.cells[int(k)]
        return float(c[1] + self.x0 + 0.5), float(c[0] + self.y0 + 0.5)

    # ------------------------------------------------------------------ graph
    def graph(self, jaywalk: bool = False):
        key = "jay" if jaywalk else "std"
        if key in self._graph_cache:
            return self._graph_cache[key]
        cost = self.cost_jay if jaywalk else self.cost
        rows, cols, vals = [], [], []
        iy, ix = self.cells[:, 0], self.cells[:, 1]
        src = self.idx[iy, ix]
        for dx, dy, w in OFFS:
            jx, jy = ix + dx, iy + dy
            ok = (jx >= 0) & (jx < self.nx) & (jy >= 0) & (jy < self.ny)
            jx, jy = np.clip(jx, 0, self.nx - 1), np.clip(jy, 0, self.ny - 1)
            ok &= self.walk[jy, jx]
            if abs(dx) + abs(dy) == 2:  # no corner cutting
                ok &= self.walk[iy, jx] & self.walk[jy, ix]
            dst = self.idx[jy, jx]
            c = w * 0.5 * (cost[iy, ix] + cost[jy, jx])
            rows.append(src[ok])
            cols.append(dst[ok])
            vals.append(c[ok])
        n = len(self.cells)
        G = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
        self._graph_cache[key] = G
        return G

    def paths_to(
        self, goal_xy: tuple[float, float], sources: list[tuple[float, float]], jaywalk: bool = False
    ) -> list[list[tuple[float, float]] | None]:
        """Shortest paths from each source to a common goal (Dijkstra from the goal over the reversed graph)."""
        G = self.graph(jaywalk)
        gx, gy = self.snap(*goal_xy)
        gc = self.cell(gx, gy)
        gi = int(self.idx[gc[1], gc[0]])
        _, pred = dijkstra(G.T.tocsr(), directed=True, indices=gi, return_predecessors=True)
        out = []
        for sx, sy in sources:
            sx, sy = self.snap(sx, sy)
            sc = self.cell(sx, sy)
            si = int(self.idx[sc[1], sc[0]])
            if si < 0 or (pred[si] < 0 and si != gi):
                out.append(None)
                continue
            seq = [si]
            while seq[-1] != gi:
                p = int(pred[seq[-1]])
                if p < 0:
                    break
                seq.append(p)
            pts = [(float(self.cells[i][1] + self.x0 + 0.5), float(self.cells[i][0] + self.y0 + 0.5)) for i in seq]
            out.append(self.pull(pts))
        return out

    def path(
        self, a: tuple[float, float], b: tuple[float, float], jaywalk: bool = False
    ) -> list[tuple[float, float]] | None:
        return self.paths_to(b, [a], jaywalk)[0]

    def los(self, a: tuple[float, float], b: tuple[float, float], cost_aware: bool = True, margin: float = 0.6) -> bool:
        n = int(math.dist(a, b) * 2) + 2
        xs = np.linspace(a[0], b[0], n)
        ys = np.linspace(a[1], b[1], n)
        ix = np.floor(xs - self.x0).astype(int)
        iy = np.floor(ys - self.y0).astype(int)
        if ix.min() < 0 or iy.min() < 0 or ix.max() >= self.nx or iy.max() >= self.ny:
            return False
        if not self.walk[iy, ix].all():
            return False
        if cost_aware and (self.cost[iy, ix] > 2.0).any():  # do not shortcut across the carriageway
            return False
        return bool((self.dist[iy, ix] >= margin).all())

    def pull(self, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Forward greedy string pulling: extend each segment while the straight line stays walkable, off the
        carriageway and ≥ 0.6 m from obstacles."""
        if len(pts) <= 2:
            return pts
        out = [pts[0]]
        i = 0
        n = len(pts)
        while i < n - 1:
            j = i + 1
            while j + 1 < n and self.los(pts[i], pts[j + 1]):
                j += 1
            out.append(pts[j])
            i = j
        return out

    # ------------------------------------------------------------------ fields for the social-force model
    def obstacle_force(self, xy: np.ndarray, r: float = 0.3, range_m: float = 1.0) -> np.ndarray:
        ix = np.clip(np.floor(xy[:, 0] - self.x0).astype(int), 0, self.nx - 1)
        iy = np.clip(np.floor(xy[:, 1] - self.y0).astype(int), 0, self.ny - 1)
        d = self.dist[iy, ix] - 0.5  # distance from cell centre to the nearest unwalkable cell edge
        g = np.stack([self.gx[iy, ix], self.gy[iy, ix]], 1)
        gn = np.linalg.norm(g, axis=1, keepdims=True) + 1e-9
        g = g / gn
        mag = 2.5 * np.exp((r - d) / 0.25)
        mag = np.where(d < range_m, mag, 0.0)
        return g * mag[:, None]

    def safe_point(self, x: float, y: float, min_road_dist: float = 1.8, r: float = 6.0) -> tuple[float, float]:
        """Nearest walkable cell centre at least ``min_road_dist`` m from the carriageway (for standing / goals)."""
        x, y = self.snap(x, y, r=r)
        c = self.cell(x, y)
        if c is not None and self.rdist[c[1], c[0]] >= min_road_dist:
            return x, y
        R = int(r)
        best, bd = (x, y), 1e18
        for dy in range(-R, R + 1):
            iy = c[1] + dy
            if not 0 <= iy < self.ny:
                continue
            for dx in range(-R, R + 1):
                ix = c[0] + dx
                if 0 <= ix < self.nx and self.walk[iy, ix] and self.rdist[iy, ix] >= min_road_dist:
                    d = dx * dx + dy * dy
                    if d < bd:
                        bd, best = d, (self.x0 + ix + 0.5, self.y0 + iy + 0.5)
        return best

    def random_point_near(self, x: float, y: float, r: float, rng, min_road_dist: float = 1.5) -> tuple[float, float]:
        """A uniformly chosen walkable cell centre within ``r`` m of (x, y) and off the carriageway; spreads crowds
        over the walkable part of a park or square instead of collapsing them onto one snapped point."""
        c = self.cell(x, y)
        if c is None:
            return self.safe_point(x, y, min_road_dist, r)
        R = int(r)
        x0, x1 = max(0, c[0] - R), min(self.nx, c[0] + R + 1)
        y0, y1 = max(0, c[1] - R), min(self.ny, c[1] + R + 1)
        sub = self.walk[y0:y1, x0:x1] & (self.rdist[y0:y1, x0:x1] >= min_road_dist)
        iy, ix = np.nonzero(sub)
        if len(ix) == 0:
            return self.safe_point(x, y, min_road_dist, r)
        d2 = (ix + x0 - c[0]) ** 2 + (iy + y0 - c[1]) ** 2
        ok = d2 <= R * R
        if not ok.any():
            return self.safe_point(x, y, min_road_dist, r)
        k = int(rng.integers(int(ok.sum())))
        jx, jy = ix[ok][k] + x0, iy[ok][k] + y0
        return float(jx + self.x0 + 0.5), float(jy + self.y0 + 0.5)

    def road_force(self, xy: np.ndarray, r: float = 0.3, range_m: float = 1.2) -> np.ndarray:
        """Push walkers that are not crossing away from the carriageway edge."""
        ix = np.clip(np.floor(xy[:, 0] - self.x0).astype(int), 0, self.nx - 1)
        iy = np.clip(np.floor(xy[:, 1] - self.y0).astype(int), 0, self.ny - 1)
        d = self.rdist[iy, ix] - 0.5
        g = np.stack([self.rgx[iy, ix], self.rgy[iy, ix]], 1)
        g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
        mag = np.where(d < range_m, 2.0 * np.exp((r - d) / 0.35), 0.0)
        return g * mag[:, None]

    def on_road(self, xy: np.ndarray) -> np.ndarray:
        ix = np.clip(np.floor(xy[:, 0] - self.x0).astype(int), 0, self.nx - 1)
        iy = np.clip(np.floor(xy[:, 1] - self.y0).astype(int), 0, self.ny - 1)
        return self.road[iy, ix]

    def road_at(self, x: float, y: float) -> int:
        c = self.cell(x, y)
        return int(self.road_id[c[1], c[0]]) if c else -1
