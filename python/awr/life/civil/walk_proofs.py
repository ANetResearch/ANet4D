"""Bounded continuous-path hints across activities, never cached permission.

The ordinary planner remains authoritative when it finds a route. These hints
only supply geometry when its node raster cannot represent a known passage.
Every reuse, in either direction, checks current terrain and exact bodies.
Local endpoint connectors are measured segments, never snapped positions.
"""

import hashlib
from collections import OrderedDict

import numpy as np

from .walking import polyline


class WalkProofs:
    def __init__(self, max_paths=2048, max_points=262144, connector_m=1.5):
        if max_paths < 1 or max_points < 2 or not 0 < connector_m <= 1.5:
            raise ValueError("Positive bounded path storage and local connectors required")
        self.max_paths, self.max_points, self.connector_m = max_paths, max_points, connector_m
        self.paths = OrderedDict()
        self._endpoints = None
        self.points = 0
        self.stats = dict(remembered=0, evicted=0, queries=0, candidates=0, reused=0)

    def remember(self, points):
        p = np.array(points, dtype=np.float64, copy=True)
        if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
            raise ValueError("Finite XY path required")
        if len(p) < 2 or len(p) > self.max_points:
            return
        p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-7]]
        if len(p) < 2:
            return
        digest = hashlib.sha256(min(p.tobytes(), p[::-1].tobytes())).hexdigest()
        if digest in self.paths:
            self.paths.move_to_end(digest)
            return
        p.flags.writeable = False
        self.paths[digest] = p
        self._endpoints = None
        self.points += len(p)
        self.stats["remembered"] += 1
        while len(self.paths) > self.max_paths or self.points > self.max_points:
            _, old = self.paths.popitem(last=False)
            self.points -= len(old)
            self.stats["evicted"] += 1

    def plan(self, start, end, network, free):
        self.stats["queries"] += 1
        start, end = np.asarray(start, float), np.asarray(end, float)
        if not self.paths:
            return None
        if self._endpoints is None:
            self._endpoint_keys = tuple(self.paths)
            self._endpoints = np.array([[p[0], p[-1]] for p in self.paths.values()])
        candidates = []
        for reverse in (False, True):
            ends = self._endpoints[:, ::-1] if reverse else self._endpoints
            near = (np.linalg.norm(ends[:, 0] - start, axis=1) <= self.connector_m) & (
                np.linalg.norm(ends[:, 1] - end, axis=1) <= self.connector_m
            )
            for index in np.flatnonzero(near):
                digest = self._endpoint_keys[index]
                p = self.paths[digest]
                q = p[::-1] if reverse else p
                length = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
                length += float(np.linalg.norm(start - q[0]) + np.linalg.norm(end - q[-1]))
                candidates.append((length, digest, reverse))
        for _, digest, reverse in sorted(candidates):
            self.stats["candidates"] += 1
            p = self.paths[digest]
            if reverse:
                p = p[::-1]
            route = polyline(np.vstack([start, p, end]), "retained-continuous-walk")
            if route is None:
                continue
            points = route["points"]
            if not all(network.clear_segment(a, b, free) for a, b in zip(points[:-1], points[1:], strict=False)):
                continue
            self.paths.move_to_end(digest)
            self.stats["reused"] += 1
            route["retained_geometry_sha256"] = digest
            return route
        return None
