"""Physical neighbourhood capacity for independently named activity slots.

Two facility approaches can name almost the same standing space. Distinct slot
IDs do not make them simultaneously usable. Cache neighbourhoods once, then
check both reservations and the last accepted pedestrian observation.
"""

import math

import numpy as np
from scipy.spatial import cKDTree


class SpatialActivityCapacity:
    def __init__(self, places, *, spacing_m):
        if not math.isfinite(spacing_m) or spacing_m <= 0:
            raise ValueError("Positive physical slot spacing required")
        self.places = {p.id: p for p in places}
        if len(self.places) != len(places):
            raise ValueError("Unique activity places required")
        standing = sorted((p for p in places if p.kind != "berth"), key=lambda p: p.id)
        points = np.array([p.xy for p in standing])
        tree = cKDTree(points) if len(points) else None
        self.neighbours = {}
        for i, p in enumerate(standing):
            self.neighbours[p.id] = tuple(
                standing[j]
                for j in tree.query_ball_point(points[i], spacing_m)
                if np.linalg.norm(points[i] - points[j]) < spacing_m
            )
        self.stats = dict(
            reservation_rejections=0,
            physical_rejections=0,
            observation_updates=0,
            close_place_pairs=sum(len(v) - 1 for v in self.neighbours.values()) // 2,
        )
        self.names = ()
        self.xy = np.empty((0, 2))
        self.radii = np.empty(0)
        self.active = np.empty(0, dtype=bool)
        self.tree = None

    def observe(self, names, xy, radii, active):
        xy = np.asarray(xy, float)
        radii = np.asarray(radii, float)
        active = np.asarray(active, bool)
        if (
            xy.shape != (len(names), 2)
            or radii.shape != (len(names),)
            or active.shape != (len(names),)
            or len(set(names)) != len(names)
            or not np.isfinite(xy).all()
            or not np.isfinite(radii).all()
            or np.any(radii <= 0)
        ):
            raise ValueError("Complete finite pedestrian observation required")
        self.names = tuple(names)
        self.xy = xy.copy()
        self.radii = radii.copy()
        self.active = active.copy()
        self.index = {name: i for i, name in enumerate(names)}
        self.tree = cKDTree(self.xy) if len(xy) else None
        self.stats["observation_updates"] += 1

    def available(self, director, place, actor):
        if place.id not in self.places or place != self.places[place.id]:
            raise ValueError("Capacity map changed")
        if place.kind == "berth":
            return True  # Vehicle admission owns its geometry.
        for neighbour in self.neighbours[place.id]:
            if any(director.holders.get(slot) not in (None, actor) for slot in neighbour.slots):
                self.stats["reservation_rejections"] += 1
                return False
        if self.tree is not None and actor in self.index:
            i = self.index[actor]
            reach = float(self.radii[i] + self.radii.max() + 0.15)
            for j in self.tree.query_ball_point(place.xy, reach):
                if (
                    j != i
                    and self.active[j]
                    and np.linalg.norm(self.xy[j] - place.xy) <= self.radii[i] + self.radii[j] + 0.15
                ):
                    self.stats["physical_rejections"] += 1
                    return False
        return True
