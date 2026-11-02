"""Parking legality from verified geometry, applied only to intentional stops.

A vehicle that is queueing, braking for a pedestrian or merely stationary is not
parked. The simulation emits ``park`` events only with parking intent (initial
roadside position before departure, delivery stop, end of trip). This module
evaluates such a pose against geometry that exists in the inputs:

* SUMO-derived crossings (``semantics["crossings"]``, physical paint not
  surveyed): no parking within ``crossing_margin_m`` of a crossing;
* SUMO junction areas (priority / right-before-left nodes);
* SUMO walking areas and pedestrian lanes (sidewalks);
* carriageway obstruction: less than ``min_free_width_m`` of driving surface
  remains beside the vehicle on its cross-section.

Rules are a scenario convention, not a legal opinion about the surveyed town.
"""

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from awr.life.civil.collision import rectangle


class ParkingRules:
    def __init__(self, semantics, lane_network, crossing_margin_m=5.0, min_free_width_m=3.0):
        self.crossing_margin_m = float(crossing_margin_m)
        self.min_free_width_m = float(min_free_width_m)
        crossings = [
            LineString(c["ends"]).buffer(c["width_m"] / 2 + self.crossing_margin_m)
            for c in semantics.get("crossings", [])
            if len(c.get("ends", [])) >= 2
        ]
        self.crossings = unary_union(crossings) if crossings else Polygon()
        junctions = [j["polygon"] for j in getattr(lane_network, "junctions", [])]
        self.junctions = unary_union(junctions) if junctions else Polygon()
        self.walking = lane_network.walking
        self.driving = lane_network.driving
        for geom in (self.crossings, self.junctions, self.walking, self.driving):
            shapely.prepare(geom)

    def describe(self):
        return dict(
            schema="awr.life.parking_rules.v1",
            crossing_margin_m=self.crossing_margin_m,
            min_free_width_m=self.min_free_width_m,
            rules=["within_crossing_margin", "in_junction", "on_sidewalk", "obstructs_carriageway"],
            geometry=dict(
                crossings_m2=round(float(self.crossings.area), 1),
                junctions_m2=round(float(self.junctions.area), 1),
                walking_m2=round(float(self.walking.area), 1),
                driving_m2=round(float(self.driving.area), 1),
            ),
            note="Scenario convention over SUMO-derived geometry; crossings are not surveyed paint.",
        )

    def free_width(self, xy, yaw, size):
        """Driving surface remaining on the cross-section through the vehicle centre."""
        xy = np.asarray(xy, float)
        normal = np.array([-np.sin(yaw), np.cos(yaw)])
        section = LineString([xy - normal * 15.0, xy + normal * 15.0])
        carriageway = section.intersection(self.driving)
        if carriageway.is_empty:
            return 0.0, 0.0
        # Passing room must be contiguous on one side of the vehicle; a car in
        # the middle of a narrow street leaves no usable lane on either side.
        remaining = carriageway.difference(Polygon(rectangle(xy, yaw, size)).buffer(0.05))
        parts = getattr(remaining, "geoms", [remaining])
        free = max((float(g.length) for g in parts if not g.is_empty), default=0.0)
        return float(carriageway.length), free

    def evaluate(self, xy, yaw, size):
        foot = Polygon(rectangle(np.asarray(xy, float), float(yaw), np.asarray(size, float)))
        reasons = []
        if foot.intersects(self.crossings):
            reasons.append("within_crossing_margin")
        if foot.intersects(self.junctions):
            reasons.append("in_junction")
        if foot.intersection(self.walking).area > 0.2 * foot.area:
            reasons.append("on_sidewalk")
        width, free = self.free_width(xy, yaw, size)
        on_carriageway = foot.intersection(self.driving).area > 0.5 * foot.area
        if on_carriageway and free < self.min_free_width_m:
            reasons.append("obstructs_carriageway")
        return dict(
            legal=not reasons,
            reasons=reasons,
            where="carriageway" if on_carriageway else "roadside",
            carriageway_width_m=round(width, 2),
            free_width_m=round(free, 2),
            slot=None,
        )
