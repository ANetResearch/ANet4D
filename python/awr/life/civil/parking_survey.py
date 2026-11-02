"""Conservative off-road lot geometry for offline connection exploration.

This is explicitly not a substitute for SUMO sidewalk/junction parking rules.
It can reject known conflicts without certifying the absence of unmapped ones.
The candidate must still pass body support, aisle and entrance checks separately.
"""

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from awr.life.civil.collision import rectangle


class SurveyedLotGeometry:
    def __init__(self, semantics, crossing_margin_m=5.0):
        margin = float(crossing_margin_m)
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("Crossing margin must be finite and nonnegative")
        lots = semantics.get("verified_parking_regions", [])
        if not lots or any(l.get("verified") is not True or not l.get("evidence") for l in lots):
            raise ValueError("Explicit verified parking regions and evidence are required")
        self.lots = [Polygon(l["polygon"]) for l in lots]
        if any(p.is_empty or not p.is_valid or p.area <= 0 for p in self.lots):
            raise ValueError("Invalid parking region")
        self.exclusions = {
            "mapped_carriageway": unary_union(
                [LineString(r["line"]).buffer(r["width_m"] / 2, cap_style=2) for r in semantics.get("roads", [])]
            ),
            "mapped_pedestrian_path": unary_union(
                [LineString(r["line"]).buffer(r["width_m"] / 2, cap_style=2) for r in semantics.get("paths", [])]
            ),
            "mapped_building": unary_union([Polygon(r["polygon"]) for r in semantics.get("buildings", [])]),
            "within_crossing_margin": unary_union(
                [
                    LineString(r["ends"]).buffer(r["width_m"] / 2 + margin)
                    for r in semantics.get("crossings", [])
                    if len(r.get("ends", [])) >= 2
                ]
            ),
        }

    def evaluate(self, xy, yaw, size):
        body = Polygon(rectangle(np.asarray(xy, float), float(yaw), np.asarray(size, float)))
        reasons = []
        if not any(lot.covers(body) for lot in self.lots):
            reasons.append("outside_verified_parking_region")
        reasons += [name for name, area in self.exclusions.items() if body.intersects(area)]
        # `legal` is the existing candidate-filter interface. Consumers must
        # retain the explicit unverified status instead of treating it as release.
        return dict(
            legal=not reasons,
            reasons=reasons,
            where="parking_lot",
            slot=None,
            verification="offline_geometry_candidate_only",
            parking_legality_verified=False,
        )

    def describe(self):
        return dict(
            schema="awr.life.parking_survey_geometry.v1",
            parking_legality_verified=False,
            release_evidence=False,
            limitations=[
                "SUMO sidewalk and junction geometry unavailable.",
                "Only explicitly mapped off-road lots are considered.",
                "Missing restrictions are unknown, not certified absent.",
            ],
        )
