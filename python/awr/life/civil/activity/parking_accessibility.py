"""Reject parking envelopes that occupy an existing resident's home approach.

This is a conservative necessary condition. Clear home points do not prove a
connected sidewalk, facility access or a physically executable parking route.
"""

import math

import numpy as np
from shapely.geometry import Point

from ..curb_allocation import footprint


def home_clearance_conflicts(nodes, size, places, turn_radius_m):
    size = np.asarray(size, float)
    if (
        size.shape != (3,)
        or not np.isfinite(size).all()
        or np.any(size <= 0)
        or not math.isfinite(turn_radius_m)
        or turn_radius_m <= 0
    ):
        raise ValueError("Finite full vehicle and resident turning envelopes required")
    homes = sorted((p for p in places if p.kind == "home"), key=lambda p: p.id)
    if len({h.id for h in homes}) != len(homes):
        raise ValueError("Unique existing homes required")
    rows = []
    for name, node in sorted(nodes.items()):
        if node["kind"] != "berth":
            continue
        pose = np.asarray(node["pose"], float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("Finite berth pose required")
        # The walking body already includes its own safety margin. Do not add
        # the driving planner's default 0.4m road-traffic margin a second time.
        polygon = footprint(dict(xy=pose[:2], yaw=pose[2]), size, margin_m=0.0)
        for home in homes:
            distance = float(polygon.distance(Point(home.xy)))
            if distance <= turn_radius_m:
                rows.append(
                    dict(
                        berth=name,
                        home=home.id,
                        home_xy=list(home.xy),
                        body_distance_m=distance,
                        required_turn_clearance_m=turn_radius_m,
                        reason="parked_body_blocks_reserved_home_approach",
                    )
                )
    return rows
