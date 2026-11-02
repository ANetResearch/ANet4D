"""Explicit scenario convention for short manoeuvres at legal two-way curbs.

This declares a bounded candidate domain, not surveyed free space or a legal
claim about local traffic law. Native terrain, whole-body and occupancy gates
still apply. The original whole-road resource remains exclusive throughout.
"""

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import substring

POLICY = "bounded_two_way_curb_v1"
REVERSE_LIMIT_M = 6.0
REVERSE_SPEED_MPS = 1.0


def domain(node, road):
    if node.get("curb_maneuver_policy") != POLICY:
        raise ValueError("Unknown explicit curb policy")
    if road.get("oneway") is not False or node.get("parking_legality", {}).get("legal") is not True:
        return None
    line = LineString(road["line"])
    width = float(road["width_m"])
    if not np.isfinite(width) or width <= 0 or not line.is_valid or line.length <= 20:
        return None
    point = Point(node["pose"][:2])
    station = line.project(point)
    low = max(10.0, station - 10.0)
    high = min(line.length - 10.0, station + 10.0)
    if high <= low:
        return None
    area = substring(line, low, high).buffer(width / 2, cap_style=2)
    area = area.intersection(line.buffer(width / 2, cap_style=2))
    return area if area.covers(point) else None


def reverse_within_limit(maneuver):
    breaks = np.r_[0, np.flatnonzero(np.diff(maneuver.gear)) + 1, len(maneuver.gear)]
    return all(
        maneuver.gear[a] != -1 or maneuver.arc[b] - maneuver.arc[a] <= REVERSE_LIMIT_M + 1e-8
        for a, b in zip(breaks[:-1], breaks[1:], strict=False)
    )
