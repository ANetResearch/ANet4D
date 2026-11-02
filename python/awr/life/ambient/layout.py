"""Parked cars of the stage that queries count and read: stalls of a lot row and kerb places of a street.

A parked car stands for the whole recording. Its body (catalogue box) and tyre contacts must lie on vehicle support,
its body keeps LANE_CLEAR_M from the car lanes of the stage network and PARKED_GAP_M from the other parked cars, and
the people's walkway is cut around it (awr.life.ambient.walkway).
"""

from __future__ import annotations

import json

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, box

# Tyre contacts (body frame, m) of the compiled vehicle assets, from awr.life.civil.contacts.vehicle_contacts.
CONTACTS = {
    "sedan": [
        [1.549, 0.0],
        [-1.549, 0.0],
        [0.0, 0.807],
        [0.0, -0.807],
        [-1.387, -0.81],
        [-1.387, 0.81],
        [1.416, -0.81],
        [1.416, 0.81],
    ],
    "hatchback": [
        [1.365, 0.0],
        [-1.365, 0.0],
        [0.0, 0.783],
        [0.0, -0.783],
        [-1.337, -0.775],
        [-1.337, 0.775],
        [1.246, -0.775],
        [1.246, 0.775],
    ],
    "suv": [
        [1.523, 0.0],
        [-1.523, 0.0],
        [0.0, 0.827],
        [0.0, -0.827],
        [-1.496, -0.84],
        [-1.496, 0.84],
        [1.393, -0.84],
        [1.393, 0.84],
    ],
    "van": [
        [1.825, 0.0],
        [-1.825, 0.0],
        [0.0, 0.971],
        [0.0, -0.971],
        [-1.699, -0.907],
        [-1.699, 0.907],
        [1.701, -0.907],
        [1.701, 0.907],
    ],
    "pickup": [
        [1.821, 0.0],
        [-1.821, 0.0],
        [0.0, 0.858],
        [0.0, -0.858],
        [-1.761, -0.885],
        [-1.761, 0.885],
        [1.827, -0.885],
        [1.827, 0.885],
    ],
}
PARKED_GAP_M = 0.6
LANE_CLEAR_M = 0.5  # from a lane centre line to a parked body, beyond a moving car's half width (1.12 m)
CAR_HALF_WIDTH_M = 1.12
KERB_SPACING_M = 6.5  # between the middles of kerb-parked cars


def body(xy, yaw, size):
    """The body rectangle of a car."""
    r = box(-size[0] / 2, -size[1] / 2, size[0] / 2, size[1] / 2)
    return affinity.translate(affinity.rotate(r, yaw, use_radians=True, origin=(0, 0)), *xy)


def supported(xy, yaw, size, vtype, support):
    """Corners, side middles and tyre contacts of the body on vehicle support."""
    L, W = size[0] / 2, size[1] / 2
    pts = np.array([[-L, -W], [L, -W], [L, W], [-L, W], [0, -W], [0, W], [-L, 0], [L, 0]] + list(CONTACTS[vtype]))
    c, s = np.cos(yaw), np.sin(yaw)
    world = np.asarray(xy) + pts @ np.array([[c, s], [-s, c]])
    return bool(support.supported(world, "vehicle").all())


def lane_lines(stage_geometry):
    """Centre lines of the car lanes (edges and junction connections) of a compiled stage."""
    geo = json.loads(open(stage_geometry).read()) if isinstance(stage_geometry, str) else stage_geometry
    lines = [LineString(p) for p in geo["edges"].values() if len(p) > 1]
    return lines + [LineString(c["points"]) for c in geo["connections"] if len(c["points"]) > 1]


def clear_of_lanes(xy, yaw, size, lanes):
    b = body(xy, yaw, size)
    return all(l.distance(b) >= CAR_HALF_WIDTH_M + LANE_CLEAR_M for l in lanes)


def row_stalls(row, support, sizes, lanes, *, shifts=np.arange(-1.5, 1.51, 0.25)):
    """Stalls of a lot row (a row of the semantics' verified parking regions: start, end, heading_rad, spacing_m):
    the middle of each stall along the row, shifted along the car heading by the shift that holds most stalls for
    every vehicle type of `sizes` (vtype -> size)."""
    a, b = np.array(row["start"]), np.array(row["end"])
    u = (b - a) / np.hypot(*(b - a))
    yaw = float(row["heading_rad"])
    h = np.array([np.cos(yaw), np.sin(yaw)])
    n = int(np.hypot(*(b - a)) // row["spacing_m"]) + 1
    best = []
    for shift in shifts:
        got = []
        for k in range(n):
            xy = a + u * k * row["spacing_m"] + h * shift
            if all(
                supported(xy, yaw, size, vt, support) and clear_of_lanes(xy, yaw, size, lanes)
                for vt, size in sizes.items()
            ):
                got.append((k, xy))
        if len(got) > len(best):
            best = got
    return [dict(stall=k, xy=[float(v) for v in xy], yaw=yaw) for k, xy in best]


def kerb_places(line, support, size, vtype, lanes, *, side=1, offsets=np.arange(1.0, 6.01, 0.25), step=0.5):
    """Places along a street centre line (shapely LineString) for a car parallel to the street, on the left of the
    line's direction (side +1) or its right (-1): at each station the offset furthest out that holds the body, as
    (arc, xy, yaw)."""
    out = []
    for s in np.arange(size[0] / 2, line.length - size[0] / 2, step):
        p = np.array(line.interpolate(s).coords[0])
        q = np.array(line.interpolate(min(s + 0.5, line.length)).coords[0])
        t = (q - p) / max(np.hypot(*(q - p)), 1e-9)
        n = np.array([-t[1], t[0]]) * side
        yaw = float(np.arctan2(t[1], t[0]))
        for off in offsets[::-1]:
            xy = p + n * off
            if supported(xy, yaw, size, vtype, support) and clear_of_lanes(xy, yaw, size, lanes):
                out.append((float(s), xy, yaw))
                break
    return out
