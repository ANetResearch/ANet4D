"""Reconnect an observed walker to the conservative planning grid.

Local avoidance may legitimately leave the grid's extra inset while the whole
body remains on measured ground. Such a pose is not a valid A* source. Search a
bounded set of nearby grid entrances, checking the actual connector instead of
moving the person to a grid node or relaxing the planning mask. This runs only
on the throttled congestion-replanning path, never per body per tick.
"""

import math

import numpy as np
from shapely.geometry import MultiPoint, Polygon

from awr.life.civil.body import FULL_WALK_MPS
from awr.life.civil.grid_search import compiled_segment_check

from . import kernels as K

REACH_M = 3.0
MAX_NODES = 64
MAX_PLANS = 4
CLEARANCE_M = 0.03


def corners(xy, heading, speed, ext):
    x, y, hl, hw = K.box(*xy, heading, speed, ext)
    c, s = math.cos(heading), math.sin(heading)
    return np.array(
        [(x + a * hl * c - b * hw * s, y + a * hl * s + b * hw * c) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    )


def connector_clear(crowd, i, a, b, heading, obstacles):
    """Rest rotation followed by a supported, full-body straight connector.

    Translation/body clearance is a swept convex hull; support uses the complete
    rectangle translation. Rotation is sampled for planning only; Crowd still
    owns every realised turn and its mandatory per-tick checks.
    """
    delta = b - a
    if np.linalg.norm(delta) < 1e-6:
        return False
    direction = math.atan2(delta[1], delta[0])
    check = compiled_segment_check()
    # Exact cell-boundary traversal on the controller's legal-centre mask,
    # never on a newly invented or widened walking surface.
    if check is None:
        return False
    q = (a - np.array([crowd.fx0, crowd.fy0])) / crowd.fstep
    end = (b - np.array([crowd.fx0, crowd.fy0])) / crowd.fstep
    if not check(*q, *end, crowd.free, np.empty((0, 0), np.float32), math.inf):
        return False
    turn = K.wrap(direction - heading)
    for u in np.linspace(0.0, 1.0, max(2, int(math.ceil(abs(turn) / 0.05)) + 1)):
        h = heading + turn * u
        if not K.body_supported(*a, h, 0.0, crowd.ext[i], crowd.z_ok, crowd.walk, crowd.sx0, crowd.sy0, crowd.scell):
            return False
        shape = Polygon(corners(a, h, 0.0, crowd.ext[i]))
        if any(shape.distance(obstacle) < CLEARANCE_M for obstacle in obstacles):
            return False
    for velocity in (0.0, FULL_WALK_MPS):
        if not K.translated_body_supported(
            a, b, direction, velocity, crowd.ext[i], crowd.z_ok, crowd.walk, crowd.sx0, crowd.sy0, crowd.scell
        ):
            return False
        swept = MultiPoint(
            np.vstack([corners(a, direction, velocity, crowd.ext[i]), corners(b, direction, velocity, crowd.ext[i])])
        ).convex_hull
        if any(swept.distance(obstacle) < CLEARANCE_M for obstacle in obstacles):
            return False
    return True


def reconnect(router, crowd, i, goal, xy, yaw, speed, active, avoid):
    """Return a normal walking route from the current pose, or None.

    All observed neighbours (including moving people and parked cars) participate
    in connector clearance. Future motion still needs the shared controller. At
    most four graph queries are allowed; failed candidates cannot flood A*.
    """
    if not active[i] or crowd.kinds[i] != "person" or abs(speed[i]) > 0.05:
        return None
    network = router.network
    dist, ids = network.tree.query(xy[i], k=min(MAX_NODES, len(network.xy)))
    reach = REACH_M + crowd.radius[i] + crowd.radius
    near = np.flatnonzero(active & (np.arange(len(xy)) != i) & (np.linalg.norm(xy - xy[i], axis=1) <= reach))
    obstacles = [Polygon(corners(xy[j], yaw[j], abs(speed[j]), crowd.ext[j])) for j in near]
    queries = 0
    for distance, node in zip(np.atleast_1d(dist), np.atleast_1d(ids), strict=False):
        if distance > REACH_M:
            break
        target = network.xy[node]
        if not router.free[tuple(network.nodes[node])]:
            continue
        if not connector_clear(crowd, i, xy[i], target, yaw[i], obstacles):
            continue
        queries += 1
        route = router.plan(target, goal, avoid=avoid)
        if route is not None:
            points = np.vstack([xy[i], route["points"]])
            return dict(
                route, points=points, arc=np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            )
        if queries >= MAX_PLANS:
            break
    return None
