"""Bounded offline forward-car search through a measured local corridor.

Analytic two-arc connections miss paths that need several alternating bends.
Search uses exact continuous poses and finite-radius forward primitives; a
quantized pose key only bounds exploration, never snaps emitted geometry.
Every primitive and final connection is admitted against actual support,
semantic swept area and supplied parked bodies. Failure is not an impossibility
proof. Traffic permissions/reservations remain the caller's responsibility.
"""

import heapq
import math
from collections import Counter

import numpy as np

from .driving_graph import route_arrays
from .vehicle_connections import PreparedPermission, admit, angle_delta, candidates


def primitive(pose, curvature, length=1.5, spacing=0.25, *, gear=1):
    pose = np.asarray(pose, float)
    if (
        pose.shape != (3,)
        or not np.isfinite(np.r_[pose, curvature, length, spacing]).all()
        or min(length, spacing) <= 0
        or gear not in (-1, 1)
    ):
        raise ValueError("Invalid driving primitive")
    s = np.linspace(0.0, length, max(1, math.ceil(length / spacing)) + 1)
    yaw = pose[2] + gear * curvature * s
    if abs(curvature) < 1e-12:
        points = pose[:2] + gear * s[:, None] * np.array([math.cos(pose[2]), math.sin(pose[2])])
    else:
        points = (
            pose[:2] + np.column_stack([np.sin(yaw) - math.sin(pose[2]), math.cos(pose[2]) - np.cos(yaw)]) / curvature
        )
    return route_arrays(points, yaw, np.full(len(points) - 1, gear))


def connect(
    surface,
    start,
    goal,
    size,
    *,
    allowed_area,
    obstacles=(),
    max_expansions=2000,
    max_length_m=100.0,
    resolution_m=0.5,
    heading_bins=72,
    primitive_length_m=1.5,
    guide_points=None,
):
    """Return (route or None, counters), without wall-time-dependent decisions.

    Optional static navigation guidance changes only search ordering. It is not
    an admissible shortest-path heuristic, motion geometry or a corridor permit:
    every emitted primitive, join and final curve still passes the original
    body, support, steering and obstacle checks. Budget pruning uses Euclidean
    distance, not guide length; a detouring guide cannot falsely prune a short
    physical connection. The guide must use the exact requested XY endpoints.
    """
    start, goal, size = map(lambda x: np.asarray(x, float), (start, goal, size))
    if (
        start.shape != (3,)
        or goal.shape != (3,)
        or size.shape != (3,)
        or not np.isfinite(np.r_[start, goal, size, max_length_m, resolution_m, primitive_length_m]).all()
        or np.any(size <= 0)
        or min(max_length_m, resolution_m, primitive_length_m) <= 0
        or not isinstance(max_expansions, int)
        or max_expansions < 1
        or not isinstance(heading_bins, int)
        or heading_bins < 8
        or allowed_area is None
        or allowed_area.is_empty
    ):
        raise ValueError("Invalid measured driving search request")
    counts = Counter()
    guide = None if guide_points is None else RouteGuide(guide_points, start[:2], goal[:2])

    def heuristic(pose):
        direct = float(np.linalg.norm(goal[:2] - pose[:2]))
        if guide is None:
            return direct
        counts["guide_queries"] += 1
        return max(direct, guide.remaining(pose[:2]))

    if np.linalg.norm(start[:2] - goal[:2]) <= 1e-7 and abs(angle_delta(start[2], goal[2])) <= 1e-7:
        return None, {"already_at_goal": 1}
    # No route can repair an inadmissible endpoint without moving it. Reject
    # these before expanding thousands of poses, using the same physical body,
    # obstacle clearance and ground contacts as route admission.
    from .curb_allocation import footprint
    from .grounded import support_mask

    permitted = PreparedPermission(allowed_area)
    obstacles = tuple(obstacles)
    for name, pose in (("start", start), ("goal", goal)):
        body = footprint(dict(xy=pose[:2], yaw=pose[2]), size)
        if not permitted.covers(footprint(dict(xy=pose[:2], yaw=pose[2]), size, margin_m=0.0)):
            return None, {name + "_outside_permitted_area": 1}
        if any(body.intersects(obstacle) for obstacle in obstacles):
            return None, {name + "_static_obstacle": 1}
        if not support_mask(surface, pose[None, :2], pose[None, 2], size, "vehicle")[0]:
            return None, {name + "_unsupported": 1}
    radius = max(4.0, size[0] * 0.65 / math.tan(math.radians(32.0)))
    turns = (0.0, -0.5 / radius, 0.5 / radius, -1.0 / radius, 1.0 / radius)

    def key(pose):
        return (
            *np.rint((pose[:2] - start[:2]) / resolution_m).astype(int),
            int(round((pose[2] % (2 * math.pi)) * heading_bins / (2 * math.pi))) % heading_bins,
        )

    poses = [start]
    parents = [-1]
    legs = [None]
    costs = [0.0]
    best = {key(start): 0.0}
    queue = [(heuristic(start), 0)]
    # A local road adapter must supply an appropriate domain. A broad unrelated
    # parking area is never silently used as a road shortcut by this function.
    while queue and counts["expanded"] < max_expansions:
        _, index = heapq.heappop(queue)
        pose, cost = poses[index], costs[index]
        if cost > best.get(key(pose), math.inf) + 1e-10:
            continue
        counts["expanded"] += 1
        distance = float(np.linalg.norm(goal[:2] - pose[:2]))
        if distance <= 12.0 or index == 0:
            exact = index > 0 and distance <= 1e-7 and abs(angle_delta(pose[2], goal[2])) <= 1e-7
            remaining = min(max_length_m - cost, 2 * distance + 15.0)
            finishes = (
                [None] if exact else (candidates(pose, goal, size, max_length_m=remaining) if remaining > 1e-9 else [])
            )
            for finish in finishes:
                counts["finish_candidates"] += 1
                ok, reason = (
                    (True, None)
                    if finish is None
                    else admit(surface, finish, size, allowed_area=permitted, obstacles=obstacles)
                )
                if not ok:
                    counts["finish_" + reason] += 1
                    continue
                chain = [] if finish is None else [finish]
                cursor = index
                while parents[cursor] >= 0:
                    chain.append(legs[cursor])
                    cursor = parents[cursor]
                chain.reverse()
                pp = np.concatenate([r["points"][1 if i else 0 :] for i, r in enumerate(chain)])
                yy = np.concatenate([r["yaw"][1 if i else 0 :] for i, r in enumerate(chain)])
                route = route_arrays(pp, yy)
                # Endpoints/joins are never quantized to the search lattice.
                if np.linalg.norm(pp[-1] - goal[:2]) > 1e-6 or abs(angle_delta(yy[-1], goal[2])) > 1e-6:
                    raise RuntimeError("Forward search lost the exact goal pose")
                ok, reason = admit(surface, route, size, allowed_area=permitted, obstacles=obstacles)
                if not ok:
                    counts["assembled_" + reason] += 1
                    continue
                counts["accepted"] += 1
                route["construction"] = "bounded_forward_pose_search"
                return route, dict(counts)
        for turn in turns:
            leg = primitive(pose, turn, length=primitive_length_m)
            end = np.r_[leg["points"][-1], leg["yaw"][-1]]
            length = float(leg["arc"][-1])
            new_cost = cost + length
            if new_cost + np.linalg.norm(goal[:2] - end[:2]) > max_length_m:
                counts["length_budget"] += 1
                continue
            k = key(end)
            if new_cost >= best.get(k, math.inf) - 1e-10:
                continue
            counts["primitive_candidates"] += 1
            ok, reason = admit(surface, leg, size, allowed_area=permitted, obstacles=obstacles)
            if not ok:
                counts[reason] += 1
                continue
            best[k] = new_cost
            next_index = len(poses)
            poses.append(end)
            parents.append(index)
            legs.append(leg)
            costs.append(new_cost)
            heapq.heappush(queue, (new_cost + heuristic(end), next_index))
    counts["budget_exhausted" if queue else "search_exhausted"] += 1
    return None, dict(counts)


class RouteGuide:
    """Bounded polyline-distance priority from a static navigation backend.

    This lightweight object never alters allowed area or generated car poses.
    Recast-derived guides are typically a few straight-path corners, not a
    dense per-tick route; cap size to keep per-expansion overhead explicit.
    """

    def __init__(self, points, start, goal):
        from shapely.geometry import LineString

        p = np.asarray(points, float)
        if (
            p.ndim != 2
            or p.shape[1] != 2
            or not 2 <= len(p) <= 4096
            or not np.isfinite(p).all()
            or np.linalg.norm(p[0] - start) > 1e-6
            or np.linalg.norm(p[-1] - goal) > 1e-6
            or not LineString(p).is_simple
        ):
            raise ValueError("Finite simple guide with exact XY endpoints required (2..4096 points)")
        lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
        if np.any(lengths <= 1e-9):
            raise ValueError("Navigation guide contains duplicate points")
        self.starts = p[:-1].copy()
        self.delta = np.diff(p, axis=0)
        self.lengths = lengths
        self.after = np.cumsum(lengths[::-1])[::-1] - lengths

    def remaining(self, xy):
        u = np.clip(np.sum((xy - self.starts) * self.delta, axis=1) / self.lengths**2, 0.0, 1.0)
        distance = np.linalg.norm(self.starts + u[:, None] * self.delta - xy, axis=1)
        k = int(np.argmin(distance))
        return float(distance[k] + (1.0 - u[k]) * self.lengths[k] + self.after[k])
