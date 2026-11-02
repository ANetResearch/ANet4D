"""Finite, deterministic car-connection candidates on measured support.

Two constant-curvature arcs and their common tangent propose forward motion.
Short straight reverse departures/arrivals permit parking manoeuvres; every
gear change is an explicit required stop. Candidates are not a navigation
oracle: collision, permission, support and traffic admission remain mandatory.
Failure of this bounded family is not proof that no physical route exists.
"""

from __future__ import annotations

import math

import numpy as np


def direction(yaw):
    return np.array([math.cos(yaw), math.sin(yaw)])


def angle_delta(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def _arc(point, yaw, turn, radius, angle, spacing):
    length = radius * angle
    s = np.linspace(0.0, length, max(1, math.ceil(length / spacing)) + 1)
    angles = yaw + turn * s / radius
    points = np.asarray(point) + radius / turn * np.column_stack(
        [np.sin(angles) - math.sin(yaw), math.cos(yaw) - np.cos(angles)]
    )
    return points, angles


# Shared by curve construction and persisted graph loading. This rejects
# zero-length samples; it is not a parking or vehicle-motion accuracy target.
MIN_ROUTE_STEP_M = 1e-10


def _straight(a, b, yaw, spacing):
    points = np.linspace(a, b, max(1, math.ceil(np.linalg.norm(b - a) / spacing)) + 1)
    return points, np.full(len(points), yaw)


def _join(pieces):
    points, yaw, gears = [], [], []
    for pp, yy, gear in pieces:
        if np.linalg.norm(pp[-1] - pp[0]) < 1e-10 and np.ptp(yy) < 1e-10:
            continue
        if points:
            if np.linalg.norm(points[-1] - pp[0]) > 1e-7 or abs(angle_delta(yaw[-1], yy[0])) > 1e-7:
                raise ValueError("Disconnected pose or heading at manoeuvre join")
            points.extend(pp[1:])
            yaw.extend(yy[1:])
        else:
            points.extend(pp)
            yaw.extend(yy)
        gears.extend([gear] * (len(pp) - 1))
    if len(points) < 2:
        return None
    points = np.asarray(points)
    yaw = np.unwrap(yaw)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(distances <= MIN_ROUTE_STEP_M):
        raise ValueError("Zero-distance pose change in manoeuvre")
    arc = np.r_[0.0, np.cumsum(distances)]
    gear = np.array(gears, dtype=np.int8)
    changes = np.flatnonzero(np.diff(gear)) + 1
    return dict(
        points=points, yaw=yaw, arc=arc, gear=gear, required_stops_m=arc[changes].tolist(), smooth_vehicle_heading=True
    )


def tangent_paths(start, goal, radius, *, spacing=0.25, max_length_m=None):
    """All oriented common-tangent (CSC) candidates, without environment claims.

    The circle centres are offset to each pose's left/right. At a tangent
    heading t with left normal n, contact = centre - turn*radius*n.
    Thus n dot (c2-c1) = (turn2-turn1)*radius. Both solutions are inspected;
    only the one with positive forward tangent length is admitted.
    """
    p0, p1 = np.asarray(start[:2], float), np.asarray(goal[:2], float)
    y0, y1 = float(start[2]), float(goal[2])
    if not np.isfinite(np.r_[p0, p1, y0, y1, radius, spacing]).all() or min(radius, spacing) <= 0:
        raise ValueError("Invalid manoeuvre geometry")
    limit = math.inf if max_length_m is None else float(max_length_m)
    if max_length_m is not None and (not math.isfinite(limit) or limit <= 0):
        raise ValueError("Positive finite candidate length budget required")
    if np.linalg.norm(p1 - p0) > limit + 1e-8:
        return []
    n0, n1 = direction(y0 + math.pi / 2), direction(y1 + math.pi / 2)
    out = []
    # Straight travel also handles coincident equal-radius centres.
    delta = p1 - p0
    if np.linalg.norm(delta) > 1e-8 and abs(angle_delta(y1, y0)) < 1e-8:
        tangent = direction(y0)
        if abs(tangent[0] * delta[1] - tangent[1] * delta[0]) < 1e-8 and np.dot(tangent, delta) > 0:
            route = _join([(*_straight(p0, p1, y0, spacing), 1)])
            route["construction"] = "straight"
            out.append(route)
    for first in (-1, 1):
        for last in (-1, 1):
            c0, c1 = p0 + first * radius * n0, p1 + last * radius * n1
            d = c1 - c0
            length = np.linalg.norm(d)
            if length < 1e-8:
                # A single same-turn circle can join coincident circle centres.
                if first == last:
                    angle = (first * (y1 - y0)) % (2 * math.pi)
                    if radius * angle > limit + 1e-8:
                        continue
                    pp, yy = _arc(p0, y0, first, radius, angle, spacing)
                    if np.linalg.norm(pp[-1] - p1) < 1e-7:
                        route = _join([(pp, yy, 1)])
                        if route:
                            route["construction"] = "single_arc"
                            out.append(route)
                continue
            ratio = (last - first) * radius / length
            if abs(ratio) > 1.0:
                continue
            base = math.atan2(d[1], d[0])
            offset = math.acos(ratio)
            for normal_yaw in (base - offset, base + offset):
                normal = direction(normal_yaw)
                tangent = direction(normal_yaw - math.pi / 2)
                a, b = c0 - first * radius * normal, c1 - last * radius * normal
                if np.dot(b - a, tangent) < -1e-8:
                    continue
                tangent_yaw = math.atan2(tangent[1], tangent[0])
                angle0 = (first * (tangent_yaw - y0)) % (2 * math.pi)
                angle1 = (last * (y1 - tangent_yaw)) % (2 * math.pi)
                # Round-off at an exact tangent must not insert a full circle.
                if abs(angle_delta(tangent_yaw, y0)) < 1e-10:
                    angle0 = 0.0
                if abs(angle_delta(y1, tangent_yaw)) < 1e-10:
                    angle1 = 0.0
                # Reject analytically before allocating sampled circles. A
                # gentle turn may have a huge fitted radius; its unused CSC
                # alternatives can contain many kilometres of full circles.
                if radius * (angle0 + angle1) + np.linalg.norm(b - a) > limit + 1e-8:
                    continue
                pieces = [
                    (*_arc(p0, y0, first, radius, angle0, spacing), 1),
                    (*_straight(a, b, tangent_yaw, spacing), 1),
                    (*_arc(b, tangent_yaw, last, radius, angle1, spacing), 1),
                ]
                route = _join(pieces)
                if route:
                    route["construction"] = f"CSC:{first}:{last}"
                    out.append(route)
    return sorted(out, key=lambda r: float(r["arc"][-1]))


def three_point_path(start, goal, radius, *, spacing=0.25):
    """Symmetric forward/reverse/forward return between opposite parallel poses.

    This family ends at the same longitudinal station. For heading changes
    (a, pi-2a, a), lateral displacement is 2R(1-2cos(a)); all pose changes
    therefore follow finite-radius motion rather than an in-place rotation.
    """
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    heading = direction(start[2])
    normal = direction(start[2] + math.pi / 2)
    delta = goal[:2] - start[:2]
    lateral = float(np.dot(delta, normal))
    if (
        abs(float(np.dot(delta, heading))) > 1e-7
        or abs(abs(angle_delta(goal[2], start[2])) - math.pi) > 1e-7
        or abs(lateral) >= 2 * radius
    ):
        return None
    side = 1 if lateral >= 0 else -1
    a = math.acos(0.5 - abs(lateral) / (4 * radius))
    p, yaw = start[:2], float(start[2])
    pieces = []
    for angle, gear in ((a, 1), (math.pi - 2 * a, -1), (a, 1)):
        pp, yy = _arc(p, yaw, side, radius, angle, spacing)
        if gear == -1:
            pp = 2 * p - pp
        pieces.append((pp, yy, gear))
        p, yaw = pp[-1], float(yy[-1])
    route = _join(pieces)
    route["construction"] = "three_point_return"
    return route


def candidates(
    start,
    goal,
    size,
    *,
    reverse_departure_m=(0.0,),
    reverse_arrival_m=(0.0,),
    allow_three_point=False,
    max_length_m=100.0,
    spacing=0.25,
):
    """Propose short connections, preserving exact endpoint positions/headings.

    Reversing is explicitly enabled by the map adapter only in parking areas.
    No turnaround is enabled just because two road endpoints are close.
    """
    start, goal, size = np.asarray(start, float), np.asarray(goal, float), np.asarray(size, float)
    options = tuple(reverse_departure_m), tuple(reverse_arrival_m)
    if (
        start.shape != (3,)
        or goal.shape != (3,)
        or size.shape != (3,)
        or not np.isfinite(np.r_[start, goal, size, max_length_m, spacing, *options[0], *options[1]]).all()
        or np.any(size <= 0)
        or min(max_length_m, spacing) <= 0
        or any(v < 0 for values in options for v in values)
    ):
        raise ValueError("Invalid car connection request")
    # Margin below the existing 35-degree limit avoids boundary rounding.
    radius = max(1 / 0.25, float(size[0]) * 0.65 / math.tan(math.radians(32.0)))
    result = []
    if allow_three_point:
        for scale in (1.0, 1.25, 1.5):
            turn = three_point_path(start, goal, radius * scale, spacing=spacing)
            if turn is not None and turn["arc"][-1] <= max_length_m:
                result.append(turn)
    for departure in options[0]:
        a = start.copy()
        a[:2] -= departure * direction(a[2])
        for arrival in options[1]:
            b = goal.copy()
            b[:2] += arrival * direction(b[2])
            # All-reverse straight parking is a separate finite candidate.
            if departure == 0 and arrival == 0 and abs(angle_delta(start[2], goal[2])) < 1e-8:
                delta = goal[:2] - start[:2]
                heading = direction(start[2])
                if abs(heading[0] * delta[1] - heading[1] * delta[0]) < 1e-8 and np.dot(delta, heading) < 0:
                    # This family is available only if the caller explicitly
                    # allowed a reverse option; road-only requests stay forward.
                    if np.linalg.norm(delta) <= max((*options[0], *options[1]), default=0.0):
                        reverse = _join([(*_straight(start[:2], goal[:2], start[2], spacing), -1)])
                        if reverse["arc"][-1] <= max_length_m:
                            reverse["construction"] = "reverse_straight"
                            result.append(reverse)
            for scale in (1.0, 1.25, 1.5):
                for middle in tangent_paths(a, b, radius * scale, spacing=spacing):
                    if departure + arrival + middle["arc"][-1] > max_length_m:
                        continue
                    route = _join(
                        [
                            (*_straight(start[:2], a[:2], start[2], spacing), -1),
                            (middle["points"], middle["yaw"], 1),
                            (*_straight(b[:2], goal[:2], goal[2], spacing), -1),
                        ]
                    )
                    route["construction"] = middle["construction"]
                    result.append(route)
    # Shortest geometric candidates first; validation determines feasibility.
    return sorted(result, key=lambda r: (float(r["arc"][-1]), len(r["required_stops_m"])))


class PreparedPermission:
    """Reuse the identical permission tolerance for a batch of candidate curves.

    Buffering a town's entire road union for every search primitive is costly.
    The underlying geometry remains immutable; preparation only indexes it.
    """

    def __init__(self, area):
        import shapely

        if area is None or area.is_empty or not area.is_valid:
            raise ValueError("A valid nonempty permitted area is required")
        self.geometry = area.buffer(1e-7)
        shapely.prepare(self.geometry)
        self.is_empty = False  # DrivingRoutes accepts this validated edge permission.

    def covers(self, corridor):
        return self.geometry.covers(corridor)


def admit(surface, route, size, *, allowed_area=None, obstacles=()):
    """Same support/steering contract as curb trips, plus semantic swept area.

    The complete physical body must stay inside the permitted road/parking
    area. Its .4 m collision clearance is checked against obstacles, not against
    the semantic road boundary: a clear shoulder is not an occupied body.
    Ground support still checks its own padded footprint and actual tyres.
    Obstacles are Shapely polygons including parked vehicles. Dynamic traffic
    admission is separate, and must reserve conflicts before executing a route.
    """
    from awr.life.civil.curb import valid_route
    from awr.life.civil.curb_allocation import swept

    if not valid_route(surface, route, size):
        return False, "unsupported_or_steering"
    if allowed_area is not None:
        permitted = allowed_area if isinstance(allowed_area, PreparedPermission) else allowed_area.buffer(1e-7)
        if not permitted.covers(swept(route, size, margin_m=0.0)):
            return False, "outside_permitted_manoeuvre_area"
    obstacles = tuple(obstacles)
    if obstacles:
        corridor = swept(route, size)
        if any(corridor.intersects(obstacle) for obstacle in obstacles):
            return False, "static_obstacle"
    return True, None
