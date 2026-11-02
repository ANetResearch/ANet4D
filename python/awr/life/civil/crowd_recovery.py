"""Space-aware, deterministic short retreat selection for stalled pedestrians.

The normal walking goal and controller are retained. Candidates must clear the
actual static walking geometry and all observed body footprints along the
accelerating retreat. Nearby retreats reserve disjoint swept regions. This is
proposal selection, not a licence to ignore subsequent moving-body checks.
"""

import math

import numpy as np
from shapely.geometry import MultiPoint

from . import body
from .collision import obb_overlap, rectangle

ANGLES = (0.0, 0.6, -0.6, 1.1, -1.1)


def retreat_samples(start, direction, distance, speed, *, max_speed=0.2, dt=0.1):
    points = []
    speeds = []
    progress = 0.0
    left = distance
    while left > 0.005:
        speed = min(max_speed, left / dt, speed + dt)
        step = speed * dt
        for u in np.linspace(0.0, 1.0, 6):
            points.append(start + direction * (progress + u * step))
            speeds.append(speed)
        progress += step
        left -= step
    return np.asarray(points), np.asarray(speeds)


def retreat_corridor(start, yaw, ext, direction, distance, speed):
    # Extents grow monotonically with speed. The convex hull of endpoint
    # rectangles contains the whole fixed-heading translation and gait blend.
    ends = np.array([start, start + direction * distance])
    centres, sizes = body.boxes(ends, np.full(2, yaw), np.full(2, speed + 0.01), ext)
    return MultiPoint(np.concatenate([rectangle(c, yaw, s) for c, s in zip(centres, sizes, strict=False)])).convex_hull


def choose_backsteps(
    xy,
    yaw,
    speeds,
    ext,
    centres,
    sizes,
    people,
    eligible,
    static_clear,
    *,
    active=None,
    near_m=1.0,
    distance=0.6,
    max_speed=0.2,
    dt=0.1,
):
    """Choose complete short retreats, with priority only among feasible peers.

    `static_clear(i,end)` must use the unchanged actual static walking map.
    Inputs describe one common observed frame; no job or pose is changed here.
    `active` maps currently retreating indices to (direction, remaining distance).
    """
    xy, yaw, speeds, ext, centres, sizes = map(np.asarray, (xy, yaw, speeds, ext, centres, sizes))
    people, eligible = np.asarray(people, bool), np.asarray(eligible, bool)
    n = len(xy)
    active = active or {}
    if (
        xy.shape != (n, 2)
        or yaw.shape != (n,)
        or speeds.shape != (n,)
        or ext.shape != (n, 2, 4)
        or centres.shape != (n, 2)
        or sizes.shape != (n, 3)
        or people.shape != (n,)
        or eligible.shape != (n,)
        or np.any(eligible & ~people)
        or not all(np.isfinite(a).all() for a in (xy, yaw, speeds, ext, centres, sizes))
        or np.any(speeds < 0)
        or not all(math.isfinite(v) and v > 0 for v in (near_m, distance, max_speed, dt))
        or dt > 0.1
        or max_speed > 0.5
        or any(i < 0 or i >= n for i in active)
    ):
        raise ValueError("Finite common body frame and bounded retreat parameters required")
    reservations = []
    for i, (direction, left) in sorted(active.items()):
        direction = np.asarray(direction, float)
        if (
            direction.shape != (2,)
            or not np.isfinite(direction).all()
            or abs(np.linalg.norm(direction) - 1) > 1e-6
            or not math.isfinite(left)
            or left < 0
        ):
            raise ValueError("Invalid active retreat")
        if left > 0.005:
            reservations.append(retreat_corridor(xy[i], yaw[i], ext[i], direction, left, max(max_speed, speeds[i])))
    selected = {}
    seen = set()
    obstacle_radius = np.linalg.norm(sizes[:, :2], axis=1) / 2
    for trigger in np.flatnonzero(eligible):
        if trigger in active or trigger in selected:
            continue
        distances = np.linalg.norm(xy - xy[trigger], axis=1)
        distances[(np.arange(n) == trigger) | ~people] = np.inf
        other = int(np.argmin(distances))
        if distances[other] >= near_m or speeds[other] > 0.05 or other in active or other in selected:
            continue
        pair = tuple(sorted((int(trigger), other)))
        if pair in seen:
            continue
        seen.add(pair)
        candidates = [i for i in sorted(pair, reverse=True) if eligible[i] and speeds[i] <= 0.05]
        for i in candidates:
            partner = other if i == trigger else int(trigger)
            separation = float(np.linalg.norm(xy[i] - xy[partner]))
            if separation <= 1e-8:
                continue  # Initial overlap cannot be repaired by a teleport.
            away = (xy[i] - xy[partner]) / separation
            radius = float(body.turn_radius(ext[i, 1]))
            neighbours = np.flatnonzero(
                (np.arange(n) != i)
                & (np.linalg.norm(centres - xy[i], axis=1) <= distance + radius + obstacle_radius + 0.01)
            )
            for angle in ANGLES:
                c, s = math.cos(angle), math.sin(angle)
                direction = np.array([c * away[0] - s * away[1], s * away[0] + c * away[1]])
                if not static_clear(i, xy[i] + distance * direction):
                    continue
                corridor = retreat_corridor(xy[i], yaw[i], ext[i], direction, distance, max_speed)
                if any(corridor.intersects(r) for r in reservations):
                    continue
                pp, vv = retreat_samples(xy[i], direction, distance, float(speeds[i]), max_speed=max_speed, dt=dt)
                aa = np.full(len(pp), yaw[i])
                bc, bs = body.boxes(pp, aa, np.maximum(speeds[i], vv) + 0.01, ext[i])
                if (
                    len(neighbours)
                    and obb_overlap(
                        bc[:, None],
                        aa[:, None],
                        bs[:, None],
                        centres[neighbours][None],
                        yaw[neighbours][None],
                        sizes[neighbours][None],
                    ).any()
                ):
                    continue
                selected[i] = dict(
                    direction=direction,
                    distance_m=distance,
                    other=partner,
                    candidate_angle=angle,
                    preferred_actor=candidates[0],
                    priority_fallback=i != candidates[0],
                )
                reservations.append(corridor)
                break
            if i in selected:
                break
    return selected
