"""Bounded offline multi-point turns inside an explicitly permitted area.

This is not a road-following policy. The caller must reserve the entire turn
area, limit execution speed and stop at every returned gear cusp. Geometry is
admitted against body containment, wheel support and parked-body clearance.
No route is a proof of impossibility; this bounded search is not optimal.
"""

import heapq
import math
from collections import Counter

import numpy as np

from .curb_allocation import footprint, swept
from .driving_graph import route_arrays
from .driving_search import primitive
from .reeds_shepp import candidates
from .vehicle_connections import PreparedPermission, admit, angle_delta


def reverse_contained(route, size, permission):
    """Every reversed body envelope stays in its explicit manoeuvring area.

    Forward travel may leave a parking area onto the connected street. Reverse
    travel may not inherit that larger permission. Include both cusp poses and
    the continuous swept envelope, not just reversed segment centre points.
    """
    reverse = np.asarray(route["gear"]) < 0
    boundaries = np.flatnonzero(np.diff(np.r_[False, reverse, False]))
    for first, last in zip(boundaries[::2], boundaries[1::2], strict=False):
        leg = dict(points=np.asarray(route["points"])[first : last + 1], yaw=np.asarray(route["yaw"])[first : last + 1])
        if not permission.covers(swept(leg, size, margin_m=0.0)):
            return False
    return True


def reverse_run(route, initial_m=0.0, limit_m=math.inf):
    """Check contiguous reverse distance and return its trailing run length."""
    run = float(initial_m)
    longest = run
    for distance, gear in zip(np.diff(route["arc"]), route["gear"], strict=False):
        run = run + float(distance) if gear < 0 else 0.0
        longest = max(longest, run)
        if longest > limit_m + 1e-8:
            return False, run
    return True, run


def connect(
    surface,
    start,
    goal,
    size,
    *,
    maneuver_area,
    obstacles=(),
    max_expansions=2000,
    max_length_m=80.0,
    max_gear_changes=6,
    gear_change_cost_m=0.25,
    resolution_m=0.5,
    heading_bins=72,
    primitive_length_m=1.5,
    reverse_area=None,
    candidate_admission=None,
    turn_radius_m=None,
    allowed_gears=(1, -1),
    max_reverse_leg_m=None,
    terminal_reverse_reserve_m=0.0,
):
    """Return an exact-pose candidate and deterministic search counters.

    Gear penalties rank the search; the separate travelled-length budget and
    final arc never include them. Search states distinguish arrival gear and
    gear-change count. No other caller gets reverse travel implicitly.
    An optional reverse_area restricts backing to a subset of maneuver_area;
    it applies to primitives, analytic finishes and the assembled route.
    An explicit candidate_admission with pose/route methods permits an alternate
    reference (e.g. a native vehicle rear axle). Both receive the permissions and
    obstacles. Such output is always candidate-only and requires its physical
    backend's independent admission; it cannot claim this legacy support proof.
    Optional reverse-leg limits include the prefix/analytic-finish seam. A
    terminal reserve accounts for a caller's subsequent reverse alignment;
    it does not grant any additional length or reverse permission.
    """
    start, goal, size = [np.asarray(v, float) for v in (start, goal, size)]
    allowed_gears = tuple(allowed_gears)
    if max_reverse_leg_m is not None and (not np.isfinite(max_reverse_leg_m) or max_reverse_leg_m <= 0):
        raise ValueError("Positive finite reverse-leg limit required")
    if (
        not np.isfinite(terminal_reverse_reserve_m)
        or terminal_reverse_reserve_m < 0
        or (
            terminal_reverse_reserve_m and (max_reverse_leg_m is None or terminal_reverse_reserve_m > max_reverse_leg_m)
        )
    ):
        raise ValueError("Terminal reverse reserve must fit the explicit reverse-leg limit")
    if (
        allowed_gears not in ((1,), (-1,), (1, -1))
        or any(type(g) is not int for g in allowed_gears)
        or (turn_radius_m is not None and (not math.isfinite(turn_radius_m) or turn_radius_m <= 0))
    ):
        raise ValueError("Explicit positive radius and canonical gear permission required")
    if (
        any(v.shape != (3,) for v in (start, goal, size))
        or not np.isfinite(
            np.r_[start, goal, size, max_length_m, gear_change_cost_m, resolution_m, primitive_length_m]
        ).all()
        or np.any(size <= 0)
        or min(max_length_m, resolution_m, primitive_length_m) <= 0
        or gear_change_cost_m < 0
        or type(max_expansions) is not int
        or max_expansions < 1
        or type(max_gear_changes) is not int
        or max_gear_changes < 0
        or type(heading_bins) is not int
        or heading_bins < 8
        or maneuver_area is None
        or maneuver_area.is_empty
    ):
        raise ValueError("Invalid explicit manoeuvre search request")
    if np.linalg.norm(start[:2] - goal[:2]) <= 1e-7 and abs(angle_delta(start[2], goal[2])) <= 1e-7:
        return None, {"already_at_goal": 1}
    from .grounded import support_mask

    permission = PreparedPermission(maneuver_area)
    reverse_permission = None
    if reverse_area is not None:
        reverse_permission = PreparedPermission(reverse_area)
        if not permission.covers(reverse_area):
            raise ValueError("Reverse permission must be inside the overall manoeuvre area")
    obstacles = tuple(obstacles)
    if candidate_admission is not None:
        if surface is not None or not all(callable(getattr(candidate_admission, k, None)) for k in ("pose", "route")):
            raise ValueError("Alternate candidate gates require no legacy surface and explicit pose/route methods")

    def checked(result):
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[0], (bool, np.bool_))
            or (result[1] is not None and not isinstance(result[1], str))
            or (not result[0] and not result[1])
        ):
            raise ValueError("Candidate gate must return a boolean and failure reason")
        return result

    def route_admission(route):
        if candidate_admission is not None:
            return checked(candidate_admission.route(route, size, permission, reverse_permission, obstacles))
        if reverse_permission is not None and not reverse_contained(route, size, reverse_permission):
            return False, "reverse_outside_permitted_area"
        return admit(surface, route, size, allowed_area=permission, obstacles=obstacles)

    for name, pose in (("start", start), ("goal", goal)):
        if candidate_admission is not None:
            ok, reason = checked(candidate_admission.pose(pose, size, permission, obstacles))
            if not ok:
                return None, {name + "_" + reason: 1}
            continue
        slot = dict(xy=pose[:2], yaw=pose[2])
        if not permission.covers(footprint(slot, size, margin_m=0.0)):
            return None, {name + "_outside_permitted_area": 1}
        if any(footprint(slot, size).intersects(obstacle) for obstacle in obstacles):
            return None, {name + "_static_obstacle": 1}
        if not support_mask(surface, pose[None, :2], pose[None, 2], size, "vehicle")[0]:
            return None, {name + "_unsupported": 1}
    radius = max(4.0, size[0] * 0.65 / math.tan(math.radians(32.0))) if turn_radius_m is None else float(turn_radius_m)
    turns = (0.0, -0.5 / radius, 0.5 / radius, -1.0 / radius, 1.0 / radius)
    counts = Counter()

    def key(pose, gear, changes, reverse_m=0.0):
        base = (
            *np.rint((pose[:2] - start[:2]) / resolution_m).astype(int),
            int(round((pose[2] % (2 * math.pi)) * heading_bins / (2 * math.pi))) % heading_bins,
            gear,
            changes,
        )
        # Paths reaching the same pose with different remaining backing
        # allowance are not interchangeable. Legacy searches retain their key.
        return base if max_reverse_leg_m is None else (*base, round(reverse_m, 10))

    def heuristic(pose):
        return max(float(np.linalg.norm(goal[:2] - pose[:2])), radius * abs(angle_delta(pose[2], goal[2])))

    # (pose, parent index, preceding leg, distance, gear, changes, reverse run)
    states = [(start, -1, None, 0.0, 0, 0, 0.0)]
    best = {key(start, 0, 0): 0.0}
    queue = [(heuristic(start), 0)]
    while queue and counts["expanded"] < max_expansions:
        _, index = heapq.heappop(queue)
        pose, parent, leg, travelled, gear, changes, reverse_m = states[index]
        if travelled > best.get(key(pose, gear, changes, reverse_m), math.inf) + 1e-10:
            continue
        counts["expanded"] += 1
        distance = float(np.linalg.norm(goal[:2] - pose[:2]))
        exact = index > 0 and distance <= 1e-7 and abs(angle_delta(pose[2], goal[2])) <= 1e-7
        # Analytic finishing is deterministic and amortized, not done ten times
        # for every expanded primitive. It cannot loosen any admission checks.
        if exact or index == 0 or (counts["expanded"] % 8 == 0 and distance <= 12.0):
            remaining = max_length_m - travelled
            finishes = (
                [None]
                if exact
                else (
                    candidates(
                        pose,
                        goal,
                        size,
                        max_length_m=remaining,
                        turn_radius_m=turn_radius_m,
                        allowed_gears=allowed_gears,
                    )
                    if remaining > 1e-9
                    else []
                )
            )
            for finish in finishes:
                counts["finish_candidates"] += 1
                added_changes = (
                    0
                    if finish is None
                    else (len(finish["required_stops_m"]) + int(gear != 0 and gear != finish["gear"][0]))
                )
                if changes + added_changes > max_gear_changes:
                    counts["gear_change_budget"] += 1
                    continue
                if max_reverse_leg_m is not None:
                    ok, tail = (
                        (True, reverse_m) if finish is None else reverse_run(finish, reverse_m, max_reverse_leg_m)
                    )
                    if not ok or tail + terminal_reverse_reserve_m > max_reverse_leg_m + 1e-8:
                        counts["finish_reverse_leg_budget"] += 1
                        continue
                ok, reason = (True, None) if finish is None else route_admission(finish)
                if not ok:
                    counts["finish_" + reason] += 1
                    continue
                chain = [] if finish is None else [finish]
                cursor = index
                while states[cursor][1] >= 0:
                    chain.append(states[cursor][2])
                    cursor = states[cursor][1]
                chain.reverse()
                route = route_arrays(
                    np.concatenate([r["points"][1 if i else 0 :] for i, r in enumerate(chain)]),
                    np.concatenate([r["yaw"][1 if i else 0 :] for i, r in enumerate(chain)]),
                    np.concatenate([r["gear"] for r in chain]),
                )
                if (
                    np.linalg.norm(route["points"][-1] - goal[:2]) > 1e-6
                    or abs(angle_delta(route["yaw"][-1], goal[2])) > 1e-6
                ):
                    raise RuntimeError("Manoeuvre search lost exact goal pose")
                if len(route["required_stops_m"]) > max_gear_changes or route["arc"][-1] > max_length_m + 1e-8:
                    raise RuntimeError("Manoeuvre assembly exceeded search budget")
                ok, reason = route_admission(route)
                if not ok:
                    counts["assembled_" + reason] += 1
                    continue
                if max_reverse_leg_m is not None:
                    ok, tail = reverse_run(route, 0.0, max_reverse_leg_m)
                    if not ok or tail + terminal_reverse_reserve_m > max_reverse_leg_m + 1e-8:
                        raise RuntimeError("Manoeuvre assembly exceeded reverse-leg budget")
                route["construction"] = "bounded_explicit_maneuver_pose_search"
                if candidate_admission is not None:
                    route["candidate_only"] = True
                    route["native_physical_admission_required"] = True
                counts["accepted"] += 1
                return route, dict(counts)
        for next_gear in allowed_gears:
            next_changes = changes + int(gear != 0 and gear != next_gear)
            if next_changes > max_gear_changes:
                counts["gear_change_budget"] += 1
                continue
            for turn in turns:
                leg = primitive(pose, turn, length=primitive_length_m, gear=next_gear)
                end = np.r_[leg["points"][-1], leg["yaw"][-1]]
                length = travelled + float(leg["arc"][-1])
                reverse_next = (reverse_m + float(leg["arc"][-1])) if next_gear < 0 else 0.0
                if max_reverse_leg_m is not None and reverse_next > max_reverse_leg_m + 1e-8:
                    counts["reverse_leg_budget"] += 1
                    continue
                if length + np.linalg.norm(goal[:2] - end[:2]) > max_length_m:
                    counts["length_budget"] += 1
                    continue
                k = key(end, next_gear, next_changes, reverse_next)
                if length >= best.get(k, math.inf) - 1e-10:
                    continue
                counts["primitive_candidates"] += 1
                ok, reason = route_admission(leg)
                if not ok:
                    counts[reason] += 1
                    continue
                best[k] = length
                next_index = len(states)
                states.append((end, index, leg, length, next_gear, next_changes, reverse_next))
                heapq.heappush(queue, (length + next_changes * gear_change_cost_m + heuristic(end), next_index))
    counts["budget_exhausted" if queue else "search_exhausted"] += 1
    return None, dict(counts)
