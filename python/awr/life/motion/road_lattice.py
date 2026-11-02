"""Bounded offline forward corridor planning around measured support gaps.

Layered dynamic programming over source-road samples and lateral offsets.
Every transition is an analytic rear-axle CSC curve with actual steering and
full planar-body permission checks. No online A*, road widening or actor script.
"""

import math
from collections import Counter

import numpy as np

from awr.life.civil.vehicle_connections import PreparedPermission, tangent_paths

from .local_road_search import MeasuredAxleAdmission
from .maneuver import Maneuver, body_to_rear


def plan(
    profile,
    support,
    baseline,
    permission,
    *,
    station_m=8.0,
    offsets_m=(0.0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5),
    max_transitions=4096,
    steering_fraction=0.8,
):
    offsets = np.asarray(offsets_m, float)
    if (
        not np.isfinite(station_m)
        or not 3 <= station_m <= 15
        or offsets.ndim != 1
        or not 1 <= len(offsets) <= 9
        or not np.isfinite(offsets).all()
        or (abs(offsets) > 2).any()
        or len(set(offsets)) != len(offsets)
        or 0.0 not in offsets
        or type(max_transitions) is not int
        or not 1 <= max_transitions <= 8192
        or baseline.arc[-1] > 300
        or (baseline.gear != 1).any()
    ):
        raise ValueError("Bounded forward road lattice required")
    gate = MeasuredAxleAdmission(profile, support, steering_fraction=steering_fraction)
    if abs(baseline.rear_x - gate.rear) > 1e-8:
        raise ValueError("Actual rear axle required")
    area = PreparedPermission(permission)
    radius = 1.01 * gate.wheelbase / (math.tan(profile.max_steer) * steering_fraction)
    # Integer sample selection preserves source tangent; no interpolation/snap.
    indices = sorted(
        {
            0,
            len(baseline.points) - 1,
            *[
                int(np.argmin(abs(baseline.arc - x)))
                for x in np.arange(station_m, baseline.arc[-1] - station_m * 0.5, station_m)
            ],
        }
    )
    counts = Counter()
    layers = []
    for number, index in enumerate(indices):
        base = baseline.body_pose(index)
        normal = np.array([-np.sin(base[2]), np.cos(base[2])])
        layer = []
        for offset in (0.0,) if number in (0, len(indices) - 1) else offsets:
            pose = base.copy()
            pose[:2] += offset * normal
            rear = body_to_rear(pose, gate.rear)
            ok, reason = gate.pose(rear, gate.size, area, ())
            if ok:
                layer.append((float(offset), pose, rear))
            else:
                counts["pose_" + str(reason)] += 1
        layers.append(layer)
    report = dict(
        candidate_only=True,
        release_evidence=False,
        station_indices=indices,
        counts=counts,
        limits=dict(station_m=station_m, offsets_m=offsets.tolist(), max_transitions=max_transitions),
        turn_radius_m=radius,
        steering_fraction=steering_fraction,
    )
    if any(not layer for layer in layers):
        report["reason"] = "empty_supported_layer"
        return None, report
    costs = {0: 0.0}
    back = []
    for level, (before, after) in enumerate(zip(layers, layers[1:], strict=False)):
        current = {}
        parents = {}
        for b, (offset_b, pose_b, rear_b) in enumerate(after):
            for a in sorted(costs):
                if counts["transitions"] >= max_transitions:
                    report["reason"] = "transition_budget"
                    return None, report
                counts["transitions"] += 1
                offset_a, pose_a, rear_a = before[a]
                advance = pose_b[:2] - pose_a[:2]
                distance = float(np.linalg.norm(advance))
                if distance < 1e-6:
                    counts["coincident_stations"] += 1
                    continue
                routes = tangent_paths(rear_a, rear_b, radius, spacing=0.15, max_length_m=distance * 1.3 + 1.0)
                chosen = None
                for route in routes:
                    counts["curves"] += 1
                    if np.linalg.norm(route["points"][-1] - rear_b[:2]) > 1e-8:
                        counts["endpoint_drift"] += 1
                        continue
                    ok, reason = gate.route(route, gate.size, area, None, ())
                    if not ok:
                        counts[str(reason)] += 1
                        continue
                    m = Maneuver(route["points"], route["yaw"], route["gear"], gate.rear)
                    # No loops or station reversal hidden inside a layer edge.
                    from .road_guides import body_route

                    progress = (body_route(m)["points"] - pose_a[:2]) @ advance / distance
                    if (np.diff(progress) < -1e-8).any() or progress.min() < -1e-8 or progress.max() > distance + 1e-8:
                        counts["station_reversal"] += 1
                        continue
                    chosen = m
                    break
                if chosen is None:
                    continue
                cost = costs[a] + float(chosen.arc[-1]) + 0.05 * abs(offset_b - offset_a)
                if b not in current or cost < current[b]:
                    current[b] = cost
                    parents[b] = (a, chosen)
        back.append(parents)
        costs = current
        if not costs:
            report.update(reason="no_supported_transition", failed_layer=level + 1)
            return None, report
    cursor = min(costs, key=lambda x: (costs[x], x))
    parts = []
    selected = []
    for level in range(len(back) - 1, -1, -1):
        selected.append(layers[level + 1][cursor][0])
        cursor, m = back[level][cursor]
        parts.append(m)
    selected.append(layers[0][cursor][0])
    parts.reverse()
    selected.reverse()
    m = Maneuver(
        np.concatenate([p.points[1 if i else 0 :] for i, p in enumerate(parts)]),
        np.concatenate([p.yaw[1 if i else 0 :] for i, p in enumerate(parts)]),
        np.concatenate([p.gear for p in parts]),
        gate.rear,
    )
    # Recheck assembled seam geometry as a whole, including the exact endpoints.
    ok, reason = gate.route(dict(points=m.points, yaw=m.yaw, gear=m.gear), gate.size, area, None, ())
    if not ok:
        report.update(reason="assembled_" + str(reason))
        return None, report
    if any(np.linalg.norm(m.body_pose(i)[:2] - baseline.body_pose(i)[:2]) > 1e-8 for i in (0, -1)):
        report["reason"] = "assembled_endpoint_drift"
        return None, report
    report.update(reason="candidate_found", selected_offsets_m=selected, length_m=float(m.arc[-1]))
    return m, report
