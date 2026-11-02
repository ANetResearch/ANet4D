"""Bounded offline curve repair around an observed native failure.

Preserve the source route outside an exact sample window. This changes a
candidate guide, never a physical pose, road permission, or support surface.
"""

import heapq

import numpy as np
import shapely

from .guide_support import check_centers
from .local_road_search import search
from .maneuver import Maneuver
from .road_guides import body_envelope, body_route


def extend_chain(document, edge_id, surface, clearance, *, max_extra_m=30.0, max_states=1000):
    """Move unsafe artificial seams to existing safe ports on the same guide.

    Follow directed graph edges, preserving road, offset, direction and traffic
    resources. This never crosses a junction or makes a new physical pose.
    The returned chain is still only a repair input, not a valid route.
    """
    if (
        not np.isfinite(max_extra_m)
        or not 0 < max_extra_m <= 50
        or type(max_states) is not int
        or not 1 <= max_states <= 2000
    ):
        raise ValueError("Bounded source-chain extension required")
    nodes, edges = document["nodes"], document["edges"]
    original = edges[edge_id]

    def family(node):
        return tuple(nodes[node].get(k) for k in ("road", "direction", "guide_offset_m", "geometry_reference"))

    key = family(original["source"])
    if original["kind"] != "travel" or family(original["target"]) != key or not key[0] or not key[1]:
        raise ValueError("A single directed source road guide is required")
    resource = tuple(sorted(original["traffic_resources"]))
    eligible = {
        k: e
        for k, e in edges.items()
        if e["kind"] == "travel"
        and family(e["source"]) == key
        and family(e["target"]) == key
        and tuple(sorted(e["traffic_resources"])) == resource
    }

    def safe(node):
        p = nodes[node]["pose"]
        return bool(surface.supported([p[:2]], "vehicle")[0]) and clearance.pose(p)[0]

    def walk(start, reverse):
        queue = [(0.0, (), start)]
        seen = set()
        expansions = 0
        while queue and expansions < max_states:
            distance, path, node = heapq.heappop(queue)
            if node in seen:
                continue
            seen.add(node)
            expansions += 1
            if safe(node):
                return list(reversed(path)) if reverse else list(path), distance, expansions
            for name, e in sorted(eligible.items()):
                if name == edge_id or name in path:
                    continue
                if e["target" if reverse else "source"] != node:
                    continue
                length = float(e["route"]["arc"][-1])
                if not np.isfinite(length) or length <= 0:
                    raise ValueError("Positive finite source edge required")
                if distance + length <= max_extra_m:
                    heapq.heappush(queue, (distance + length, (*path, name), e["source" if reverse else "target"]))
        return None, None, expansions

    before, lb, nb = walk(original["source"], True)
    after, la, na = walk(original["target"], False)
    info = dict(
        candidate_only=True,
        max_extra_m_per_end=max_extra_m,
        expanded_states=nb + na,
        before_m=lb,
        after_m=la,
        reason="existing_safe_ports" if before is not None and after is not None else "no_safe_source_port",
    )
    return (None if before is None or after is None else before + [edge_id] + after), info


def repair(
    route, observed_xy, profile, surface, permission, clearance, *, paddings=(8.0, 12.0, 16.0), max_expansions=512
):
    observed = np.asarray(observed_xy, float)
    paddings = tuple(paddings)
    if (
        observed.shape != (2,)
        or not np.isfinite(observed).all()
        or not 1 <= len(paddings) <= 4
        or not np.isfinite(paddings).all()
        or tuple(sorted(set(paddings))) != paddings
        or any(not 2 <= x <= 30 for x in paddings)
        or type(max_expansions) is not int
        or not 1 <= max_expansions <= 2000
        or clearance is None
        or not np.all(route.gear == 1)
        or route.arc[-1] > 1000
    ):
        raise ValueError("Bounded forward-only native repair with full mesh gate required")
    body = body_route(route)["points"]
    distances = np.linalg.norm(body - observed, axis=1)
    if distances.min() > 10:
        raise ValueError("Observed failure is too far from this source guide")
    nearest = int(np.argmin(distances))
    rejected = np.flatnonzero(~surface.supported(body, "vehicle"))
    lower = min(nearest, int(rejected[0])) if len(rejected) else nearest
    upper = max(nearest, int(rejected[-1])) if len(rejected) else nearest
    trials = []
    seen = set()
    result = None
    for padding in paddings:
        a = max(0, int(np.searchsorted(route.arc, route.arc[lower] - padding, "right")) - 1)
        b = min(len(body) - 1, int(np.searchsorted(route.arc, route.arc[upper] + padding, "left")))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        length = float(route.arc[b] - route.arc[a])
        row = dict(samples=[a, b], padding_m=padding, replaced_length_m=length, candidate_found=False)
        trials.append(row)
        if length > 130:
            row["reason"] = "local_window_length_budget"
            continue
        local, info = search(
            profile,
            surface,
            route.body_pose(a),
            route.body_pose(b),
            permission,
            max_expansions=max_expansions,
            max_length_m=length + 20,
            body_clearance=clearance,
        )
        row["search"] = info
        if local is None:
            continue
        rebuilt = Maneuver(
            np.concatenate([route.points[:a], local.points, route.points[b + 1 :]]),
            np.r_[route.yaw[:a], local.yaw, route.yaw[b + 1 :]],
            np.r_[route.gear[:a], local.gear, route.gear[b:]],
            route.rear_x,
        )
        for old, new in ((route.body_pose(0), rebuilt.body_pose(0)), (route.body_pose(-1), rebuilt.body_pose(-1))):
            if np.linalg.norm(old[:2] - new[:2]) > 1e-8 or abs(np.angle(np.exp(1j * (old[2] - new[2])))) > 1e-8:
                raise RuntimeError("Repair changed original source endpoints")
        points = body_route(rebuilt)["points"]
        row["support"] = check_centers(points, surface)
        row["body_permission"] = bool(permission.covers(body_envelope(rebuilt, profile)))
        stations = shapely.line_locate_point(shapely.LineString(body), shapely.points(points))
        row["source_forward_progress"] = bool(np.all(np.diff(stations) >= -1e-6))
        row["whole_mesh"] = clearance.route(points, rebuilt.yaw)
        if (
            row["support"]["rejected_samples"]
            or not row["body_permission"]
            or not row["source_forward_progress"]
            or not row["whole_mesh"][0]
        ):
            row["reason"] = "assembled_route_rejected"
            continue
        row["candidate_found"] = True
        result = rebuilt
        break
    return result, dict(
        candidate_only=True,
        release_evidence=False,
        trials=trials,
        observed_xy=observed.tolist(),
        nearest_sample=nearest,
        original_rejected_samples=len(rejected),
        limits=dict(paddings=paddings, max_expansions=max_expansions),
        limitations=[
            "Exact unchanged source endpoints and retained outside-window guide.",
            "Native replay, fixed occupancy, visual semantics and full activities remain unverified.",
        ],
    )
