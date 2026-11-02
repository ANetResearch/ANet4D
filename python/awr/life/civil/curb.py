"""Measured curb berths and continuous parking manoeuvres.

Road metadata proposes geometry; it never grants support. All poses, including
the pull-out/pull-in, pass the same tyre, body and steering checks as driving.
On streets too narrow for two lanes beside parked cars, the remaining corridor
is shared. A scenario must not schedule opposing traffic in that corridor at
the same time. No street is widened and no missing support is filled here.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u**3 * (10.0 - 15.0 * u + 6.0 * u * u)


def sampled_route(points, road, *, endpoint_headings=None):
    """Resample at .35 m, preserving endpoint positions and smooth heading.

    Explicit arc length avoids rounding the destination down to a .35 m tick.
    The yaw array is used by both scalar and batched vehicle interpolation.
    """
    points = np.asarray(points, float)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]
    points = points[keep]
    if len(points) < 2:
        raise ValueError("A driving route needs distinct points")
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    ds = np.r_[np.arange(0.0, arc[-1], 0.35), arc[-1]]
    pp = np.column_stack([np.interp(ds, arc, points[:, k]) for k in range(2)])
    # Resampling can cut across a source-polyline vertex. The executed route
    # is pp, so all controllers and stop coordinates must use its measured
    # chord lengths, not the slightly longer source distance parameter ds.
    actual_arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pp, axis=0), axis=1))]
    tangent = np.gradient(pp, actual_arc, axis=0)
    yaw = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    if endpoint_headings is not None:
        headings = np.asarray(endpoint_headings, float)
        if headings.shape != (2,) or not np.isfinite(headings).all():
            raise ValueError("Both endpoint headings must be finite")
        # One-sided finite differences estimate the first/last chord, not the
        # exact berth tangent. Carry the admitted pose into the route before
        # support and steering validation, rather than rotating a parked body
        # at departure or tolerating a mismatched execution handoff.
        for index, heading in zip((0, -1), headings, strict=False):
            yaw[index] += (heading - yaw[index] + math.pi) % (2 * math.pi) - math.pi
    return dict(road=road, points=pp, arc=actual_arc, yaw=yaw, smooth_vehicle_heading=True)


def valid_route(surface, route, size):
    from awr.life.civil.grounded import support_mask

    if hasattr(surface, "vehicle_route_supported"):
        # Candidate proof backend can reject a complete route before doing
        # costly mesh certificates for its other poses. Preserve both steering
        # limits of the existing support_mask + valid_route contract.
        curvature = np.abs(np.diff(route["yaw"])) / np.diff(route["arc"])
        sampled_curvature = np.abs(np.diff(np.unwrap(route["yaw"]))) / 0.35
        limit = min(0.26, math.tan(math.radians(35.0)) / (float(size[0]) * 0.65))
        return bool(
            (curvature < limit).all()
            and (sampled_curvature < limit).all()
            and surface.vehicle_route_supported(route["points"], route["yaw"], size)
        )
    ok = support_mask(surface, route["points"], route["yaw"], size, "vehicle").all()
    # The old support test assumes .35 m spacing. The last interval can be
    # shorter, so also check the actual arc length for steering feasibility.
    curvature = np.abs(np.diff(route["yaw"])) / np.diff(route["arc"])
    limit = min(0.26, math.tan(math.radians(35.0)) / (float(size[0]) * 0.65))
    return bool(ok and (curvature < limit).all())


def road_layout(width, size, *, curb_gap=0.20, body_gap=0.50, lane_margin=0.15):
    """Right curb parking strip plus lanes in the remaining carriageway.

    Offsets are positive towards the right of the road's stored direction.
    A two-way assertion requires two full-width lanes *beside* parking, not
    merely two centre lines fitting on the raster.
    """
    w = float(size[1])
    band = w + curb_gap + body_gap
    available = float(width) - band
    lane_width = max(3.0, w + 2 * lane_margin)
    if available < lane_width:
        return None
    lanes = 2 if available >= 2 * lane_width else 1
    centre = -band / 2
    travel = centre + (available / 4 if lanes == 2 else 0.0)
    return dict(
        parking_offset_m=width / 2 - curb_gap - w / 2,
        travel_offset_m=travel,
        opposite_offset_m=centre - available / 4,
        remaining_width_m=available,
        lanes=lanes,
        mode="two_way_beside_parking" if lanes == 2 else "shared_narrow_corridor",
    )


def road_geometry(road):
    line = LineString(road["line"])
    ds = np.r_[np.arange(0.0, line.length, 0.35), line.length]
    import shapely

    pp = shapely.get_coordinates(shapely.line_interpolate_point(line, ds))
    tangent = np.gradient(pp, ds, axis=0)
    yaw = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    return line, ds, pp, yaw


def connected_roads(semantics):
    """Also inspect exact degree-two centreline joins of equal-width roads.

    No endpoint snapping, junction shortcut or inferred turn is permitted.
    Curves still need a supported manoeuvre after joining. Retain originals so
    a failed bend does not erase a useful fragment on either side.
    """
    from shapely.ops import linemerge

    roads = list(semantics["roads"])
    groups = {}
    for r in roads:
        if len(r["line"]) >= 2:
            groups.setdefault(float(r["width_m"]), []).append(r["line"])
    for width, lines in groups.items():
        merged = linemerge(lines)
        for j, line in enumerate(getattr(merged, "geoms", [merged])):
            if line.geom_type == "LineString" and not line.is_ring:
                roads.append(dict(id=f"joined-curb-{width:g}-{j}", line=list(line.coords), width_m=width))
    return roads


def poses_at(geometry, distances, offsets):
    _, arc, pp, yaw = geometry
    s = np.asarray(distances, float)
    p = np.column_stack([np.interp(s, arc, pp[:, k]) for k in range(2)])
    a = np.interp(s, arc, yaw)
    return p + np.column_stack([np.sin(a), -np.cos(a)]) * np.asarray(offsets)[..., None], a


def berth(geometry, s, layout, size, surface, rules=None):
    p, a = poses_at(geometry, [s], layout["parking_offset_m"])
    p, a = p[0], float(a[0])
    from awr.life.civil.grounded import support_mask

    if not support_mask(surface, p[None], np.array([a]), size, "vehicle")[0]:
        return None
    legality = rules.evaluate(p, a, size) if rules else None
    if legality is not None and not legality["legal"]:
        return None
    return dict(s=float(s), xy=p.tolist(), yaw=a, legality=legality)


def manoeuvre(geometry, start, end, layout, size, surface, road, *, stops=(), merge_m=18.0, _retry=True):
    """Forward-only curb -> travel lane -> curb with optional curb stops.

    Each intermediate stop has its own disjoint pull-in and pull-out. There is
    no jump from a travel point to a parking slot and no in-place vehicle turn.
    """
    stations = [float(start), *map(float, stops), float(end)]
    if any(b - a < 2 * merge_m + 2.0 for a, b in zip(stations, stations[1:], strict=False)):
        return None
    points = []
    anchors = []
    for start_s, end_s in zip(stations, stations[1:], strict=False):
        ds = np.r_[np.arange(start_s, end_s, 0.25), end_s]
        active = smoothstep((ds - start_s) / merge_m) * smoothstep((end_s - ds) / merge_m)
        offset = layout["parking_offset_m"] + active * (layout["travel_offset_m"] - layout["parking_offset_m"])
        p, _ = poses_at(geometry, ds, offset)
        points.extend(p if not points else p[1:])
        anchors.append(p[-1])
    _, endpoint_headings = poses_at(geometry, [start, end], layout["parking_offset_m"])
    route = sampled_route(points, road, endpoint_headings=endpoint_headings)
    if not valid_route(surface, route, size):
        # A long body swings its rear corner towards the kerb when pulling out.
        # Try a gentler forward curve; never shrink the body or waive support.
        if _retry:
            for distance in (merge_m + 7.0, merge_m + 14.0):
                longer = manoeuvre(
                    geometry, start, end, layout, size, surface, road, stops=stops, merge_m=distance, _retry=False
                )
                if longer is not None:
                    return longer
        return None
    route["dense_validated"] = True
    route["merge_m"] = float(merge_m)
    # Cumulative nearest anchors on a forward route. Stops are exact source
    # vertices but may fall between resampling points: project on segments.
    line = LineString(route["points"])
    from shapely.geometry import Point

    stop_arcs = [float(line.project(Point(p))) for p in anchors]
    route["curb_stops_m"] = stop_arcs
    return route


def candidates(surface, semantics, size, rules=None, *, min_trip=60.0, spacing=8.0, merge_m=18.0, report=None):
    """Enumerate curb-to-curb trips; do not silently substitute parked actors.

    Static berths and the corridor are returned too, for allocation and audit.
    Road fragments remain fragments. Joining roads needs a validated junction
    connector and is deliberately outside this admission function.
    """
    from awr.life.civil.grounded import segments, support_mask

    report = report if report is not None else {}
    report.update(roads=0, narrow=0, berths=0, trips=0, supported_two_way_segments=0)
    result = []
    seen = set()
    for road in connected_roads(semantics):
        if len(road["line"]) < 2:
            continue
        # Duplicate directed SUMO edges must not allocate the same road twice.
        coords = tuple(map(tuple, np.round(road["line"], 2)))
        signature = min(coords, coords[::-1])
        if signature in seen:
            continue
        seen.add(signature)
        report["roads"] += 1
        layout = road_layout(float(road["width_m"]), size)
        if layout is None:
            report["narrow"] += 1
            continue
        geometry = road_geometry(road)
        if geometry[0].length < size[0] + 4.0:
            continue
        ds = geometry[1]
        travel, yaw = poses_at(geometry, ds, layout["travel_offset_m"])
        valid = support_mask(surface, travel, yaw, size, "vehicle")
        if layout["lanes"] == 2:
            opposite, angles = poses_at(geometry, ds, layout["opposite_offset_m"])
            dual = valid & support_mask(surface, opposite, angles + math.pi, size, "vehicle")
            report["supported_two_way_segments"] += sum(int(ds[b - 1] - ds[a] >= min_trip) for a, b in segments(dual))
        berths = []
        for s in np.arange(size[0] / 2 + 2.0, geometry[0].length - size[0] / 2 - 2.0, spacing):
            slot = berth(geometry, s, layout, size, surface, rules)
            if slot is not None:
                berths.append(slot)
        report["berths"] += len(berths)
        trips = []
        # A start chooses the farthest reachable berth within the same measured
        # fragment. This is not a shortest route between arbitrary buildings.
        for a, b in segments(valid):
            slots = [s for s in berths if ds[a] <= s["s"] <= ds[b - 1]]
            for start in slots:
                for end in reversed(slots):
                    if end["s"] - start["s"] < max(min_trip, 2 * merge_m + 2.0):
                        break
                    route = manoeuvre(
                        geometry, start["s"], end["s"], layout, size, surface, road["id"], merge_m=merge_m
                    )
                    if route is not None:
                        trips.append(dict(start=start, end=end, route=route))
                        break
        report["trips"] += len(trips)
        if berths:
            result.append(
                dict(
                    road=road["id"],
                    layout=layout,
                    geometry=geometry,
                    berths=berths,
                    trips=trips,
                    corridor=travel,
                    corridor_valid=valid,
                )
            )
    return result
