"""Source-directed rear-axle road candidates, without terrain admission.

Compile once per map/profile, outside the activity tick. Source connectivity
authorizes turns; geometric proximity does not. Explicit circular fillets
replace polygon corners, never a stationary heading change. Permission is a
planar conservative body-envelope screen only: native ground, wheel, traffic
and activity admission remain mandatory before these guides can be used.
"""

import math
from collections import Counter, defaultdict

import numpy as np
import shapely
from shapely.geometry import LineString, Point

from awr.life.civil.curb_allocation import swept
from awr.life.civil.driving_graph import DrivingGraph, route_arrays
from awr.life.civil.road_corridors import source_roads
from awr.life.civil.vehicle_connections import _arc, _join, _straight, tangent_paths

from .maneuver import Maneuver, axle


def fillet_route(points, radius, *, inset=8.0, spacing=0.15):
    """Round source-polyline corners at a specified minimum turning radius.

    Preserve all source vertices as constraints on tangent construction. Tight
    consecutive corners fail explicitly; no global spline, shortcut, endpoint
    snap, reverse driving or radius reduction is used to force connectivity.
    Insets leave room for a separately checked junction maneuver.
    """
    p = np.asarray(points, float)
    ends = np.asarray(inset, float)
    if ends.ndim == 0:
        ends = np.repeat(ends, 2)
    if (
        p.ndim != 2
        or p.shape[1] != 2
        or len(p) < 2
        or not np.isfinite(p).all()
        or ends.shape != (2,)
        or not np.isfinite(ends).all()
        or np.any(ends < 0)
        or not np.isfinite([radius, spacing]).all()
        or radius <= 0
        or not 0 < spacing <= 0.5
    ):
        raise ValueError("Finite polyline and bounded positive sampling required")
    delta = np.diff(p, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if (lengths <= 1e-8).any():
        raise ValueError("Duplicate source vertices")
    headings = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    turns = np.diff(headings)
    if (np.abs(turns) >= math.pi - 1e-6).any():
        raise ValueError("Source road doubles back")
    trim = np.r_[0.0, radius * np.tan(np.abs(turns) / 2), 0.0]
    if (trim[:-1] + trim[1:] >= lengths - 1e-6).any():
        raise ValueError("Insufficient source segment for turning radius")
    direction = delta / lengths[:, None]
    pieces = []
    for i in range(len(lengths)):
        a = p[i] + trim[i] * direction[i]
        b = p[i + 1] - trim[i + 1] * direction[i]
        pieces.append((*_straight(a, b, headings[i], spacing), 1))
        if i < len(turns) and abs(turns[i]) > 1e-10:
            pieces.append((*_arc(b, headings[i], np.sign(turns[i]), radius, abs(turns[i]), spacing), 1))
    route = _join(pieces)
    if route["arc"][-1] <= ends.sum() + 1e-6:
        raise ValueError("Insufficient route for junction insets")
    if not ends.any():
        return route

    def at(distance):
        i = min(int(np.searchsorted(route["arc"], distance, side="right")) - 1, len(route["arc"]) - 2)
        ds = route["arc"][i + 1] - route["arc"][i]
        u = (distance - route["arc"][i]) / ds
        yaw = route["yaw"][i]
        change = route["yaw"][i + 1] - yaw
        # Interpolate the analytic circle, not the chord. Chord interpolation
        # near a cut end would violate the rear wheel's no-slip tangent.
        if abs(change) > 1e-10:
            radius_signed = ds / (2 * math.sin(change / 2))
            point = route["points"][i] + radius_signed * np.array(
                [math.sin(yaw + u * change) - math.sin(yaw), math.cos(yaw) - math.cos(yaw + u * change)]
            )
        else:
            point = (1 - u) * route["points"][i] + u * route["points"][i + 1]
        return point, yaw + u * change

    finish = route["arc"][-1] - ends[1]
    start_p, start_y = at(ends[0])
    finish_p, finish_y = at(finish)
    middle = (route["arc"] > ends[0] + 1e-8) & (route["arc"] < finish - 1e-8)
    return route_arrays(
        np.vstack([start_p, route["points"][middle], finish_p]), np.r_[start_y, route["yaw"][middle], finish_y]
    )


def body_route(maneuver):
    """Graph positions are body origins; controller positions are rear axles."""
    points = maneuver.points - maneuver.rear_x * np.column_stack([np.cos(maneuver.yaw), np.sin(maneuver.yaw)])
    return route_arrays(points, maneuver.yaw, maneuver.gear)


def body_envelope(maneuver, profile):
    route = body_route(maneuver)
    # Profiles need not have a centered local bounding box.
    middle = (profile.body_hi + profile.body_lo) / 2
    yaw = route["yaw"]
    c, s = np.cos(yaw), np.sin(yaw)
    route = dict(
        route, points=route["points"] + np.column_stack([c * middle[0] - s * middle[1], s * middle[0] + c * middle[1]])
    )
    return swept(route, profile.body_hi - profile.body_lo, margin_m=0.0)


def junction_radii(start, end, minimum):
    """Finite radii from the exact endpoint tangent intersection.

    The minimum steering radius is a lower bound, not the only useful turn.
    A larger radius can keep chassis overhang inside a real street corner.
    No source road, endpoint, permission or steering limit changes here.
    """
    a, b = np.asarray(start, float), np.asarray(end, float)
    turn = math.remainder(b[2] - a[2], 2 * math.pi)
    directions = np.array([[math.cos(a[2]), math.cos(b[2])], [math.sin(a[2]), math.sin(b[2])]])
    if abs(np.linalg.det(directions)) < 1e-8:
        return (minimum,)
    lengths = np.linalg.solve(directions, b[:2] - a[:2])
    if np.any(lengths <= 0):
        return (minimum,)
    fitted = float(min(lengths) / math.tan(abs(turn) / 2))
    if not math.isfinite(fitted) or fitted <= minimum + 1e-8:
        return (minimum,)
    return (minimum, (minimum + fitted) / 2, fitted)


def parallel_guide(maneuver, offset_m):
    """Parallel rear guide; keep analytic tangents and exact physical axle.

    This is only candidate geometry. Curvature and original source permission
    are checked by the caller, so shifting cannot create an independent lane.
    """
    if not np.isfinite(offset_m):
        raise ValueError("Finite lateral offset required")
    if offset_m == 0:
        return maneuver
    normal = np.column_stack([-np.sin(maneuver.yaw), np.cos(maneuver.yaw)])
    return Maneuver(maneuver.points + offset_m * normal, maneuver.yaw, maneuver.gear, maneuver.rear_x)


def compile_guides(
    semantics,
    source_graph,
    profile,
    permission,
    *,
    radius_factor=1.15,
    inset=8.0,
    spacing=0.15,
    max_join_m=60.0,
    adaptive_insets=False,
    junction_alternatives=1,
    lateral_roads=(),
    lateral_offset_limit_m=0.75,
    support=None,
    automatic_lateral=False,
):
    """Conservative center corridors with exact source-authorized junctions.

    Both directions share one exclusive resource, even if metadata says the
    road is wide. Two separate usable lanes require independent native proof.
    Turns stay in the incoming/outgoing corridor and a source-width disk at
    their shared junction, intersected with explicit supplied permission. The
    disk avoids artificial slivers between angled flat road end caps; it never
    expands supplied permission or proves terrain. No nearby parking polygon
    or unrelated road authorizes a shortcut. Candidate status cannot be
    upgraded by this function.
    """
    if (
        not np.isfinite([radius_factor, inset, spacing, max_join_m]).all()
        or not np.isfinite(lateral_offset_limit_m)
        or not 0 < lateral_offset_limit_m <= 2.0
        or type(adaptive_insets) is not bool
        or type(automatic_lateral) is not bool
        or (automatic_lateral and support is None)
        or type(junction_alternatives) is not int
        or not 1 <= junction_alternatives <= 3
        or radius_factor < 1.01
        or inset <= 0
        or not 0 < spacing <= 0.5
        or not 0 < max_join_m <= 200
    ):
        raise ValueError("Bounded compiler parameters required")
    if permission.is_empty or not permission.is_valid:
        raise ValueError("Valid explicit planar permission required")
    roads = sorted(source_roads(semantics, source_graph), key=lambda r: r["id"])
    if len({r["id"] for r in roads}) != len(roads):
        raise ValueError("Duplicate semantic road ID")
    endpoints = {}
    for r in roads:
        p = np.asarray(r["line"], float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
            raise ValueError("Finite source road geometry required")
        if not math.isfinite(r["width_m"]) or r["width_m"] <= 0:
            raise ValueError("Positive source width required")
        for node, point in ((r["source_from"], p[0]), (r["source_to"], p[-1])):
            if not isinstance(node, str) or not node:
                raise ValueError("Explicit source junction identity required")
            if node in endpoints and np.linalg.norm(endpoints[node] - point) > 0.01:
                raise ValueError("Source junction identity has inconsistent coordinates")
            endpoints[node] = point
    lateral_roads = tuple(lateral_roads)
    if (
        any(not isinstance(k, str) for k in lateral_roads)
        or len(set(lateral_roads)) != len(lateral_roads)
        or set(lateral_roads) - {r["id"] for r in roads}
    ):
        raise ValueError("Unique known source roads required for lateral candidates")
    rear, wheelbase = axle(profile)
    radius = radius_factor * wheelbase / math.tan(profile.max_steer)
    graph = DrivingGraph()
    templates = {}
    counts = Counter()
    rejected = []
    used_insets = {}
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    areas = {}
    by_id = {r["id"]: r for r in roads}
    for r in roads:
        areas[r["id"]] = LineString(r["line"]).buffer(r["width_m"] / 2, cap_style=2)
    shapely.prepare(permission)

    def permitted(m, area):
        envelope = body_envelope(m, profile)
        return bool(permission.covers(envelope) and area.covers(envelope))

    for r in roads:
        counts["source_roads"] += 1
        for direction in ("forward",) if r["oneway"] else ("forward", "reverse"):
            counts["directed_road_attempts"] += 1
            reverse = direction == "reverse"
            line = r["line"][::-1] if reverse else r["line"]
            try:
                ends = np.array([inset, inset])
                if adaptive_insets:
                    # Rear-axle route endpoints need different clearances: the
                    # chassis front extends farther ahead than its rear behind.
                    # Keep the established inset on long roads. For short roads
                    # preserve half of their actual rounded length for travel,
                    # without shrinking body extents or changing permissions.
                    full = fillet_route(line, radius, inset=0.0, spacing=spacing)
                    length = float(full["arc"][-1])
                    if length <= 2 * inset + 1e-6:
                        required = np.maximum([rear - profile.body_lo[0], profile.body_hi[0] - rear], 0.0) + 0.2
                        ends = np.maximum(required, min(inset, length / 4))
                        counts["adaptive_inset_attempts"] += 1
                route = fillet_route(line, radius, inset=ends, spacing=spacing)
                m = Maneuver(route["points"], route["yaw"], route["gear"], rear)
            except ValueError as exc:
                rejected.append(dict(road=r["id"], direction=direction, reason=str(exc)))
                counts["road_geometry_rejected"] += 1
                continue
            offsets = (0.0,)
            if automatic_lateral or r["id"] in lateral_roads:
                # An offline alternative centreline, not added road width or
                # a second independently usable traffic resource. Leave a
                # quarter metre beyond the full body on either source side.
                room = min(
                    lateral_offset_limit_m, (r["width_m"] - (profile.body_hi[1] - profile.body_lo[1])) / 2 - 0.25
                )
                if room > 0.05:
                    offsets = (0.0, room, -room)
            original = m
            for variant, source_offset in enumerate(offsets):
                offset = -source_offset if reverse else source_offset
                try:
                    m = parallel_guide(original, offset)
                except ValueError:
                    counts["lateral_geometry_rejected"] += 1
                    continue
                if variant and (np.abs(np.diff(m.yaw)) / np.diff(m.arc) > 1 / radius + 1e-5).any():
                    counts["lateral_curvature_rejected"] += 1
                    continue
                if not permitted(m, areas[r["id"]]):
                    rejected.append(
                        dict(road=r["id"], direction=direction, offset_m=offset, reason="planar_body_outside_road")
                    )
                    counts["road_permission_rejected"] += 1
                    continue
                if support is not None:
                    from .guide_support import check_centers

                    check = check_centers(body_route(m)["points"], support)
                    if check["rejected_samples"]:
                        # Berth ports later split long source roads. Keep their
                        # supported subsegments available; one distant obstacle
                        # must not delete an otherwise usable local entrance.
                        # Preserve originals, and omit only wholly unsupported
                        # new lateral candidates. Final split-edge screening is
                        # mandatory before any native path is selected.
                        if variant and check["rejected_samples"] == check["samples"]:
                            rejected.append(
                                dict(
                                    road=r["id"],
                                    direction=direction,
                                    offset_m=offset,
                                    reason="entire_lateral_center_unsupported",
                                    support=check,
                                )
                            )
                            counts["road_support_rejected"] += 1
                            continue
                        counts["road_support_deferred_until_split"] += 1
                source, target = (r["source_to"], r["source_from"]) if reverse else (r["source_from"], r["source_to"])
                name = f"source:{r['id']}:{direction}" + (f":lateral:{variant}" if variant else "")
                a, b = name + ":in", name + ":out"
                used_insets[name] = ends.tolist()
                for node, index, junction in ((a, 0, source), (b, -1, target)):
                    graph.add_node(
                        node,
                        m.body_pose(index),
                        kind="lane",
                        road=r["id"],
                        direction=direction,
                        source_junction=junction,
                        traffic_resource="road:" + r["id"],
                        traffic_mode="oneway" if r["oneway"] else "exclusive_bidirectional",
                        guide_offset_m=offset,
                        geometry_reference="rear_axle_guide_body_origin_graph",
                    )
                graph.add_edge(name, a, b, body_route(m), kind="travel", traffic_resource="road:" + r["id"])
                templates[name] = m
                incoming[target].append(b)
                outgoing[source].append(a)
                counts["road_candidates"] += 1
                if variant:
                    counts["lateral_road_candidates"] += 1

    for junction in sorted(set(incoming) & set(outgoing)):
        for a in sorted(incoming[junction]):
            for b in sorted(outgoing[junction]):
                ra, rb = graph.nodes[a]["road"], graph.nodes[b]["road"]
                if ra == rb:  # A source dead end does not authorize a U-turn.
                    continue
                counts["junction_attempts"] += 1
                start = templates[a[:-4]].points[-1]
                end = templates[b[:-3]].points[0]
                yaw0 = graph.nodes[a]["pose"][2]
                yaw1 = graph.nodes[b]["pose"][2]
                junction_disk = Point(endpoints[junction]).buffer(max(by_id[ra]["width_m"], by_id[rb]["width_m"]) / 2)
                area = areas[ra].union(areas[rb]).union(junction_disk)
                radii = junction_radii([*start, yaw0], [*end, yaw1], radius)
                options = sorted(
                    (
                        r
                        for rad in radii
                        for r in tangent_paths(
                            [*start, yaw0], [*end, yaw1], rad, spacing=spacing, max_length_m=max_join_m
                        )
                    ),
                    key=lambda r: float(r["arc"][-1]),
                )
                counts["junction_radius_trials"] += len(radii)
                chosen = []
                seen = set()
                for route in options:
                    if route["arc"][-1] > max_join_m:
                        continue
                    try:
                        m = Maneuver(route["points"], route["yaw"], route["gear"], rear)
                    except ValueError:
                        continue
                    errors = []
                    for node, index in ((a, 0), (b, -1)):
                        actual = m.body_pose(index)
                        expected = graph.nodes[node]["pose"]
                        errors.append(
                            (
                                float(np.linalg.norm(actual[:2] - expected[:2])),
                                abs(math.remainder(float(actual[2] - expected[2]), 2 * math.pi)),
                            )
                        )
                    if max(v for pair in errors for v in pair) > 1e-8:
                        counts["junction_seam_rejected"] += 1
                        rejected.append(
                            dict(
                                junction=junction,
                                source=a,
                                target=b,
                                reason="analytic_junction_endpoint_drift",
                                endpoint_errors=errors,
                            )
                        )
                        continue
                    if permitted(m, area):
                        if support is not None:
                            from .guide_support import check_centers

                            check = check_centers(body_route(m)["points"], support)
                            if check["rejected_samples"]:
                                counts["junction_support_rejected"] += 1
                                continue
                        key = (np.round(m.points, 8).tobytes(), np.round(m.yaw, 8).tobytes(), m.gear.tobytes())
                        if key in seen:
                            continue
                        seen.add(key)
                        chosen.append(m)
                        if len(chosen) >= junction_alternatives:
                            break
                if not chosen:
                    rejected.append(
                        dict(junction=junction, source=a, target=b, reason="no_bounded_permitted_junction_curve")
                    )
                    counts["junction_rejected"] += 1
                    continue
                # Keep exact source/target poses and the same exclusive road
                # resources for every alternative. Geometry ranks candidates;
                # shortest length alone is not evidence of terrain driveability.
                base = f"junction:{junction}:{a}:{b}"
                for index, m in enumerate(chosen):
                    name = base if index == 0 else base + f":curve:{index}"
                    graph.add_edge(
                        name,
                        a,
                        b,
                        body_route(m),
                        kind="join",
                        traffic_resources=("junction:" + junction, "road:" + ra, "road:" + rb),
                    )
                    templates[name] = m
                    counts["junction_candidates"] += 1
    report = dict(
        candidate_only=True,
        release_evidence=False,
        counts=dict(counts),
        rejected=rejected,
        radius_m=radius,
        radius_factor=radius_factor,
        junction_inset_m=inset,
        spacing_m=spacing,
        adaptive_insets=adaptive_insets,
        used_insets_m=used_insets,
        junction_alternatives=junction_alternatives,
        lateral_roads=sorted(lateral_roads),
        lateral_offset_limit_m=lateral_offset_limit_m,
        automatic_lateral=automatic_lateral,
        measured_support_prefilter=support is not None,
        requires_split_support_prefilter=support is not None,
        junction_radius_family="minimum_midpoint_and_endpoint_tangent_fit",
        max_join_m=max_join_m,
        source_road_ids=sorted(by_id),
        junction_geometry="source_width_disk_intersected_with_supplied_permission",
        component_sizes=[len(c) for c in graph.components()],
        limitations=[
            "Planar body permission and source direction only; no native terrain or traffic admission.",
            "No inferred source junctions, dead-end U-turns, independent lanes or parking access.",
            "Tight short-segment bends remain rejected; this bounded family is not a reachability oracle.",
        ],
    )
    return graph, templates, report
