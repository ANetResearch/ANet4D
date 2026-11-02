"""Recover omitted source-road directions with a finite gentler-radius family.

An aggressive fillet failing whole-body containment does not prove the road
unusable. Try larger steering radii inside the SAME source corridor. These are
offline geometric candidates, not new road width, support or traffic proof.
Existing directions, nodes, edges and resource identities remain untouched.
"""

import math

import numpy as np
from shapely.geometry import LineString

from awr.life.civil.driving_graph import load_candidate

from .guide_support import check_centers
from .maneuver import Maneuver, axle
from .road_guides import body_envelope, body_route, fillet_route


def extend(
    document, roads, profile, permission, support, *, factors=(1.5, 2.0, 2.5, 3.0), max_directions=64, inset_m=8.0
):
    factors = tuple(factors)
    if (
        not 1 <= len(factors) <= 6
        or not np.isfinite(factors).all()
        or any(not 1.01 <= x <= 4 for x in factors)
        or tuple(sorted(set(factors))) != factors
        or type(max_directions) is not int
        or not 1 <= max_directions <= 128
        or not np.isfinite(inset_m)
        or not 0 < inset_m <= 12
        or permission.is_empty
        or not permission.is_valid
    ):
        raise ValueError("Finite increasing radius family and bounded source budget required")
    if len({r["id"] for r in roads}) != len(roads):
        raise ValueError("Unique source roads required")
    graph = load_candidate(document)
    templates = {k: Maneuver(**v) for k, v in document["rear_axle_templates"].items()}
    represented = {(n.get("road"), n.get("direction")) for n in graph.nodes.values() if n.get("kind") == "lane"}
    rear, wb = axle(profile)
    minimum = wb / math.tan(profile.max_steer)
    attempts = []
    added = []
    budget = 0
    for road in sorted(roads, key=lambda r: r["id"]):
        points = np.asarray(road["line"], float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 2
            or not np.isfinite(points).all()
            or not math.isfinite(road["width_m"])
            or road["width_m"] <= 0
            or not all(isinstance(road[k], str) and road[k] for k in ("source_from", "source_to"))
        ):
            raise ValueError("Finite source geometry and explicit junction identities required")
        allowed = LineString(points).buffer(road["width_m"] / 2, cap_style=2).intersection(permission)
        for direction in ("forward",) if road["oneway"] else ("forward", "reverse"):
            if (road["id"], direction) in represented:
                continue
            row = dict(road=road["id"], direction=direction, trials=[], candidate_found=False)
            attempts.append(row)
            if budget >= max_directions:
                row["reason"] = "direction_budget"
                continue
            budget += 1
            reverse = direction == "reverse"
            line = points[::-1] if reverse else points
            for factor in factors:
                trial = dict(radius_factor=factor)
                row["trials"].append(trial)
                try:
                    full = fillet_route(line, minimum * factor, inset=0.0)
                    required = np.maximum([rear - profile.body_lo[0], profile.body_hi[0] - rear], 0.0) + 0.2
                    ends = np.repeat(inset_m, 2)
                    if full["arc"][-1] <= 2 * inset_m + 1e-6:
                        ends = np.maximum(required, min(inset_m, float(full["arc"][-1]) / 4))
                    r = fillet_route(line, minimum * factor, inset=ends)
                    m = Maneuver(r["points"], r["yaw"], r["gear"], rear)
                except ValueError as exc:
                    trial["reason"] = str(exc)
                    continue
                if not allowed.covers(body_envelope(m, profile)):
                    trial["reason"] = "planar_body_outside_source_permission"
                    continue
                measured = check_centers(body_route(m)["points"], support)
                trial["support"] = measured
                if measured["rejected_samples"] == measured["samples"]:
                    trial["reason"] = "entire_center_unsupported"
                    continue
                # Partly supported roads may supply usable endpoint/subsegments.
                # Full-edge support remains required before any route executes.
                key = f"radius-road:{road['id']}:{direction}"
                if key in graph.edges:
                    raise ValueError("Recovered source key already exists")
                a, b = key + ":in", key + ":out"
                junctions = (
                    (road["source_to"], road["source_from"]) if reverse else (road["source_from"], road["source_to"])
                )
                for node, index, junction in ((a, 0, junctions[0]), (b, -1, junctions[1])):
                    graph.add_node(
                        node,
                        m.body_pose(index),
                        kind="lane",
                        road=road["id"],
                        direction=direction,
                        source_junction=junction,
                        traffic_resource="road:" + road["id"],
                        traffic_mode="oneway" if road["oneway"] else "exclusive_bidirectional",
                        guide_offset_m=0.0,
                        geometry_reference="rear_axle_guide_body_origin_graph",
                    )
                graph.add_edge(key, a, b, body_route(m), kind="travel", traffic_resource="road:" + road["id"])
                templates[key] = m
                added.append(key)
                row.update(
                    candidate_found=True,
                    edge=key,
                    radius_factor=factor,
                    requires_split_support_prefilter=bool(measured["rejected_samples"]),
                )
                trial["reason"] = "planar_candidate_found"
                break
    return (
        graph,
        templates,
        dict(
            candidate_only=True,
            release_evidence=False,
            added_roads=added,
            rows=attempts,
            directions_attempted=budget,
            limits=dict(factors=factors, max_directions=max_directions, inset_m=inset_m),
            limitations=[
                "Absent source-road directions only; existing guides are unchanged.",
                "Original width, permission, one-way rule and exclusive resources are preserved.",
                "A partly supported candidate must be split and rescreened; it is not an executable road.",
                "Source junction connections, fixed occupancy, native physics and purpose remain unverified.",
            ],
        ),
    )
