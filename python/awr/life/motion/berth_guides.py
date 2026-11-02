"""Compile fixed mapped berths into a source-directed axle road graph.

Parking rows/curb metadata authorize candidate access to named roads. They do
not prove occupied capacity, native driveability, pedestrian doors or facility
purpose. All candidates remain subject to the same physical/traffic gates.
No actor agenda, spawn time, teleport or per-person trajectory is generated.
"""

import math
from collections import Counter
from copy import deepcopy

import numpy as np
from shapely.geometry import LineString, Point, Polygon, mapping

from awr.life.civil.driving_graph import load_candidate
from awr.life.civil.road_corridors import source_roads

from .maneuver import Maneuver, axle, berth_candidates, body_to_rear
from .road_guides import body_envelope, body_route


def without_access(document):
    """Explicitly rebuild access after road geometry changes, preserving roads.

    Old access proofs remain in their parent artifact; they cannot be silently
    inherited by a map with new mid-road geometry or different parking policy.
    """
    graph = load_candidate(document)
    berths = {k for k, v in graph.nodes.items() if v.get("kind") == "berth"}
    removed = set()
    for k, e in graph.edges.items():
        touches = e["source"] in berths or e["target"] in berths
        if e["kind"] in ("entry", "exit"):
            if not touches:
                raise ValueError("Access edge has no berth")
            removed.add(k)
        elif touches:
            raise ValueError("Cannot discard a non-access berth connection")
    out = deepcopy(document)
    out["nodes"] = {k: v for k, v in out["nodes"].items() if k not in berths}
    for field in ("edges", "rear_axle_templates"):
        out[field] = {k: v for k, v in out[field].items() if k not in removed}
    return out, dict(
        retired_berths=len(berths),
        retired_access_edges=len(removed),
        retained_road_edges=len(out["edges"]),
        old_physical_admission_inherited=False,
    )


def access_roads(node, roads, parking_regions):
    region = node.get("region", "")
    if region.startswith("curb:"):
        name = region[len("curb:") :]
        if name not in roads or node.get("parking_legality", {}).get("legal") is not True:
            return [], None
        # Legacy labels can call a slot "curb" even inside a surveyed parking
        # area. Use that area's explicit source-aisle authority only when it
        # contains this mapped berth; proximity to an unrelated lot is not enough.
        area = None
        if "pose" in node:
            for region in parking_regions.values():
                if region.get("verified") is not True or roads[name]["source_osm_id"] not in region.get(
                    "source_aisles", []
                ):
                    continue
                polygon = Polygon(region["polygon"])
                if polygon.is_empty or not polygon.is_valid:
                    raise ValueError("Mapped parking permission must be valid")
                if polygon.covers(Point(node["pose"][:2])):
                    area = polygon if area is None else area.union(polygon)
        if node.get("curb_maneuver_policy") is not None:
            from .curb_maneuver import domain

            area = domain(node, roads[name])
        return [name], area
    area = parking_regions.get(region)
    if not area or area.get("verified") is not True:
        return [], None
    sources = set(area.get("source_aisles", []))
    names = sorted(k for k, r in roads.items() if r["source_osm_id"] in sources)
    polygon = Polygon(area["polygon"])
    if polygon.is_empty or not polygon.is_valid:
        raise ValueError("Mapped parking permission must be valid")
    return names, polygon


def compile_access(
    document,
    inventory,
    semantics,
    source_graph,
    profile,
    permission,
    *,
    berth_ids=None,
    offsets_m=(6.0, 12.0, 18.0),
    candidates_per_port=2,
    max_access_length_m=50.0,
    max_distance_m=20.0,
    max_berths=100,
    curve_trials_per_port=32,
    use_reeds_shepp=False,
    search_expansions=0,
    max_search_ports=4,
    steering_fraction=0.8,
    allow_curb_maneuvers=False,
    support=None,
):
    """Return a new graph; split travel edges only at exact existing samples.

    For each permitted road direction, an exit merges downstream of the berth's
    nearest sample and an entry leaves the lane upstream. Curves may reverse
    only inside the access maneuver, with explicit gear-stop locations. Whole
    road resources are retained conservatively until traffic capacity is proven.
    Candidate enumeration is bounded and performed at map compilation time.
    """
    offsets = np.asarray(offsets_m, float)
    if (
        offsets.ndim != 1
        or not 1 <= len(offsets) <= 8
        or not np.isfinite(offsets).all()
        or np.any(offsets <= 0)
        or len(set(offsets)) != len(offsets)
        or type(candidates_per_port) is not int
        or not 1 <= candidates_per_port <= 4
        or type(curve_trials_per_port) is not int
        or not 1 <= curve_trials_per_port <= 256
        or type(use_reeds_shepp) is not bool
        or type(allow_curb_maneuvers) is not bool
        or type(search_expansions) is not int
        or not 0 <= search_expansions <= 2000
        or type(max_search_ports) is not int
        or not 0 <= max_search_ports <= 100
        or not np.isfinite(steering_fraction)
        or not 0 < steering_fraction <= 1
        or not np.isfinite([max_access_length_m, max_distance_m]).all()
        or not 0 < max_distance_m <= 50
        or not 0 < max_access_length_m <= 100
        or type(max_berths) is not int
        or not 1 <= max_berths <= 1000
    ):
        raise ValueError("Explicit bounded berth/port/candidate budget required")
    if permission.is_empty or not permission.is_valid:
        raise ValueError("Valid explicit road permission required")
    graph = load_candidate(document)
    templates = {k: Maneuver(**v) for k, v in document["rear_axle_templates"].items()}
    if set(templates) != set(graph.edges):
        raise ValueError("Every parent edge requires its rear template")
    rear, wheelbase = axle(profile)
    for key in templates:
        # Avoid reconstructing the complete graph for each edge. Exact body
        # positions/headings and the physical axle are checked here.
        m = templates[key]
        r = body_route(m)
        q = graph.edges[key]["route"]
        if (
            abs(m.rear_x - rear) > 1e-8
            or r["points"].shape != q["points"].shape
            or not np.allclose(r["points"], q["points"], rtol=0, atol=1e-8)
            or not np.allclose(np.cos(r["yaw"]), np.cos(q["yaw"]), rtol=0, atol=1e-8)
            or not np.allclose(np.sin(r["yaw"]), np.sin(q["yaw"]), rtol=0, atol=1e-8)
            or not np.array_equal(r["gear"], q["gear"])
        ):
            raise ValueError("Parent body graph and rear template differ")
    raw_roads = source_roads(semantics, source_graph)
    raw_regions = semantics.get("verified_parking_regions", [])
    if (
        len({r["id"] for r in raw_roads}) != len(raw_roads)
        or len({r["id"] for r in raw_regions}) != len(raw_regions)
        or len({r["berth"] for r in inventory["rows"]}) != len(inventory["rows"])
    ):
        raise ValueError("Duplicate road, region or inventory berth identity")
    roads = {r["id"]: r for r in raw_roads}
    regions = {r["id"]: r for r in raw_regions}
    rows = {r["berth"]: r for r in inventory["rows"] if r["physical_berth_pass"] is True}
    selected = sorted(rows if berth_ids is None else berth_ids)
    if not selected or len(selected) > max_berths or len(set(selected)) != len(selected) or set(selected) - set(rows):
        raise ValueError("Unique physically screened berth roster within budget required")
    travels = sorted(k for k, e in graph.edges.items() if e["kind"] == "travel")
    counts = Counter()
    rejected = []
    additions = []
    splits = {}
    searches = []
    curb_domains = {}
    for name in selected:
        node = dict(rows[name]["mapped_node"])
        pose = np.asarray(node["pose"], float)
        if node.get("kind") != "berth" or pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("Fixed finite mapped berth required")
        allowed_roads, park = access_roads(node, roads, regions)
        if allow_curb_maneuvers and park is None and allowed_roads and node.get("region", "").startswith("curb:"):
            from .curb_maneuver import POLICY, REVERSE_LIMIT_M, REVERSE_SPEED_MPS

            proposed = {**node, "curb_maneuver_policy": POLICY}
            _, area = access_roads(proposed, roads, regions)
            if area is not None:
                node = proposed
                park = area
                curb_domains[name] = dict(
                    source_roads=allowed_roads,
                    geometry=mapping(area),
                    policy=POLICY,
                    max_reverse_leg_m=REVERSE_LIMIT_M,
                    reverse_speed_mps=REVERSE_SPEED_MPS,
                )
        if not allowed_roads:
            rejected.append(dict(berth=name, reason="no_verified_source_road_access_metadata"))
            continue
        counts["selected_berths"] += 1
        graph.add_node(name, pose, **{k: v for k, v in node.items() if k != "pose"})
        for edge_id in travels:
            edge = graph.edges[edge_id]
            road_id = graph.nodes[edge["source"]].get("road")
            if road_id not in allowed_roads:
                continue
            lane = templates[edge_id]
            body = edge["route"]["points"]
            near = int(np.argmin(np.linalg.norm(body - pose[:2], axis=1)))
            if np.linalg.norm(body[near] - pose[:2]) > max_distance_m:
                continue
            road_area = LineString(roads[road_id]["line"]).buffer(roads[road_id]["width_m"] / 2, cap_style=2)
            area = road_area if park is None else road_area.union(park)
            for kind, sign in (("exit", 1), ("entry", -1)):
                indices = set()
                for offset in sorted(offsets):
                    station = lane.arc[near] + sign * offset
                    if station < 0 or station > lane.arc[-1]:
                        continue
                    indices.add(int(np.argmin(abs(lane.arc - station))))
                for sample in sorted(indices):
                    counts["port_attempts"] += 1
                    lane_pose = lane.body_pose(sample)
                    start, end = (pose, lane_pose) if kind == "exit" else (lane_pose, pose)
                    options = berth_candidates(
                        profile,
                        start,
                        end,
                        reverse_departure_m=(0.0, 2.0, 4.0) if kind == "exit" else (0.0,),
                        reverse_arrival_m=(0.0, 2.0, 4.0) if kind == "entry" else (0.0,),
                        allow_three_point=True,
                        max_length_m=max_access_length_m,
                        approach_m=1.5 * wheelbase if kind == "entry" else 0.0,
                    )
                    if use_reeds_shepp and park is not None:
                        from awr.life.civil.reeds_shepp import candidates as rs_candidates

                        rs = []
                        for runway in (1.5 * wheelbase, -1.5 * wheelbase) if kind == "entry" else (0.0,):
                            rs.extend(
                                rs_candidates(
                                    body_to_rear(start, rear),
                                    body_to_rear(end, rear),
                                    profile.body_hi - profile.body_lo,
                                    max_length_m=max_access_length_m,
                                    spacing=0.15,
                                    arrival_m=runway,
                                    turn_radius_m=(
                                        1.01 * wheelbase / (math.tan(profile.max_steer) * steering_fraction)
                                        if name in curb_domains
                                        else None
                                    ),
                                )
                            )
                        for r in rs:
                            # Merge into the through road in forward gear.
                            # Entry may begin by backing only within the mapped
                            # maneuver area: concatenation inserts the actual
                            # forward/reverse stop at this exact lane port.
                            if kind == "exit" and r["gear"][-1] != 1:
                                continue
                            options.append(Maneuver(r["points"], r["yaw"], r["gear"], rear))
                        options.sort(key=lambda m: (float(m.arc[-1]), int(np.count_nonzero(np.diff(m.gear)))))
                    accepted = 0
                    seen = set()
                    for index, m in enumerate(options[:curve_trials_per_port]):
                        counts["curve_attempts"] += 1
                        # Reserve physical steering authority for feedback at
                        # parking cusps. This tightens geometric admission; it
                        # does not alter the vehicle's actual steering limits.
                        if (
                            np.abs(np.diff(m.yaw)) / np.diff(m.arc)
                            > math.tan(profile.max_steer) * steering_fraction / wheelbase + 1e-8
                        ).any():
                            counts["steering_reserve"] += 1
                            continue
                        # Analytic families can return the same sampled path
                        # to roundoff. Do not spend both alternative slots on it.
                        key = (
                            tuple(np.round(m.points, 8).ravel()),
                            tuple(np.round(np.unwrap(m.yaw - m.yaw[0]), 8)),
                            m.gear.tobytes(),
                        )
                        if key in seen:
                            counts["duplicate_curve"] += 1
                            continue
                        seen.add(key)
                        # Reversing requires a surveyed parking area or the
                        # explicit bounded curb domain, not the whole road.
                        if (m.gear < 0).any():
                            if park is None:
                                counts["reverse_permission_required"] += 1
                                continue
                            if name in curb_domains:
                                from .curb_maneuver import reverse_within_limit

                                if not reverse_within_limit(m):
                                    counts["curb_reverse_leg_limit"] += 1
                                    continue
                            changes = np.r_[0, np.flatnonzero(np.diff(m.gear)) + 1, len(m.gear)]
                            reverse_ok = True
                            for a, b in zip(changes, changes[1:], strict=False):
                                if m.gear[a] == -1:
                                    reverse = Maneuver(m.points[a : b + 1], m.yaw[a : b + 1], m.gear[a:b], m.rear_x)
                                    if not park.covers(body_envelope(reverse, profile)):
                                        reverse_ok = False
                                        break
                            if not reverse_ok:
                                counts["reverse_outside_parking"] += 1
                                continue
                        envelope = body_envelope(m, profile)
                        if not permission.covers(envelope) or not area.covers(envelope):
                            counts["outside_access_permission"] += 1
                            continue
                        if support is not None:
                            from .guide_support import check_centers

                            if check_centers(body_route(m)["points"], support)["rejected_samples"]:
                                counts["measured_access_support"] += 1
                                continue
                        additions.append((name, edge_id, sample, kind, index, m))
                        splits.setdefault(edge_id, set()).add(sample)
                        accepted += 1
                        counts["access_candidates"] += 1
                        if accepted >= candidates_per_port:
                            break
                    if (
                        not accepted
                        and use_reeds_shepp
                        and park is not None
                        and search_expansions
                        and len(searches) < max_search_ports
                    ):
                        from .axle_admission import search_access

                        m, attempts = search_access(
                            profile,
                            start,
                            end,
                            kind,
                            area.intersection(permission),
                            park,
                            max_expansions=search_expansions,
                            max_length_m=max_access_length_m,
                            steering_fraction=steering_fraction,
                            support=support,
                            max_reverse_leg_m=REVERSE_LIMIT_M if name in curb_domains else None,
                        )
                        searches.append(
                            dict(
                                berth=name,
                                road_edge=edge_id,
                                sample=sample,
                                kind=kind,
                                attempts=attempts,
                                candidate_found=m is not None,
                            )
                        )
                        if m is not None and name in curb_domains:
                            from .curb_maneuver import reverse_within_limit

                            if not reverse_within_limit(m):
                                counts["curb_reverse_leg_limit"] += 1
                                m = None
                                searches[-1]["candidate_found"] = False
                                searches[-1]["final_reason"] = "curb_reverse_leg_limit"
                        if m is not None:
                            additions.append((name, edge_id, sample, kind, -1, m))
                            splits.setdefault(edge_id, set()).add(sample)
                            accepted += 1
                            counts["access_candidates"] += 1
                            counts["bounded_search_candidates"] += 1
                    if not accepted:
                        rejected.append(
                            dict(
                                berth=name,
                                road_edge=edge_id,
                                sample=sample,
                                kind=kind,
                                reason="no_bounded_permitted_access_curve",
                                generated=len(options),
                                checked=min(len(options), curve_trials_per_port),
                                curve_budget_exhausted=len(options) > curve_trials_per_port,
                            )
                        )
    ports = {}
    for edge_id, samples in sorted(splits.items()):
        edge = graph.edges.pop(edge_id)
        lane = templates.pop(edge_id)
        graph.links[edge["source"]].remove(edge_id)
        indices = sorted({0, len(lane.points) - 1, *samples})
        for sample in indices:
            port = (
                edge["source"]
                if sample == 0
                else edge["target"]
                if sample == len(lane.points) - 1
                else f"{edge_id}:port:{sample}"
            )
            if port not in graph.nodes:
                metadata = deepcopy(graph.nodes[edge["source"]])
                metadata.pop("pose")
                metadata.pop("source_junction", None)
                metadata.pop("source_sample", None)
                graph.add_node(port, lane.body_pose(sample), **metadata, source_sample=sample)
            ports[edge_id, sample] = port
        for a, b in zip(indices, indices[1:], strict=False):
            key = f"{edge_id}:segment:{a}:{b}"
            m = Maneuver(lane.points[a : b + 1], lane.yaw[a : b + 1], lane.gear[a:b], lane.rear_x)
            graph.add_edge(
                key,
                ports[edge_id, a],
                ports[edge_id, b],
                body_route(m),
                kind="travel",
                blockers=edge["blockers"],
                traffic_resource=edge["traffic_resource"],
                traffic_resources=edge["traffic_resources"],
            )
            templates[key] = m
    for berth, edge_id, sample, kind, index, m in additions:
        port = ports[edge_id, sample]
        source, target = (berth, port) if kind == "exit" else (port, berth)
        key = f"access:{berth}:{edge_id}:{sample}:{kind}:{index}"
        graph.add_edge(
            key,
            source,
            target,
            body_route(m),
            kind=kind,
            traffic_resources=("berth:" + berth, graph.nodes[port]["traffic_resource"]),
        )
        templates[key] = m
    return (
        graph,
        templates,
        dict(
            candidate_only=True,
            release_evidence=False,
            selected_berths=selected,
            counts=dict(counts),
            rejected=rejected,
            split_road_edges=sorted(splits),
            bounded_searches=searches,
            curb_maneuver_domains=curb_domains,
            measured_access_support=support is not None,
            limits=dict(
                offsets_m=offsets.tolist(),
                candidates_per_port=candidates_per_port,
                max_access_length_m=max_access_length_m,
                max_distance_m=max_distance_m,
                curve_trials_per_port=curve_trials_per_port,
                terminal_alignment_m=1.5 * wheelbase,
                steering_curvature_fraction=steering_fraction,
                search_expansions=search_expansions,
                max_search_ports=max_search_ports,
                allow_curb_maneuvers=allow_curb_maneuvers,
            ),
            planner="bounded_CSC_and_rsplan_1.0.10" if use_reeds_shepp else "bounded_CSC",
            limitations=[
                "Planar source-authorized access only; native physical arrival/hold remains mandatory.",
                "No joint occupied parking capacity, door access, facility purpose or population admission.",
            ],
        ),
    )
