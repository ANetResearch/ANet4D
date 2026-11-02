"""Source-derived straight bay clearance before analytic parking turns.

Uses pinned rsplan families at offline map compilation time. Original bodies,
neighbours, endpoints and permissions remain unchanged; candidates are not
native motion certificates. No actor-specific itinerary is produced.
"""

import hashlib
import math
from collections import Counter, defaultdict

import numpy as np
import shapely

from awr.life.civil.driving_graph import load_candidate
from awr.life.civil.reeds_shepp import candidates

from .local_road_search import MeasuredAxleAdmission
from .maneuver import Maneuver, axle, body_to_rear
from .road_guides import body_envelope, body_route


def source_runway(start, node):
    """Read a collinear bay-to-aisle measurement, without snapping its pose."""
    if (
        not start.get("parking_lot")
        or start["parking_lot"] != node.get("region")
        or start.get("legality", {}).get("legal") is not True
        or start.get("access_evidence", {}).get("kind") != "supported_straight_bay_to_aisle"
    ):
        raise ValueError("Explicit source parking bay and measured aisle access required")
    pose = np.asarray(node["pose"], float)
    original = np.r_[start["xy"], start["yaw"]]
    end = np.asarray(start["access_evidence"]["end_xy"], float)
    if (
        pose.shape != (3,)
        or end.shape != (2,)
        or not np.isfinite(np.r_[pose, original, end]).all()
        or np.linalg.norm(pose[:2] - original[:2]) > 1e-6
        or abs(math.remainder(pose[2] - original[2], 2 * math.pi)) > 1e-6
    ):
        raise ValueError("Source bay pose changed")
    heading = np.array([math.cos(pose[2]), math.sin(pose[2])])
    distance = float((end - pose[:2]) @ heading)
    if not 1.0 <= abs(distance) <= 15.0 or np.linalg.norm(end - pose[:2] - distance * heading) > 1e-6:
        raise ValueError("Measured source access must be a bounded straight runway")
    return distance


def repair(
    document,
    allocations,
    screened,
    profile,
    support,
    bodies,
    area_for_edge,
    parking_regions,
    *,
    max_ports=4,
    max_candidates_per_port=96,
):
    if (
        type(max_ports) is not int
        or not 1 <= max_ports <= 12
        or type(max_candidates_per_port) is not int
        or not 1 <= max_candidates_per_port <= 256
    ):
        raise ValueError("Bounded offline parking search required")
    graph = load_candidate(document)
    templates = {k: Maneuver(**v) for k, v in document["rear_axle_templates"].items()}
    rear, wheelbase = axle(profile)
    gate = MeasuredAxleAdmission(profile, support, steering_fraction=0.8)
    radius = 1.01 * wheelbase / (math.tan(profile.max_steer) * 0.8)
    names = sorted(bodies)
    polygons = [bodies[n].polygon() for n in names]
    tree = shapely.STRtree(polygons)
    logs = []
    added = []
    for row in sorted(screened, key=lambda r: r["actor"]):
        actor, berth = row["actor"], row["berth"]
        if row["exit"]:
            continue
        node = graph.nodes[berth]
        start = allocations[actor]["start"]
        if not start.get("parking_lot"):
            continue
        distance = source_runway(start, node)
        region = parking_regions.get(node["region"])
        if region is None or region.get("verified") is not True:
            raise ValueError("Verified source parking domain required")
        park = shapely.Polygon(region["polygon"])
        ports = {}
        for key in graph.links[berth]:
            edge = graph.edges[key]
            if edge["kind"] == "exit":
                ports.setdefault(edge["target"], key)
        # Existing surveyed lane ports only; round-robin over both directions
        # before spending the budget on neighbouring points on one lane.
        groups = defaultdict(list)
        for port, key in ports.items():
            lane = graph.nodes[port]
            groups[(lane.get("road"), lane.get("direction"))].append(
                (float(np.linalg.norm(np.asarray(lane["pose"][:2]) - node["pose"][:2])), port, key)
            )
        ordered = [sorted(v) for _, v in sorted(groups.items())]
        jobs = [g[i] for i in range(max((len(g) for g in ordered), default=0)) for g in ordered if i < len(g)][
            :max_ports
        ]
        entry = dict(actor=actor, berth=berth, measured_runway_m=distance, ports=[], added=[])
        if not jobs:
            entry["reason"] = "no_existing_source_exit_port"
        for _, port, reference in jobs:
            goal = graph.nodes[port]["pose"]
            domain = area_for_edge(graph, reference)
            families = [
                candidates(
                    body_to_rear(node["pose"], rear),
                    body_to_rear(goal, rear),
                    gate.size,
                    departure_m=distance * f,
                    max_length_m=50.0,
                    spacing=0.15,
                    turn_radius_m=radius,
                )
                for f in (0.5, 0.75, 1.0)
            ]
            options = [f[i] for i in range(max((len(f) for f in families), default=0)) for f in families if i < len(f)]
            reasons = Counter()
            found = None
            for route in options[:max_candidates_per_port]:
                if route["gear"][-1] != 1:
                    reasons["nonforward_lane_merge"] += 1
                    continue
                m = Maneuver(route["points"], route["yaw"], route["gear"], rear)
                envelope = body_envelope(m, profile).buffer(0.4)
                obstacles = [polygons[int(i)] for i in tree.query(envelope) if names[int(i)] != actor]
                ok, reason = gate.route(route, gate.size, domain, park.intersection(domain), obstacles)
                if not ok:
                    reasons[reason] += 1
                    continue
                found = m
                break
            checked = sum(reasons.values()) + int(found is not None)
            entry["ports"].append(
                dict(
                    port=port,
                    reference=reference,
                    generated=len(options),
                    checked=checked,
                    reasons=dict(reasons),
                    candidate_found=found is not None,
                    budget_exhausted=found is None and len(options) > max_candidates_per_port,
                )
            )
            if found is not None:
                key = "parking-runway:" + hashlib.sha256((berth + "|" + port).encode()).hexdigest()[:24]
                old = graph.edges[reference]
                graph.add_edge(
                    key,
                    berth,
                    port,
                    body_route(found),
                    kind="exit",
                    blockers=old["blockers"],
                    traffic_resources=old["traffic_resources"],
                )
                templates[key] = found
                entry["added"].append(key)
                added.append(key)
        logs.append(entry)
    return (
        graph,
        templates,
        dict(
            rows=logs,
            added_edges=added,
            candidates_repaired=sum(bool(r["added"]) for r in logs),
            candidate_only=True,
            release_evidence=False,
            limits=dict(
                max_ports=max_ports, max_candidates_per_port=max_candidates_per_port, runway_fractions=[0.5, 0.75, 1.0]
            ),
            planner="rsplan-1.0.10 analytic families with source-derived departure runways",
            limitations=[
                "Planar occupied access and measured center prefilter only; native driving still mandatory.",
                "Existing source lane ports only; an absent port is not invented.",
                "Every other original parked body retained; no fleet, destinations or traffic admission.",
            ],
        ),
    )
