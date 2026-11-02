"""Additional exact lane samples for geometric connection planning.

A coarse graph's first/last sample need not be a feasible turn landing. Add
physical ports along the original directed fragment, never shift a road or
snap an actor. Every new travel edge is independently re-admitted. The parent
graph is unchanged; this remains map compilation, not population admission.
"""

import math
from collections import Counter
from copy import deepcopy

import numpy as np

from .curb_allocation import footprint, swept
from .driving_graph import route_arrays
from .grounded import support_mask
from .road_corridors import directions, fragments, source_roads
from .vehicle_connections import PreparedPermission, admit, angle_delta


def add_lane_ports(
    parent, road_probe, semantics, source_graph, surface, size, allowed_area, obstacles, *, road_ids, spacing_m=1.0
):
    """Return a copied candidate graph and an audit of exact new geometry.

    `obstacles` is an explicitly supplied mapping of actor IDs to body polygons;
    intersections are recorded, not hidden by dropping parked vehicles. Parent
    blocker annotations are retained with their original parent provenance.
    Source directions, road metadata and existing exact sample poses must agree.
    """
    if not math.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError("Positive finite lane-port spacing required")
    roads = {r["id"]: r for r in source_roads(semantics, source_graph)}
    selected = set(road_ids)
    if not selected or selected - set(roads):
        raise ValueError("Known source roads required")
    size = np.asarray(size, float)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("Actual positive vehicle dimensions required")
    source_fragments = fragments(road_probe)
    for node in parent.nodes.values():
        if node.get("kind") != "lane":
            continue
        fi = node.get("fragment")
        direction = node.get("direction")
        sample = node.get("sample")
        if (
            not isinstance(fi, int)
            or not 0 <= fi < len(source_fragments)
            or direction not in directions(source_fragments[fi])
        ):
            raise ValueError("Parent lane graph requires its exact source fragment roster")
        fragment = source_fragments[fi]
        xy = np.asarray(fragment[direction + "_xy"])
        yaw = fragment[direction + "_yaw"]
        if (
            not isinstance(sample, int)
            or not 0 <= sample < len(xy)
            or np.linalg.norm(node["pose"][:2] - xy[sample]) > 1e-7
            or abs(angle_delta(node["pose"][2], yaw[sample])) > 1e-7
        ):
            raise ValueError("Existing lane port differs from its source sample")
    permission = PreparedPermission(allowed_area)
    graph = deepcopy(parent)
    counts = Counter()
    added_nodes = []
    added_edges = []
    for fi, fragment in enumerate(source_fragments):
        if fragment["road"] not in selected:
            continue
        road = roads[fragment["road"]]
        resource = "road:" + road["id"]
        if fragment.get("traffic_resource") not in (None, resource):
            raise ValueError("Fragment traffic resource differs from source road")
        available = directions(fragment)
        if road["oneway"] and "reverse" in available:
            raise ValueError("Source one-way road has reverse geometry")
        for direction in available:
            xy = np.asarray(fragment[direction + "_xy"], float)
            yaw = np.asarray(fragment[direction + "_yaw"], float)
            whole = route_arrays(xy, yaw)
            samples = {}
            for name, node in graph.nodes.items():
                if node.get("kind") != "lane" or node.get("fragment") != fi or node.get("direction") != direction:
                    continue
                sample = node.get("sample")
                if (
                    not isinstance(sample, int)
                    or not 0 <= sample < len(xy)
                    or np.linalg.norm(node["pose"][:2] - xy[sample]) > 1e-7
                    or abs(angle_delta(node["pose"][2], yaw[sample])) > 1e-7
                ):
                    raise ValueError("Existing lane port differs from its source sample")
                if sample in samples:
                    raise ValueError("Ambiguous names for one source lane sample")
                samples[sample] = name
            last = -math.inf
            for sample, (point, heading) in enumerate(zip(xy, yaw, strict=False)):
                if sample in samples:
                    last = float(whole["arc"][sample])
                    continue
                if sample != len(xy) - 1 and whole["arc"][sample] - last < spacing_m - 1e-9:
                    continue
                counts["pose_candidates"] += 1
                if not support_mask(surface, point[None], np.array([heading]), size, "vehicle")[0]:
                    counts["unsupported_pose"] += 1
                    continue
                if not permission.covers(footprint(dict(xy=point, yaw=heading), size, margin_m=0.0)):
                    counts["outside_permission_pose"] += 1
                    continue
                name = f"lane:{fi}:{direction}:{sample}"
                if name in graph.nodes:
                    raise ValueError("Source lane port identity collision")
                graph.add_node(
                    name,
                    [*point, heading],
                    kind="lane",
                    fragment=fi,
                    direction=direction,
                    sample=sample,
                    traffic_mode=fragment.get("traffic_mode", "separated_two_way"),
                    traffic_resource=resource,
                    port_origin="exact_source_sample",
                )
                samples[sample] = name
                added_nodes.append(name)
                last = float(whole["arc"][sample])
            ordered = sorted(samples)
            for a, b in zip(ordered, ordered[1:], strict=False):
                source, target = samples[a], samples[b]
                if any(graph.edges[e]["target"] == target for e in graph.links[source]):
                    continue
                route = route_arrays(xy[a : b + 1], yaw[a : b + 1])
                counts["travel_candidates"] += 1
                ok, reason = admit(surface, route, size, allowed_area=permission)
                if not ok:
                    counts["travel_" + reason] += 1
                    continue
                corridor = swept(route, size)
                blockers = [key for key, polygon in sorted(obstacles.items()) if corridor.intersects(polygon)]
                name = f"port-travel:{fi}:{direction}:{a}:{b}"
                graph.add_edge(name, source, target, route, kind="travel", blockers=blockers, traffic_resource=resource)
                added_edges.append(name)
    return graph, dict(
        counts=counts,
        added_nodes=added_nodes,
        added_edges=added_edges,
        source_roads=sorted(selected),
        spacing_m=spacing_m,
        release_evidence=False,
    )
