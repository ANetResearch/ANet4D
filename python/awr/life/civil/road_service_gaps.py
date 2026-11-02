"""Locate missing directed transitions without inventing traversable geometry.

For each missing service, compare nodes actually reachable from its origins
with nodes that can actually reach its destinations. Berths remain terminals.
Close frontier poses are candidates for physical review, never permission to
merge identities, join disconnected paths or widen a road.
"""

import heapq
import json
import math
from collections import Counter, defaultdict, deque

from .road_service_geometry import Stationing


def _reach(starts, links, kinds):
    previous = {n: None for n in starts}
    queue = deque(sorted(starts))
    while queue:
        node = queue.popleft()
        if kinds[node] == "berth" and previous[node] is not None:
            continue
        for edge, other in links[node]:
            if other not in previous:
                previous[other] = (node, edge)
                queue.append(other)
    return previous


def _path(previous, node):
    edges = []
    while previous[node] is not None:
        node, edge = previous[node]
        edges.append(edge)
    return node, edges[::-1]


def inspect(document, source_graph, contract, *, closest=3):
    if type(closest) is not int or not 1 <= closest <= 20:
        raise ValueError("Use a bounded positive number of frontier candidates")
    nodes = document["nodes"]
    edges = document["edges"]
    kinds = {n: d["kind"] for n, d in nodes.items()}
    forward = {n: [] for n in nodes}
    backward = {n: [] for n in nodes}
    road_nodes = defaultdict(set)
    for n, d in nodes.items():
        if d["kind"] == "lane":
            road_nodes[(d.get("traffic_resource"), d.get("direction"))].add(n)
    for eid, e in sorted(edges.items()):
        a, b = e["source"], e["target"]
        forward[a].append((eid, b))
        backward[b].append((eid, a))
    source_roads = {}
    for f in source_graph["features"]:
        p = f["properties"]
        if p.get("feature") == "edge" and "vehicle" in p.get("agents", []):
            if p["edge_id"] in source_roads:
                raise ValueError("Duplicate source road")
            source_roads[p["edge_id"]] = Stationing(f["geometry"]["coordinates"])
    interfaces = contract["interfaces"]
    cache = {}
    projections = {}
    rows = []

    def reach(key, reverse=False):
        ck = (key, reverse)
        if ck not in cache:
            cache[ck] = _reach(interfaces[key], backward if reverse else forward, kinds)
        return cache[ck]

    def station(name, road, direction):
        key = (name, road, direction)
        if key not in projections:
            projections[key] = source_roads[road].project(nodes[name]["pose"], direction)
        return projections[key]

    for c in sorted(contract["connections"], key=lambda c: c["id"]):
        aroad, _, aside, adir = json.loads(c["source"])
        broad, _, bside, bdir = json.loads(c["target"])
        if not (aroad == broad and adir == bdir and aside == "outgoing" and bside == "incoming"):
            continue
        source, target = reach(c["source"]), reach(c["target"], True)
        reached = sorted(set(source) & set(interfaces[c["target"]]))
        if reached:
            start, path = _path(source, reached[0])
            rows.append(
                dict(id=c["id"], status="whole_network_path", source_node=start, target_node=reached[0], edges=path)
            )
            continue
        candidates = road_nodes[("road:" + aroad, adir)]
        left, right = sorted(candidates & source.keys()), sorted(candidates & target.keys())

        def pairs(positive_only):
            for a in left:
                pa = nodes[a]["pose"]
                sa = station(a, aroad, adir)
                for b in right:
                    pb = nodes[b]["pose"]
                    sb = station(b, broad, bdir)
                    alo, ahi = sa["station_range_m"]
                    blo, bhi = sb["station_range_m"]
                    delta = [blo - ahi, bhi - alo] if adir == "forward" else [alo - bhi, ahi - blo]
                    if positive_only and delta[0] <= 1e-6:
                        continue
                    yield math.dist(pa[:2], pb[:2]), a, b, delta

        def describe(pair):
            distance, a, b, delta = pair
            pa, pb = nodes[a]["pose"], nodes[b]["pose"]
            start, approach = _path(source, a)
            end, reversed_exit = _path(target, b)
            # Backward BFS records reversed traversal; restore original edges.
            return dict(
                source_node=a,
                target_node=b,
                distance_m=distance,
                source_pose=pa,
                target_pose=pb,
                directed_station_delta_range_m=delta,
                yaw_difference_deg=abs(math.degrees((pb[2] - pa[2] + math.pi) % (2 * math.pi) - math.pi)),
                origin_node=start,
                approach_edges=approach,
                destination_node=end,
                departure_edges=reversed_exit[::-1],
                physically_verified=False,
            )

        rows.append(
            dict(
                id=c["id"],
                status="missing_transition",
                road=aroad,
                direction=adir,
                origin_reachable_nodes=len(source),
                destination_coreachable_nodes=len(target),
                same_road_origin_nodes=len(left),
                same_road_destination_nodes=len(right),
                nearest=[describe(p) for p in heapq.nsmallest(closest, pairs(False))],
                nearest_positive_station=[describe(p) for p in heapq.nsmallest(closest, pairs(True))],
            )
        )
    return dict(
        schema="awr.life.road_service_gaps.v1",
        rows=rows,
        summary=dict(
            status_counts=dict(Counter(r["status"] for r in rows)),
            missing_without_same_road_frontiers=sum(
                r["status"] == "missing_transition" and not r["nearest"] for r in rows
            ),
        ),
        admitted=False,
        scope="Whole-network terminal-aware topology. Closest pairs are bounded diagnostic "
        "candidates, not an exhaustive connector search or physical permission. Source/destination legs "
        "are real edge paths; the gap has no edge. No occupancy, map, engine or population changed.",
    )
