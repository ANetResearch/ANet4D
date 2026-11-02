"""Offline berth-cycle nomination over real edges and explicitly unplanned turns.

Unplanned links are topology hypotheses only. They have no motion geometry,
never enter DrivingGraph, and must be compiled and physically checked before
execution. No berth relocation, resident schedule or per-actor script is used.
"""

import hashlib
import heapq
import math
from collections import defaultdict

import numpy as np
from shapely.geometry import Point, Polygon

from .missing_turns import source_turn_pairs


def nominate(
    graph,
    roads,
    blocked_edges,
    parking_regions=(),
    *,
    max_results=12,
    min_separation_m=20.0,
    max_berths=400,
    purpose_screen=None,
    deferred_turn_pairs=(),
    require_supported_edges=False,
):
    if (
        type(max_results) is not int
        or not 1 <= max_results <= 100
        or type(max_berths) is not int
        or not 1 <= max_berths <= 1000
        or not np.isfinite(min_separation_m)
        or min_separation_m < 0
    ):
        raise ValueError("Bounded cycle nomination required")
    if type(require_supported_edges) is not bool:
        raise ValueError("Explicit support-screen policy required")
    blocked = set(blocked_edges)
    if blocked - set(graph.edges):
        raise ValueError("Unknown measured-support edge")
    berths = sorted(n for n, v in graph.nodes.items() if v["kind"] == "berth")
    if len(berths) > max_berths:
        raise ValueError("Berth nomination budget exceeded")
    parking = []
    for r in parking_regions:
        if r.get("verified") is not True:
            continue
        area = Polygon(r["polygon"])
        if area.is_empty or not area.is_valid:
            raise ValueError("Invalid verified parking area")
        parking.append(area)
    membership = {n: {i for i, p in enumerate(parking) if p.covers(Point(graph.nodes[n]["pose"][:2]))} for n in berths}
    hypothetical = {}
    adj = {n: sorted(graph.links[n]) for n in graph.nodes}
    # Compact traversal data deliberately cannot be loaded as executable routes.
    edges = {
        k: dict(source=e["source"], target=e["target"], kind=e["kind"], length_m=float(e["route"]["arc"][-1]))
        for k, e in graph.edges.items()
    }
    for pair in source_turn_pairs(graph, roads):
        a, b = pair["source"], pair["target"]
        key = "unplanned-turn:" + hashlib.sha256((a + "|" + b).encode()).hexdigest()[:24]
        if key in edges:
            raise ValueError("Unplanned identity collides with an executable edge")
        hypothetical[key] = pair
        edges[key] = dict(
            source=a,
            target=b,
            kind="unplanned",
            length_m=float(np.linalg.norm(graph.nodes[a]["pose"][:2] - graph.nodes[b]["pose"][:2])),
        )
        adj[a].append(key)
    deferred_pairs = {tuple(p) for p in deferred_turn_pairs}
    by_pair = {(p["source"], p["target"]): k for k, p in hypothetical.items()}
    if len(deferred_pairs) > 8192 or deferred_pairs - set(by_pair):
        raise ValueError("Unknown deferred source turn")
    deferred = {by_pair[p] for p in deferred_pairs}
    searches = {}
    for source in berths:
        best = {source: (0, 0, 0, 0.0)}
        prev = {}
        queue = [(0, 0, 0, 0.0, source)]
        while queue:
            failed, bad, missing, d, node = heapq.heappop(queue)
            if (failed, bad, missing, d) != best[node]:
                continue
            if node != source and graph.nodes[node]["kind"] == "berth":
                continue
            for key in sorted(adj[node]):
                if require_supported_edges and key in blocked:
                    continue
                e = edges[key]
                if e["kind"] == "exit" and node != source:
                    continue
                absent = key in hypothetical
                target = e["target"]
                value = (
                    failed + int(key in deferred),
                    bad + int(key in blocked) + int(absent),
                    missing + int(absent),
                    d + e["length_m"],
                )
                if value < best.get(target, (math.inf,) * 4):
                    best[target] = value
                    prev[target] = key
                    heapq.heappush(queue, (*value, target))
        searches[source] = (best, prev)
    rows = []
    excluded = defaultdict(int)
    purpose_rejections = defaultdict(int)
    topology_pairs = 0
    for i, a in enumerate(berths):
        for b in berths[i + 1 :]:
            if graph.nodes[a].get("region") == graph.nodes[b].get("region"):
                excluded["same_region"] += 1
                continue
            if membership[a] & membership[b]:
                excluded["same_verified_parking_area"] += 1
                continue
            separation = float(np.linalg.norm(graph.nodes[a]["pose"][:2] - graph.nodes[b]["pose"][:2]))
            if separation < min_separation_m:
                excluded["below_separation"] += 1
                continue
            if b not in searches[a][0] or a not in searches[b][0]:
                excluded["no_source_topology_roundtrip"] += 1
                continue
            costs = [searches[a][0][b], searches[b][0][a]]
            paths = []
            topology_pairs += 1
            purpose = None
            if purpose_screen is not None:
                purpose = purpose_screen.evaluate(a, b, [costs[0][3], costs[1][3]])
                for reason, count in purpose["rejected"].items():
                    purpose_rejections[reason] += count
                if not purpose["eligible"]:
                    excluded["no_credible_activity_option"] += 1
                    continue
            for source, target in ((a, b), (b, a)):
                path = []
                cursor = target
                while cursor != source:
                    key = searches[source][1][cursor]
                    path.append(key)
                    cursor = edges[key]["source"]
                paths.append(path[::-1])
            missing = sorted({k for path in paths for k in path if k in hypothetical})
            rows.append(
                dict(
                    pair=[a, b],
                    separation_m=separation,
                    ranking_cost=[sum(c[j] for c in costs) for j in range(1, 4)],
                    previously_failed_turns=[hypothetical[k] for k in missing if k in deferred],
                    topology_paths=paths,
                    required_turns=[hypothetical[k] for k in missing],
                    support_rejected_edges=sorted({k for path in paths for k in path if k in blocked}),
                )
            )
            if purpose is not None:
                rows[-1]["purpose_screen"] = purpose

    def rank(row):
        unknown = purpose_screen is not None and all(
            x["known_walk_roundtrip_s"] is None for x in row["purpose_screen"]["options"]
        )
        return (len(row["previously_failed_turns"]), unknown, row["ranking_cost"], row["pair"])

    rows.sort(key=rank)
    return dict(
        candidate_only=True,
        release_evidence=False,
        topology_only=True,
        executable=False,
        berths=len(berths),
        dijkstra_runs=len(searches),
        unplanned_turns=len(hypothetical),
        possible_pairs=len(rows),
        excluded=dict(excluded),
        rows=rows[:max_results],
        topology_pairs=topology_pairs,
        deferred_turns=len(deferred),
        require_supported_edges=require_supported_edges,
        no_screened_cycle=require_supported_edges and not rows,
        support_rejected_edge_count=len(blocked),
        pairs_revisiting_failed_proposals=sum(bool(r["previously_failed_turns"]) for r in rows),
        purpose_screened=purpose_screen is not None,
        purpose_rejections=dict(purpose_rejections),
        known_walk_comparison_pairs=sum(
            any(x["known_walk_roundtrip_s"] is not None for x in r["purpose_screen"]["options"]) for r in rows
        )
        if purpose_screen
        else 0,
        limits=dict(
            max_results=max_results, min_separation_m=min_separation_m, max_berths=max_berths, max_unplanned_turns=8192
        ),
        ranking=["fewer previously failed proposals (no topology exclusion)"]
        + (["known walking alternative before unknown"] if purpose_screen else [])
        + [
            "support-rejected edges plus unplanned turns",
            "unplanned turns",
            "length with straight-line lower bounds for unplanned turns",
        ],
        limitations=[
            "Ranking is a bounded candidate-family heuristic, not geometric reachability or native admission.",
            "Support rejection applies to the current sampled candidate graph, not all possible curves within real roads.",
            "Distinct regions and parking areas do not establish different facility purposes or pedestrian access.",
        ],
    )
