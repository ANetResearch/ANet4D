"""Offline occupied-source destination reachability, without berth shortcuts.

Two graph searches per source produce outward/return candidate paths for every
vacant destination. No native admission or actor-specific agenda is inferred.
"""

import heapq
import math

import numpy as np


def road_tree(graph, home, blocked, *, reverse=False):
    if home not in graph.nodes or graph.nodes[home]["kind"] != "berth":
        raise ValueError("Known home berth required")
    blocked = set(blocked)
    if blocked - set(graph.edges):
        raise ValueError("Unknown blocked edge")
    links = {}
    for key, edge in sorted(graph.edges.items()):
        if key in blocked:
            continue
        a, b = edge["source"], edge["target"]
        if reverse:
            a, b = b, a
        # Every berth except home is a terminal, never a road shortcut.
        if a != home and graph.nodes[a]["kind"] == "berth":
            continue
        links.setdefault(a, []).append((b, key, float(edge["route"]["arc"][-1])))
    best = {home: 0.0}
    previous = {}
    queue = [(0.0, home)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != best[node]:
            continue
        for target, key, length in links.get(node, ()):
            if cost + length < best.get(target, math.inf):
                best[target] = cost + length
                previous[target] = (node, key)
                heapq.heappush(queue, (cost + length, target))
    return best, previous


def path_to(previous, home, target, *, reverse=False):
    path = []
    node = target
    while node != home:
        if node not in previous:
            return None
        node, key = previous[node]
        path.append(key)
    return path if reverse else path[::-1]


def destinations(graph, home, vacant, blocked, *, minimum_distance_m=20.0, minimum_trip_m=50.0):
    if not np.isfinite([minimum_distance_m, minimum_trip_m]).all() or minimum_distance_m <= 0 or minimum_trip_m <= 0:
        raise ValueError("Positive finite trip policy required")
    vacant = sorted(set(vacant))
    if any(n not in graph.nodes or graph.nodes[n]["kind"] != "berth" for n in vacant):
        raise ValueError("Known vacant berths required")
    forward, fp = road_tree(graph, home, blocked)
    backward, bp = road_tree(graph, home, blocked, reverse=True)
    rows = []
    for n in vacant:
        if n == home or n not in forward or n not in backward:
            continue
        distance = float(np.linalg.norm(graph.nodes[n]["pose"][:2] - graph.nodes[home]["pose"][:2]))
        if distance < minimum_distance_m or min(forward[n], backward[n]) < minimum_trip_m:
            continue
        rows.append(
            dict(
                berth=n,
                separation_m=distance,
                outward_m=forward[n],
                return_m=backward[n],
                outward=path_to(fp, home, n),
                returning=path_to(bp, home, n, reverse=True),
            )
        )
    pairs = [
        (a["berth"], b["berth"])
        for i, a in enumerate(rows)
        for b in rows[i + 1 :]
        if np.linalg.norm(graph.nodes[a["berth"]]["pose"][:2] - graph.nodes[b["berth"]]["pose"][:2])
        >= minimum_distance_m
    ]
    return dict(
        destinations=rows,
        has_two_separated_destinations=bool(pairs),
        destination_pairs=pairs,
        outward_reachable=len(forward),
        return_reachable=len(backward),
    )
