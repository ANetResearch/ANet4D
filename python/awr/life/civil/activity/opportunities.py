"""Screen real-home/car/facility associations before population allocation.

This is geometry screening, not an occupancy or execution certificate. Distinct
standing slots at one physical facility do not become distinct destinations.
The matching count is an upper bound for these candidates before collision,
remaining-fleet and driver scheduling admission, never a runnable-car count.
"""

import heapq
import math

import numpy as np


def screen_car_activities(places, graph, *, min_trip_m=50.0, min_separation_m=20.0, blocked_edges=()):
    if not math.isfinite(min_trip_m) or min_trip_m <= 0 or not math.isfinite(min_separation_m) or min_separation_m <= 0:
        raise ValueError("Positive finite meaningful-trip distances required")
    blocked = set(blocked_edges)
    if blocked - set(graph.edges):
        raise ValueError("Unknown blocked edge")
    places = list(places)
    by_id = {p.id: p for p in places}
    if len(by_id) != len(places):
        raise ValueError("Duplicate activity place")
    homes = {}
    facilities = {}
    for place in places:
        if place.kind in ("berth", "parking", "parking_lot"):
            continue
        for berth in place.parking:
            if (
                berth not in by_id
                or by_id[berth].kind != "berth"
                or by_id[berth].access_xy is None
                or berth not in graph.nodes
                or graph.nodes[berth]["kind"] != "berth"
            ):
                raise ValueError("Activity requires a bound exact berth")
            (homes if place.kind == "home" else facilities).setdefault(berth, []).append(place)
    trees = {}

    def tree(source, unobstructed=False):
        key = (source, unobstructed)
        if key in trees:
            return trees[key]
        distance = {source: 0.0}
        previous = {}
        queue = [(0.0, source)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != distance[node]:
                continue
            # A berth is a terminal, never a way along the road (DrivingGraph.path).
            if node != source and graph.nodes[node]["kind"] == "berth":
                continue
            for name in sorted(graph.links[node]):
                if not unobstructed and name in blocked:
                    continue
                edge = graph.edges[name]
                end = edge["target"]
                length = cost + float(edge["route"]["arc"][-1])
                if length < distance.get(end, math.inf):
                    distance[end] = length
                    previous[end] = name
                    heapq.heappush(queue, (length, end))
        trees[key] = (distance, previous)
        return trees[key]

    def path(source, target):
        distance, previous = tree(source)
        if target not in distance:
            return None
        result = []
        node = target
        while node != source:
            name = previous[node]
            result.append(name)
            node = graph.edges[name]["source"]
        return result[::-1]

    def separated(a, b):
        return np.linalg.norm(graph.nodes[a]["pose"][:2] - graph.nodes[b]["pose"][:2]) >= min_separation_m

    rows = []
    for home in sorted(homes):
        options = []
        distance, _ = tree(home)
        for destination in sorted(facilities):
            if destination == home or not separated(home, destination):
                continue
            back, _ = tree(destination)
            outward_m = distance.get(destination)
            return_m = back.get(home)
            if outward_m is None or return_m is None or min(outward_m, return_m) < min_trip_m:
                continue
            if blocked:
                # Congestion must not promote a short parking shuffle into a
                # meaningful trip simply by forcing a long detour.
                baseline, _ = tree(home, True)
                reverse, _ = tree(destination, True)
                if min(baseline.get(destination, math.inf), reverse.get(home, math.inf)) < min_trip_m:
                    continue
            outward = path(home, destination)
            returning = path(destination, home)
            old_blockers = sorted({b for e in (*outward, *returning) for b in graph.edges[e]["blockers"]})
            for facility in facilities[destination]:
                options.append(
                    dict(
                        berth=destination,
                        place=facility.id,
                        group=facility.group or facility.id,
                        kind=facility.kind,
                        outward_m=outward_m,
                        return_m=return_m,
                        outward_edges=outward,
                        return_edges=returning,
                        original_allocation_blockers=old_blockers,
                    )
                )
        options.sort(
            key=lambda o: (
                len(o["original_allocation_blockers"]),
                o["outward_m"] + o["return_m"],
                o["group"],
                o["place"],
                o["berth"],
            )
        )
        selected = []
        # Pair search avoids a greedy first destination hiding a valid pair.
        for i, a in enumerate(options):
            for b in options[i + 1 :]:
                if a["group"] != b["group"] and separated(a["berth"], b["berth"]):
                    selected = [a, b]
                    break
            if selected:
                break
        rows.append(
            dict(
                home_berth=home,
                home_places=sorted(p.id for p in homes[home]),
                service_pair=selected,
                qualifies=bool(selected),
                options=len(options),
            )
        )
    # A home may be near many berths. It still supplies at most one driver.
    # Likewise, one berth cannot hold several drivers' cars at initial time.
    adjacency = {r["home_berth"]: r["home_places"] for r in rows if r["qualifies"]}
    owners = {}

    def augment(berth, seen):
        for home in adjacency[berth]:
            if home in seen:
                continue
            seen.add(home)
            if home not in owners or augment(owners[home], seen):
                owners[home] = berth
                return True
        return False

    for berth in sorted(adjacency):
        augment(berth, set())
    return dict(
        rows=rows,
        home_berths=len(homes),
        qualifying_berths=len(adjacency),
        distinct_home_berth_matching=len(owners),
        matching=[dict(home_place=h, berth=b) for h, b in sorted(owners.items())],
        routing_sources=len(trees),
        blocked_edges=len(blocked),
        release_evidence=False,
    )
