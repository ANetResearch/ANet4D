"""Recover supported lane directions without replacing existing graph identities.

Only exact coincident poses on the same source road/layout/direction connect.
There is no coordinate snap, missing-ground interpolation or inferred U-turn.
The extended graph remains a candidate requiring occupancy and traffic checks.
"""

from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

from .curb_allocation import swept
from .driving_graph import route_arrays
from .road_corridors import directions
from .vehicle_connections import PreparedPermission, admit, angle_delta


def additions(existing, measured):
    """Keep directed intervals not entirely covered by previously admitted runs."""
    covered = defaultdict(list)
    for fragment in existing:
        for direction in directions(fragment):
            covered[fragment["road"], fragment["traffic_mode"], direction].append(fragment["station_m"])
    result = []
    for fragment in measured:
        available = directions(fragment)
        if len(available) != 1:
            raise ValueError("Extension survey must measure each direction independently")
        start, end = fragment["station_m"]
        reach = start
        for lo, hi in sorted(covered[fragment["road"], fragment["traffic_mode"], available[0]]):
            if lo > reach + 1e-8:
                break
            reach = max(reach, hi)
        if reach < end - 1e-8:
            result.append(fragment)
    return result


def extend(graph, existing, extra, surface, size, allowed_area, obstacles=()):
    permission = PreparedPermission(allowed_area)
    counts = Counter()
    by_road = defaultdict(list)
    for name, node in graph.nodes.items():
        if node["kind"] != "lane":
            continue
        fragment = existing[node["fragment"]]
        direction, sample = node["direction"], node["sample"]
        xy, yaw = fragment[direction + "_xy"][sample], fragment[direction + "_yaw"][sample]
        if np.linalg.norm(np.asarray(xy) - node["pose"][:2]) > 1e-6 or abs(angle_delta(yaw, node["pose"][2])) > 1e-6:
            raise ValueError("Parent map and road survey do not share exact poses")
        by_road[fragment["road"], fragment["traffic_mode"], direction].append(name)
    for number, fragment in enumerate(extra, len(existing)):
        available = directions(fragment)
        if len(available) != 1:
            raise ValueError("Extension requires one independently measured direction")
        direction = available[0]
        points = np.asarray(fragment[direction + "_xy"])
        yaw = np.asarray(fragment[direction + "_yaw"])
        tree = cKDTree(points)
        anchors = defaultdict(list)
        for name in by_road[fragment["road"], fragment["traffic_mode"], direction]:
            pose = graph.nodes[name]["pose"]
            distance, k = tree.query(pose[:2])
            k = int(k)
            if distance <= 1e-6 and abs(angle_delta(pose[2], yaw[k])) <= 1e-6:
                anchors[k].append(name)
                counts["reused_exact_anchors"] += 1
        selected = sorted(set(anchors) | set(range(0, len(points), 28)) | {len(points) - 1})
        for k in selected:
            if not anchors[k]:
                name = f"lane:{number}:{direction}:{k}"
                graph.add_node(
                    name,
                    [*points[k], yaw[k]],
                    kind="lane",
                    fragment=number,
                    direction=direction,
                    sample=k,
                    traffic_mode=fragment["traffic_mode"],
                    traffic_resource=fragment["traffic_resource"],
                )
                anchors[k].append(name)
                counts["new_nodes"] += 1
        for a, b in zip(selected, selected[1:], strict=False):
            route = route_arrays(points[a : b + 1], yaw[a : b + 1])
            ok, reason = admit(surface, route, size, allowed_area=permission)
            if not ok:
                counts["rejected:" + reason] += 1
                continue
            corridor = swept(route, size)
            blockers = [name for name, body in obstacles if corridor.intersects(body)]
            for source in sorted(anchors[a]):
                for target in sorted(anchors[b]):
                    eid = f"extension:{number}:{a}:{b}:{source}:{target}"
                    graph.add_edge(
                        eid,
                        source,
                        target,
                        route,
                        kind="travel",
                        blockers=blockers,
                        traffic_resource=fragment["traffic_resource"],
                    )
                    counts["new_edges"] += 1
    return dict(counts)
