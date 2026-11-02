"""Reusable directed car routes with exact pose joins and explicit reverse stops.

The graph stores admitted geometry, not entity-specific trajectories. A caller
selects destinations from activity needs and reserves traffic resources before
execution. Removing blocked edges changes routing without moving any actor.
Probe import remains candidate-only until population and physical admission.
"""

from __future__ import annotations

import copy
import heapq
import math
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from awr.life.common.schema import schema_id

from .vehicle_connections import MIN_ROUTE_STEP_M, angle_delta


def route_arrays(points, yaw, gear=None):
    points, yaw = np.asarray(points, float), np.unwrap(np.asarray(yaw, float))
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 2
        or yaw.shape != (len(points),)
        or not np.isfinite(points).all()
        or not np.isfinite(yaw).all()
    ):
        raise ValueError("Invalid driving route")
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    gears = np.ones(len(ds), dtype=np.int8) if gear is None else np.asarray(gear)
    if np.any(ds <= MIN_ROUTE_STEP_M) or gears.shape != ds.shape or not np.isin(gears, (-1, 1)).all():
        raise ValueError("Driving route contains a jump in place or invalid gears")
    arc = np.r_[0.0, np.cumsum(ds)]
    return dict(
        points=points,
        yaw=yaw,
        arc=arc,
        gear=gears.astype(np.int8),
        required_stops_m=arc[np.flatnonzero(np.diff(gears)) + 1].tolist(),
        smooth_vehicle_heading=True,
    )


# Cost of a metre of a lane both directions share, against a metre of a lane of one direction (DrivingGraph.cost).
SHARED_LANE_COST = 1.25
TURN_BACK_COST_M = 50.0  # a turn back (three-point turn): the way a car drives in the time it takes, about 8 s


class DrivingGraph:
    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False) and name in ("nodes", "edges", "links", "_frozen"):
            raise ValueError("Frozen driving geometry requires a new backend snapshot")
        object.__setattr__(self, name, value)

    def __deepcopy__(self, memo):
        # A new backend can copy a frozen graph without sharing mutable arrays.
        def thaw(value):
            if isinstance(value, Mapping):
                return {k: thaw(v) for k, v in value.items()}
            if isinstance(value, tuple):
                return tuple(thaw(v) for v in value)
            return copy.deepcopy(value, memo)

        result = type(self)()
        memo[id(self)] = result
        result.nodes = thaw(self.nodes)
        result.edges = thaw(self.edges)
        result.links = {k: list(v) for k, v in self.links.items()}
        result.candidate_only = self.candidate_only
        return result

    def freeze(self):
        """Seal this backend's private geometry; occupancy remains separate.

        Prevent same-size edge/cost edits from silently invalidating route and
        physical-admission caches. Compilers keep their original mutable graph.
        """
        if getattr(self, "_frozen", False):
            return self

        def seal(value):
            if isinstance(value, Mapping):
                return MappingProxyType({k: seal(v) for k, v in value.items()})
            if isinstance(value, (list, tuple)):
                return tuple(seal(v) for v in value)
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
            return value

        self.nodes, self.edges, self.links = (seal(v) for v in (self.nodes, self.edges, self.links))
        self._frozen = True
        return self

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.links = {}
        self.candidate_only = True

    def add_node(self, name, pose, **metadata):
        if getattr(self, "_frozen", False):
            raise ValueError("Frozen driving geometry requires a new backend snapshot")
        pose = np.asarray(pose, float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("Invalid car node pose")
        if name in self.nodes:
            old = self.nodes[name]["pose"]
            if np.linalg.norm(pose[:2] - old[:2]) > 1e-7 or abs(angle_delta(pose[2], old[2])) > 1e-7:
                raise ValueError("Conflicting poses for the same driving node")
            return
        self.nodes[name] = dict(pose=pose, **metadata)
        self.links[name] = []

    def add_edge(
        self,
        name,
        source,
        target,
        route,
        *,
        kind,
        blockers=(),
        traffic_resource=None,
        traffic_resources=(),
        source_junction=None,
    ):
        if getattr(self, "_frozen", False):
            raise ValueError("Frozen driving geometry requires a new backend snapshot")
        if name in self.edges or source not in self.nodes or target not in self.nodes:
            raise ValueError("Invalid driving edge identity")
        if source_junction is not None and (
            kind != "join" or not isinstance(source_junction, str) or not source_junction
        ):
            raise ValueError("Explicit source junction belongs only to a join")
        for node, index in ((source, 0), (target, -1)):
            pose = self.nodes[node]["pose"]
            if (
                np.linalg.norm(pose[:2] - route["points"][index]) > 1e-6
                or abs(angle_delta(pose[2], route["yaw"][index])) > 1e-6
            ):
                raise ValueError("Driving graph edge does not join the exact node pose")
        # A join occupies both adjoining corridor resources. Never discard a
        # narrow-road reservation just because this edge is a turn/bay access.
        resources = set(traffic_resources)
        resources.update(
            v
            for v in (
                traffic_resource,
                self.nodes[source].get("traffic_resource"),
                self.nodes[target].get("traffic_resource"),
            )
            if v is not None
        )
        self.edges[name] = dict(
            source=source,
            target=target,
            route=route,
            kind=kind,
            blockers=tuple(blockers),
            traffic_resource=traffic_resource,
            traffic_resources=tuple(sorted(resources)),
        )
        if source_junction is not None:
            self.edges[name]["source_junction"] = source_junction
        self.links[source].append(name)

    def cost(self, edge_id):
        """Path cost of an edge: its length, the share of it on a lane that both directions share (traffic mode
        exclusive_bidirectional: half for each end of a travel or join edge on such a lane) counting SHARED_LANE_COST a
        metre. A route keeps to a lane of its own direction where the road has one, as drivers keep to their side, so
        that cars meeting on a shared lane can pass where the road has room for both, and takes the shared lane where
        the road has no other. A change of lane counts like the lanes it joins: otherwise a chain of changes between the
        overlapping layouts of a shared road would carry a route along it at the cost of its length. A turn back
        (kind 'return') adds TURN_BACK_COST_M: it is slow, reverses across the road and holds it meanwhile."""
        edge = self.edges[edge_id]
        length = float(edge["route"]["arc"][-1])
        if edge["kind"] == "return":
            return length + TURN_BACK_COST_M
        if edge["kind"] not in ("travel", "join"):
            return length
        shared = sum(
            self.nodes[edge[end]].get("traffic_mode") == "exclusive_bidirectional" for end in ("source", "target")
        )
        return length * (1.0 + (SHARED_LANE_COST - 1.0) * shared / 2.0)

    def path(self, source, target, *, blocked_edges=()):
        """Least cost (`cost`) over available geometry; no nearest-node snapping. A berth is a terminal: a path starts or
        ends there and never passes through one (driving into a parking place and out again is not a way along the
        road; a lane change through a curb berth would otherwise undercut a shared lane)."""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Unknown driving destination")
        blocked = set(blocked_edges)
        if blocked - set(self.edges):
            raise ValueError("Unknown blocked driving edge")
        queue = [(0.0, source)]
        distance = {source: 0.0}
        previous = {}
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != distance[node]:
                continue
            if node == target:
                route = []
                while node != source:
                    edge = previous[node]
                    route.append(edge)
                    node = self.edges[edge]["source"]
                return route[::-1]
            if node != source and self.nodes[node]["kind"] == "berth":
                continue
            for edge_id in sorted(self.links[node]):
                if edge_id in blocked:
                    continue
                edge = self.edges[edge_id]
                end = edge["target"]
                candidate = cost + self.cost(edge_id)
                if candidate < distance.get(end, math.inf):
                    distance[end] = candidate
                    previous[end] = edge_id
                    heapq.heappush(queue, (candidate, end))
        return None

    def concatenate(self, edge_ids):
        if not edge_ids:
            return None
        pieces = [self.edges[k] for k in edge_ids]
        if any(a["target"] != b["source"] for a, b in zip(pieces, pieces[1:], strict=False)):
            raise ValueError("Cannot concatenate disconnected driving edges")
        if any(self.nodes[p["target"]]["kind"] == "berth" for p in pieces[:-1]):
            raise ValueError("A driving route cannot pass through a terminal berth")
        points = np.concatenate([p["route"]["points"][1 if i else 0 :] for i, p in enumerate(pieces)])
        yaw = np.concatenate([p["route"]["yaw"][1 if i else 0 :] for i, p in enumerate(pieces)])
        gear = np.concatenate([p["route"]["gear"] for p in pieces])
        return route_arrays(points, yaw, gear)

    def components(self, *, blocked_edges=()):
        """Strongly connected parts of the map as `path` drives it, a berth being a terminal: the edges between lane
        nodes (and those from berth to berth) link the parts, and a berth of no part with another node belongs to the
        part that an entry into it comes from and an exit from it leads to (the largest, if several do), else forms a
        part of its own."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        names = sorted(self.nodes)
        index = {name: i for i, name in enumerate(names)}
        blocked = set(blocked_edges)
        if blocked - set(self.edges):
            raise ValueError("Unknown blocked driving edge")
        berth = {name for name in names if self.nodes[name]["kind"] == "berth"}
        pairs = [
            (index[e["source"]], index[e["target"]])
            for k, e in self.edges.items()
            if k not in blocked and (e["source"] in berth) == (e["target"] in berth)
        ]
        rows, cols = zip(*pairs, strict=False) if pairs else ([], [])
        matrix = csr_matrix((np.ones(len(pairs)), (rows, cols)), shape=(len(names), len(names)))
        _, labels = connected_components(matrix, directed=True, connection="strong")
        labels = [int(v) for v in labels]
        sizes = {}
        for name, label in zip(names, labels, strict=False):
            sizes[label] = sizes.get(label, 0) + 1
        entries, exits = {}, {}
        for k, e in self.edges.items():
            if k in blocked:
                continue
            if e["target"] in berth and e["source"] not in berth:
                entries.setdefault(e["target"], set()).add(labels[index[e["source"]]])
            if e["source"] in berth and e["target"] not in berth:
                exits.setdefault(e["source"], set()).add(labels[index[e["target"]]])
        result = {}
        for name, label in zip(names, labels, strict=False):
            if name in berth and sizes[label] == 1:
                common = entries.get(name, set()) & exits.get(name, set())
                if common:
                    label = min(common, key=lambda c: (-sizes[c], c))
            result.setdefault(label, []).append(name)
        return sorted(result.values(), key=lambda component: (-len(component), component))


def from_probes(road_probe, parking_probe, node_spacing_m=None):
    """Compile measured candidate connections; do not infer missing street joins.

    Lane nodes stand at the ends of every lane fragment, at the lane poses of parking and return connections and,
    with `node_spacing_m`, about every `node_spacing_m` metres along each lane where no lane node stands within half
    that distance (marked `waypoint`): the places where the map compiler may join a lane to another lane, so that a
    car can change between the overlapping layouts of a road (scripts/sandbox/compile_driving_candidate.py)."""
    if road_probe["asset"] != parking_probe["asset"] or not np.allclose(
        road_probe["size_m"], parking_probe["size_m"], atol=1e-8, rtol=0
    ):
        raise ValueError("Driving probes use different vehicle bodies")
    from .road_corridors import directions
    from .road_corridors import fragments as read_fragments

    graph = DrivingGraph()
    fragments = read_fragments(road_probe)
    samples = {}

    def lane_node(fragment, direction, sample, **extra):
        road = fragments[fragment]
        point, yaw = road[direction + "_xy"][sample], road[direction + "_yaw"][sample]
        name = f"lane:{fragment}:{direction}:{sample}"
        graph.add_node(
            name,
            [*point, yaw],
            kind="lane",
            fragment=fragment,
            direction=direction,
            sample=sample,
            traffic_mode=road.get("traffic_mode", "separated_two_way"),
            traffic_resource=road.get("traffic_resource"),
            **extra,
        )
        samples.setdefault((fragment, direction), set()).add(sample)
        return name

    for fragment, road in enumerate(fragments):
        for direction in directions(road):
            lane_node(fragment, direction, 0)
            lane_node(fragment, direction, len(road[direction + "_xy"]) - 1)
    for node in parking_probe["lane_nodes"]:
        ident = lane_node(node["fragment"], node["direction"], node["sample"])
        graph.add_node(ident, node["pose"])  # reject a mismatching probe coordinate
    for number, connection in enumerate(road_probe["return_connections"]):
        fragment = connection["road_fragment"]
        ends = []
        for pose in (connection["start_pose"], connection["end_pose"]):
            matches = []
            for direction in directions(fragments[fragment]):
                xy = np.asarray(fragments[fragment][direction + "_xy"])
                k = int(np.argmin(np.linalg.norm(xy - np.asarray(pose[:2]), axis=1)))
                if (
                    np.linalg.norm(xy[k] - pose[:2]) < 1e-6
                    and abs(angle_delta(fragments[fragment][direction + "_yaw"][k], pose[2])) < 1e-6
                ):
                    matches.append((direction, k))
            if len(matches) != 1:
                raise ValueError("Return connection does not meet one exact lane pose")
            ends.append(lane_node(fragment, *matches[0]))
        route = route_arrays(connection["points"], connection["yaw"], connection.get("gear"))
        graph.add_edge(
            f"return:{number}", *ends, route, kind="return", blockers=connection["blocked_by_existing_allocation"]
        )
    for bay in parking_probe["bays"]:
        slot = bay["slot"]
        name = "berth:" + bay["bay"]
        graph.add_node(name, [*slot["xy"], slot["yaw"]], kind="berth", region=bay["region"])
        alternatives = bay.get("connection_alternatives") or {
            mode: [connection] for mode, connection in bay["connections"].items()
        }
        for mode, options in alternatives.items():
            if mode not in ("exit", "entry"):
                raise ValueError("Unknown parking connection kind")
            for number, connection in enumerate(options):
                ends = (name, connection["lane_node"]) if mode == "exit" else (connection["lane_node"], name)
                route = route_arrays(connection["points"], connection["yaw"], connection["gear"])
                graph.add_edge(
                    name + ":" + mode + ":" + str(number),
                    *ends,
                    route,
                    kind=mode,
                    blockers=connection["blocked_by_existing_allocation"],
                )
    if node_spacing_m is not None:
        if not math.isfinite(node_spacing_m) or node_spacing_m <= 0:
            raise ValueError("Lane node spacing must be positive")
        for fragment, road in enumerate(fragments):
            for direction in directions(road):
                xy = np.asarray(road[direction + "_xy"], float)
                arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
                for mark in np.arange(node_spacing_m, arc[-1] - node_spacing_m / 2, node_spacing_m):
                    # A lane node within half the spacing already offers the place.
                    if any(abs(arc[k] - mark) < node_spacing_m / 2 for k in samples.get((fragment, direction), ())):
                        continue
                    lane_node(fragment, direction, int(np.searchsorted(arc, mark)), waypoint=True)
    for (fragment, direction), selected in sorted(samples.items()):
        road = fragments[fragment]
        selected = sorted(selected)
        for a, b in zip(selected, selected[1:], strict=False):
            route = route_arrays(road[direction + "_xy"][a : b + 1], road[direction + "_yaw"][a : b + 1])
            graph.add_edge(
                f"travel:{fragment}:{direction}:{a}:{b}",
                lane_node(fragment, direction, a),
                lane_node(fragment, direction, b),
                route,
                kind="travel",
                traffic_resource=road.get("traffic_resource"),
            )
    return graph


def load_candidate(document):
    """Load compiler output as geometry, explicitly retaining candidate status."""
    if schema_id(document.get("schema")) != "awr.life.driving_map_candidate.v1":
        raise ValueError("Unknown driving map schema")
    graph = DrivingGraph()
    for name, node in sorted(document["nodes"].items()):
        graph.add_node(name, node["pose"], **{k: v for k, v in node.items() if k != "pose"})
    for name, edge in sorted(document["edges"].items()):
        source = edge["route"]
        # Reconstruct arc length and gear stops rather than trusting costs from
        # a persisted document. Physical re-admission is still a separate step.
        route = route_arrays(source["points"], source["yaw"], source["gear"])
        graph.add_edge(
            name,
            edge["source"],
            edge["target"],
            route,
            kind=edge["kind"],
            blockers=edge["blockers"],
            traffic_resource=edge.get("traffic_resource"),
            traffic_resources=edge.get("traffic_resources", ()),
            source_junction=edge.get("source_junction"),
        )
    return graph


def summarize_berths(graph):
    """Recompute geometric round-trip evidence; never carry stale components."""
    from collections import Counter

    components = []
    for group in graph.components():
        berths = [n for n in group if graph.nodes[n]["kind"] == "berth"]
        if len(berths) < 2:
            continue
        opportunities = []
        for start in berths:
            choices = []
            for goal in berths:
                if np.linalg.norm(graph.nodes[start]["pose"][:2] - graph.nodes[goal]["pose"][:2]) < 20.0:
                    continue
                outward = graph.path(start, goal)
                returning = graph.path(goal, start)
                if outward is None or returning is None:
                    continue
                out_m = sum(graph.edges[e]["route"]["arc"][-1] for e in outward)
                back_m = sum(graph.edges[e]["route"]["arc"][-1] for e in returning)
                if min(out_m, back_m) >= 50.0:
                    choices.append(
                        dict(
                            destination=goal,
                            outward_m=float(out_m),
                            return_m=float(back_m),
                            outward_edges=outward,
                            return_edges=returning,
                        )
                    )
            opportunities.append(dict(berth=start, returnable_destinations=choices))
        components.append(
            dict(
                nodes=len(group),
                berths=berths,
                regions=dict(Counter(graph.nodes[n]["region"] for n in berths)),
                qualifying_origin_berths=sum(bool(r["returnable_destinations"]) for r in opportunities),
                opportunities=opportunities,
            )
        )
    return components
