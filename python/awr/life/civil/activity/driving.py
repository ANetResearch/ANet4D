"""Measured car routes and transactional departure admission for activities.

This bridge never writes an actor pose. A motion controller must execute the
nominal schedule with live collision checks and acknowledge actual arrival.
Traffic admission is progressive (awr.life.civil.activity.traffic_runs): a car
reserves each conflict run of its route (a crossing or merging turn, the
opposite lane of a narrow road, a berth manoeuvre) before entering it and
releases it once it has left; it waits for a run only at a safe pose on its
route, where its body obstructs no other flow. Same-lane traffic follows by the
controller's live stopping arc.
"""

import copy
import hashlib
import heapq
import json
import math
import random
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from shapely.ops import unary_union

from ..curb_allocation import footprint, swept
from ..driving_motion import Limits, RouteExecution, Schedule
from ..vehicle_connections import admit, angle_delta
from .director import Route
from .traffic_runs import TrafficRuns

THROUGH_MIN_M = 2000.0  # travel lanes the turn-back-free part of a map needs before tours keep to it
DOUBLE_BACK_M = (
    300.0  # a tour drives no road back within this distance along it where a waypoint avoids that (plan_tour)
)
MEETING_M = 30.0  # length of a way along which a plan meets another car's way coming the other way, that it avoids
MEETING_DETOUR_M = 400.0  # extra length a plan drives to avoid such a meeting (_meeting_free)
HOLD_WINDOW_S = (
    60.0  # a departure waits while its way would meet another car's way within this time of it (_departure_held)
)
HOLD_MAX_S = 120.0  # ... for at most this long at one berth
HOLD_SPEED_MPS = 4.0  # speed at which the times of those meetings are estimated


def _components(nodes, links):
    """The strongly connected components of a directed graph (Tarjan, iterative): lists of nodes."""
    index, low, on, stack, out = {}, {}, set(), [], []
    for root in nodes:
        if root in index:
            continue
        index[root] = low[root] = len(index)
        stack.append(root)
        on.add(root)
        work = [(root, iter(links.get(root, ())))]
        while work:
            v, ahead = work[-1]
            for w in ahead:
                if w not in index:
                    index[w] = low[w] = len(index)
                    stack.append(w)
                    on.add(w)
                    work.append((w, iter(links.get(w, ()))))
                    break
                if w in on:
                    low[v] = min(low[v], index[w])
            else:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[v])
                if low[v] == index[v]:
                    component = []
                    while True:
                        w = stack.pop()
                        on.discard(w)
                        component.append(w)
                        if w == v:
                            break
                    out.append(component)
    return out


@dataclass(frozen=True)
class Body:
    pose: tuple[float, float, float]
    size: tuple[float, float, float]

    def __post_init__(self):
        object.__setattr__(self, "pose", tuple(self.pose))
        object.__setattr__(self, "size", tuple(self.size))
        if (
            len(self.pose) != 3
            or len(self.size) != 3
            or not all(math.isfinite(x) for x in (*self.pose, *self.size))
            or min(self.size) <= 0
        ):
            raise ValueError("Invalid observed vehicle body")

    def polygon(self):
        return footprint(dict(xy=self.pose[:2], yaw=self.pose[2]), self.size)


class DrivingRoutes:
    """A frozen geometry snapshot with explicitly updated current occupancy.

    `areas` must provide the original permitted polygon for every edge; a
    town-wide union must not replace a turn's own street domain. `occupancy`
    covers a fixed roster. None explicitly means the actor is absent from this
    snapshot, never that its old blocking pose can silently be ignored.
    One instance supports one actual vehicle body size. Separate maps/backends
    are necessary for larger vehicles until their geometry is admitted too.
    """

    def __init__(
        self,
        graph,
        surface,
        size,
        areas,
        occupancy,
        *,
        limits=Limits(),
        cache_plan_results=True,
        static_tree_capacity=16,
        index_occupancy=True,
        tour_legs=2,
        tour_spacing_m=120.0,
        no_standing=(),
    ):
        if not isinstance(static_tree_capacity, int) or static_tree_capacity < 0:
            raise ValueError("Static tree capacity must be a nonnegative integer")
        self.graph = copy.deepcopy(graph).freeze()
        self.static_tree_capacity = static_tree_capacity
        self._static_trees = OrderedDict()
        self._berth_index = None
        self.surface = surface
        self.size = np.array(size, float)
        if self.size.shape != (3,) or not np.isfinite(self.size).all() or np.any(self.size <= 0):
            raise ValueError("Actual vehicle dimensions required")
        if set(areas) != set(graph.edges) or any(a is None or a.is_empty for a in areas.values()):
            raise ValueError("Every edge requires its own explicit permitted area")
        self.areas = dict(areas)
        self.limits = limits
        self.roster = frozenset(occupancy)
        required = {actor for e in graph.edges.values() for actor in e["blockers"]}
        if not required <= self.roster:
            raise ValueError("Occupancy omits known map blockers")
        self.cache_plan_results = cache_plan_results
        self.index_occupancy = index_occupancy
        self._occupancy_index = None
        # pending() -> {car: destination berth} of admitted journeys not yet departed (Director.pending_destinations).
        self.pending = None
        self._pending = {}
        self.plan_results = {}
        self._revision_paths = OrderedDict()  # (vehicle, splice node, target, avoid) -> path or None (revise)
        self.planning_snapshot = None
        self.occupancy = {}
        self._reach = {}  # static indices of the frozen map
        self._returns = None  # its turn-back edges (_turn_backs)
        self.update_occupancy(occupancy)
        self.geometry = {}
        self.plans = {}
        self.compiled_plans = {}
        self.active = {}
        self.executions = {}
        self.completed_motion = []
        self.revisions = {}  # vehicle -> [(old evidence, new evidence, splice arc, reason)] not yet bound (revise)
        self.path_failures = {}  # vehicle -> its last path search without a way (_path): what removed the edges
        self.roads = None  # ConnectedRoads of the map: parking keeps the roads passable (keeps_roads); None: unchecked
        self.traffic = TrafficRuns(self, no_standing)  # no_standing: polygons cars do not stop on (crossings)
        # Tours (route mode 'tour'): `tour_legs` waypoints of the lane network, each at least `tour_spacing_m` from the
        # point before it, drawn per car and per completed trip.
        if not isinstance(tour_legs, int) or tour_legs < 1 or not math.isfinite(tour_spacing_m) or tour_spacing_m < 0:
            raise ValueError("Tours need at least one waypoint and a finite spacing")
        self.tour_legs, self.tour_spacing_m = tour_legs, float(tour_spacing_m)
        self._through = None  # lane nodes a tour may pass through without turning back (_through_nodes)
        self.trips_done = {}
        self._holds = {}  # (vehicle, evidence) -> time its departure was first held (_departure_held)
        self._hold_index = None  # (tick time, _ways_index()) for _departure_held
        self._reaching_cache = {}
        self._reverse = None
        self.stats = dict(
            edge_checks=0,
            edge_cache_hits=0,
            plan_cache_hits=0,
            static_tree_builds=0,
            static_tree_hits=0,
            static_tree_evictions=0,
            snapshot_plan_hits=0,
            snapshot_negative_hits=0,
            occupancy_rejections=0,
            traffic_rejections=0,
            plans_computed=0,
            plan_wall_s=0.0,
            path_searches=0,
            keeps_ways_wall_s=0.0,
            occupancy_index_builds=0,
            occupancy_index_queries=0,
            occupancy_index_hits=0,
        )

    def update_occupancy(self, occupancy):
        if set(occupancy) != self.roster or any(b is not None and not isinstance(b, Body) for b in occupancy.values()):
            raise ValueError("A complete observed body snapshot is required")
        for vehicle in getattr(self, "active", {}):
            body = occupancy[vehicle]
            if body is None or not np.allclose(body.size, self.size, atol=1e-8, rtol=0.0):
                raise ValueError("An active vehicle cannot disappear or change its physical body")
        moved = {actor for actor, body in occupancy.items() if self.occupancy.get(actor) != body}
        # Adapted from Claude b42ebcc: active trips are represented at their
        # reserved destination in planning, while departures use the live body.
        # Pose-only updates of those cars therefore do not change planning.
        # Active-trip membership/destination changes invalidate in plan().
        if moved - getattr(self, "active", {}).keys():
            self.plan_results.clear()
            self._revision_paths.clear()
            self._occupancy_index = None
        self.occupancy = dict(occupancy)
        bodies = dict(getattr(self, "bodies", {}))
        for actor in moved:
            bodies.pop(actor, None)
            if occupancy[actor] is not None:
                bodies[actor] = occupancy[actor].polygon()
        self.bodies = bodies

    def _edge(self, edge):
        if edge in self.geometry:
            self.stats["edge_cache_hits"] += 1
            return self.geometry[edge]
        self.stats["edge_checks"] += 1
        route = self.graph.edges[edge]["route"]
        ok, reason = admit(self.surface, route, self.size, allowed_area=self.areas[edge])
        self.geometry[edge] = (ok, reason, swept(route, self.size) if ok else None)
        return self.geometry[edge]

    def _clear(self, vehicle, edges):
        """Whether every edge is admitted and clear of the bodies standing outside a trip (cars in a trip are
        traffic: the runs and the live stopping arc keep them apart)."""
        for edge in edges:
            ok, _, corridor = self._edge(edge)
            if not ok:
                return False
            if any(
                actor != vehicle and actor not in self.active and corridor.intersects(body)
                for actor, body in self.bodies.items()
            ):
                return False
        return True

    def route(self, mode, start, end, entity):
        """Director-compatible callback; other modes belong to the walk backend.

        The Director supplies XY coordinates. Ambiguous coincident berths are
        rejected rather than guessing a parking heading. This never snaps a
        nearby street point into the berth graph.
        """
        if mode not in ("drive", "tour"):
            return None
        endpoints = []
        for xy in (start, end):
            xy = np.asarray(xy, float)
            if xy.shape != (2,) or not np.isfinite(xy).all():
                raise ValueError("Finite exact berth position required")
            names = self._berths_at(xy)
            if len(names) != 1:
                return None
            endpoints.append(names[0])
        return self.plan_tour(entity, *endpoints) if mode == "tour" else self.plan(entity, *endpoints)

    def _berths_at(self, xy):
        """Exact berth lookup on sealed geometry, adapted from Claude c925a9a.

        The tree only narrows candidates. Keep the original norm test and all
        coincident berths: proximity is neither permission to snap nor a way to
        choose between ambiguous headings. Memory scales with map size, not time.
        """
        if self._berth_index is None:
            from scipy.spatial import cKDTree

            names = sorted(n for n, d in self.graph.nodes.items() if d["kind"] == "berth")
            points = np.array([self.graph.nodes[n]["pose"][:2] for n in names]).reshape(-1, 2)
            self._berth_index = (names, cKDTree(points))
        names, tree = self._berth_index
        # Slightly expand the broad phase only; exact acceptance is unchanged.
        candidates = tree.query_ball_point(xy, 1e-7 + 1e-14)
        return [
            names[k] for k in sorted(candidates) if np.linalg.norm(self.graph.nodes[names[k]]["pose"][:2] - xy) <= 1e-7
        ]

    def plan(self, vehicle, source, target):
        """Plan exact berth-to-berth geometry; no reservation or pose mutation.

        Cars already executing an admitted trip are future occupants of their
        reserved destination berths, not permanent blockers at a transient
        point on the road. Departures still recheck *actual* occupancy and all
        traffic reservations atomically in acquire; a plan is not permission
        to move through another car.
        """
        if vehicle not in self.occupancy or self.occupancy[vehicle] is None:
            raise ValueError("Owned vehicle missing from current observed population")
        if not np.allclose(self.occupancy[vehicle].size, self.size, atol=1e-8, rtol=0.0):
            raise ValueError("Vehicle body differs from admitted map body")
        for node in (source, target):
            if node not in self.graph.nodes or self.graph.nodes[node]["kind"] != "berth":
                raise ValueError("Activity drive must connect exact mapped berths")
        if source == target:
            return None
        if target not in self._reachable(source):
            # No path in the frozen map at all; blocking edges only removes paths.
            self.stats["unreachable_in_map"] = self.stats.get("unreachable_in_map", 0) + 1
            return None
        # Planning is pure for one complete body/reservation snapshot. Many
        # facility slots share the same berths, including blocked paths. Reuse
        # both outcomes only while that snapshot and motion limits are equal.
        # acquire() still observes bodies and reservations afresh; this cache
        # never grants permission to depart.
        self._snapshot()
        key = (vehicle, source, target)
        if self.cache_plan_results and key in self.plan_results:
            result = self.plan_results[key]
            self.stats["snapshot_plan_hits"] += 1
            self.stats["snapshot_negative_hits"] += int(result is None)
            return result
        began = time.perf_counter()
        result = self._plan_current_snapshot(vehicle, source, target)
        self.stats["plans_computed"] += 1
        self.stats["plan_wall_s"] += time.perf_counter() - began
        if self.cache_plan_results:
            self.plan_results[key] = result
        return result

    def _snapshot(self):
        """Start a new planning snapshot when the active trips, the pending journeys or the limits changed."""
        pending = {} if self.pending is None else dict(self.pending())
        snapshot = (tuple(sorted(self.active.items())), tuple(sorted(pending.items())), self.limits)
        if snapshot != self.planning_snapshot:
            self._pending = pending
            self.plan_results.clear()
            self._revision_paths.clear()
            self._occupancy_index = None
            self.planning_snapshot = snapshot

    def _reaching(self, target):
        """Nodes from which `target` is reachable in the frozen map, ignoring occupancy; a berth other than `target`
        starts a path but is not passed through (DrivingGraph.path)."""
        if target not in self._reaching_cache:
            if self._reverse is None:
                self._reverse = {}
                for edge in self.graph.edges.values():
                    self._reverse.setdefault(edge["target"], []).append(edge["source"])
            seen = {target}
            stack = [target]
            while stack:
                for node in self._reverse.get(stack.pop(), ()):
                    if node not in seen:
                        seen.add(node)
                        if self.graph.nodes[node]["kind"] != "berth":
                            stack.append(node)
            self._reaching_cache[target] = frozenset(seen)
        return self._reaching_cache[target]

    def plan_tour(self, vehicle, source, target):
        """A berth-to-berth drive through waypoints of the lane network (see __init__), or None; like plan(), pure for
        one planning snapshot. The waypoints are drawn from the lane nodes reachable from `source` from which `target`
        is reachable, by a stream of the car and its number of completed trips; a waypoint without a clear way is
        drawn again (eight times at most) and a tour keeps the legs it found. Each leg drives without a turn back where
        it can: a cruising driver goes round a block rather than turning across the road and back along the street it
        came by (which holds both directions of it, often a shared lane, for the car's way out and in again). A leg
        that meets another car's way coming the other way (`_meetings`) is drawn again too, after one that drives a
        road back; the way on to `target` keeps clear of such meetings where it can (`_meeting_free`). Nor does
        a tour drive a road back the way it came, within DOUBLE_BACK_M along it, where another waypoint of the draws
        avoids that (`_doubles_back`; for the last waypoint with the way on to `target`): a waypoint down a street with
        a block at its end only, reached round that block, sends the car out and back along the street just as a turn
        back does, and one conflict run then holds both directions of it (cruise5 r11f v8/v9, vehicle-043: 250 m out
        and back held for the four cars that came the other way). A tour that ends near where it began drives some
        roads back further on; the runs between hold each direction in turn."""
        if vehicle not in self.occupancy or self.occupancy[vehicle] is None:
            raise ValueError("Owned vehicle missing from current observed population")
        for node in (source, target):
            if node not in self.graph.nodes or self.graph.nodes[node]["kind"] != "berth":
                raise ValueError("Activity drive must connect exact mapped berths")
        if source == target or target not in self._reachable(source):
            return None
        self._snapshot()
        key = ("tour", vehicle, source, target, self.trips_done.get(vehicle, 0))
        if self.cache_plan_results and key in self.plan_results:
            self.stats["snapshot_plan_hits"] += 1
            return self.plan_results[key]
        began = time.perf_counter()
        rng = random.Random(int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:8], "big"))
        reach = self._reachable(source) & self._reaching(target)
        through = self._through_nodes()
        pool = sorted(n for n in (reach & through or reach) if self.graph.nodes[n]["kind"] == "lane")
        legs, at = [], source
        for k in range(self.tour_legs if pool else 0):
            best = None  # (doubles back, meets, draw, waypoint, edges)
            for draw in range(8):
                waypoint = pool[rng.randrange(len(pool))]
                if (
                    math.dist(self.graph.nodes[waypoint]["pose"][:2], self.graph.nodes[at]["pose"][:2])
                    < self.tour_spacing_m
                ):
                    continue
                edges = self._path(vehicle, at, waypoint, turn_backs=False) or self._path(vehicle, at, waypoint)
                if not edges:
                    continue
                onward = (
                    edges
                    if k < self.tour_legs - 1
                    else edges
                    + (
                        self._path(vehicle, waypoint, target, turn_backs=False)
                        or self._path(vehicle, waypoint, target)
                        or []
                    )
                )
                doubles = self._doubles_back([e for leg in legs for e in leg], onward)
                meets = self._meetings(vehicle, onward)[1] >= MEETING_M
                self.stats["tour_waypoints_doubling_back"] = self.stats.get("tour_waypoints_doubling_back", 0) + int(
                    doubles
                )
                self.stats["tour_waypoints_meeting"] = self.stats.get("tour_waypoints_meeting", 0) + int(meets)
                if best is None or (doubles, meets) < best[:2]:
                    best = (doubles, meets, draw, waypoint, edges)
                if not doubles and not meets:
                    break
            if best is not None:
                at, edges = best[3:]
                legs.append(edges)
        result = None
        while result is None:
            last = self._meeting_free(vehicle, at, target, turn_backs=False) or self._meeting_free(vehicle, at, target)
            if last is not None:
                result = self._compile(vehicle, source, target, [e for leg in legs for e in leg] + last)
            elif not legs:
                break
            else:
                legs.pop()
                at = self.graph.edges[legs[-1][-1]]["target"] if legs else source
        self.stats["tours_computed"] = self.stats.get("tours_computed", 0) + 1
        self.stats["plan_wall_s"] += time.perf_counter() - began
        if self.cache_plan_results:
            self.plan_results[key] = result
        return result

    def _meeting_free(self, vehicle, source, target, *, turn_backs=True):
        """`_path`, but where that path meets another car's way coming the other way for MEETING_M or more
        (`_meetings`), the path with the fewest metres of such meetings found by keeping off them (twice at most) and
        costing at most MEETING_DETOUR_M more. Two cars that meet on a road too narrow for both take it in turn, the
        whole stretch the other holds (TrafficRuns); the town has the roads to go round such a stretch instead
        (cruise r11f v9/v10, cruise5 r11f v8: cars waited 60 s for the far end of 180-320 m of a shared lane)."""
        base = self._path(vehicle, source, target, turn_backs=turn_backs)
        if base is None:
            return None
        meet, length = self._meetings(vehicle, base)
        if length < MEETING_M:
            return base
        best, best_length = base, length
        limit = sum(self.graph.cost(e) for e in base) + MEETING_DETOUR_M
        avoid = set(meet)
        for _ in range(2):
            other = self._path(vehicle, source, target, turn_backs=turn_backs, avoid=avoid)
            if other is None or sum(self.graph.cost(e) for e in other) > limit:
                break
            meet, length = self._meetings(vehicle, other)
            if length < best_length:
                best, best_length = other, length
            if length < MEETING_M:
                break
            avoid |= meet
        self.stats["meetings_avoided" if best is not base else "meetings_kept"] = (
            self.stats.get("meetings_avoided" if best is not base else "meetings_kept", 0) + 1
        )
        return best

    def _meetings(self, vehicle, edges):
        """The edges of `edges` whose corridor meets the corridor of an edge of another active car's remaining way that
        runs along one of the same roads (`traffic_resource` of a lane end) in the other direction, and their length.
        On a road with a lane each way the corridors of the two directions do not meet; on a shared lane, or where
        parked cars leave one lane, they do. Crossings of other roads are no meetings."""
        entries, tree = self._meeting_index()
        out, length = set(), 0.0
        if tree is None:
            return out, length
        for e in edges:
            flows = self._edge_flows(e)
            ok, _, corridor = self._edge(e)
            if not flows or not ok:
                continue
            for k in tree.query(corridor, predicate="intersects"):
                other, theirs = entries[k][:2]
                if other != vehicle and any(r == r2 and d != d2 for r, d in flows for r2, d2 in theirs):
                    out.add(e)
                    length += float(self.graph.edges[e]["route"]["arc"][-1])
                    break
        return out, length

    def _meeting_index(self):
        """(car, flows, corridor, arc) of the edges of the active cars' remaining ways (from where each was when the
        index was built; arc: how far along it each edge begins) and an STRtree of the corridors; kept with the plans of
        the planning snapshot."""
        self._snapshot()
        if self.cache_plan_results and ("meeting index",) in self.plan_results:
            return self.plan_results[("meeting index",)]
        index = self._ways_index()
        if self.cache_plan_results:
            self.plan_results[("meeting index",)] = index
        return index

    def _ways_index(self):
        """(car, flows, corridor, arc) of the edges of the active cars' remaining ways from where each is now, and an
        STRtree of the corridors (_meeting_index)."""
        from shapely.strtree import STRtree

        entries = []
        for other, evidence in sorted(self.active.items()):
            edges, trip = self.plans[evidence]["edges"], self.traffic.trips.get(other)
            course = self.traffic.course(evidence)
            arc = trip["arc"] if trip is not None else 0.0
            first = course.edge_index(arc) if trip is not None else 0
            for k in range(first, len(edges)):
                f = edges[k]
                flows = self._edge_flows(f)
                ok, _, corridor = self._edge(f)
                if flows and ok:
                    entries.append((other, flows, corridor, max(float(course.ends[k - 1]) if k else 0.0, arc) - arc))
        return entries, STRtree([e[2] for e in entries]) if entries else None

    def _departure_held(self, vehicle, evidence):
        """Whether a departure waits at its berth (acquire): its way meets, for MEETING_M or more, the way of a car in a
        trip coming the other way along one of the same roads (_meetings), where the two cars would reach the meeting
        within HOLD_WINDOW_S of each other (each at HOLD_SPEED_MPS from where it is). The street holds one direction
        at a time where the two meet (a shared lane, or one lane left beside parked cars), so a car that set out would
        stand in its trip for the other car's whole stretch: it waits parked instead, as a driver waits for an oncoming
        car to clear a narrow street before pulling out. A departure waits HOLD_MAX_S at most; without the tick time
        (prepare_revisions) none waits."""
        now = self.traffic.now
        if now is None:
            return False
        first = self._holds.setdefault((vehicle, evidence), now)
        if now - first >= HOLD_MAX_S:
            return False
        # From where the cars are in this tick, not where they were when the planning snapshot began.
        if self._hold_index is None or self._hold_index[0] != now:
            self._hold_index = (now, self._ways_index())
        entries, tree = self._hold_index[1]
        if tree is None:
            return False
        meeting, mine = {}, 0.0
        for e in self.plans[evidence]["edges"]:
            flows = self._edge_flows(e)
            ok, _, corridor = self._edge(e)
            length = float(self.graph.edges[e]["route"]["arc"][-1])
            if flows and ok:
                for k in tree.query(corridor, predicate="intersects"):
                    other, theirs, _, at = entries[k]
                    if (
                        other != vehicle
                        and any(r == r2 and d != d2 for r, d in flows for r2, d2 in theirs)
                        and abs(mine - at) <= HOLD_WINDOW_S * HOLD_SPEED_MPS
                    ):
                        meeting[other] = meeting.get(other, 0.0) + length
                        break
            mine += length
        held = any(length >= MEETING_M for length in meeting.values())
        if held:
            self.stats["departure_hold_ticks"] = self.stats.get("departure_hold_ticks", 0) + 1
            if now == first:
                self.stats["departures_held"] = self.stats.get("departures_held", 0) + 1
        return held

    def _edge_flows(self, edge):
        """The (road, direction) of the lane ends of `edge`."""
        flows = set()
        for end in ("source", "target"):
            node = self.graph.nodes[self.graph.edges[edge][end]]
            if node.get("traffic_resource") is not None and node.get("direction") is not None:
                flows.add((node["traffic_resource"], node["direction"]))
        return flows

    def _doubles_back(self, driven, edges):
        """Whether `edges`, driven after `driven`, drive a road (the flow of a travel edge, TrafficRuns._lane) in the
        direction opposite to one in which the route drives it within DOUBLE_BACK_M before or after."""
        seen, arc, flows = {}, 0.0, []
        for k, e in enumerate(list(driven) + list(edges)):
            edge = self.graph.edges[e]
            node = self.graph.nodes[edge["source"]]
            if (
                edge["kind"] == "travel"
                and node.get("traffic_resource") is not None
                and node.get("direction") is not None
            ):
                flows.append((node["traffic_resource"], node["direction"], arc, k >= len(driven)))
            arc += float(edge["route"]["arc"][-1])
        for road, direction, at, new in flows:
            for other, s, was_new in seen.get(road, ()):
                if other != direction and (new or was_new) and abs(at - s) <= DOUBLE_BACK_M:
                    return True
            seen.setdefault(road, []).append((direction, at, new))
        return False

    def way_out_nodes(self, count=3):
        """`count` lane nodes spread over the largest strongly connected part of the map (farthest-point sampling
        from its middle): a car with a way to one of them is not shut off from the main road network (Director.way_out).
        Lane nodes, not berths: a berth another car parks at is not a road closed."""
        part = max(_components(self.graph.nodes, self._lane_links(turn_backs=True)), key=len, default=())
        lanes = sorted(n for n in part if self.graph.nodes[n]["kind"] == "lane")
        if not lanes:
            return ()
        xy = np.array([self.graph.nodes[n]["pose"][:2] for n in lanes], float)
        far = np.linalg.norm(xy - xy.mean(axis=0), axis=1)
        chosen = [int(np.argmin(far))]
        while len(chosen) < min(count, len(lanes)):
            far = np.min([np.linalg.norm(xy - xy[k], axis=1) for k in chosen], axis=0)
            chosen.append(int(np.argmax(far)))
        return tuple(lanes[k] for k in chosen)

    def _through_nodes(self):
        """The lane nodes of the largest strongly connected part of the map without turn backs (edges of kind
        'return'): between any two of them a car can drive without turning back. Cruising drivers pass through streets
        rather than into dead ends and out again, which also keeps them off single-lane spurs they would hold both ways.
        Empty when that part has less than THROUGH_MIN_M of travel lanes (a tree-like network: tours use every lane
        node)."""
        if self._through is None:
            best = max(_components(self.graph.nodes, self._lane_links(turn_backs=False)), key=len, default=())
            best = frozenset(best)
            length = sum(
                float(e["route"]["arc"][-1])
                for e in self.graph.edges.values()
                if e["kind"] == "travel" and e["source"] in best and e["target"] in best
            )
            self._through = best if length >= THROUGH_MIN_M else frozenset()
            self.stats["through_travel_m"] = round(length, 1)
        return self._through

    def _turn_backs(self):
        """The turn-back edges (kind 'return') of the map."""
        if self._returns is None:
            self._returns = frozenset(k for k, e in self.graph.edges.items() if e["kind"] == "return")
        return self._returns

    def _lane_links(self, *, turn_backs):
        """{node: [node]} of the edges between lane nodes (with or without turn backs): the map a path drives between
        its ends, where a berth is a terminal (DrivingGraph.path)."""
        links = {}
        for edge in self.graph.edges.values():
            if (
                (turn_backs or edge["kind"] != "return")
                and self.graph.nodes[edge["source"]]["kind"] != "berth"
                and self.graph.nodes[edge["target"]]["kind"] != "berth"
            ):
                links.setdefault(edge["source"], []).append(edge["target"])
        return links

    def _reachable(self, source):
        """Nodes reachable from `source` in the frozen map, ignoring occupancy (a static pre-filter); a berth other
        than `source` ends a path (DrivingGraph.path)."""
        if source not in self._reach:
            seen = {source}
            stack = [source]
            while stack:
                node = stack.pop()
                if node != source and self.graph.nodes[node]["kind"] == "berth":
                    continue
                for edge in self.graph.links[node]:
                    end = self.graph.edges[edge]["target"]
                    if end not in seen:
                        seen.add(end)
                        stack.append(end)
            self._reach[source] = frozenset(seen)
        return self._reach[source]

    def _static_path(self, source, target, *, turn_backs=True):
        """Exact unblocked Dijkstra paths, shared across facility destinations.

        Adapted from Claude's 90c58af; bounded LRU and sealed geometry make the
        cache lifetime explicit. Dynamic occupancy and reservations are NOT
        cached here. Zero capacity retains the original search for comparisons.
        """
        if source not in self.graph.nodes or target not in self.graph.nodes:
            raise ValueError("Unknown driving destination")
        if not self.static_tree_capacity:
            return self.graph.path(source, target, blocked_edges=() if turn_backs else self._turn_backs())
        key = source if turn_backs else (source, "without turn backs")
        if key in self._static_trees:
            previous = self._static_trees.pop(key)
            self.stats["static_tree_hits"] += 1
        else:
            queue = [(0.0, source)]
            distance, previous = {source: 0.0}, {}
            while queue:
                cost, node = heapq.heappop(queue)
                if cost != distance[node]:
                    continue
                if node != source and self.graph.nodes[node]["kind"] == "berth":
                    continue
                for edge_id in sorted(self.graph.links[node]):
                    edge = self.graph.edges[edge_id]
                    if not turn_backs and edge["kind"] == "return":
                        continue
                    end = edge["target"]
                    candidate = cost + self.graph.cost(edge_id)
                    if candidate < distance.get(end, math.inf):
                        distance[end], previous[end] = candidate, edge_id
                        heapq.heappush(queue, (candidate, end))
            self.stats["static_tree_builds"] += 1
        self._static_trees[key] = previous
        if len(self._static_trees) > self.static_tree_capacity:
            self._static_trees.popitem(last=False)
            self.stats["static_tree_evictions"] += 1
        if target != source and target not in previous:
            return None
        route, node = [], target
        while node != source:
            edge = previous[node]
            route.append(edge)
            node = self.graph.edges[edge]["source"]
        return route[::-1]

    def _planning_bodies(self):
        """The bodies a plan must clear: the observed bodies; a car in an active trip at its reserved destination
        instead of its pose on the road; a car with an admitted journey that has not departed (`pending`) both where
        it stands and at its destination, which it will occupy."""
        planning_bodies = dict(self.bodies)
        for other, evidence in self.active.items():
            plan = self.plans[evidence]
            destination = self.graph.nodes[plan["target"]]["pose"]
            planning_bodies[other] = Body(tuple(destination), self.occupancy[other].size).polygon()
        for other, target in sorted(self._pending.items()):
            if other in self.active or other not in planning_bodies:
                continue
            if target not in self.graph.nodes or self.graph.nodes[target]["kind"] != "berth":
                raise ValueError("A pending journey must end at an exact mapped berth")
            destination = Body(tuple(self.graph.nodes[target]["pose"]), self.occupancy[other].size).polygon()
            planning_bodies[other] = unary_union([planning_bodies[other], destination])
        return planning_bodies

    def _planning_blocked(self, vehicle):
        """Reuse exact edge/body intersections for one occupancy/reservation state.

        Adapted from Claude c925a9a's _edge_hits. The graph and admitted edge
        corridors are fixed. Updates to body pose/size/presence or active trip
        destinations discard the whole table; entries never cross snapshots.
        acquire() still checks live bodies and traffic reservations directly.
        Disable index_occupancy for a reference comparison, not weaker checks.
        """
        if not self.index_occupancy:
            bodies = self._planning_bodies()
            return lambda edge, corridor: any(
                actor != vehicle and corridor.intersects(body) for actor, body in bodies.items()
            )
        if self._occupancy_index is None:
            from shapely.strtree import STRtree

            bodies = self._planning_bodies()
            actors = sorted(bodies)
            self._occupancy_index = (actors, STRtree([bodies[a] for a in actors]), {})
            self.stats["occupancy_index_builds"] += 1
        actors, tree, memo = self._occupancy_index

        def blocked(edge, corridor):
            if edge not in memo:
                memo[edge] = frozenset(actors[int(j)] for j in tree.query(corridor, predicate="intersects"))
                self.stats["occupancy_index_queries"] += 1
            else:
                self.stats["occupancy_index_hits"] += 1
            return bool(memo[edge] - {vehicle})

        return blocked

    def _path(self, vehicle, source, target, extra=None, *, turn_backs=True, avoid=()):
        """Edges of the least-cost path (DrivingGraph.cost) clear of every planning body other than the vehicle's
        (and of the polygon `extra`), or None; without `turn_backs`, a path with no turn back; never through the edges
        `avoid` (_meeting_free).

        Failed edges are removed monotonically. When a path fails, every edge already known to fail for this vehicle
        in the planning snapshot (found by earlier plans) is removed with it. All removed edges are invalid and the
        returned path crosses none, so its cost is the least clear cost, as when the failed edges of one path are
        removed at a time; among paths of equal cost the choice may differ."""
        occupied = self._planning_blocked(vehicle)
        blocked = set(avoid)
        blockers, unsupported = set(), 0
        # Another exact graph path may still exist; nearby unconnected berths never serve as substitutes.
        while True:
            self.stats["path_searches"] += 1
            edges = (
                self.graph.path(source, target, blocked_edges=blocked if turn_backs else blocked | self._turn_backs())
                if blocked
                else self._static_path(source, target, turn_backs=turn_backs)
            )
            if edges is None:
                if avoid:
                    return None
                if extra is None:
                    self.stats["blocked_by_bodies" if blocked else "no_path"] = (
                        self.stats.get("blocked_by_bodies" if blocked else "no_path", 0) + 1
                    )
                self.path_failures[vehicle] = dict(
                    source=source,
                    target=target,
                    removed_edges=len(blocked),
                    unsupported_edges=unsupported,
                    blocking_bodies=sorted(blockers),
                    extra=extra is not None,
                )
                return None
            invalid = set()
            for edge in edges:
                ok, _, corridor = self._edge(edge)
                if not ok:
                    invalid.add(edge)
                    unsupported += 1
                elif occupied(edge, corridor) or (extra is not None and corridor.intersects(extra)):
                    invalid.add(edge)
                    self.stats["occupancy_rejections"] += 1
                    if self._occupancy_index is not None:
                        blockers |= self._occupancy_index[2].get(edge, frozenset()) - {vehicle}
            if invalid:
                blocked.update(invalid)
                blocked.update(self._known_invalid(vehicle))
                continue
            return edges

    def _known_invalid(self, vehicle):
        """Edges known, in the current planning snapshot, to fail for `vehicle`: unsupported geometry, or a corridor
        that a planning body other than the vehicle's intersects."""
        memo = self._occupancy_index[2] if self._occupancy_index is not None else {}
        return {e for e, actors in memo.items() if actors - {vehicle}} | {
            e for e, (ok, _, _) in self.geometry.items() if not ok
        }

    def keeps_ways(self, vehicle, berth, needs):
        """Whether `vehicle` parked at `berth` leaves every other car each drive it still needs and has now.

        `needs`: {car: ((berth, target berths), ...)} (Director.ways_home): the departure of an admitted journey
        that has not left, and the way home of a car that stands, or is about to park, away from its owner's home
        parking. A way is a path on the current planning snapshot; with this car's body at `berth` in place it must
        still exist. A drive that has no way now is not made worse by this parking."""
        began = time.perf_counter()
        try:
            return self._keeps_ways(vehicle, berth, needs)
        finally:
            self.stats["keeps_ways_wall_s"] += time.perf_counter() - began

    def keeps_roads(self, vehicle, berth):
        """Whether `vehicle` parked at `berth` keeps the roads passable (ConnectedRoads.keeps): every two lanes that
        reach each other with the planning bodies other than its own in place still do with its body at `berth`, as
        for every car parked in the scene. True when no ConnectedRoads is bound."""
        if self.roads is None:
            return True
        began = time.perf_counter()
        try:
            self._snapshot()
            key = ("roads", vehicle, berth)
            if self.cache_plan_results and key in self.plan_results:
                return self.plan_results[key]
            standing = self.plan_results.get(("roads standing", vehicle)) if self.cache_plan_results else None
            if standing is None:
                standing = self.roads.standing(b for a, b in sorted(self._planning_bodies().items()) if a != vehicle)
                if self.cache_plan_results:
                    self.plan_results[("roads standing", vehicle)] = standing
            kept = self.roads.keeps(Body(tuple(self.graph.nodes[berth]["pose"]), tuple(self.size)).polygon(), standing)
            self.stats["roads_checked"] = self.stats.get("roads_checked", 0) + 1
            self.stats["roads_refused"] = self.stats.get("roads_refused", 0) + int(not kept)
            if self.cache_plan_results:
                self.plan_results[key] = kept
            return kept
        finally:
            self.stats["keeps_roads_wall_s"] = self.stats.get("keeps_roads_wall_s", 0.0) + time.perf_counter() - began

    def _keeps_ways(self, vehicle, berth, needs):
        self._snapshot()
        body = None
        for car, legs in sorted(needs.items()):
            if car == vehicle or car not in self.occupancy or self.occupancy[car] is None:
                continue
            for source, targets in legs:
                targets = tuple(t for t in targets if t != source and t in self._reachable(source))
                ways = self._ways(car, source, targets)
                if not ways:
                    self.stats["ways_home_already_closed"] = self.stats.get("ways_home_already_closed", 0) + 1
                    continue
                if body is None:
                    body = Body(tuple(self.graph.nodes[berth]["pose"]), tuple(self.size)).polygon()
                if any(not any(self._edge(e)[2].intersects(body) for e in edges) for edges in ways):
                    continue
                key = ("kept", car, source, targets, berth)
                kept = self.plan_results.get(key) if self.cache_plan_results else None
                if kept is None:
                    kept = any(self._path(car, source, t, extra=body) is not None for t in targets)
                    if self.cache_plan_results:
                        self.plan_results[key] = kept
                if not kept:
                    self.stats["ways_home_closed"] = self.stats.get("ways_home_closed", 0) + 1
                    return False
        return True

    def _ways(self, car, source, targets):
        """The ways of `car` from `source` to `targets` in the current planning snapshot (_keeps_ways): a berth target
        needs a drive to park there (plan); a lane node, a way to it (_path). Kept with the plans of the snapshot: every
        parking a car considers asks for the same ways of the other cars."""
        key = ("ways", car, source, targets)
        if self.cache_plan_results and key in self.plan_results:
            return self.plan_results[key]
        ways = [
            self.plans[r.evidence]["edges"]
            for r in (self.plan(car, source, t) for t in targets if self.graph.nodes[t]["kind"] == "berth")
            if r is not None
        ]
        ways += [
            p
            for p in (self._path(car, source, t) for t in targets if self.graph.nodes[t]["kind"] != "berth")
            if p is not None
        ]
        if self.cache_plan_results:
            self.plan_results[key] = ways
        return ways

    def _plan_current_snapshot(self, vehicle, source, target):
        edges = self._meeting_free(vehicle, source, target)
        if edges is None:
            return None
        return self._compile(vehicle, source, target, edges)

    def _compile(self, vehicle, source, target, edges):
        """The Route of the clear edge sequence `edges` from berth `source` to berth `target`."""
        # Repeated facility slots often select exactly the same berth path.
        # Reuse only its frozen schedule/corridor, AFTER checking this call's
        # occupancy and selected edges. A cache hit never admits departure.
        key = (vehicle, source, target, tuple(edges), self.limits)
        evidence = self.compiled_plans.get(key)
        if evidence is not None:
            self.stats["plan_cache_hits"] += 1
            schedule = self.plans[evidence]["schedule"]
            return Route(schedule.length_m, schedule.duration_s, evidence)
        route = self.graph.concatenate(edges)
        schedule = Schedule(route, self.limits)
        payload = dict(
            vehicle=vehicle,
            source=source,
            target=target,
            edges=edges,
            size=self.size.tolist(),
            points=route["points"].tolist(),
            yaw=route["yaw"].tolist(),
            gear=route["gear"].tolist(),
            limits=self.limits.__dict__,
        )
        evidence = "measured-drive:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        self.plans[evidence] = dict(
            vehicle=vehicle,
            source=source,
            target=target,
            edges=tuple(edges),
            schedule=schedule,
            corridor=swept(route, self.size),
            resources=frozenset(r for e in edges for r in self.graph.edges[e]["traffic_resources"]),
        )
        self.compiled_plans[key] = evidence
        return Route(schedule.length_m, schedule.duration_s, evidence)

    @staticmethod
    def _at(observed_pose, expected, speed):
        p = np.asarray(observed_pose, float)
        if p.shape != (3,) or not np.isfinite(p).all() or not math.isfinite(speed):
            raise ValueError("Finite measured pose and speed required")
        return (
            np.linalg.norm(p[:2] - expected[:2]) <= 0.1
            and abs(angle_delta(p[2], expected[2])) <= 0.05
            and abs(speed) <= 0.08
        )

    def acquire(self, evidence, observed_pose, speed):
        """Admit a trip after the real driver is ready to depart.

        A plan may have become occupied since activity selection: the route is
        rechecked against the bodies standing outside a trip, and the first run
        is reserved when it begins within a car length of the berth (traffic_runs). The departure
        waits while its way would soon meet an oncoming car's (_departure_held). The
        caller still owns driver/berth transactions.
        """
        plan = self.plans[evidence]
        vehicle = plan["vehicle"]
        if vehicle in self.active:
            return self.active[vehicle] == evidence
        body = self.occupancy[vehicle]
        if body is None or not np.allclose(body.size, self.size, atol=1e-8, rtol=0.0):
            return False
        source_pose = self.graph.nodes[plan["source"]]["pose"]
        if not self._at(body.pose, source_pose, 0.0):
            return False
        if not self._at(observed_pose, source_pose, speed):
            return False
        if not self._clear(vehicle, plan["edges"]):
            self.stats["occupancy_rejections"] += 1
            return False
        if self._departure_held(vehicle, evidence):
            return False
        if not self.traffic.depart(vehicle, evidence):
            self.stats["traffic_rejections"] += 1
            return False
        self._holds.pop((vehicle, evidence), None)
        self.active[vehicle] = evidence
        # A trip beginning and ending between two plans leaves an equal snapshot key but moves a body: begin anew.
        self.planning_snapshot = None
        return True

    def revise(self, vehicle, avoid, *, reason, detour_m=None, check=None):
        """Take the remaining route of an active car round the polygon `avoid` (a body about to stand across it: a car
        parking; the parts of a run other cars keep from it: congestion) where the car can still change its way. True
        when its remaining route clears `avoid` or the car now follows a new route; False, nothing changed, otherwise.
        With `detour_m`, a new way costs at most that much more than the present route beyond the splice; with `check`,
        TrafficRuns.revise also asks it of the recomputed trips (a congestion revision must shorten the wait).

        The new route keeps the present one up to a lane node (the splice) and goes on to the same destination berth by
        the least-cost path clear of the planning bodies and of `avoid`; where the route it replaces beyond the splice
        has no turn back, by one without a turn back if there is one (a detour round a parked car does not send a car
        down a street and back: cruise5 r11f v4, vehicle-043 taken round a parking car by a three-point turn at a
        junction mouth held that street both ways for 265 m). The splice lies before the first edge whose
        corridor meets `avoid`, beyond the end of every run the car holds and beyond its stopping distance and one
        second of travel: the car leaves no reserved run and need not brake harder for the new way. The latest such
        splice with a clear path is taken. The other trips are recomputed against the new route as for a departure
        (TrafficRuns.revise); if one of them would be granted a run it may not reserve, the car keeps its route. A
        moving car's execution continues on a copy prepared before any reservation changes (its `prepare_reroute`); an
        execution without one keeps its route.
        Each change is recorded in `revisions` for the activity layer, which binds the drive step to the new
        evidence."""
        old = self.active.get(vehicle)
        trip = self.traffic.trips.get(vehicle)
        if old is None or trip is None:
            return False
        plan = self.plans[old]
        edges = plan["edges"]
        course = self.traffic.course(old)
        arc, speed = trip["arc"], trip["speed"]
        execution = self.executions.get(vehicle)
        if execution is not None:
            arc, speed = max(arc, float(execution.arc_m)), max(speed, float(execution.speed_mps))
        current = course.edge_index(arc)
        hit = [m for m in range(current, len(edges)) if self._edge(edges[m])[2].intersects(avoid)]
        if not hit:
            return True
        if execution is not None and not hasattr(execution, "prepare_reroute"):
            return False
        braking = self.limits.braking_mps2
        earliest = max(
            [arc + speed * speed / (2.0 * braking) + speed + 1e-6] + [r.s1 for r in trip["held"] if r.s1 > arc]
        )
        self._snapshot()
        for k in range(hit[0] - 1, current - 1, -1):
            splice = float(course.ends[k])
            if splice < earliest:
                break
            node = self.graph.edges[edges[k]]["target"]
            if self.graph.nodes[node]["kind"] != "lane":
                continue
            # The searches (a way found or none) are kept until the planning bodies change; the speed and the runs
            # held, which decide the earliest splice, and the traffic are checked anew each time.
            key = (vehicle, node, plan["target"], avoid.wkb)
            if key in self._revision_paths:
                self._revision_paths.move_to_end(key)
                path = self._revision_paths[key]
                self.stats["revision_path_hits"] = self.stats.get("revision_path_hits", 0) + 1
            else:
                straight = not any(self.graph.edges[e]["kind"] == "return" for e in edges[k + 1 :])
                path = (
                    self._path(vehicle, node, plan["target"], extra=avoid, turn_backs=False) if straight else None
                ) or self._path(vehicle, node, plan["target"], extra=avoid)
                self._revision_paths[key] = path
                if len(self._revision_paths) > 128:
                    self._revision_paths.popitem(last=False)
            if path is None:
                continue
            if detour_m is not None and (
                sum(self.graph.cost(e) for e in path) > sum(self.graph.cost(e) for e in edges[k + 1 :]) + detour_m
            ):
                self.stats["revisions_too_long"] = self.stats.get("revisions_too_long", 0) + 1
                continue
            new = self._compile(vehicle, plan["source"], plan["target"], list(edges[: k + 1]) + list(path)).evidence
            # The execution is prepared on a copy first: a refused prefix or schedule leaves route and traffic as
            # they were.
            proposed = None
            if execution is not None:
                try:
                    proposed = execution.prepare_reroute(self.plans[new]["schedule"].route, splice)
                except ValueError:
                    self.stats["revisions_execution_refused"] = self.stats.get("revisions_execution_refused", 0) + 1
                    continue
            if not self.traffic.revise(vehicle, new, check):
                self.stats["revisions_refused"] = self.stats.get("revisions_refused", 0) + 1
                return False
            if proposed is not None:
                self.executions[vehicle] = proposed
            self.active[vehicle] = new
            self.planning_snapshot = None
            self.revisions.setdefault(vehicle, []).append((old, new, splice, reason))
            self.stats["revisions"] = self.stats.get("revisions", 0) + 1
            return True
        self.stats["revisions_without_way"] = self.stats.get("revisions_without_way", 0) + 1
        return False

    def prepare_revisions(self, t_s=None):
        """Take cars round the bodies of cars about to park across their way and, given the tick time `t_s`, round
        congestion (TrafficRuns.prepare_revisions), once per tick on the accepted world state, before any car senses or
        moves."""
        self.traffic.prepare_revisions(t_s)

    def traffic_limit(self, vehicle, arc_m, speed_mps):
        """The arc an active car must stop before for its next conflict run, or None (traffic_runs)."""
        return self.traffic.limit(vehicle, arc_m, speed_mps, self.limits.braking_mps2)

    def nominal_at(self, evidence, elapsed_s):
        plan = self.plans[evidence]
        if self.active.get(plan["vehicle"]) != evidence:
            raise ValueError("Cannot execute a trip without current traffic admission")
        return plan["schedule"].at(elapsed_s)

    def advance_motion(self, evidence, observed_pose, speed, dt, *, stop_before_arc_m=None):
        """Propose a physical control tick from the authoritative body snapshot.

        The backend supplies a live clearance-adjusted stopping arc, applies
        the proposal only after its swept collision check, and submits that
        actual pose/speed on the next tick. A refused/modified proposal cannot
        silently advance this execution. It must be diagnosed, not repaired by
        snapping or freezing the nominal schedule. Riders/transfers are handled
        by the shared activity executor, not invented by this geometry bridge.
        """
        plan = self.plans[evidence]
        vehicle = plan["vehicle"]
        if self.active.get(vehicle) != evidence:
            raise ValueError("Live motion requires current traffic admission")
        execution = self.executions.get(vehicle)
        if execution is None:
            execution = RouteExecution(plan["schedule"].route, self.limits)
        state = execution.state()
        p = np.asarray(observed_pose, float)
        body = self.occupancy[vehicle]
        expected = np.r_[state["xy"], state["yaw"]]

        def same_pose(value):
            q = np.asarray(value, float)
            return (
                q.shape == (3,)
                and np.isfinite(q).all()
                and np.linalg.norm(q[:2] - expected[:2]) <= 1e-6
                and abs(angle_delta(q[2], expected[2])) <= 1e-6
            )

        if (
            body is None
            or not same_pose(body.pose)
            or not same_pose(p)
            or not math.isfinite(speed)
            or abs(speed - state["speed_mps"]) > 1e-6
        ):
            raise ValueError("Measured motion differs from the previous accepted execution state")
        traffic = self.traffic_limit(vehicle, state["arc_m"], state["speed_mps"])
        if traffic is not None:
            stop_before_arc_m = traffic if stop_before_arc_m is None else min(stop_before_arc_m, traffic)
        result = execution.advance(dt, stop_before_arc_m=stop_before_arc_m)
        self.executions[vehicle] = execution
        return result

    def can_arrive(self, evidence, observed_pose, speed, *, require_executed=False):
        """Read-only arrival predicate shared by transactional activity handoff."""
        plan = self.plans[evidence]
        vehicle = plan["vehicle"]
        if self.active.get(vehicle) != evidence:
            raise ValueError("Trip has no active traffic reservation")
        target = self.graph.nodes[plan["target"]]["pose"]
        body = self.occupancy[vehicle]
        if body is None or not self._at(body.pose, target, 0.0):
            return False
        if not self._at(observed_pose, target, speed):
            return False
        execution = self.executions.get(vehicle)
        if require_executed and execution is None:
            return False
        if execution is not None and not execution.arrived:
            return False
        if require_executed:
            state = execution.state()
            expected = np.r_[state["xy"], state["yaw"]]
            for p in (body.pose, observed_pose):
                if (
                    np.linalg.norm(np.asarray(p)[:2] - expected[:2]) > 1e-6
                    or abs(angle_delta(p[2], expected[2])) > 1e-6
                ):
                    return False
            if abs(speed - state["speed_mps"]) > 1e-6:
                return False
        return True

    def arrive(self, evidence, observed_pose, speed):
        """Release only after the real body has parked at the exact destination."""
        if not self.can_arrive(evidence, observed_pose, speed):
            return False
        plan = self.plans[evidence]
        vehicle = plan["vehicle"]
        execution = self.executions.get(vehicle)
        del self.active[vehicle]
        self.planning_snapshot = None
        self.traffic.arrive(vehicle)
        self.trips_done[vehicle] = self.trips_done.get(vehicle, 0) + 1
        if execution is not None:
            self.completed_motion.append(
                dict(
                    vehicle=vehicle,
                    evidence=evidence,
                    elapsed_s=execution.time_s,
                    length_m=execution.arc_m,
                    brake_count=execution.brake_count,
                )
            )
            del self.executions[vehicle]
        return True
