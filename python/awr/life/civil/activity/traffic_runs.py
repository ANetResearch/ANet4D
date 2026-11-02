"""Progressive traffic admission: a car holds only the conflict runs it is passing.

A car's route is divided, by the poses of its body along it, into safe and unsafe stretches. Conflicts are taken
against the traffic actually present: the remaining routes of the other cars in a trip (from the edge before the one
each is on), not every manoeuvre the map offers. A driving map superposes alternative lanes of one carriageway, lane
changes, junction turns and berth manoeuvres; measured against all of them, every pose of a town street is unsafe,
each route is a single run, and cars can only drive one at a time.

A pose is unsafe when the body meets a corridor of another car's remaining route on an edge that does not carry the
same flow as the edge the car is on. The same flow is: the edge itself, the travel edges of the same road in the
same direction (whichever layout of the carriageway: centre lane, lane beside parked cars, separated lane) and the
changes between those layouts that overlap (no point shifted a body width from the lane left: no car can be beside
one changing), the beginning of an edge that leaves either end of the edge, and the end of an edge that enters its
start; a car on any of those follows or leads this car. So does a car on the next edge of a route both cars take (a
route came to its edge by the edge of the other car), and a car on the edge another route takes after an edge this
route came by (the two routes have parted after one common edge): a route that alternates between layouts of one
carriageway changes lanes by joins shifted more than a body width, whose corridors the neighbouring parts of the
same route meet (cq3p v7: vehicle-053 waited 60 s for a 204 m run of vehicle-040, which it was following along the
same edges). Any other corridor - the opposite direction of a narrow road, a merge into a lane a
car may drive in beside, a crossing or merging turn into another road, a turn back, the sweep of a berth manoeuvre -
is another flow: a car standing there could obstruct the other car, and the live following distance does not see a
car beside. So is a body on a no-standing area (a pedestrian crossing): cars pass
it without stopping on it. A run is a maximal unsafe stretch; safe gaps shorter than MERGE_M join their neighbouring
runs.

A car stands only at safe poses (or behind a live obstacle). It enters a run only after reserving the whole run:

- conflict: no other car holds a run with a part ahead of its body (see below) that meets one of this run's parts on
  an edge of another flow, and no other car's body stands in the run on an edge of another flow - unless that car
  yields: it can still stop (beyond its stopping distance and a second of travel) before the first of its parts that
  meets the stretch this car will drive or stand on up to the end of the run and the standing room after it. It then
  stops there until this car no longer holds a run that meets its own; a car yields to one car at a time and
  at most YIELDS_PER_RUN times for the runs it holds. A yielding car waits for a car that reserved after it, so
  waits for yields cannot form a cycle;
- front: no other car ahead of it on its route before the end of the run will wait there for a reservation (at the
  stop of the first run ahead of it that it does not hold). Before the run, the front car reserves first; in the run
  (on this car's flow), this car would stand behind it in the run while it waits for a run that may meet this one.
  Nor does a car waiting for a run that meets this one stand (at that run's stop) in the standing room after the
  run: this car would stand behind it while it waits for this car to clear its way (cq3p-d12-s3 v1: vehicle-041 followed vehicle-050 into a dead end
  where both turn back; vehicle-050 stood in the turn, waiting for its way out past vehicle-041);
- order: no car that has waited longer, for a run that conflicts with this one, is still waiting on a conflict, or
  on its room while only cars driving on stand in it (none of them waiting, yielding or slower than MOVING_MPS: they
  clear it, and a car they wait for no longer waits for the turn of the car behind) - unless this car's own body stands
  in that run, or it is one of the cars in that room (that car waits for this one to move). A car keeps its turn while
  it waits at the same stop, as its run is laid out anew with the traffic;
- room: the safe stretch after the run has a standing place for it (one per car length and GAP_M) beside the cars
  standing in that stretch and those holding runs into it; a run ending at the destination berth needs none, the
  activity reserved the berth;
- parking: a run ending at the destination berth only when the parked body meets no other active car's remaining
  route. A car whose remaining route it meets is first taken round it where it can still change its way
  (DrivingRoutes.revise: a new way from a lane node beyond the runs it holds and its stopping distance). These changes
  are made for all cars at once at the start of a tick (prepare_revisions), before any car looks ahead or
  moves, so that no car moves by a layout another car's change has replaced.

Congestion is relieved by going round it, as dynamic rerouting does in traffic simulation (SUMO's rerouting device:
a vehicle takes a faster route from where it can still change its way), never by moving a car out of the way. At the
start of a tick (prepare_revisions), a car whose next run, up to CONGESTION_AHEAD_M beyond its reach, is kept from it
(_obstruction) is expected to wait for it: the time the cars keeping it need to leave it, less the time this car needs
to reach its stop. A car holding a conflicting run keeps it until its body has passed the last of its parts that meet
this run; a car standing in the run, or ahead of this car up to the end of the run, keeps it until it moves; a car that
waits for a reservation or yields keeps it indefinitely. When the expected wait is at least CONGESTION_WAIT_S, the car
is taken round where the run is kept from it (DrivingRoutes.revise) by a way that costs at most what the
wait would (the wait at the cruising speed, and CONGESTION_DETOUR_M) and on which the expected wait for the next run
within reach is shorter by at least half of CONGESTION_WAIT_S (a way round into the stretch the same cars are driving
to is no way round); without one it waits for the run as before. A car looks again for a way round the same cars at
the same stop after CONGESTION_RETRY_S.

The runs of a trip are recomputed when the traffic changes (a trip begins or ends, another car moves on to its next
edge). A car keeps the runs it holds (the recomputed runs that overlap them) and is granted those it can no longer
stop before. When another car departs, the cars already driving are recomputed first: if one of them would be granted
a run it may not reserve, the departure waits. A departure also needs its first car length and gap clear of the
bodies of the cars in a trip, and its first run when that begins within a car length and gap of the berth: the driver
waits parked rather than pulling out only to stand at the run (cq3p v1, vehicle-045 stood 60 s in its trip 3.5 m out of
its kerb berth, waiting for the room of a run another car stood in). A run is released when the car's pose
has left it, and each of its parts (the sweep over at most PART_M of one edge) once the body has left that part
behind: the car never stands there again in this run, and a body still on it is seen by the conflict rule. Nothing here moves a car: the stop arc it returns is one more limit for the motion controller, which
still stops for every live obstacle (pedestrians, cars ahead).
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import LineString

from ..curb_allocation import footprint, swept
from ..vehicle_connections import angle_delta

STEP_M = 0.5  # spacing of the sampled body poses along a route
GAP_M = 1.0  # spacing between standing cars
MERGE_M = 1.0  # a safe gap shorter than this joins its neighbouring runs
PART_M = 20.0  # longest stretch of one edge swept by one part of a run (released as the car leaves it behind)
YIELDS_PER_RUN = 2  # times a car holding runs may stop inside them to let another car through
MOVING_MPS = 0.5  # a car in the room of a waiting car keeps that car's turn while it drives on at least this fast
ON_ROUTE_M = 0.5  # a pose lies on a route stretch within this distance ...
ON_ROUTE_RAD = 0.6  # ... and heading difference
KEEP_COURSES = 32  # geometry kept for routes not in a trip (departures that wait, plans not taken)
CONGESTION_WAIT_S = (
    10.0  # an expected wait this long for the next run sends a car round what keeps it (module docstring)
)
CONGESTION_AHEAD_M = 80.0  # ... for a next run up to this far beyond the car's reach
CONGESTION_MIN_MPS = 2.0  # speed below which a moving car's times are counted at this speed
CONGESTION_DETOUR_M = 400.0  # a way round costs at most this much more than the rest of the route it replaces
CONGESTION_RETRY_S = 10.0  # a car looks again for a way round the same cars at the same stop after this long


@dataclass(eq=False)
class Run:
    s0: float  # first unsafe sampled arc
    s1: float  # first safe sampled arc after the run (the route end for the last run)
    stop: float  # last safe sampled arc before the run (where a car waits for it)
    parts: tuple  # ((edge, polygon), ...): the body swept over the run, per edge and PART_M of it
    room: tuple | None  # (a, b, capacity, line) of the safe stretch after the run, None at the route end
    exit_pose: tuple | None  # pose at s1
    tree: object = field(default=None, repr=False)
    ends: tuple = ()  # the arc at which each part ends (none: no part is left behind before s1)
    prev: tuple = ()  # the route edge before each part's edge (None at the route start; none: unknown)


@dataclass
class Course:
    """The static geometry of one planned route."""

    route: dict
    edges: tuple
    ends: np.ndarray  # end arc of each route edge
    arcs: np.ndarray  # sampled arcs
    index: np.ndarray  # route edge index of each sample
    bodies: np.ndarray  # body polygon at each sample
    crossing: np.ndarray  # whether the body at a sample is on a no-standing area
    corridors: tuple  # ((end arc, admitted corridor), ...) per route edge
    sweeps: tuple  # body swept over each route edge
    tree: object  # STRtree of `sweeps`

    def edge_index(self, arc):
        return min(int(np.searchsorted(self.ends, arc - 1e-9, side="left")), len(self.edges) - 1)

    def edge_at(self, arc):
        return self.edges[self.edge_index(arc)]


@dataclass
class Layout:
    runs: list
    course: Course


def _slice(route, lo, hi):
    arc = route["arc"]
    lo, hi = max(float(lo), 0.0), min(float(hi), float(arc[-1]))
    s = np.unique(np.r_[lo, arc[(arc > lo) & (arc < hi)], hi])
    if len(s) < 2:
        s = np.array([lo, min(lo + 1e-3, float(arc[-1]))])
    return dict(
        points=np.column_stack([np.interp(s, arc, route["points"][:, k]) for k in (0, 1)]),
        yaw=np.interp(s, arc, route["yaw"]),
    )


def _span(run):
    return [round(run.stop, 1), round(run.s0, 1), round(run.s1, 1)]


def _pose(route, s):
    return (
        float(np.interp(s, route["arc"], route["points"][:, 0])),
        float(np.interp(s, route["arc"], route["points"][:, 1])),
        float(np.interp(s, route["arc"], route["yaw"])),
    )


class TrafficRuns:
    """The runs of the active trips of one DrivingRoutes backend (see the module docstring)."""

    def __init__(self, routes, no_standing=()):
        """`no_standing`: polygons where a car does not stop (pedestrian crossings)."""
        self.routes = routes
        self.no_standing = shapely.STRtree(list(no_standing)) if len(no_standing) else None
        self.length = float(routes.size[0])
        self.courses = OrderedDict()  # evidence -> Course, most recently used last
        self._meet = {}  # (evidence, other evidence) -> last conflicting edge index of the other route per sample
        self._runs_memo = {}  # (evidence, first sample, last sample, next run's first sample) -> Run
        self.trips = {}  # vehicle -> dict(evidence, arc, speed, unsafe, runs, held)
        self.waiting = {}  # vehicle -> dict(order, run, reason)
        self.yields = {}  # vehicle holding runs -> dict(waiter, stop arc) it stops before while the waiter passes
        self._yielded = {}  # vehicle -> yields given since it last reserved a run
        self._room_by = ()  # the cars in the room of the run last refused for room (_refusal)
        self.now = None  # the tick time given to prepare_revisions (congestion is relieved only with one)
        self._relieved = {}  # vehicle -> ((stop, cars), time) of its last search for a way round congestion
        # Changes of reservation state since the caller last took them (evidence for the tape, no effect here):
        # a run reserved (with the cars yielding to it), a wait begun or its reason changed (with the cars concerned),
        # a yield ended.
        self.journal = []
        self._order = 0
        self._after, self._before, self._raw, self._flows = {}, {}, {}, {}
        self.events = []
        self.stats = dict(
            layouts=0,
            layout_wall_s=0.0,
            relayouts=0,
            acquired=0,
            released=0,
            refused_conflict=0,
            refused_front=0,
            refused_order=0,
            refused_room=0,
            refused_parking=0,
            refused_committed=0,
            committed=0,
            max_run_m=0.0,
        )

    # Static geometry.

    def _lane(self, edge):
        """The flow of an edge along a road: the road and direction of a travel edge (the layouts of one road -
        centre lane, lanes beside parked cars, lanes followed on the measured carriageway - carry one flow each way)
        or of a change between two layouts of that road in that direction that overlap (`_shift` less than a body
        width), else the lane of a travel edge; None for any other edge (a merge into a lane beside, a turn to another
        road, a turn back, a berth manoeuvre)."""
        if edge not in self._flows:
            graph = self.routes.graph
            e = graph.edges[edge]
            ends = [graph.nodes[e["source"]], graph.nodes[e["target"]]]
            flow = None
            if e["kind"] in ("travel", "join") and all(n.get("kind") == "lane" for n in ends):
                keys = {(n.get("traffic_resource"), n.get("direction")) for n in ends}
                resource, direction = ends[0].get("traffic_resource"), ends[0].get("direction")
                if (
                    len(keys) == 1
                    and resource is not None
                    and direction is not None
                    and (e["kind"] == "travel" or self._shift(edge) < float(self.routes.size[1]))
                ):
                    flow = ("road", resource, direction)
                elif e["kind"] == "travel" and ends[0].get("fragment") is not None and direction is not None:
                    flow = (ends[0]["fragment"], ends[0]["direction"])
            self._flows[edge] = flow
        return self._flows[edge]

    def _shift(self, edge):
        """Lateral shift of a join: the largest distance of a point of it from the lane it leaves (that lane's travel
        edges from the join's start on, continued straight along its last heading) - every point, not only the end, as
        a join may swing out further on its way. Less than a body width, the join keeps overlapping that lane and no car
        can be beside one changing between them; a larger shift merges into, or passes, a lane a car may be driving in
        beside it, which the live following distance does not see."""
        graph = self.routes.graph
        e = graph.edges[edge]
        node = e["source"]
        lane = (graph.nodes[node].get("fragment"), graph.nodes[node].get("direction"))
        start = graph.nodes[node]["pose"]
        points, heading = [np.asarray(start[:2], float)], float(start[2])
        reach, length = float(e["route"]["arc"][-1]) + float(self.routes.size[0]), 0.0
        visited = set()  # a lane may close on itself (codex 57d320b)
        while length < reach and node not in visited:
            visited.add(node)
            ahead = [
                f
                for f in sorted(graph.links[node])
                if graph.edges[f]["kind"] == "travel"
                and (
                    graph.nodes[graph.edges[f]["target"]].get("fragment"),
                    graph.nodes[graph.edges[f]["target"]].get("direction"),
                )
                == lane
            ]
            if not ahead or lane[0] is None:
                break
            route = graph.edges[ahead[0]]["route"]
            points.extend(np.asarray(route["points"][1:, :2], float))
            heading = float(route["yaw"][-1])
            length += float(route["arc"][-1])
            node = graph.edges[ahead[0]]["target"]
        points.append(points[-1] + (reach + self.length) * np.array([math.cos(heading), math.sin(heading)]))
        end = graph.nodes[e["target"]]["pose"]
        line = LineString(points)
        way = np.r_[np.asarray(e["route"]["points"], float)[:, :2], np.asarray(end[:2], float)[None]]
        return float(np.max(shapely.distance(line, shapely.points(way))))

    def _corridor(self, edge):
        """The swept corridor of `edge` (the geometry DrivingRoutes._edge gives an admitted edge)."""
        if edge not in self._raw:
            self._raw[edge] = swept(self.routes.graph.edges[edge]["route"], self.routes.size)
        return self._raw[edge]

    def _partial(self, edge, after):
        """The corridor of `edge` beyond one car length and a gap from its start (after) or before as much from its
        end (not after); None when the edge is not longer than that."""
        memo = self._after if after else self._before
        if edge not in memo:
            route = self.routes.graph.edges[edge]["route"]
            reach = self.length + GAP_M
            total = float(route["arc"][-1])
            memo[edge] = (
                None
                if total <= reach
                else swept(
                    _slice(route, reach, total) if after else _slice(route, 0.0, total - reach), self.routes.size
                )
            )
        return memo[edge]

    def _same_flow(self, e, f, body):
        """Whether the body of a car on edge `e` meeting the corridor of edge `f` leaves `f`'s traffic in its flow."""
        if f == e:
            return True
        lane = self._lane(e)
        if lane is not None and lane == self._lane(f):
            return True
        E, F = self.routes.graph.edges[e], self.routes.graph.edges[f]
        if F["source"] in (E["source"], E["target"]):
            rest = self._partial(f, True)
            if rest is None or not rest.intersects(body):
                return True
        if F["target"] == E["source"]:
            rest = self._partial(f, False)
            if rest is None or not rest.intersects(body):
                return True
        return False

    def _one_flow(self, e, f, p, pe, pf):
        """Whether a body `p` on edge `e`, which its route came to by `pe`, meeting the corridor of edge `f` of a route
        that came to it by `pf` leaves `f`'s traffic in its flow: _same_flow, or one route came to its edge by the
        other's edge, or both came by one edge (module docstring)."""
        if f == pe or e == pf or (pe is not None and pe == pf):
            return True
        return self._same_flow(e, f, p)

    def course(self, evidence):
        if evidence in self.courses:
            self.courses.move_to_end(evidence)
            return self.courses[evidence]
        self._prune()
        routes = self.routes
        plan = routes.plans[evidence]
        graph, size = routes.graph, routes.size
        route, edges = plan["schedule"].route, tuple(plan["edges"])
        ends = np.cumsum([float(graph.edges[e]["route"]["arc"][-1]) for e in edges])
        total = float(route["arc"][-1])
        S = np.unique(np.r_[np.arange(0.0, total, STEP_M), total])
        k = np.minimum(np.searchsorted(ends, S - 1e-9, side="left"), len(edges) - 1)
        bodies = np.array(
            [footprint(dict(xy=p[:2], yaw=p[2]), size) for p in (_pose(route, s) for s in S)], dtype=object
        )
        crossing = np.zeros(len(S), bool)
        if self.no_standing is not None:
            crossing[np.unique(self.no_standing.query(bodies, predicate="intersects")[0])] = True
        sweeps = tuple(self._corridor(e) for e in edges)
        corridors = tuple((float(ends[m]), routes._edge(e)[2]) for m, e in enumerate(edges))
        out = self.courses[evidence] = Course(
            route, edges, ends, S, k, bodies, crossing, corridors, sweeps, shapely.STRtree(list(sweeps))
        )
        return out

    def _meets(self, evidence, other):
        """Per sample of `evidence`'s route, the last index of an edge of `other`'s route whose corridor the body
        meets on another flow (-1 for none)."""
        key = (evidence, other)
        if key not in self._meet:
            began = time.perf_counter()
            c, o = self.course(evidence), self.course(other)
            last = np.full(len(c.arcs), -1, int)
            for i, m in sorted(
                zip(*o.tree.query(c.bodies, predicate="intersects"), strict=False), key=lambda h: (h[0], -h[1])
            ):
                k = c.index[i]
                if m > last[i] and not self._one_flow(
                    c.edges[k],
                    o.edges[m],
                    c.bodies[i],
                    c.edges[k - 1] if k > 0 else None,
                    o.edges[m - 1] if m > 0 else None,
                ):
                    last[i] = m
            self._meet[key] = last
            self.stats["layout_wall_s"] += time.perf_counter() - began
        return self._meet[key]

    # Runs against the present traffic.

    def _unsafe(self, vehicle):
        trip = self.trips[vehicle]
        unsafe = self.course(trip["evidence"]).crossing.copy()
        for other in sorted(self.trips):
            if other == vehicle:
                continue
            t = self.trips[other]
            first = max(self.course(t["evidence"]).edge_index(t["arc"]) - 1, 0)
            unsafe |= self._meets(trip["evidence"], t["evidence"]) >= first
        return unsafe

    def _runs(self, evidence, unsafe):
        c = self.course(evidence)
        S, route, size = c.arcs, c.route, self.routes.size
        total = float(S[-1])
        idx = np.flatnonzero(unsafe)
        if not len(idx):
            return []
        groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        merged = [[groups[0][0], groups[0][-1]]]
        for g in groups[1:]:
            if S[g[0]] - S[merged[-1][1]] <= MERGE_M + STEP_M:
                merged[-1][1] = g[-1]
            else:
                merged.append([g[0], g[-1]])
        runs = []
        for n, (a, b) in enumerate(merged):
            nxt = merged[n + 1][0] if n + 1 < len(merged) else None
            key = (evidence, int(a), int(b), nxt)
            if key not in self._runs_memo:
                s0, s1 = float(S[a]), float(S[min(b + 1, len(S) - 1)])
                stop = float(S[a - 1]) if a > 0 else 0.0
                parts, ends, prev = [], [], []
                for m in range(len(c.edges)):
                    lo = 0.0 if m == 0 else float(c.ends[m - 1])
                    hi = float(c.ends[m])
                    if hi <= max(stop, 0.0) or lo >= s1:
                        continue
                    lo, hi = max(lo, stop), min(hi, s1)
                    cuts = np.linspace(lo, hi, max(int(math.ceil((hi - lo) / PART_M)), 1) + 1)
                    for u, v in zip(cuts[:-1], cuts[1:], strict=False):
                        parts.append((c.edges[m], swept(_slice(route, float(u), float(v)), size)))
                        ends.append(float(v))
                        prev.append(c.edges[m - 1] if m > 0 else None)
                room = exit_pose = None
                if b + 1 < len(S):
                    end = float(S[nxt - 1]) if nxt is not None else total
                    capacity = int((end - s1) // (self.length + GAP_M)) + 1
                    room = (s1, end, capacity, LineString(_slice(route, s1, max(end, s1 + 1e-3))["points"]))
                    exit_pose = _pose(route, s1)
                self._runs_memo[key] = Run(
                    s0, s1, stop, tuple(parts), room, exit_pose, ends=tuple(ends), prev=tuple(prev)
                )
                self.stats["layouts"] += 1
                self.stats["max_run_m"] = max(self.stats["max_run_m"], s1 - s0)
            runs.append(self._runs_memo[key])
        return runs

    def _refresh(self, vehicle, *, check=False):
        """Recompute the runs of a trip against the present traffic, keeping the runs it holds and granting those it
        can no longer stop before. With `check`, False (and the trip unchanged) when such a run may not be granted."""
        trip = self.trips[vehicle]
        unsafe = self._unsafe(vehicle)
        if trip["unsafe"] is not None and np.array_equal(unsafe, trip["unsafe"]):
            return True
        runs = self._runs(trip["evidence"], unsafe)
        arc, speed = trip["arc"], trip["speed"]
        braking = self.routes.limits.braking_mps2
        held, granted = [], []
        for run in runs:
            if run.s1 <= arc + 1e-6 and run.room is not None:
                continue
            kept = [h for h in trip["held"] if run.s0 < h.s1 and h.s0 < run.s1]
            if kept:
                held.append(run)
                if check and not any(h.s0 <= run.s0 and run.s1 <= h.s1 for h in kept):
                    granted.append(run)  # grown by the new traffic
            elif run.stop - arc < speed * speed / (2.0 * braking) - 1e-6:
                held.append(run)
                granted.append(run)
        if check:
            for run in granted:
                if self._refusal(vehicle, run, arc, committed=True) is not None:
                    return False
        self.stats["committed"] += sum(
            1 for run in held if not any(run.s0 < h.s1 and h.s0 < run.s1 for h in trip["held"])
        )
        trip.update(unsafe=unsafe, runs=runs, held=held)
        self.stats["relayouts"] += 1
        w = self.waiting.get(vehicle)
        if w is not None:
            w["run"] = next((r for r in runs if r not in held and (r.s1 > arc + 1e-6 or r.room is None)), None)
        return True

    def layout(self, key):
        """The present runs of a car in a trip (`key`: the car, or the evidence of its trip); for the evidence of a
        plan not in a trip, the runs its route would have against the present traffic."""
        if key not in self.trips:
            vehicle = self.routes.plans[key]["vehicle"]
            trip = self.trips.get(vehicle)
            if trip is None or trip["evidence"] != key:
                saved = trip
                self.trips[vehicle] = dict(evidence=key, arc=0.0, speed=0.0, unsafe=None, runs=[], held=[])
                try:
                    return Layout(self._runs(key, self._unsafe(vehicle)), self.course(key))
                finally:
                    if saved is None:
                        del self.trips[vehicle]
                    else:
                        self.trips[vehicle] = saved
            key = vehicle
        self._refresh(key)
        trip = self.trips[key]
        return Layout(list(trip["runs"]), self.course(trip["evidence"]))

    # Reservations.

    def _parts_tree(self, run):
        if run.tree is None:
            run.tree = shapely.STRtree([q for _, q in run.parts])
        return run.tree

    def _behind(self, run, arc):
        """Whether each part of `run` lies behind a body at `arc` on its route: the body (a car length from the end of
        the part) can no longer meet the stretch it sweeps."""
        if arc is None or not run.ends:
            return None
        return [end + self.length <= arc for end in run.ends]

    def _meeting_parts(self, a, b, a_arc=None, b_arc=None):
        """The pairs (i, j) of parts of runs `a` and `b` that meet on edges of another flow; `a_arc`, `b_arc`: the arcs
        of the cars holding them (the parts they have left behind are not counted)."""
        if not a.parts or not b.parts:
            return
        a_behind, b_behind = self._behind(a, a_arc), self._behind(b, b_arc)
        for i, j in zip(*self._parts_tree(b).query([p for _, p in a.parts], predicate="intersects"), strict=False):
            if (a_behind is not None and a_behind[i]) or (b_behind is not None and b_behind[j]):
                continue
            (e, p), (f, q) = a.parts[i], b.parts[j]
            pe, pf = (a.prev[i] if a.prev else None), (b.prev[j] if b.prev else None)
            if not (self._one_flow(e, f, p, pe, pf) and self._one_flow(f, e, q, pf, pe)):
                yield int(i), int(j)

    def _conflicts(self, a, b, a_arc=None, b_arc=None):
        """Whether runs `a` and `b` have parts that meet on edges of another flow (_meeting_parts)."""
        return next(self._meeting_parts(a, b, a_arc, b_arc), None) is not None

    def _occupies(self, other, run):
        """Whether the body of `other` (in a trip) stands in `run` on an edge of another flow."""
        body = self.routes.bodies.get(other)
        if body is None or other not in self.trips:
            return False
        trip = self.trips[other]
        course = self.course(trip["evidence"])
        k = course.edge_index(trip["arc"])
        edge, before = course.edges[k], (course.edges[k - 1] if k > 0 else None)
        return any(
            q.intersects(body) and not self._one_flow(edge, e, body, before, run.prev[n] if run.prev else None)
            for n, (e, q) in enumerate(run.parts)
        )

    def _on(self, line, route_yaw_at, pose):
        p = shapely.Point(pose[:2])
        if line.distance(p) > ON_ROUTE_M:
            return None
        s = line.project(p)
        return s if abs(angle_delta(pose[2], route_yaw_at(s))) <= ON_ROUTE_RAD else None

    def _pose_of(self, vehicle):
        body = self.routes.occupancy.get(vehicle)
        return None if body is None else body.pose

    def _wait_pose(self, vehicle):
        """Where a car in a trip will next wait for a reservation: the stop of the first run ahead of it that it does
        not hold (its present pose when it waits there already); None when it holds every run ahead of it."""
        self._refresh(vehicle)
        trip = self.trips[vehicle]
        arc = trip["arc"]
        stop = self._yield_stop(vehicle)
        if stop is not None:
            return _pose(self.course(trip["evidence"]).route, max(stop, arc))
        run = next((r for r in trip["runs"] if (r.s1 > arc + 1e-6 or r.room is None) and r not in trip["held"]), None)
        return None if run is None else _pose(self.course(trip["evidence"]).route, max(run.stop, arc))

    def _zone(self, vehicle, run, arc):
        """Where a car at `arc` will drive up to the end of `run`, and the standing room after it (None at the route
        end), as STRtrees."""
        route = self.course(self.trips[vehicle]["evidence"]).route
        way = [q for _, q in run.parts]
        if run.stop > arc + 1e-3:
            way.append(swept(_slice(route, arc, run.stop), self.routes.size))
        room = None
        if run.room is not None:
            # The bodies standing anywhere in the room (with the clearance of every swept part), not a band round it:
            # a car passing in the lane beside the room does not follow this one into it.
            a, b = run.room[0], run.room[1]
            room = shapely.STRtree([swept(_slice(route, a, max(b, a + 1e-3)), self.routes.size)])
        return shapely.STRtree(way), room

    def _yield_point(self, holder, zone, through):
        """The arc a car holding runs would stop before to let another car through its way and into the standing room
        after it (_zone): the start of the first of its remaining parts that meets that way. None (it does not yield)
        unless that start lies beyond its stopping distance and a second of travel, none of its remaining parts meets
        that room (it never follows the other car into the room, where that car may wait for a run of its own), and
        the other car has less to drive to its room (`through`) than this one to leave the way behind."""
        trip = self.trips[holder]
        if self._yield_stop(holder) is not None or self._yielded.get(holder, 0) >= YIELDS_PER_RUN:
            return None
        way, room = zone
        arc, speed = trip["arc"], trip["speed"]
        reach = arc + speed * speed / (2.0 * self.routes.limits.braking_mps2) + speed + 1.0
        first = last = math.inf
        for held in trip["held"]:
            behind = self._behind(held, arc) or [False] * len(held.parts)
            for k, (_, q) in enumerate(held.parts):
                if behind[k]:
                    continue
                if room is not None and len(room.query(q, predicate="intersects")):
                    return None
                if len(way.query(q, predicate="intersects")):
                    if first == math.inf:
                        first = held.stop if k == 0 or not held.ends else held.ends[k - 1]
                    last = held.ends[k] if held.ends else held.s1
        stop = first - 0.25
        if not (math.isfinite(stop) and stop >= reach and through < last + self.length - arc):
            return None
        return stop

    def _yield_stop(self, holder):
        """The arc a car stops before while it yields (_yield_point), None when it does not; a yield ends when the car
        it let through no longer holds a run that meets the parts the yielding car has still to drive."""
        y = self.yields.get(holder)
        if y is None:
            return None
        waiter = self.trips.get(y["waiter"])
        trip = self.trips.get(holder)
        if waiter is not None and trip is not None:
            for mine in trip["held"]:
                for theirs in waiter["held"]:
                    a_behind, b_behind = self._behind(mine, trip["arc"]), self._behind(theirs, waiter["arc"])
                    for i, j in zip(
                        *self._parts_tree(theirs).query([q for _, q in mine.parts], predicate="intersects"),
                        strict=False,
                    ):
                        if not ((a_behind is not None and a_behind[i]) or (b_behind is not None and b_behind[j])):
                            return y["stop"]
        del self.yields[holder]
        self.journal.append(dict(kind="traffic_yield_ended", vehicle=holder, waiter=y["waiter"]))
        return None

    def _refusal(self, vehicle, run, arc, *, committed=False):
        """None when `run` may be reserved now, else the reason (see the module docstring). A `committed` run (one
        the car cannot stop before) is checked for conflict, room and parking only."""
        trip = self.trips[vehicle]
        route = self.course(trip["evidence"]).route
        zone = None
        yields = []
        for other in sorted(self.trips):
            if other == vehicle:
                continue
            if self._occupies(other, run):
                return "conflict"
            held_arc = self.trips[other]["arc"]
            if any(self._conflicts(run, held, arc, held_arc) for held in self.trips[other]["held"]):
                if committed:
                    return "conflict"
                zone = self._zone(vehicle, run, arc) if zone is None else zone
                stop = self._yield_point(other, zone, run.s1 - arc)
                if stop is None:
                    return "conflict"
                yields.append((other, stop))
        if not committed:
            # Front: another car ahead of this one on its route, up to the end of the run, that will wait for a
            # reservation there. Before the run it reserves first; in the run (on this car's flow) it would hold this
            # car in the run while it waits for a run of its own, which may meet this one.
            if run.s1 > arc + 1e-6:
                ahead = LineString(_slice(route, arc, run.s1)["points"])

                def yaw(s, a=arc):
                    return float(np.interp(a + s, route["arc"], route["yaw"]))

                for other in sorted(self.trips):
                    if other == vehicle:
                        continue
                    pose = self._pose_of(other)
                    s = None if pose is None else self._on(ahead, yaw, pose)
                    if s is None or s <= 0.1:
                        continue
                    wait = self._wait_pose(other)
                    if wait is not None and self._on(ahead, yaw, wait) is not None:
                        return "front"
            if run.room is not None:
                a, b, capacity, line = run.room

                def yaw(s, a=a):
                    return float(np.interp(a + s, route["arc"], route["yaw"]))

                for other, w in sorted(self.waiting.items()):
                    if other == vehicle or other not in self.trips or w["run"] is None:
                        continue
                    t = self.trips[other]
                    wait = _pose(self.course(t["evidence"]).route, max(w["run"].stop, t["arc"]))
                    if self._on(line, yaw, wait) is not None and self._conflicts(run, w["run"]):
                        return "front"
            mine = self.waiting.get(vehicle)
            order = mine["order"] if mine is not None else math.inf
            for other, w in sorted(self.waiting.items()):
                if other == vehicle or w["order"] >= order or other not in self.trips or w["run"] is None:
                    continue
                if w["reason"] == "room":
                    by = w.get("room_by", ())
                    if (
                        not by
                        or vehicle in by
                        or any(
                            o not in self.trips
                            or o in self.waiting
                            or o in self.yields
                            or self.trips[o]["speed"] < MOVING_MPS
                            for o in by
                        )
                    ):
                        continue
                elif w["reason"] != "conflict":
                    continue
                if self._conflicts(run, w["run"]) and not self._occupies(vehicle, w["run"]):
                    return "order"
        if run.room is not None:
            a, b, capacity, line = run.room

            def yaw(s, a=a):
                return float(np.interp(a + s, route["arc"], route["yaw"]))

            by = []
            for other in sorted(self.trips):
                if other == vehicle:
                    continue
                held = self.trips[other]["held"]
                if held and held[-1].exit_pose is not None and self._on(line, yaw, held[-1].exit_pose) is not None:
                    by.append(other)
                    continue
                pose = self._pose_of(other)
                if pose is not None and self._on(line, yaw, pose) is not None:
                    by.append(other)
            if len(by) >= capacity:
                self._room_by = tuple(by)
                return "room"
        else:
            target = self.routes.graph.nodes[self.routes.plans[trip["evidence"]]["target"]]["pose"]
            parked = footprint(dict(xy=target[:2], yaw=target[2]), self.routes.size)
            for other in sorted(self.trips):
                if other == vehicle:
                    continue
                t = self.trips[other]
                for end, corridor in self.course(t["evidence"]).corridors:
                    if end > t["arc"] and corridor is not None and corridor.intersects(parked):
                        return "parking"
        self._yields = yields
        return None

    def _try(self, vehicle, run, arc):
        trip = self.trips[vehicle]
        self._yields, self._room_by = [], ()
        reason = self._refusal(vehicle, run, arc)
        if reason is None:
            trip["held"] = sorted(trip["held"] + [run], key=lambda r: r.s0)
            self.waiting.pop(vehicle, None)
            self.stats["acquired"] += 1
            self._yielded[vehicle] = 0
            for other, stop in self._yields:
                self.yields[other] = dict(waiter=vehicle, stop=stop)
                self._yielded[other] = self._yielded.get(other, 0) + 1
                self.stats["yields"] = self.stats.get("yields", 0) + 1
            self.journal.append(
                dict(
                    kind="traffic_reserved",
                    vehicle=vehicle,
                    arc_m=round(arc, 1),
                    run=_span(run),
                    yielding=[dict(vehicle=o, stop_m=round(q, 1)) for o, q in self._yields],
                )
            )
            self._yields = []
            return True
        w = self.waiting.get(vehicle)
        changed = w is None or w["run"] is not run or w["reason"] != reason
        if w is None or w["run"] is None or abs(w["run"].stop - run.stop) > 1e-6:
            self._order += 1
            w = self.waiting[vehicle] = dict(order=self._order, run=run, reason=reason)
        w.update(run=run, reason=reason, room_by=self._room_by if reason == "room" else ())
        self.stats["refused_" + reason] += 1
        if changed:
            row = self.describe(vehicle)
            self.journal.append(
                dict(
                    kind="traffic_wait",
                    vehicle=vehicle,
                    **{k: row[k] for k in ("arc_m", "held")},
                    **row.get("waiting", {}),
                )
            )
        return False

    def depart(self, vehicle, evidence):
        """Admit a departure from the berth (see the module docstring)."""
        c = self.course(evidence)
        start = swept(_slice(c.route, 0.0, self.length + GAP_M), self.routes.size)
        if any(
            other != vehicle and body is not None and start.intersects(body)
            for other, body in sorted(self.routes.bodies.items())
            if other in self.trips
        ):
            self.stats["refused_departure"] = self.stats.get("refused_departure", 0) + 1
            return False
        saved = {v: dict(t) for v, t in self.trips.items()}
        saved_waiting = {v: dict(w) for v, w in self.waiting.items()}
        self.trips[vehicle] = dict(evidence=evidence, arc=0.0, speed=0.0, unsafe=None, runs=[], held=[])

        def undo():
            self.trips.clear()
            self.trips.update(saved)
            mine = self.waiting.get(vehicle)
            self.waiting.clear()
            self.waiting.update(saved_waiting)
            if mine is not None:
                self.waiting[vehicle] = mine

        for other in sorted(self.trips):
            if other != vehicle and not self._refresh(other, check=True):
                undo()
                self.stats["refused_committed"] += 1
                return False
        self._refresh(vehicle)
        runs = self.trips[vehicle]["runs"]
        if runs and runs[0].stop <= self.length + GAP_M:
            if not self._try(vehicle, runs[0], 0.0):
                undo()
                return False
        self.waiting.pop(vehicle, None)
        return True

    def revise(self, vehicle, evidence, check=None):
        """Put a car in a trip on the route of `evidence`, which follows its present route beyond the runs it holds
        (DrivingRoutes.revise). The trips are recomputed as for a departure: False, nothing changed, when another car
        or this one would be granted a run it may not reserve, or when `check()` (called on the recomputed trips)
        refuses the new route."""
        trip = self.trips[vehicle]
        old = trip["evidence"]
        saved = {v: dict(t) for v, t in self.trips.items()}
        saved_waiting = {v: dict(w) for v, w in self.waiting.items()}
        trip.update(evidence=evidence, unsafe=None)
        for other in sorted(self.trips, key=lambda v: (v == vehicle, v)):
            if not self._refresh(other, check=True):
                self.trips.clear()
                self.trips.update(saved)
                self.waiting.clear()
                self.waiting.update(saved_waiting)
                self._forget(evidence)
                self.stats["refused_revision"] = self.stats.get("refused_revision", 0) + 1
                return False
        if check is not None and not check():
            self.trips.clear()
            self.trips.update(saved)
            self.waiting.clear()
            self.waiting.update(saved_waiting)
            self._forget(evidence)
            self.stats["revision_not_better"] = self.stats.get("revision_not_better", 0) + 1
            return False
        self._forget(old)
        self.stats["revised"] = self.stats.get("revised", 0) + 1
        return True

    def _clear_parking(self, vehicle):
        """Take the other cars whose remaining route meets the body this car will park at its destination round it
        (DrivingRoutes.revise); whether every one of them now clears it."""
        revise = getattr(self.routes, "revise", None)
        if revise is None:
            return False
        target = self.routes.graph.nodes[self.routes.plans[self.trips[vehicle]["evidence"]]["target"]]["pose"]
        parked = footprint(dict(xy=target[:2], yaw=target[2]), self.routes.size)
        return all(
            revise(other, parked, reason="parking:" + vehicle) for other in sorted(self.trips) if other != vehicle
        )

    def prepare_revisions(self, t_s=None):
        """Change routes where the traffic asks for it, once per tick on the accepted state, before any car looks
        ahead or moves: a change of route rebuilds every layout, which must not happen while limit goes through one or
        after another car has looked ahead by it. Nothing is reserved here; a car whose way stays obstructed waits for
        it in limit. The changes made for one car stand when another cannot change its way.

        For each car whose next run within reach (as in limit) waits for parking only, the cars whose remaining route
        meets its parked body are taken round it (_clear_parking). Then, given the tick time `t_s`, each car expected
        to wait long for its next run is taken round what keeps it from that run (_relieve, the module docstring)."""
        for vehicle, trip in sorted(self.trips.items()):
            execution = self.routes.executions.get(vehicle)
            if execution is not None:
                trip.update(arc=float(execution.arc_m), speed=float(execution.speed_mps))
        braking = self.routes.limits.braking_mps2
        for vehicle in sorted(self.trips):
            self._refresh(vehicle)
            trip = self.trips[vehicle]
            arc, speed = trip["arc"], trip["speed"]
            reach = speed * speed / (2.0 * braking) + speed + 6.0
            for run in trip["runs"]:
                if (run.s1 <= arc + 1e-6 and run.room is not None) or run in trip["held"]:
                    continue
                if run.stop - arc > reach:
                    break
                reason = self._refusal(vehicle, run, arc)
                if reason == "parking":
                    self._clear_parking(vehicle)
                if reason is not None:
                    break  # the layout gone through is stale after a change of route
        self.now = t_s
        if t_s is not None:
            for vehicle in sorted(self.trips):
                self._relieve(vehicle, braking)

    def _obstruction(self, vehicle, run, arc):
        """What keeps a car at `arc` from `run`: (seconds, cars, polygons) - the time the cars keeping it are expected
        to need to leave it (infinite for a car that waits for a reservation, yields or stands), those cars, and where
        they keep it: the areas where the parts of the run meet the parts of their runs, and the bodies of the cars
        standing in it or ahead (see the module docstring). Only those areas are gone round: the whole parts of the run
        would reach back over the stretch before it, where the car may still change its way."""
        parts = [q for _, q in run.parts]
        keep, cars, polygons = 0.0, [], []
        for other in sorted(self.trips):
            if other == vehicle:
                continue
            t = self.trips[other]
            stuck = (self.waiting.get(other) or {}).get("run") is not None or other in self.yields
            speed = max(t["speed"], CONGESTION_MIN_MPS)
            for held in t["held"]:
                pairs = list(self._meeting_parts(run, held, arc, t["arc"]))
                if not pairs:
                    continue
                last = max(held.ends[j] if held.ends else held.s1 for _, j in pairs)
                keep = max(keep, math.inf if stuck else max(last + self.length - t["arc"], 0.0) / speed)
                cars.append(other)
                polygons.extend(parts[i].intersection(held.parts[j][1]) for i, j in pairs)
            if self._occupies(other, run):
                keep = max(keep, math.inf if stuck or t["speed"] < MOVING_MPS else 2.0 * self.length / speed)
                cars.append(other)
                polygons.append(self.routes.bodies[other])
        if run.s1 > arc + 1e-6:
            route = self.course(self.trips[vehicle]["evidence"]).route
            ahead = LineString(_slice(route, arc, run.s1)["points"])

            def yaw(s, a=arc):
                return float(np.interp(a + s, route["arc"], route["yaw"]))

            for other in sorted(self.trips):
                if other == vehicle or other in cars:
                    continue
                pose = self._pose_of(other)
                s = None if pose is None else self._on(ahead, yaw, pose)
                if s is None or s <= 0.1:
                    continue
                t = self.trips[other]
                if (
                    (self.waiting.get(other) or {}).get("run") is not None
                    or other in self.yields
                    or t["speed"] < MOVING_MPS
                ):
                    keep = math.inf
                    cars.append(other)
                    polygons.append(self.routes.bodies[other])
        return keep, sorted(set(cars)), polygons

    def _relieve(self, vehicle, braking):
        """Take a car round what keeps it from its next run up to CONGESTION_AHEAD_M beyond its reach, when it is
        expected to wait for that run at least CONGESTION_WAIT_S (the module docstring); whether its route changed."""

        def expected():
            # (wait for the next run within reach, the cars keeping it, where), or None when no run is in reach
            self._refresh(vehicle)
            trip = self.trips[vehicle]
            arc, speed = trip["arc"], trip["speed"]
            run = next(
                (r for r in trip["runs"] if not ((r.s1 <= arc + 1e-6 and r.room is not None) or r in trip["held"])),
                None,
            )
            if run is None or run.stop - arc > speed * speed / (2.0 * braking) + speed + 6.0 + CONGESTION_AHEAD_M:
                return None
            keep, cars, polygons = self._obstruction(vehicle, run, arc)
            return keep - max(run.stop - arc, 0.0) / max(speed, CONGESTION_MIN_MPS), cars, polygons, run

        now = expected()
        if now is None:
            return False
        wait, cars, polygons, run = now
        arc = self.trips[vehicle]["arc"]
        if not polygons or wait < CONGESTION_WAIT_S:
            return False
        key = (round(run.stop, 1), tuple(cars))
        last = self._relieved.get(vehicle)
        if last is not None and last[0] == key and self.now - last[1] < CONGESTION_RETRY_S:
            return False
        self._relieved[vehicle] = (key, self.now)
        revise = getattr(self.routes, "revise", None)
        if revise is None:
            return False
        self.stats["congestion_searches"] = self.stats.get("congestion_searches", 0) + 1
        old = self.trips[vehicle]["evidence"]
        detour = min(CONGESTION_DETOUR_M, wait * self.routes.limits.forward_mps)

        def better():
            # The way round must shorten the expected wait (cq3p v3, vehicle-050 was sent round vehicle-053's run into
            # the dead end vehicle-053 was driving to and stood 60 s there).
            after = expected()
            return after is None or after[0] < wait - CONGESTION_WAIT_S / 2.0

        revise(
            vehicle, shapely.union_all(polygons), reason="congestion:" + ",".join(cars), detour_m=detour, check=better
        )
        if self.trips[vehicle]["evidence"] == old:
            return False
        self.stats["congestion_revised"] = self.stats.get("congestion_revised", 0) + 1
        self.journal.append(
            dict(
                kind="traffic_congestion_reroute",
                vehicle=vehicle,
                arc_m=round(arc, 1),
                run=_span(run),
                cars=cars,
                wait_s=None if math.isinf(wait) else round(wait, 1),
            )
        )
        return True

    def _forget(self, evidence):
        for key in [k for k in self._meet if evidence in k]:
            del self._meet[key]
        for key in [k for k in self._runs_memo if k[0] == evidence]:
            del self._runs_memo[key]
        self.courses.pop(evidence, None)

    def _prune(self):
        """Keep the geometry of the routes in a trip and of the KEEP_COURSES most recent others."""
        live = {t["evidence"] for t in self.trips.values()}
        idle = [e for e in self.courses if e not in live]
        for evidence in idle[: max(len(idle) - KEEP_COURSES + 1, 0)]:
            self._forget(evidence)

    def cancel(self, vehicle):
        trip = self.trips.pop(vehicle, None)
        self.waiting.pop(vehicle, None)
        self.yields.pop(vehicle, None)
        self._yielded.pop(vehicle, None)
        self._relieved.pop(vehicle, None)
        if trip is not None:
            self._forget(trip["evidence"])

    def limit(self, vehicle, arc, speed, braking):
        """The arc the car must stop before for its next run, or None; releases the runs it has left and reserves
        the next one when the car comes within its stopping distance and a margin of it."""
        trip = self.trips[vehicle]
        trip["arc"], trip["speed"] = float(arc), float(speed)
        self._refresh(vehicle)
        left = [r for r in trip["held"] if arc >= r.s1 - 1e-6 and r.room is not None]
        if left:
            trip["held"] = [r for r in trip["held"] if r not in left]
            self.stats["released"] += len(left)
        reach = speed * speed / (2.0 * braking) + speed + 6.0
        stop = self._yield_stop(vehicle)
        limit = None if stop is None else max(stop, arc)
        for run in trip["runs"]:
            if (run.s1 <= arc + 1e-6 and run.room is not None) or run in trip["held"]:
                continue
            if run.stop - arc > reach:
                break
            if not self._try(vehicle, run, arc):
                return max(run.stop, arc) if limit is None else min(limit, max(run.stop, arc))
        return limit

    def arrive(self, vehicle):
        self.cancel(vehicle)

    def describe(self, vehicle):
        """For a diagnostic: a car's arc, the runs it holds and, when it waits, the run, the reason and the cars
        concerned (holding a conflicting run, standing in it, waiting longer for a conflicting run)."""
        trip = self.trips.get(vehicle)
        if trip is None:
            return None

        def span(r):
            return [round(r.stop, 1), round(r.s0, 1), round(r.s1, 1)]

        out = dict(
            arc_m=round(trip["arc"], 1),
            speed_mps=round(trip["speed"], 2),
            runs=len(trip["runs"]),
            held=[span(r) for r in trip["held"]],
        )
        if self.yields.get(vehicle) is not None:
            out["yielding"] = dict(to=self.yields[vehicle]["waiter"], stop_m=round(self.yields[vehicle]["stop"], 1))
        w = self.waiting.get(vehicle)
        if w is not None and w["run"] is not None:
            run = w["run"]
            others = [o for o in sorted(self.trips) if o != vehicle]
            out["waiting"] = dict(
                reason=w["reason"],
                run=span(run),
                holding=[
                    o
                    for o in others
                    if any(self._conflicts(run, h, trip["arc"], self.trips[o]["arc"]) for h in self.trips[o]["held"])
                ],
                standing=[o for o in others if self._occupies(o, run)],
                in_room=list(w.get("room_by", ())),
                earlier=[
                    o
                    for o, x in sorted(self.waiting.items())
                    if o in others
                    and x["order"] < w["order"]
                    and x["run"] is not None
                    and self._conflicts(run, x["run"])
                ],
            )
        return out
