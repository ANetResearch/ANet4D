"""The environment-independent tick of the life runtime.

LifeEngine advances admitted bodies (people, vehicles, street objects) by one 0.1 s tick:

1. decisions: every person's agenda (LifeAgenda: the Director's walk and car journeys) is ticked with the
   measured pose and speed;
2. boarding and alighting at the car side: a person standing at the car's access point beside the parked car
   boards (its body is then contained in the vehicle and inactive); at the destination berth it reappears at the
   access point only when that place is free and supported;
3. driving: a driver's car advances through DriverExecution (trip admission and corridor reservation in
   DrivingRoutes) with LiveDrive and the live stop lookahead (awr.life.motion.vehicles);
4. walking: static routes (LifeRouter), standoffs, congestion rerouting (awr.life.motion.reroute), courtesy (a person
   at a stay steps aside for a walker held up by it, awr.life.motion.courtesy), path following and the crowd step, in
   which walkers give way to the driving cars. Walkers see a driving car along its route at its speed; a car that
   has stood ASK_AFTER_S in its trip they see setting off at WAITING_CAR_MPS (`car_intent_mps`), so that in a dense
   crowd they keep out of the way it is about to take rather than walk into it while it stands (cq3p v5,
   vehicle-042: walkers stepped into the way of the standing car and held it 60 s). A walker already in that way
   (its pose box within the car's half width and the crowd margin of the route ahead, `in_way`) sees the standing
   car where it stands, and leaves the way as it would beside any standing body: predicted setting off, the car
   would meet every step it can take, and it froze in front of the car (cq3p-s3 v1, person-064 before
   vehicle-053, which then waited 60 s). So does the person whose body sets the car's stop (its live lookahead,
   awr.life.motion.vehicles.stop_blocker): a walker the car waits for does not wait for the car (cq3p-d8-s4 v1,
   person-022, 0.75 m beside the front of vehicle-042 and inside the room the car keeps from people, waited for
   the car to set off while the car waited for that walker). A walker's route, planned or replanned, goes round the cars standing in
   their trips and behind them: their bodies and their ways ahead to CAR_AHEAD_M beyond the front cost
   (1+JAM_FACTOR) times their static cost (`car_cells`), as standing people do in congestion rerouting; the static
   snapshot holds only parked cars, and a route through a standing car held a walker against its front corner and
   the car 60 s (cq3p-d8-s3 v2, person-009 before vehicle-043). A walker whose path runs through them within its
   next JAM_AHEAD_M plans its way anew at once, at most every ROUND_RETRY_S (`_through_cars`), instead of waiting
   to be held up (congestion rerouting waits AFTER_S for that, and stepping aside restarts the wait).

The engine consumes an environment through its objects only - surface and support, walking network and router,
Director places and activities, driving graph and berths - and contains no map, place or actor specific rule.
It never writes a pose other than through these motion rules.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
from shapely.geometry import Point

from awr.life.civil import body as bodies
from awr.life.civil.activity.driving import Body
from awr.life.civil.collision import collision_pairs
from awr.life.civil.curb_allocation import footprint

from . import kernels as K
from .courtesy import ASK_AFTER_S, CAR_AHEAD_M, Courtesy
from .crowd import DT, TURN_RATE_RPS, Crowd, PathStore
from .mixed_safety import free_door
from .reroute import AHEAD_M as JAM_AHEAD_M
from .reroute import FOOTPRINT_M, JAM_FACTOR, Reroutes, _samples
from .standoff import Standoffs
from .vehicles import (
    LOOKAHEAD_S,
    LOOKAHEAD_STEP_M,
    PERSON_ROOM_M,
    PERSON_ROOM_S,
    VEHICLE_ROOM_M,
    LiveDrive,
    predict_route,
    stop_arc,
    stop_blocker,
)

ACCESS_RADIUS_M = 0.3  # board / alight: person at the access point, car at its berth
DOOR_REACH_M = 1.5  # the access point lies within this distance of the parked car's footprint
ARRIVAL_M = 0.05  # arrival radius of the agendas (a walk step or a stay is reached within it)
RECONNECT_AFTER_S = 2.0  # a held-up walker whose way to its next path point is shut by static geometry plans again
RECONNECT_RETRY_S = 5.0
WAITING_CAR_MPS = 1.0  # walkers expect a car that has stood ASK_AFTER_S in its trip to set off at this speed
ROUND_RETRY_S = 10.0  # a walker whose path runs through a standing car plans anew at most this often
STANDING_CAR_MPS = 0.05  # a car about to set off below this speed is seen standing by the walkers in its way


def car_intent_mps(speed, waited_ticks):
    """The speed at which walkers predict a car in its trip along its route (module docstring)."""
    return max(speed, WAITING_CAR_MPS) if waited_ticks >= round(ASK_AFTER_S / DT) else speed


class LifeEngine:
    def __init__(
        self,
        *,
        ids,
        kinds,
        ext,
        xy,
        yaw,
        surface,
        walking,
        router,
        agendas,
        vmax,
        occupancy,
        vehicle_ids=None,
        sizes=None,
        driving=None,
        driver_execution=None,
        events=None,
        route_cache=None,
        transfer_wait_s=60.0,
        walk_space=None,
    ):
        self.ids = list(ids)
        self.n = len(self.ids)
        self.kinds = np.asarray(kinds)
        self.ext = np.ascontiguousarray(ext, np.float64)
        self.xy = np.array(xy, float)
        self.yaw = np.array(yaw, float)
        self.speed = np.zeros(self.n)
        self.crowd = Crowd(self.ext, self.kinds, surface, walking)
        self.standoffs = Standoffs(self.crowd)
        self.courtesy = Courtesy(self.crowd)
        self.last_motion = (
            None  # the last crowd step's inputs, prior state, result and cars (failure_evidence.save_crowd)
        )
        self.router = router
        if not hasattr(router, "avoid_cells"):
            raise ValueError("LifeEngine requires a router with congestion costs (awr.life.motion.route.LifeRouter)")
        self.reroutes = Reroutes(router, self.ext)
        self.agendas = dict(agendas)  # body index -> agenda
        self.vmax = np.asarray(vmax, float)
        self.occupancy = occupancy
        self.index = {a: i for i, a in enumerate(self.ids)}
        self.vehicle_ids = dict(vehicle_ids or {})  # body index -> vehicle id (Director / DrivingRoutes)
        self.vehicle_index = {v: i for i, v in self.vehicle_ids.items()}
        self.sizes = sizes
        self.driving = driving
        self.drivers = driver_execution
        # Every rigid scene body (parked and driven cars, street objects) is in the driving occupancy.
        self.rigid_ids = {i: self.vehicle_ids.get(i, self.ids[i]) for i in range(self.n) if self.kinds[i] != "person"}
        if self.driving is not None:
            if set(self.rigid_ids.values()) != self.driving.roster:
                raise ValueError("Driving occupancy must include every rigid scene body")
            if self.sizes is None or any(i not in self.sizes for i in self.rigid_ids):
                raise ValueError("Driving occupancy requires measured sizes for every rigid body")
        self.departed_drives = {}  # vehicle id -> route evidence, from observed movement (not permission to depart)
        if not math.isfinite(transfer_wait_s) or transfer_wait_s <= 0:
            raise ValueError("transfer_wait_s must be finite and positive")
        self.transfer_wait_s = float(transfer_wait_s)
        self.transfer_wait_since = {}
        self.last_time = None
        self.failed = None
        self.events = events if events is not None else []
        self.route_cache = route_cache if route_cache is not None else {}
        self.riding = np.zeros(self.n, bool)
        self.container = np.full(self.n, -1, int)
        self.active = np.ones(self.n, bool)
        self.paths = PathStore(self.n)
        self.route_retry = {}
        self.reconnect_at = {}
        self.standing_ways = []  # (car, poses (m, 2), yaws (m,)) of the cars standing in their trips this tick
        self.round_at = {}  # walker: time from which it may plan round standing cars again (_through_cars)
        self._car_cells = None  # (router, cells): car_cells of this tick
        self.stall_run = np.zeros(self.n, int)
        self.car_wait = np.zeros(self.n, int)  # ticks a car on a trip has stood (awr.life.motion.courtesy)
        self.delayed = set()
        self.alight_wait = set()
        self.tangent = self.yaw.copy()
        self.stats = dict(
            route_plans=0,
            route_cache_hits=0,
            route_failures=0,
            decisions=0,
            boardings=0,
            alightings=0,
            drive_ticks=0,
            drive_waits=0,
            drives_completed=0,
        )
        # Resident cars standing parked are static walking geometry while they stand (awr.life.motion.parked_space).
        self.walk_space = walk_space
        self._update_walk_space()
        if self.walk_space is not None and self.driving is not None:
            self.walk_space.activity.director.parking_walk = self._parking_walk

    def _parking_walk(self, vehicle, berth, start, goal, person):
        """The Director's proof of the walk from a berth with the car parked there (ParkedSpace.plan_after_parking;
        adapted from codex 454fd00): a Route, or None."""
        from ..civil.activity.director import Route

        if not self.walk_space.activity.walk_allowed(start, goal):
            return None
        car = self.vehicle_index[vehicle]
        pose = self.driving.graph.nodes[berth]["pose"]
        route = self.walk_space.plan_after_parking(car, (*pose, *self.sizes[car]), start, goal)
        if route is None:
            return None
        points = np.asarray(route["points"], float)
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        speed = float(self.walk_space.activity.life.get("planning_speed_mps", 1.05))
        return Route(
            length,
            length / speed,
            "measured-parked-walk:" + hashlib.sha256(points.tobytes()).hexdigest(),
            points=points,
        )

    # -- helpers -------------------------------------------------------------------------------------------------
    def _activity_walk(self, i, goal):
        """The walking geometry the backend proved for walker i's present walk step (Route.points), when every
        segment of it, and a local connection of at most 1.5 m from the walker to its start, is clear now; else None.
        (codex/autonomous-mixed-life ccf9ed2: a continuous path proved beside a parked car may have no connected cells
        in the coarse node raster, and planning it anew there strands the walker.) Never moves the walker."""
        director = getattr(self.agendas.get(i), "director", None)
        intent = None if director is None else director.active.get(self.ids[i])
        if intent is None:
            return None
        step = intent.step
        proof = step.route
        if (
            step.kind != "walk"
            or step.entity != self.ids[i]
            or proof is None
            or not proof.points
            or math.dist(step.xy, goal) > 1e-6
            or math.dist(proof.points[-1], goal) > 1e-6
            or math.dist(self.xy[i], proof.points[0]) > 1.5
        ):
            return None
        points = np.vstack([self.xy[i], np.asarray(proof.points, float)])
        points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-7]]
        network, free = self.router.network, self.router.free
        if len(points) < 2 or not all(
            network.clear_segment(a, b, free) for a, b in zip(points[:-1], points[1:], strict=False)
        ):
            self.stats["walk_proofs_invalidated"] = self.stats.get("walk_proofs_invalidated", 0) + 1
            return None
        self.stats["walk_proofs_reused"] = self.stats.get("walk_proofs_reused", 0) + 1
        return dict(points=points, evidence=proof.evidence)

    def car_cells(self):
        """Router cells within reroute.FOOTPRINT_M of the cars standing in their trips this tick, at their poses
        and along their routes ahead to CAR_AHEAD_M beyond the front (module docstring)."""
        cached = getattr(self, "_car_cells", None)
        if cached is None or cached[0] is not self.router:
            cells = np.empty(0, np.int64)
            if getattr(self, "standing_ways", None) and hasattr(self.router, "avoid_bodies"):
                pts = np.concatenate([p for _, p, _ in self.standing_ways])
                yaws = np.concatenate([q for _, _, q in self.standing_ways])
                ext = np.concatenate([np.repeat(self.ext[v][None], len(p), 0) for v, p, _ in self.standing_ways])
                cells = np.unique(self.router.avoid_bodies(pts, yaws, np.zeros(len(pts)), ext, FOOTPRINT_M))
            self._car_cells = (self.router, cells)
        return self._car_cells[1]

    def _through_cars(self, i, avoided):
        """Whether walker i's path runs through the cells flagged in `avoided` (car_cells) within its next
        JAM_AHEAD_M."""
        path = self.paths.path(i)
        return self.reroutes._exposure(_samples(path, path.s, JAM_AHEAD_M), avoided) > 0.0

    def _round_cars(self, start, goal, rr, cells):
        """`rr` (a route from start to goal or None), or a route round the cells of standing cars when `rr`
        passes through them and such a route exists."""
        if not len(cells):
            return rr
        if rr is not None:
            avoided = np.zeros(len(self.router.cost), bool)
            avoided[cells] = True
            pts = np.asarray(rr["points"], float)
            if sum(self.router.avoided_m(a, b, avoided) for a, b in zip(pts[:-1], pts[1:], strict=False)) <= 0.0:
                return rr
        self.stats["round_car_plans"] = self.stats.get("round_car_plans", 0) + 1
        alt = self.router.plan(start, goal, avoid=(cells, JAM_FACTOR))
        return rr if alt is None else alt

    def route(self, i, goal, *, t=None):
        key = (tuple(self.xy[i]), tuple(goal))
        start = np.asarray(self.xy[i])
        cells = self.car_cells()
        rr = self.route_cache.get(key)
        if rr is not None:
            self.stats["route_cache_hits"] += 1
            rr = self._round_cars(start, np.asarray(goal), rr, cells)
        else:
            rr = self._activity_walk(i, goal)
            if rr is None:
                self.stats["route_plans"] += 1
                rr = self.router.plan(start, np.asarray(goal))
            rr = self._round_cars(start, np.asarray(goal), rr, cells)
        if rr is None:
            return None
        points = np.vstack([self.xy[i], np.asarray(rr["points"])[1:]])
        if t is not None:
            self.events.append(
                dict(
                    kind="walking_route_assigned",
                    t_s=round(t, 6),
                    actors=[self.ids[i]],
                    activity_route_evidence=rr.get("evidence"),
                    start=points[0].tolist(),
                    goal=points[-1].tolist(),
                )
            )
        return points

    def target(self, i):
        """Where walker i is going: the agenda's current walk target (LifeAgenda), else its visit's position."""
        ag = self.agendas[i]
        return ag.target_xy if hasattr(ag, "target_xy") else ag.destination.xy

    def ended_short(self, i, xy):
        """Whether walker i has finished its path without being within the arrival radius of its target: the path
        ended away from it (a courtesy walk back to a spot the stay had been reached from, a target that moved)."""
        return (
            self.paths.remaining(i) < 0.05
            and float(np.linalg.norm(np.asarray(xy[i], float) - np.asarray(self.target(i), float))) > ARRIVAL_M
        )

    def _update_walk_space(self, t=None):
        """Report the resident cars standing parked now (outside an active trip) to the walking space; on a change,
        walk on the new static router and plan again every remaining path that a newly placed body obstructs (its
        exact body test, ParkedSpace.blocks_path). Departures only open space; motion safety stays with the
        controller on every tick."""
        if self.walk_space is None or self.driving is None:
            return
        parked = {
            v: (float(self.xy[v, 0]), float(self.xy[v, 1]), float(self.yaw[v]), *(float(c) for c in self.sizes[v]))
            for v, vid in self.vehicle_ids.items()
            if vid not in self.driving.active
        }
        changed = self.walk_space.update(parked)
        if changed is None:
            return
        self.router, _ = changed
        self.reroutes.router = self.router
        paths = self.paths
        for i in range(self.n):
            if i in paths:
                f, c, k = int(paths.first[i]), int(paths.count[i]), int(paths.k[i])
                if self.walk_space.blocks_path(np.vstack([self.xy[i], paths.pts[f + k + 1 : f + c]])):
                    paths.pop(i)
                    self.stats["paths_closed_by_parking"] = self.stats.get("paths_closed_by_parking", 0) + 1
                    if t is not None:
                        self.events.append(
                            dict(
                                kind="walking_path_closed_by_parking",
                                t_s=round(t, 6),
                                actors=[self.ids[i]],
                                xy=self.xy[i].tolist(),
                                parked_actors=[self.ids[v] for v in self.walk_space.changed_bodies],
                            )
                        )

    def connected(self, i, xy):
        """Whether walker i's straight way from where it stands to its next path point is clear of static geometry
        (the router's grid with its exact bodies, WalkNetwork.clear_segment). Local avoidance can leave a walker
        beside its path where that way is shut, e.g. behind the corner of a parked car."""
        paths = self.paths
        f, c, k = int(paths.first[i]), int(paths.count[i]), int(paths.k[i])
        nxt = paths.pts[f + min(k + 1, c - 1)]
        return bool(self.router.network.clear_segment(np.asarray(xy[i], float), nxt, self.router.free))

    def vehicle_pose(self, v):
        return np.r_[self.xy[v], self.yaw[v]]

    def _sync_driving(self):
        if self.driving is None:
            return
        self.driving.update_occupancy(
            {vid: Body(tuple(self.vehicle_pose(v)), tuple(self.sizes[v])) for v, vid in self.rigid_ids.items()}
        )

    def _transfer_ready(self, i, v, step):
        """Whether person i's own car v is parked, at rest, at the berth admitted for it, with the transfer step's
        access point beside it (the person does not board or reappear at a car that is elsewhere or still moving)."""
        director = self.agendas[i].director
        vehicle = director.vehicles[self.vehicle_ids[v]]
        berth = director.places[vehicle.berth]
        pose = self.driving.graph.nodes[vehicle.berth]["pose"]
        return (
            berth.access_xy is not None
            and math.dist(step.xy, berth.access_xy) <= 1e-6
            and step.partner_xy is not None
            and math.dist(step.partner_xy, berth.xy) <= 1e-6
            and self.driving._at(self.vehicle_pose(v), pose, abs(float(self.speed[v])))
            and footprint(dict(xy=self.xy[v], yaw=self.yaw[v]), self.sizes[v], margin_m=0.0).distance(
                Point(berth.access_xy)
            )
            <= DOOR_REACH_M
        )

    def _wait_transfer(self, i, t, reason):
        """A transfer that cannot happen yet waits, reported once, for at most transfer_wait_s."""
        since = self.transfer_wait_since.setdefault(i, t)
        if i not in self.alight_wait:
            self.agendas[i].emit("journey_delayed", t, place=self.agendas[i].destination.place, reason=reason)
            self.alight_wait.add(i)
        self.agendas[i].director.active[self.ids[i]].service_started = None
        if t - since > self.transfer_wait_s + 1e-7:
            raise RuntimeError(f"Transfer blocked for too long: {self.ids[i]} ({reason})")

    def _clear_transfer_wait(self, i):
        self.alight_wait.discard(i)
        self.transfer_wait_since.pop(i, None)

    def in_way(self, cars, xy, yaw, speed):
        """For Crowd.step (`seen_standing`): for every active person whose pose box lies within a car's half width
        plus the crowd margin of the route ahead of that car (`cars`: {car: route points ahead}), the car; -1
        elsewhere (module docstring)."""
        seen = np.full(len(xy), -1, np.int64)
        people = np.flatnonzero((self.kinds == "person") & self.active)
        for v, ahead in sorted(cars.items()):
            reach = float(np.abs(self.ext[v, 1, 2:]).max()) + self.crowd.margin
            for i in people:
                if seen[i] >= 0 or float(np.linalg.norm(ahead - xy[i], axis=1).min()) > reach + self.crowd.radius[i]:
                    continue
                bx, by, hl, hw = K.box(xy[i, 0], xy[i, 1], yaw[i], speed[i], self.ext[i])
                c, s = math.cos(yaw[i]), math.sin(yaw[i])
                dx, dy = ahead[:, 0] - bx, ahead[:, 1] - by
                out = np.hypot(
                    np.maximum(np.abs(dx * c + dy * s) - hl, 0.0), np.maximum(np.abs(-dx * s + dy * c) - hw, 0.0)
                )
                if float(out.min()) < reach:
                    seen[i] = v
        return seen

    def seen_standing(self, set_off, cars, xy, yaw, speed):
        """Crowd.step's `seen_standing`: for the cars about to set off (`set_off`: {car: route points ahead}), the
        people in their ways (`in_way`) and the person whose body sets each car's stop (`cars[car]['blocker']`) see
        the car where it stands (module docstring)."""
        seen = self.in_way(set_off, xy, yaw, speed)
        for v in sorted(set_off):
            b = cars[v].get("blocker", -1)
            if b >= 0 and self.kinds[b] == "person" and seen[b] < 0:
                seen[b] = v
        return seen

    def _owned(self, i):
        actor = self.ids[i]
        vid = self.agendas[i].director.actors[actor].vehicle
        return vid, self.vehicle_index[vid]

    # -- one tick ------------------------------------------------------------------------------------------------
    def step(self, t, wall=None):
        """Advance by one tick from time t. Returns (new_xy, new_yaw, new_speed, velocity, state, container, held):
        the velocity is the tick's displacement over DT from the pose at which the body was (re)placed this tick, as
        recorded on the tape; state 1 moving, 2 still, 3 contained in the vehicle `container`.
        A tick is a transaction: time advances by exactly DT, and a world whose tick failed (with agendas possibly
        partly advanced) refuses to continue."""
        if self.failed is not None:
            raise RuntimeError(f"LifeEngine already failed: {self.failed}")
        if not math.isfinite(t) or (self.last_time is not None and abs(t - self.last_time - DT) > 1e-7):
            raise ValueError("LifeEngine time must advance by exactly DT")
        try:
            result = self._step(t, wall)
        except Exception as error:
            self.failed = f"{type(error).__name__}: {error}"
            raise
        self.last_time = t
        return result

    def _step(self, t, wall=None):
        import time

        clock = time.perf_counter
        wall = wall if wall is not None else {}
        n = self.n
        xy, yaw, speed = self.xy, self.yaw, self.speed
        des_yaw, des_speed = yaw.copy(), np.zeros(n)
        goal_cap = np.full(n, np.inf)
        self.standing_ways = []
        self._car_cells = None
        a = clock()
        travel, boarding, driving, alighting, standing = [], [], [], [], []
        courteous = set(self.courtesy.walking)
        for i, ag in self.agendas.items():
            phase = ag.tick(t, xy[i], float(speed[i]), self.occupancy, arrival_radius=ARRIVAL_M)
            if phase == "next_destination":
                self.paths.pop(i)
                self.route_retry.pop(i, None)
                self.stats["decisions"] += 1
            if phase in ("travel", "next_destination"):
                travel.append(i)
            elif phase == "paused" and i in courteous:
                pass  # walking aside or back (awr.life.motion.courtesy)
            else:
                self.paths.pop(i)
                {"board": boarding, "drive": driving, "alight": alighting}.get(phase, []).append(i)
                if phase not in ("board", "drive", "alight", "paused") and self.active[i] and not self.riding[i]:
                    standing.append(i)
        b = clock()
        wall["decisions"] = wall.get("decisions", 0.0) + b - a

        # Boarding: the person stands at the access point beside its parked car.
        for i in boarding:
            vid, v = self._owned(i)
            director = self.agendas[i].director
            step = director.active[self.ids[i]].step
            if not self._transfer_ready(i, v, step) or not free_door(
                self.crowd, i, xy[i], yaw[i], xy, yaw, speed, self.active, carrier=v
            ):
                self._wait_transfer(i, t, "boarding_access_not_free_or_car_not_parked")
                continue
            self._clear_transfer_wait(i)
            if director.acknowledge(
                self.ids[i],
                t,
                tuple(xy[i]),
                float(speed[i]),
                vehicle_xy=tuple(xy[v]),
                vehicle_speed=abs(float(speed[v])),
                radius=ACCESS_RADIUS_M,
            ):
                # Record the observed door-side position before containment
                # moves the person's stored pose to the vehicle centre.
                transfer_xy = xy[i].tolist()
                self.riding[i] = True
                self.container[i] = v
                self.active[i] = False
                xy[i] = xy[v]
                speed[i] = 0.0
                self.events.append(
                    dict(
                        kind="board",
                        t_s=round(t, 2),
                        actors=[self.ids[i]],
                        vehicle=vid,
                        xy=transfer_xy,
                        vehicle_xy=xy[v].tolist(),
                        source="observed_containment_transfer",
                    )
                )
                self.stats["boardings"] += 1
        # Alighting: the person reappears at the access point when it is free; then stands there.
        for i in alighting:
            vid, v = self._owned(i)
            director = self.agendas[i].director
            step = director.active[self.ids[i]].step
            if self.riding[i]:
                spot = np.asarray(step.xy, float)
                face = math.atan2(spot[1] - xy[v, 1], spot[0] - xy[v, 0])
                if self.container[i] != v or director.vehicles[vid].occupant != self.ids[i]:
                    raise ValueError("Alighting requires the actual contained owner")
                if not self._transfer_ready(i, v, step) or not free_door(
                    self.crowd, i, spot, face, xy, yaw, speed, self.active, carrier=v
                ):
                    self._wait_transfer(i, t, "alighting_access_not_free_or_car_not_parked")
                    xy[i] = xy[v]
                    continue
                self._clear_transfer_wait(i)
                self.riding[i] = False
                self.container[i] = -1
                self.active[i] = True
                xy[i] = spot
                yaw[i] = face
                speed[i] = 0.0
                self.events.append(
                    dict(
                        kind="alight",
                        t_s=round(t, 2),
                        actors=[self.ids[i]],
                        vehicle=vid,
                        xy=xy[i].tolist(),
                        vehicle_xy=xy[v].tolist(),
                        source="observed_containment_transfer",
                    )
                )
                self.stats["alightings"] += 1
            director.acknowledge(
                self.ids[i],
                t,
                tuple(xy[i]),
                float(speed[i]),
                vehicle_xy=tuple(xy[v]),
                vehicle_speed=abs(float(speed[v])),
                radius=ACCESS_RADIUS_M,
            )
        c0 = clock()
        wall["transfers"] = wall.get("transfers", 0.0) + c0 - b

        # Driving: the drivers' cars advance first; walkers then give way to them. The recorded vehicle speed is
        # signed (negative in reverse gear); the driving backend observes its magnitude, the gear is its own state.
        driven_ids, driven_xy, driven_yaw, driven_speed, preds = [], [], [], [], []
        cars = {}  # the cars on a trip: route ahead and standing time (courtesy)
        set_off = {}  # standing cars that walkers see setting off: route ahead (in_way)
        finished = []
        drive_events = []  # depart / park, published after the tick's motion
        if driving:
            self._sync_driving()
            # Cars are taken round cars about to park across their way and round congestion, and their drives bound
            # to the new routes, before any car looks ahead or moves: every car then moves by one traffic layout in
            # this tick.
            self.driving.prepare_revisions(t)
            for i in driving:
                vid, v = self._owned(i)
                actor = self.ids[i]
                intent = self.agendas[i].director.active[actor]
                for old, new, splice, reason in self.drivers.follow_revision(actor):
                    if self.departed_drives.get(vid) == old:
                        self.departed_drives[vid] = new
                    drive_events.append(
                        dict(
                            kind="reroute",
                            t_s=round(t, 6),
                            actors=[vid],
                            driver=actor,
                            intent_id=intent.id,
                            purpose=intent.purpose,
                            place=intent.place,
                            berth=self.driving.plans[new]["target"],
                            evidence=new,
                            replaces=old,
                            splice_m=round(splice, 3),
                            reason=reason,
                            xy=xy[v].tolist(),
                            source="traffic_revision",
                        )
                    )
            mover = (self.kinds == "person") & self.active
            pred_people = K.predict_bodies(
                np.ascontiguousarray(xy[:, 0]),
                np.ascontiguousarray(xy[:, 1]),
                yaw,
                speed,
                self.crowd.aim,
                mover & (speed > 0.0),
                self.crowd.times,
                TURN_RATE_RPS,
            )
            for i in driving:
                vid, v = self._owned(i)
                actor = self.ids[i]
                intent = self.agendas[i].director.active[actor]
                evidence = intent.step.route.evidence
                execution = self.driving.executions.get(vid)
                if actor in self.drivers.bindings and execution is not None and execution.arrived:
                    if self.departed_drives.get(vid) != evidence:
                        raise RuntimeError("Physical arrival has no observed departure")
                    if self.drivers.finish(actor, t, tuple(xy[i]), vid, self.vehicle_pose(v), abs(float(speed[v]))):
                        self.stats["drives_completed"] += 1
                        finished.append(i)
                        target = self.driving.plans[evidence]["target"]
                        drive_events.append(
                            dict(
                                kind="park",
                                t_s=round(t, 6),
                                actors=[vid],
                                driver=actor,
                                intent_id=intent.id,
                                purpose=intent.purpose,
                                place=intent.place,
                                berth=target,
                                evidence=evidence,
                                xy=xy[v].tolist(),
                                source="observed_drive_execution",
                                legal=self.driving.graph.nodes[target].get("parking_legality", {}).get("legal"),
                            )
                        )
                        del self.departed_drives[vid]
                    continue
                if execution is None:
                    plan = self.driving.plans[evidence]
                    execution = self.driving.executions[vid] = LiveDrive(plan["schedule"].route, self.driving.limits)
                r = execution.route
                s0, v0 = execution.arc_m, execution.speed_mps
                if vid in self.driving.active:
                    front = s0 + float(np.abs(self.ext[v, 1, :2]).max())
                    ahead = [
                        np.column_stack(
                            [np.interp(s, r["arc"], r["points"][:, 0]), np.interp(s, r["arc"], r["points"][:, 1])]
                        )
                        for s in (
                            np.arange(s0, front + CAR_AHEAD_M + 1e-9, 0.25),
                            np.arange(s0, front + CAR_AHEAD_M + 4.0 + 1e-9, 0.25),
                        )
                    ]
                    self.car_wait[v] = self.car_wait[v] + 1 if abs(float(speed[v])) < 0.2 else 0
                    cars[v] = dict(ahead=ahead[0], longer=ahead[1], waited=int(self.car_wait[v]))
                    if self.car_wait[v]:
                        s = np.arange(s0, front + CAR_AHEAD_M + 1e-9, 1.0)
                        self.standing_ways.append(
                            (
                                v,
                                np.column_stack(
                                    [
                                        np.interp(s, r["arc"], r["points"][:, 0]),
                                        np.interp(s, r["arc"], r["points"][:, 1]),
                                    ]
                                ),
                                np.interp(s, r["arc"], r["yaw"]),
                            )
                        )
                length = v0 * v0 / (2.0 * self.driving.limits.braking_mps2) + v0 * LOOKAHEAD_S + 4.0
                ignore = np.zeros(n, bool)
                ignore[i] = True
                look = (
                    v,
                    r["points"],
                    r["yaw"],
                    r["arc"],
                    s0,
                    v0,
                    length,
                    self.ext,
                    np.ascontiguousarray(xy[:, 0]),
                    np.ascontiguousarray(xy[:, 1]),
                    yaw,
                    speed,
                    self.kinds == "person",
                    self.active,
                    ignore,
                    *pred_people,
                    self.crowd.times,
                    PERSON_ROOM_M,
                    PERSON_ROOM_S,
                    VEHICLE_ROOM_M,
                    LOOKAHEAD_STEP_M,
                )
                stop = stop_arc(*look)
                if v in cars and cars[v]["waited"] >= round(ASK_AFTER_S / DT):
                    cars[v]["blocker"] = int(stop_blocker(*look))  # asked to step aside (Courtesy.update)
                q = self.drivers.advance(
                    actor,
                    t,
                    tuple(xy[i]),
                    vid,
                    self.vehicle_pose(v),
                    abs(float(speed[v])),
                    DT,
                    stop_before_arc_m=None if math.isinf(stop) else stop,
                )
                if q is None:
                    self.stats["drive_waits"] += 1
                    if self.driving.executions.get(vid) is execution and vid not in self.driving.active:
                        del self.driving.executions[vid]  # not admitted yet: no execution state kept
                    continue
                self.stats["drive_ticks"] += 1
                driven_ids.append(v)
                driven_xy.append(q["xy"])
                driven_yaw.append(q["yaw"])
                driven_speed.append(q["signed_speed_mps"])
                intent = car_intent_mps(v0, cars[v]["waited"]) if v in cars else v0
                if intent > v0 and abs(v0) < STANDING_CAR_MPS:
                    set_off[v] = cars[v]["ahead"]
                preds.append(predict_route(r["points"], r["yaw"], r["arc"], s0, intent, self.crowd.times))
        off_trip = np.ones(n, bool)
        off_trip[list(cars)] = False
        self.car_wait[off_trip] = 0
        driven = None
        if driven_ids:
            driven = (
                np.array(driven_ids, np.int64),
                np.array(driven_xy),
                np.array(driven_yaw),
                np.array(driven_speed),
                np.array([p[0] for p in preds]),
                np.array([p[1] for p in preds]),
                np.array([p[2] for p in preds]),
            )
        d0 = clock()
        wall["driving"] = wall.get("driving", 0.0) + d0 - c0

        # Walking.
        for i, pts in self.standoffs.update(
            t, travel, xy, self.stall_run, self.paths, self.tangent, self.agendas, DT
        ).items():
            self.paths.set(i, pts)
        for i, pts in self.reroutes.update(
            t,
            travel,
            xy,
            speed,
            self.active,
            self.stall_run,
            self.paths,
            self.target,
            set(self.standoffs.yielding),
            self.agendas,
            DT,
            yaw=yaw,
            extra=self.car_cells(),
        ).items():
            self.paths.set(i, pts)
        for i, pts in self.courtesy.update(
            t,
            travel,
            standing,
            xy,
            speed,
            self.active,
            self.stall_run,
            self.paths,
            self.agendas,
            DT,
            self.ids,
            cars=cars,
            yaw=yaw,
        ).items():
            self.paths.set(i, pts)
        for i in self.courtesy.aside:
            if i not in self.courtesy.walking:
                self.paths.pop(i)
        for i in self.courtesy.resumed:
            self.paths.pop(i)  # a walker that stepped aside for a car plans its way on from where it stands
        travel = [i for i in travel if i not in self.courtesy.aside] + self.courtesy.walking
        d1 = clock()
        wall["standoffs"] = wall.get("standoffs", 0.0) + d1 - d0
        cells = self.car_cells()
        avoided, near_cars = None, np.zeros(n, bool)
        if len(cells):
            avoided = np.zeros(len(self.router.cost), bool)
            avoided[cells] = True
            ways = np.concatenate([p for _, p, _ in self.standing_ways])
            for i in travel:
                near_cars[i] = float(np.min(np.sum((ways - xy[i]) ** 2, axis=1))) <= (JAM_AHEAD_M + 3.0) ** 2
        for i in travel:
            if (
                near_cars[i]
                and i in self.paths
                and i not in self.courtesy.aside
                and i not in self.standoffs.yielding
                and t >= self.round_at.get(i, -math.inf)
                and self._through_cars(i, avoided)
            ):
                self.round_at[i] = t + ROUND_RETRY_S
                p = self.route(i, np.asarray(self.target(i), float), t=t)
                if p is not None:
                    self.paths.set(i, p)
                    self.stats["round_car_paths"] = self.stats.get("round_car_paths", 0) + 1
            if (
                i in self.paths
                and i not in self.courtesy.aside
                and i not in self.standoffs.yielding
                and self.ended_short(i, xy)
            ):
                self.paths.pop(i)  # planned anew to the target
                self.stats["stale_paths"] = self.stats.get("stale_paths", 0) + 1
            if (
                i in self.paths
                and i not in self.courtesy.aside
                and i not in self.standoffs.yielding
                and self.stall_run[i] * DT >= RECONNECT_AFTER_S
                and t >= self.reconnect_at.get(i, -math.inf)
            ):
                self.reconnect_at[i] = t + RECONNECT_RETRY_S
                if not self.connected(i, xy):
                    # Planned anew from where it stands; the present path is kept when no new route exists.
                    p = self.route(i, np.asarray(self.target(i), float), t=t)
                    if p is not None:
                        self.paths.set(i, p)
                        self.stats["reconnected_paths"] = self.stats.get("reconnected_paths", 0) + 1
                    else:
                        self.stats["reconnect_failures"] = self.stats.get("reconnect_failures", 0) + 1
            if i not in self.paths and t >= self.route_retry.get(i, -math.inf):
                p = self.route(i, np.asarray(self.target(i), float), t=t)
                if p is None:
                    self.stats["route_failures"] += 1
                    self.route_retry[i] = t + 5.0
                    self.agendas[i].blocked(t, "no_static_route")
                else:
                    self.paths.set(i, p)
        d2 = clock()
        wall["routes"] = wall.get("routes", 0.0) + d2 - d1
        anchor, tangent, lane = xy.copy(), yaw.copy(), np.zeros(n)
        led = np.array([i for i in travel if i in self.paths], np.int64)
        if len(led):
            h, want, cap, an, tg, ln = self.paths.desire(led, xy, speed, self.vmax, yaw)
            goal_cap[led] = cap[led]
            ok = led[~np.isnan(h[led])]
            des_yaw[ok] = h[ok]
            anchor[ok] = an[ok]
            tangent[ok] = tg[ok]
            lane[ok] = ln[ok]
            des_speed[ok] = want[ok]
        self.tangent = tangent
        d3 = clock()
        wall["desire"] = wall.get("desire", 0.0) + d3 - d2
        mover = (self.kinds == "person") & self.active
        seen = self.seen_standing(set_off, cars, xy, yaw, speed)
        inputs = dict(
            xy=xy,
            yaw=yaw,
            speed=speed,
            active=self.active.copy(),
            mover=mover,
            des_yaw=des_yaw,
            des_speed=des_speed,
            vmax=self.vmax,
            goal_cap=goal_cap,
            anchor=anchor,
            tangent=tangent,
            lane=lane,
            seen_standing=seen,
        )
        history = dict(waited=self.crowd.waited.copy(), walked=self.crowd.walked.copy(), aim=self.crowd.aim.copy())
        new_xy, new_yaw, new_speed, held = self.crowd.step(**inputs, driven=driven)
        self.last_motion = (
            inputs,
            history,
            dict(xy=new_xy.copy(), yaw=new_yaw.copy(), speed=new_speed.copy(), held=np.array(held, np.int64)),
            driven,
        )
        # Contained occupants move with their vehicle.
        for i in np.flatnonzero(self.riding):
            v = self.container[i]
            new_xy[i] = new_xy[v]
            new_yaw[i] = new_yaw[v]
            new_speed[i] = 0.0
        d4 = clock()
        wall["motion"] = wall.get("motion", 0.0) + d4 - d3
        v_now = (new_xy - xy) / DT
        spd = np.linalg.norm(v_now, axis=1)
        state = np.where(self.riding, 3, np.where(spd > 0.025, 1, 2))
        moving_leg = np.zeros(n, bool)
        moving_leg[travel] = True
        self.stall_run = np.where(moving_leg & (spd < 0.2), self.stall_run + 1, 0)
        for i in np.flatnonzero(self.stall_run == 30):
            if int(i) not in self.delayed:
                self.agendas[i].blocked(t, "blocked_by_bodies")
                self.delayed.add(int(i))
        self.delayed -= set(np.flatnonzero(self.stall_run == 0).tolist())
        # A trip starts with the car's first observed displacement (an admitted plan alone is not a departure);
        # it ends with the executed arrival above.
        for v in driven_ids:
            vid = self.vehicle_ids[v]
            if vid in self.departed_drives or np.linalg.norm(new_xy[v] - xy[v]) <= 1e-8:
                continue
            evidence = self.driving.active[vid]
            actor = self.drivers.director.vehicles[vid].owner
            intent = self.drivers.director.active[actor]
            self.departed_drives[vid] = evidence
            drive_events.append(
                dict(
                    kind="depart",
                    t_s=round(t, 6),
                    actors=[vid],
                    driver=actor,
                    intent_id=intent.id,
                    purpose=intent.purpose,
                    place=intent.place,
                    berth=self.driving.plans[evidence]["source"],
                    evidence=evidence,
                    xy=new_xy[v].tolist(),
                    source="observed_drive_execution",
                )
            )
        traffic = getattr(self.driving, "traffic", None)
        if traffic is not None and traffic.journal:
            # Reservation changes of this tick (TrafficRuns.journal): what each car waited for, and for whom.
            drive_events.extend(dict(row, t_s=round(t, 6), actors=[row["vehicle"]]) for row in traffic.journal)
            traffic.journal.clear()
        self.events.extend(drive_events)
        self.xy, self.yaw, self.speed = new_xy, K_wrap(new_yaw), new_speed
        self._sync_driving()
        self._update_walk_space(t)
        return self.xy, self.yaw, self.speed, v_now, state, self.container.copy(), held

    def collisions(self):
        """Colliding active pairs by the tape QA predicate at the present poses (for evaluation scripts)."""
        ids = np.flatnonzero(self.active)
        c, s = bodies.boxes(self.xy[ids], self.yaw[ids], self.speed[ids], self.ext[ids], np.full(len(ids), 1.7))
        return [(int(ids[a]), int(ids[b])) for a, b in collision_pairs(c, self.yaw[ids], s)]


def K_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
