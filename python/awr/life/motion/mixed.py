"""Generic observation-driven activity loop for walkers and resident drivers.

Inputs are admitted bodies, a shared Director, measured DrivingRoutes, a Crowd
controller, and a callback resolving walking Route evidence to a polyline.
No actor-specific itinerary or recording horizon is accepted. Each tick creates
one joint body frame; vehicle and pedestrian proposals cannot overwrite one
another. On a failed physical gate this instance is terminal, not resumable by
relabeling a stop. A caller must retain the rejected frame for diagnosis.

Transfers follow the existing tape convention: a visible person waits at the
verified door, then becomes a contained occupant; after parking the person may
become visible only at a free supported door. These discrete containment events
are explicit, NOT a claim of an animated door/seat/interior implementation.

This backend is opt-in. It is not yet wired to the town population compiler or
approved for publishing; a synthetic integration run is not real-map admission.
"""

import math
import time
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point

from awr.life.civil.activity.driver_execution import DriveObservation, DriverExecution
from awr.life.civil.activity.driving import Body
from awr.life.civil.curb_allocation import footprint

from .crowd import DT, PathStore
from .mixed_safety import free_door, joint_conflicts, stopping_arc
from .reroute import Reroutes
from .standoff import Standoffs


class _DelayLog:
    def __init__(self, director, actor):
        self.director, self.actor = director, actor

    def blocked(self, t, reason):
        intent = self.director.active[self.actor]
        self.director.emit(
            "travel_delay",
            self.actor,
            t,
            reason=reason,
            intent_id=intent.id,
            purpose=intent.purpose,
            place=intent.place,
        )


@dataclass(frozen=True)
class Frame:
    time_s: float
    xy: np.ndarray
    yaw: np.ndarray
    speed: np.ndarray
    container: np.ndarray
    physics_frame: object | None = None
    physics_indices: tuple[int, ...] = ()
    # Arrays are detached copies of the last accepted frame. container=-1 is
    # visible; other values index the vehicle in the same immutable roster.


class MixedLife:
    def __init__(
        self,
        ids,
        kinds,
        sizes,
        xy,
        yaw,
        director,
        driving,
        crowd,
        walk_points,
        *,
        start_s=0.0,
        vmax=1.3,
        arrival_radius=0.05,
        max_stall_s=60.0,
        walking_separation=False,
        physics_gate=None,
        walking_router=None,
    ):
        self.ids = tuple(ids)
        self.index = {name: i for i, name in enumerate(ids)}
        self.kinds = np.asarray(kinds)
        n = len(ids)
        self.sizes = np.asarray(sizes, float).copy()
        self.xy = np.asarray(xy, float).copy()
        self.yaw = np.asarray(yaw, float).copy()
        self.speed = np.zeros(n)
        self.container = np.full(n, -1, np.int64)
        if (
            len(self.index) != n
            or n == 0
            or self.kinds.shape != (n,)
            or self.xy.shape != (n, 2)
            or self.yaw.shape != (n,)
            or self.sizes.shape != (n, 3)
            or not np.isfinite(self.xy).all()
            or not np.isfinite(self.yaw).all()
            or not np.isfinite(self.sizes).all()
            or np.any(self.sizes <= 0)
            or not math.isfinite(start_s)
            or not 0 < arrival_radius <= 0.1
            or not math.isfinite(max_stall_s)
            or max_stall_s <= 0
            or not math.isfinite(vmax)
            or vmax <= 0
        ):
            raise ValueError("Invalid mixed-world roster, geometry or motion limits")
        self.shared_physics = bool(getattr(driving, "shared_world", False))
        self.physics_gate = physics_gate
        physical_kinds = self.kinds != "person"
        if (
            crowd.ext.shape != (n, 2, 4)
            or not np.array_equal(crowd.kinds, self.kinds)
            or set(director.actors) != {self.ids[i] for i in np.flatnonzero(self.kinds == "person")}
            or set(driving.roster) != {self.ids[i] for i in np.flatnonzero(physical_kinds)}
        ):
            raise ValueError("Director, Crowd and driving occupancy must share the actual roster")
        if self.shared_physics:
            if not callable(physics_gate) or abs(start_s - driving.current.time_s) > 1e-8:
                raise ValueError("Shared physics requires a mixed-frame gate and its actual starting clock")
            if set(driving.world.kinematics) & set(director.actors):
                raise ValueError("Crowd-owned people cannot retain duplicate native kinematic bodies")
        self.director, self.driving, self.crowd = director, driving, crowd
        self.accepted_physics = driving.current if self.shared_physics else None
        self.driver = DriverExecution(director, driving)
        self.walk_points = walk_points
        from .routing import KeepRightRoutes

        self.walking_routes = KeepRightRoutes(crowd) if walking_separation else None
        self.route_records = []
        self.persons = tuple(sorted(director.actors))
        self.cars = np.flatnonzero(self.kinds == "vehicle")
        self.rigid = np.flatnonzero(self.kinds != "person")
        if self.shared_physics:
            for name, row in zip(driving.current.names, driving.current.states, strict=False):
                i = self.index[name]
                if (
                    not np.allclose(self.xy[i], row[1:3], atol=1e-8, rtol=0.0)
                    or abs((self.yaw[i] - row[8] + math.pi) % (2 * math.pi) - math.pi) > 1e-8
                ):
                    raise ValueError("Initial vehicle poses must match authoritative physics")
                self.speed[i] = row[11]
        self.vmax = np.where(self.kinds == "person", vmax, driving.limits.forward_mps)
        self.arrival_radius = arrival_radius
        self.max_stall_s = max_stall_s
        self.paths = PathStore(n)
        self.standoffs = Standoffs(crowd)
        self.reroutes = None if walking_router is None else Reroutes(walking_router, crowd.ext, crowd)
        self.delay_logs = {self.index[a]: _DelayLog(director, a) for a in self.persons}
        self.tangent = self.yaw.copy()
        self.path_keys = {}
        self.stalls = np.zeros(n)
        self.previous_velocity = np.zeros((n, 2))
        self.previous_active = np.ones(n, bool)
        self.waiting_doors = set()
        self.waiting_boards = set()
        self.door_wait_since = {}
        self.transfer_started = {}
        self.departure_wait_since = {}
        self.start_s, self.tick_index = float(start_s), 0
        self.failed = None
        self.transfers = []
        self.profile = dict(
            ticks=0,
            routes=0,
            stages_s={
                k: 0.0
                for k in ("activities", "walking_routes", "standoffs", "vehicle_motion", "crowd_motion", "validation")
            },
        )
        self.tick_wall = []
        for name, vehicle in director.vehicles.items():
            if name not in self.index or vehicle.occupant is not None or vehicle.in_trip:
                raise ValueError("Initial drivers must be visible and vehicles parked")
            i = self.index[name]
            if not np.allclose(self.sizes[i], driving.size, atol=1e-8, rtol=0.0):
                raise ValueError("Owned vehicle dimensions need a matching driving backend")
            berth = director.places[vehicle.berth]
            pose = driving.graph.nodes[vehicle.berth]["pose"]
            if not driving._at(np.r_[self.xy[i], self.yaw[i]], pose, 0.0):
                raise ValueError("Initial vehicle must be at its actual admitted berth")
            if berth.access_xy is None:
                raise ValueError("Resident vehicle requires an admitted door access")
        self._occupancy()
        conflicts = joint_conflicts(crowd, self.xy, self.yaw, self.xy, self.yaw, self.speed, self.container < 0)
        if conflicts:
            raise ValueError(f"Initial bodies overlap: {conflicts}")

    @property
    def time_s(self):
        return self.start_s + self.tick_index * DT

    def frame(self):
        physical = self.accepted_physics
        indices = tuple(self.index[n] for n in physical.names) if physical is not None else ()
        return Frame(
            self.time_s, self.xy.copy(), self.yaw.copy(), self.speed.copy(), self.container.copy(), physical, indices
        )

    def _occupancy(self):
        self.driving.update_occupancy(
            {self.ids[i]: Body(tuple(np.r_[self.xy[i], self.yaw[i]]), tuple(self.sizes[i])) for i in self.rigid}
        )

    def _transfer(self, actor, kind):
        i = self.index[actor]
        vehicle = self.director.vehicles[self.director.actors[actor].vehicle]
        c = self.index[vehicle.id]
        berth = self.director.places[vehicle.berth]
        pose = self.driving.graph.nodes[vehicle.berth]["pose"]
        if (
            not self.driving._at(np.r_[self.xy[c], self.yaw[c]], pose, self.speed[c])
            or footprint(dict(xy=self.xy[c], yaw=self.yaw[c]), self.sizes[c], margin_m=0.0).distance(
                Point(berth.access_xy)
            )
            > 1.5
        ):
            raise ValueError("Transfer requires the actual parked car and nearby admitted door")
        if kind == "alight":
            if self.container[i] == -1:
                return True  # visible exit dwell already in progress
            if self.container[i] != c or vehicle.occupant != actor:
                raise ValueError("Alighting person is not the actual contained owner")
            point = np.asarray(berth.access_xy, float)
            # Face away from the vehicle, so the resting and next walking body
            # do not deliberately start turned into the car.
            heading = math.atan2(*(point - self.xy[c])[::-1])
            if not free_door(
                self.crowd, i, point, heading, self.xy, self.yaw, self.speed, self.container < 0, carrier=c
            ):
                since = self.door_wait_since.setdefault(actor, self.time_s)
                if actor not in self.waiting_doors:
                    self.director.emit(
                        "alight_wait", actor, self.time_s, vehicle=vehicle.id, reason="door_not_physically_free"
                    )
                    self.waiting_doors.add(actor)
                if self.time_s - since > self.max_stall_s + 1e-7:
                    raise ValueError(f"Alighting blocked for too long: {actor}")
                return False
            self.xy[i], self.yaw[i], self.speed[i], self.container[i] = point, heading, 0.0, -1
            self.waiting_doors.discard(actor)
            self.door_wait_since.pop(actor, None)
        else:
            if self.container[i] != -1 or math.dist(self.xy[i], berth.access_xy) > self.arrival_radius:
                raise ValueError("Boarding requires a visible person at the admitted door")
            self.xy[i], self.yaw[i], self.speed[i], self.container[i] = self.xy[c], self.yaw[c], 0.0, c
        self.transfers.append(
            dict(
                t_s=round(self.time_s, 6),
                actor=actor,
                vehicle=vehicle.id,
                kind=kind,
                door_xy=list(berth.access_xy),
                representation="discrete_containment",
            )
        )
        return True

    def _activities(self):
        for actor in self.persons:
            i = self.index[actor]
            owner = self.director.actors[actor]
            v = self.index.get(owner.vehicle)
            self.director.observe_departure(actor, tuple(self.xy[i]), None if v is None else tuple(self.xy[v]))
            intent = self.director.active.get(actor)
            if intent is None:
                intent = self.director.choose(actor, self.time_s, tuple(self.xy[i]))
            if intent is None:
                continue
            step = intent.step
            if step.kind in ("board", "alight"):
                key = (intent.id, intent.index)
                previous = self.transfer_started.get(actor)
                if previous is None or previous[0] != key:
                    self.transfer_started[actor] = (key, self.time_s)
                elif self.time_s - previous[1] > step.duration_s + self.max_stall_s + 1e-7:
                    raise ValueError(f"Transfer acknowledgement stalled: {actor}/{step.kind}")
            else:
                self.transfer_started.pop(actor, None)
            if step.kind == "drive":
                if self.container[i] != v:
                    raise ValueError("Drive request without physical containment")
                continue
            if step.kind == "alight" and not self._transfer(actor, "alight"):
                continue
            if self.container[i] != -1:
                raise ValueError("Walking/service cannot occur while contained")
            if step.kind == "board" and not free_door(
                self.crowd, i, self.xy[i], self.yaw[i], self.xy, self.yaw, self.speed, self.container < 0, carrier=v
            ):
                # A transfer needs continuous free access throughout its dwell.
                # Never acknowledge first and discover a blocked corridor later.
                intent.service_started = None
                since = self.door_wait_since.setdefault(actor, self.time_s)
                if actor not in self.waiting_boards:
                    self.director.emit(
                        "board_wait",
                        actor,
                        self.time_s,
                        vehicle=owner.vehicle,
                        reason="door_corridor_not_physically_free",
                    )
                    self.waiting_boards.add(actor)
                if self.time_s - since > self.max_stall_s + 1e-7:
                    raise ValueError(f"Boarding blocked for too long: {actor}")
                continue
            if step.kind == "board":
                self.door_wait_since.pop(actor, None)
                self.waiting_boards.discard(actor)
            kwargs = (
                dict(vehicle_xy=tuple(self.xy[v]), vehicle_speed=float(self.speed[v]), vehicle_radius=0.1)
                if step.kind in ("board", "alight")
                else {}
            )
            if (
                self.director.acknowledge(
                    actor,
                    self.time_s,
                    tuple(self.xy[i]),
                    float(abs(self.speed[i])),
                    radius=self.arrival_radius,
                    **kwargs,
                )
                and step.kind == "board"
            ):
                self._transfer(actor, "board")

    def step(self):
        if self.failed is not None:
            raise RuntimeError("Rejected mixed world cannot continue; rebuild from frozen inputs")
        try:
            return self._step()
        except Exception as exc:
            self.failed = dict(t_s=self.time_s, error=repr(exc), accepted_frame=self.tick_index)
            if self.shared_physics and self.driving.failed is None:
                self.driving.failed = dict(self.failed, stage="mixed_frame")
            raise

    def _step(self):
        began = time.perf_counter()
        self._occupancy()
        self._activities()
        a = time.perf_counter()
        self.profile["stages_s"]["activities"] += a - began
        n = len(self.ids)
        active = self.container < 0
        travel = []
        for actor in self.persons:
            i = self.index[actor]
            intent = self.director.active.get(actor)
            if intent is None or intent.step.kind != "walk":
                self.paths.pop(i)
                self.path_keys.pop(i, None)
                self.standoffs.yielding.pop(i, None)
                continue
            key = (intent.id, intent.index)
            if self.path_keys.get(i) != key:
                points = np.asarray(
                    self.walk_points(actor, intent.step.route, tuple(self.xy[i]), intent.step.xy), float
                )
                if (
                    points.ndim != 2
                    or points.shape[1] != 2
                    or len(points) < 1
                    or not np.isfinite(points).all()
                    or np.linalg.norm(points[0] - self.xy[i]) > self.arrival_radius
                    or np.linalg.norm(points[-1] - intent.step.xy) > 1e-7
                ):
                    raise ValueError("Walking evidence does not connect the observed body to the activity")
                if self.reroutes is not None:
                    points = self.reroutes.departure(points, i, self.xy, self.yaw, self.speed, active)
                if self.walking_routes is not None:
                    import hashlib

                    original_hash = hashlib.sha256(points.tobytes()).hexdigest()
                    points = self.walking_routes(i, points)
                    self.route_records.append(
                        dict(
                            actor=actor,
                            t_s=self.time_s,
                            intent=intent.id,
                            step=intent.index,
                            source_sha256=original_hash,
                            motion_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
                            points=points.tolist(),
                        )
                    )
                self.paths.set(i, points)
                self.standoffs.yielding.pop(i, None)
                self.path_keys[i] = key
                self.profile["routes"] += 1
            travel.append(i)
        routed = time.perf_counter()
        self.profile["stages_s"]["walking_routes"] += routed - a
        stall_run = np.rint(self.stalls / DT).astype(np.int64)
        handoffs = self.standoffs.update(
            self.time_s, travel, self.xy, stall_run, self.paths, self.tangent, self.delay_logs, DT
        )
        for i, points in handoffs.items():
            self.paths.set(i, points)
        if self.reroutes is not None:
            # Both retreat and resume handoffs own their route for this tick.
            # Replanning observes the accepted frame, never proposed motion.
            for i, points in self.reroutes.update(
                self.time_s,
                travel,
                self.xy,
                self.speed,
                active,
                stall_run,
                self.paths,
                lambda i: self.director.active[self.ids[i]].step.xy,
                set(self.standoffs.yielding) | set(handoffs),
                self.delay_logs,
                DT,
                yaw=self.yaw,
            ).items():
                self.paths.set(i, points)
        routed2 = time.perf_counter()
        self.profile["stages_s"]["standoffs"] += routed2 - routed
        des_yaw, des_speed, cap = self.yaw.copy(), np.zeros(n), np.full(n, np.inf)
        anchor, tangent, lane = self.xy.copy(), self.yaw.copy(), np.zeros(n)
        walking_progress = np.zeros(n)
        if travel:
            led = np.array(travel, np.int64)
            previous_arc = self.paths.s[led].copy()
            h, want, cap, anchor, tangent, lane = self.paths.desire(
                led, self.xy, np.abs(self.speed), self.vmax, self.yaw
            )
            walking_progress[led] = np.maximum(0.0, self.paths.s[led] - previous_arc)
            ok = led[~np.isnan(h[led])]
            des_yaw[ok] = h[ok]
            des_speed[ok] = want[ok]  # Crowd owns turning and bounded blocked-turn recovery
        self.tangent = tangent.copy()
        b = time.perf_counter()
        self.profile["stages_s"]["walking_routes"] += b - routed2
        next_xy, next_yaw, next_speed = self.xy.copy(), self.yaw.copy(), self.speed.copy()
        external = np.zeros(n, bool)
        drive_moving_leg = np.zeros(n, bool)
        driving_actors = []
        physical_before = self.driving.current if self.shared_physics else None
        observations = {}
        speed_limits = {}
        for actor in self.persons:
            intent = self.director.active.get(actor)
            if intent is None or intent.step.kind != "drive":
                continue
            i = self.index[actor]
            c = self.container[i]
            evidence = intent.step.route.evidence
            limit = stopping_arc(self.driving, evidence, c, self.xy, self.yaw, self.speed, active, self.crowd)
            if self.shared_physics:
                observations[actor] = DriveObservation(
                    tuple(self.xy[i]), self.ids[c], tuple(np.r_[self.xy[c], self.yaw[c]]), float(abs(self.speed[c]))
                )
                speed_limits[self.ids[c]] = self.driving.speed_limit_for_stop(evidence, limit)
                continue
            proposal = self.driver.advance(
                actor,
                self.time_s,
                self.xy[i],
                self.ids[c],
                np.r_[self.xy[c], self.yaw[c]],
                abs(self.speed[c]),
                DT,
                stop_before_arc_m=limit,
            )
            if proposal is not None:
                self.departure_wait_since.pop(actor, None)
                next_xy[c], next_yaw[c], next_speed[c] = proposal["xy"], proposal["yaw"], proposal["signed_speed_mps"]
                external[c] = True
                drive_moving_leg[c] = True
                driving_actors.append(actor)
            else:
                since = self.departure_wait_since.setdefault(actor, self.time_s)
                if self.time_s - since > self.director.max_wait_s + 1e-7:
                    raise ValueError(f"Drive departure has no feasible continuation: {actor}")
        if self.shared_physics:
            physical = self.driver.advance_batch(self.time_s, observations, DT, speed_limits=speed_limits)
            for name, row in zip(physical.names, physical.states, strict=False):
                c = self.index[name]
                next_xy[c] = row[1:3]
                next_yaw[c] = row[8]
                next_speed[c] = row[11]
                external[c] = True
            for actor, o in observations.items():
                if actor in self.driver.bindings:
                    self.departure_wait_since.pop(actor, None)
                    driving_actors.append(actor)
                    drive_moving_leg[self.index[o.container]] = True
                else:
                    since = self.departure_wait_since.setdefault(actor, self.time_s)
                    if self.time_s - since > self.director.max_wait_s + 1e-7:
                        raise ValueError(f"Drive departure has no feasible continuation: {actor}")
        ctime = time.perf_counter()
        self.profile["stages_s"]["vehicle_motion"] += ctime - b
        mover = (self.kinds == "person") & active
        new_xy, new_yaw, new_speed, held = self.crowd.step(
            self.xy,
            self.yaw,
            self.speed,
            active,
            mover,
            des_yaw,
            des_speed,
            self.vmax,
            cap,
            anchor,
            tangent,
            lane,
            external=(external, next_xy, next_yaw, next_speed),
        )
        d = time.perf_counter()
        self.profile["stages_s"]["crowd_motion"] += d - ctime
        sw = np.maximum(np.abs(self.speed), np.abs(new_speed)) + 0.01
        conflicts = joint_conflicts(self.crowd, self.xy, self.yaw, new_xy, new_yaw, sw, active)
        if held or conflicts:
            raise ValueError(f"Mixed motion rejected: held={held}, conflicts={conflicts}")
        unsupported = [
            self.ids[i]
            for i in np.flatnonzero(mover)
            if not self.crowd._supported(i, self.xy, self.yaw, new_xy, new_yaw, sw)
        ]
        if unsupported:
            raise ValueError(f"Mixed pedestrian support rejected: {unsupported}")
        if self.shared_physics:
            if (
                self.physics_gate(self, physical_before, self.driving.samples, new_xy, new_yaw, new_speed, active)
                is not None
            ):
                raise ValueError("Mixed physics gate must raise on failure, not return an ignored verdict")
        velocity = (new_xy - self.xy) / DT
        acc = np.linalg.norm((np.round(velocity * 100) - np.round(self.previous_velocity * 100)) * 0.01, axis=1) / DT
        bad = np.flatnonzero(active & self.previous_active & (acc > np.where(self.kinds == "vehicle", 6.0, 8.0) + 1e-7))
        if len(bad):
            raise ValueError(f"Mixed acceleration rejected: {[(self.ids[i], float(acc[i])) for i in bad]}")
        displacement = np.linalg.norm(new_xy - self.xy, axis=1)
        # Lateral oscillation in front of an obstacle is not journey progress.
        # Paths project the previously accepted pose monotonically; replans and
        # retreat handoffs start their own observed arc, never count a route
        # replacement itself as movement. Native vehicles retain their actual
        # displacement test. No timeout is relaxed.
        displacement[travel] = walking_progress[travel]
        moving_leg = np.zeros(n, bool)
        moving_leg[travel] = True
        moving_leg |= drive_moving_leg
        self.stalls = np.where(moving_leg & (displacement / DT < 0.2), self.stalls + DT, 0.0)
        if np.any(self.stalls > self.max_stall_s + 1e-7):
            raise ValueError(
                f"Mixed travel stalled: {[self.ids[i] for i in np.flatnonzero(self.stalls > self.max_stall_s)]}"
            )
        self.xy, self.yaw, self.speed = new_xy, new_yaw, new_speed
        self.previous_velocity, self.previous_active = velocity, active.copy()
        for i in np.flatnonzero(self.container >= 0):
            vehicle = self.container[i]
            self.xy[i], self.yaw[i], self.speed[i] = self.xy[vehicle], self.yaw[vehicle], self.speed[vehicle]
        self.tick_index += 1
        self._occupancy()
        for actor in driving_actors:
            i = self.index[actor]
            vehicle = self.container[i]
            self.driver.finish(
                actor,
                self.time_s,
                self.xy[i],
                self.ids[vehicle],
                np.r_[self.xy[vehicle], self.yaw[vehicle]],
                abs(self.speed[vehicle]),
            )
        self.profile["stages_s"]["validation"] += time.perf_counter() - d
        self.profile["ticks"] += 1
        self.tick_wall.append(time.perf_counter() - began)
        if self.shared_physics:
            self.accepted_physics = self.driving.current
        return self.frame()

    def summary(self):
        wall = np.asarray(self.tick_wall)
        return dict(
            schema="awr.life.mixed_life_profile.v1",
            runtime="mixed_life_bullet_v1" if self.shared_physics else "mixed_life_v1",
            **self.profile,
            duration_s=self.tick_index * DT,
            wall_s=float(wall.sum()),
            failed=self.failed,
            tick_ms={
                q: float(np.percentile(wall, p) * 1000) if len(wall) else None
                for q, p in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))
            },
            physical_drives_completed=len(self.driving.completed_motion),
            transfers=len(self.transfers),
            standoffs=self.standoffs.stats,
            reroutes=None if self.reroutes is None else dict(self.reroutes.stats),
            walking_separation=None if self.walking_routes is None else dict(self.walking_routes.stats),
            admitted_for_benchmark=False,
            limitations=[
                "requires real-map population adapter and full external QA",
                "discrete containment transfers; no door/interior animation",
                "conservative whole-trip traffic reservations",
            ],
        )
