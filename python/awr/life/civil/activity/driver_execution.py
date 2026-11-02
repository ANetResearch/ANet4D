"""Join a shared Director's actual drive step to grounded vehicle execution.

This adapter never generates an itinerary, writes a pose, invents a transfer,
or acknowledges a walk/service step. The population backend supplies measured
occupancy and containment after physically executing boarding/alighting. It
applies motion proposals only after live swept collision checks. A nominal
arrival or an external pose jump cannot complete a drive through this adapter.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from .director import Route


@dataclass(frozen=True)
class DriveObservation:
    actor_xy: tuple[float, float]
    container: str
    vehicle_pose: tuple[float, float, float]
    vehicle_speed: float


class DriverExecution:
    def __init__(self, director, driving):
        self.director = director
        self.driving = driving
        # Connect the same generic planning promises for every adapter, including
        # synthetic environments. Future parking is part of route feasibility;
        # it never grants permission to depart or moves a body.
        if director.keeps_ways is None:
            director.keeps_ways = driving.keeps_ways
        if director.keeps_roads is None and getattr(driving, "roads", None) is not None:
            director.keeps_roads = driving.keeps_roads
        if not director.way_out and hasattr(driving, "way_out_nodes"):
            director.way_out = driving.way_out_nodes()
        if driving.pending is None:
            driving.pending = director.pending_destinations
        self.bindings = {}
        self.last_proposal_time = {}
        self.waiting = set()

    def _step(self, actor):
        intent = self.director.active.get(actor)
        if intent is None or intent.step.kind != "drive":
            raise ValueError("Current activity is not an executable drive step")
        owner = self.director.actors[actor]
        vehicle = self.director.vehicles.get(owner.vehicle)
        if vehicle is None or vehicle.owner != actor or intent.step.entity != vehicle.id or vehicle.occupant != actor:
            raise ValueError("Actual owner must finish boarding before driving")
        evidence = intent.step.route.evidence
        plan = self.driving.plans.get(evidence)
        if (
            plan is None
            or plan["vehicle"] != vehicle.id
            or plan["source"] != vehicle.berth
            or plan["target"] != intent.destination_berth
            or not np.allclose(self.driving.graph.nodes[plan["target"]]["pose"][:2], intent.step.xy, rtol=0, atol=1e-8)
        ):
            raise ValueError("Drive geometry does not belong to this activity and vehicle")
        binding = (intent.id, evidence)
        if actor in self.bindings and self.bindings[actor] != binding:
            raise ValueError("Cannot replace an active physical drive with a new intent")
        return intent, vehicle, evidence, binding

    def follow_revision(self, actor):
        """Bind the drive step of `actor` to the route its car follows after the backend revised it (DrivingRoutes.
        revise: the same destination berth by another way); the revisions taken, oldest first, as (old evidence, new
        evidence, splice arc, reason)."""
        intent = self.director.active.get(actor)
        if intent is None or intent.step.kind != "drive" or intent.step.route is None:
            return []
        pending = getattr(self.driving, "revisions", {}).get(intent.step.entity)
        # The whole chain is checked before the step changes: a bad link leaves the activity and the queue as they were.
        taken = []
        evidence = intent.step.route.evidence
        for old, new, splice, reason in pending or ():
            plan = self.driving.plans[new]
            if (
                old != evidence
                or plan["vehicle"] != intent.step.entity
                or plan["target"] != intent.destination_berth
                or plan["source"] != self.director.vehicles[intent.step.entity].berth
            ):
                raise ValueError("A revised drive must keep its car and destination berth")
            taken.append((old, new, splice, reason))
            evidence = new
        if taken:
            if self.driving.active.get(intent.step.entity) != evidence or self.bindings.get(actor) != (
                intent.id,
                intent.step.route.evidence,
            ):
                raise ValueError("A revised drive must match the active execution and activity binding")
            schedule = self.driving.plans[evidence]["schedule"]
            intent.steps[intent.index] = replace(
                intent.step, route=Route(schedule.length_m, schedule.duration_s, evidence)
            )
            self.bindings[actor] = (intent.id, evidence)
            pending.clear()
        return taken

    @staticmethod
    def _contained(actor_xy, container, vehicle, pose):
        xy = np.asarray(actor_xy, float)
        pose = np.asarray(pose, float)
        # Contained occupants use their carrier's authoritative XY in the
        # current tape convention. This does not place a visible body there.
        if (
            xy.shape != (2,)
            or pose.shape != (3,)
            or not np.isfinite(np.r_[xy, pose]).all()
            or container != vehicle.id
            or np.linalg.norm(xy - pose[:2]) > 1e-6
        ):
            raise ValueError("Observed driver is not contained in the owned vehicle")

    def _validate(self, actor, t, actor_xy, container, vehicle_pose, vehicle_speed, dt):
        intent, vehicle, evidence, binding = self._step(actor)
        self._contained(actor_xy, container, vehicle, vehicle_pose)
        if (
            not math.isfinite(t)
            or not math.isfinite(dt)
            or not 0 < dt <= 0.100000001
            or not math.isfinite(vehicle_speed)
        ):
            raise ValueError("Finite world clock and bounded physics tick required")
        if t < self.director._time.get(actor, -math.inf):
            raise ValueError("Driver observation predates its activity clock")
        if intent.paused_at is not None:
            raise ValueError("Physical driving cannot use the pedestrian courtesy pause")
        previous = self.last_proposal_time.get(actor)
        if previous is not None and abs(t - previous) > 1e-7:
            raise ValueError("Motion tick differs from the actual world progression")
        return intent, vehicle, evidence, binding

    def _admit(self, actor, t, actor_xy, vehicle_pose, vehicle_speed, validated):
        intent, vehicle, evidence, binding = validated
        self.director.clock(actor, t)
        if actor not in self.bindings:
            if not self.driving.acquire(evidence, vehicle_pose, vehicle_speed):
                if actor not in self.waiting:
                    self.director.emit(
                        "drive_departure_wait",
                        actor,
                        t,
                        intent_id=intent.id,
                        purpose=intent.purpose,
                        place=intent.place,
                        vehicle=vehicle.id,
                        reason="current_traffic_or_occupancy",
                        at="source_berth",
                    )
                    self.waiting.add(actor)
                return False
            self.bindings[actor] = binding
            self.waiting.discard(actor)
            self.director.emit(
                "drive_departure_admitted",
                actor,
                t,
                intent_id=intent.id,
                purpose=intent.purpose,
                place=intent.place,
                vehicle=vehicle.id,
                evidence=evidence,
            )
        # Only actual observed departure releases the old parking slot.
        self.director.observe_departure(actor, tuple(actor_xy), tuple(vehicle_pose[:2]))
        return True

    def advance(self, actor, t, actor_xy, container, vehicle_pose, vehicle_speed, dt, *, stop_before_arc_m=None):
        """Propose a legacy per-car tick; shared physics must use advance_batch."""
        if getattr(self.driving, "shared_world", False):
            raise ValueError("Shared physics requires one complete activity batch")
        validated = self._validate(actor, t, actor_xy, container, vehicle_pose, vehicle_speed, dt)
        if not self._admit(actor, t, actor_xy, vehicle_pose, vehicle_speed, validated):
            return None
        evidence = validated[2]
        proposal = self.driving.advance_motion(
            evidence, vehicle_pose, vehicle_speed, dt, stop_before_arc_m=stop_before_arc_m
        )
        self.last_proposal_time[actor] = t + dt
        return proposal

    def advance_batch(self, t, observations, dt, *, speed_limits=None, kinematic_targets=None):
        """Validate all drivers, admit in stable order, then advance the world once.

        Missing driving actors, stale observations and bad movement targets fail
        before reservations or activity clocks change. Empty batches still
        advance parked cars and external bodies. This method does not invent
        boarding/alighting observations or acknowledge completed drive steps.
        """
        if not getattr(self.driving, "shared_world", False):
            raise ValueError("Backend is not a shared physics world")
        required = {actor for actor, intent in self.director.active.items() if intent.step.kind == "drive"}
        if set(observations) != required or not all(isinstance(o, DriveObservation) for o in observations.values()):
            raise ValueError("Complete driving activity observations required")
        validated = {}
        for actor in sorted(observations):
            o = observations[actor]
            validated[actor] = self._validate(actor, t, o.actor_xy, o.container, o.vehicle_pose, o.vehicle_speed, dt)
        self.driving.validate_batch(t, observations, dt, speed_limits=speed_limits, kinematic_targets=kinematic_targets)
        admitted = {}
        for actor in sorted(observations):
            o = observations[actor]
            if self._admit(actor, t, o.actor_xy, o.vehicle_pose, o.vehicle_speed, validated[actor]):
                admitted[actor] = validated[actor][2]
        frame = self.driving.advance_batch(
            t, admitted, dt, speed_limits=speed_limits, kinematic_targets=kinematic_targets
        )
        for actor in admitted:
            self.last_proposal_time[actor] = t + dt
        return frame

    def finish(self, actor, t, actor_xy, container, vehicle_pose, vehicle_speed):
        """Commit parking and the Director's drive step from one observed frame.

        No alight request is issued before both the physical executor and the
        body snapshot agree on arrival. Subsequent alighting/walking/service
        still belongs to the population backend and shared Director.
        """
        if getattr(self.driving, "failed", None) is not None:
            raise RuntimeError("Rejected driving world cannot acknowledge arrival")
        intent, vehicle, evidence, binding = self._step(actor)
        self._contained(actor_xy, container, vehicle, vehicle_pose)
        if self.bindings.get(actor) != binding:
            return False
        if not math.isfinite(t) or abs(t - self.last_proposal_time[actor]) > 1e-7:
            raise ValueError("Arrival must use the observed post-motion world frame")
        if intent.paused_at is not None:
            return False
        if not self.driving.can_arrive(evidence, vehicle_pose, vehicle_speed, require_executed=True):
            return False
        # All conditions checked before either reservation owner is changed.
        # The caller executes this serially in its authoritative simulation loop.
        if not self.director.acknowledge(
            actor, t, tuple(actor_xy), 0.0, vehicle_xy=tuple(vehicle_pose[:2]), vehicle_speed=vehicle_speed, radius=0.1
        ):
            return False
        if not self.driving.arrive(evidence, vehicle_pose, vehicle_speed):
            raise RuntimeError("Physical arrival changed during serial activity handoff")
        del self.bindings[actor]
        del self.last_proposal_time[actor]
        return True
