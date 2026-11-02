"""Activity execution for the life runtime: the Director's walk and car journeys.

LifeAgenda extends the grounded activities' ReactiveAgenda (whose walk-only behaviour and audit events are kept
unchanged) with the Director's car journeys: walk to the car side, board, drive, alight, walk to the place, stay.
The agenda never writes a pose. Walk steps are walked by the crowd (`target_xy` is the step's position); board,
drive and alight are executed by the runtime (awr.life.motion.engine), which reports the physical observations to the
Director. A step completes only through Director.acknowledge with measured poses and speeds.
"""

from __future__ import annotations

import math

from awr.life.civil.activity.grounded import ReactiveAgenda
from awr.life.civil.agenda import Visit

CAR_STEPS = ("walk", "board", "drive", "alight", "walk", "activity")


class LifeAgenda(ReactiveAgenda):
    """ReactiveAgenda that also executes car journeys (phases 'board', 'drive', 'alight' for the runtime)."""

    def car_intent(self):
        intent = self.director.active.get(self.actor)
        return intent if intent is not None and intent.destination_berth is not None else None

    @property
    def target_xy(self):
        """Where the walker is going: the current walk step of a car journey, else the visit's position."""
        intent = self.car_intent()
        if intent is not None and intent.step.kind == "walk":
            return intent.step.xy
        return self.destination.xy

    def _choose(self, t, xy):
        import time

        began = time.perf_counter()
        try:
            intent = self.director.choose(self.actor, t, tuple(xy))
        finally:
            self.director.decision_wall_s += time.perf_counter() - began
        if intent is None:
            self.phase = "deciding"
            if not self.wait_logged:
                self.emit("activity_decision_wait", t, reason="no_due_feasible_activity")
                self.wait_logged = True
            return "wait"
        kinds = tuple(s.kind for s in intent.steps)
        if intent.destination_berth is not None and kinds != CAR_STEPS:
            raise ValueError(f"Unsupported car journey steps {kinds}")
        if intent.destination_berth is None and any(k not in ("walk", "activity") for k in kinds):
            raise ValueError(f"Unsupported journey steps {kinds}")
        self.visits.append(Visit(intent.slot, intent.steps[-1].xy, intent.purpose, intent.steps[-1].duration_s))
        self.index = len(self.visits) - 1
        self.phase = "scheduled"
        self.departure_s = t
        self.wait_logged = False
        self.emit(
            "activity_decided",
            t,
            place=intent.place,
            purpose=intent.purpose,
            mode="car" if intent.destination_berth else "walk",
        )
        return "next_destination" if any(s.kind == "walk" for s in intent.steps) else "wait"

    def tick(self, t, xy, speed, occupancy, arrival_radius=0.6):
        intent = self.car_intent()
        if intent is None or self.paused_at is not None:
            return super().tick(t, xy, speed, occupancy, arrival_radius)
        visit = self.destination
        step = intent.step
        if step.kind == "walk":
            self.director.observe_departure(self.actor, tuple(xy))
            if self.phase == "scheduled":
                self.phase = "travelling"
                self.emit("journey_started", t, place=visit.place, purpose=visit.purpose)
            if math.dist(xy, step.xy) > arrival_radius or speed > 0.08:
                return "travel"
            done = self.director.acknowledge(self.actor, t, tuple(xy), speed, radius=arrival_radius)
            if done and intent.step.kind == "activity":
                self.phase = "queueing"
                self.emit("arrived", t, place=visit.place)
            return "wait" if done else "travel"
        if step.kind in ("board", "drive", "alight"):
            return step.kind
        # The final stay: the inherited visit bookkeeping with the Director's service clock.
        if self.phase == "queueing":
            if not occupancy.acquire(visit.place, self.actor, visit.capacity):
                return "wait"
            self.phase = "servicing"
            self.service_started_s = t
            self.emit("visit_started", t, place=visit.place, purpose=visit.purpose)
        if math.dist(xy, step.xy) > arrival_radius or speed > 0.08:
            occupancy.release(visit.place, self.actor)
            self.service_started_s = None
            self.phase = "travelling"
            self.emit("visit_interrupted", t, place=visit.place)
            return "travel"
        if not self.director.acknowledge(self.actor, t, tuple(xy), speed, radius=arrival_radius):
            return "wait"
        occupancy.release(visit.place, self.actor)
        self.emit("visit_completed", t, place=visit.place, purpose=visit.purpose)
        if self.actor in self.director.active:
            raise RuntimeError("Car journey and activity clock disagree")
        return self._choose(t, xy)
