"""Purposeful finite itineraries. Motion, resource occupancy and intent are separate.

A blocked path never substitutes a random destination. Arrival requires both the
correct place and stopped motion; visits finish only after their actual dwell.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Visit:
    place: str
    xy: tuple[float, float]
    purpose: str
    dwell_s: float
    capacity: int = 1


class Occupancy:
    def __init__(self):
        self.occupants = {}

    def acquire(self, place, actor, capacity):
        people = self.occupants.setdefault(place, set())
        if actor in people:
            return True
        if len(people) >= capacity:
            return False
        people.add(actor)
        return True

    def release(self, place, actor):
        self.occupants.get(place, set()).discard(actor)


@dataclass
class Agenda:
    actor: str
    role: str
    visits: list[Visit]
    departure_s: float = 0.0
    index: int = 0
    phase: str = "scheduled"
    service_started_s: float | None = None
    paused_at: float | None = None
    events: list = field(default_factory=list)

    def __post_init__(self):
        if not self.visits:
            raise ValueError("An itinerary needs a destination")
        if any(v.dwell_s < 0 or not np.isfinite(v.xy).all() or v.capacity < 1 for v in self.visits):
            raise ValueError("Invalid visit")
        if any(a.place == b.place for a, b in zip(self.visits, self.visits[1:], strict=False)):
            raise ValueError("Adjacent visits cannot be the same place")

    @property
    def destination(self):
        return None if self.phase == "finished" else self.visits[self.index]

    def emit(self, kind, t, **kw):
        self.events.append(dict(kind=kind, actor=self.actor, t_s=round(float(t), 3), visit_index=self.index, **kw))

    def blocked(self, t, reason):
        self.emit("journey_delayed", t, place=self.destination.place, reason=reason)

    @property
    def place(self):
        return self.visits[min(self.index, len(self.visits) - 1)].place

    def pause(self, t, reason):
        """Step aside on request without faking arrival, service or completion.

        While paused the agenda state is frozen: a scheduled departure waits, a
        servicing visit does not accumulate dwell, a queueing person keeps its
        place and a finished itinerary stays finished. The caller moves the body and must bring it
        back to the same place before resuming.
        """
        if self.paused_at is not None:
            return False
        self.paused_at = float(t)
        self.emit("stepped_aside", t, place=self.place, phase=self.phase, reason=reason)
        return True

    def resume(self, t):
        if self.paused_at is None:
            return
        if self.service_started_s is not None:
            self.service_started_s += float(t) - self.paused_at
        self.emit(
            "returned_to_place", t, place=self.place, phase=self.phase, aside_s=round(float(t) - self.paused_at, 1)
        )
        self.paused_at = None

    def tick(self, t, xy, speed, occupancy, arrival_radius=0.6):
        if self.paused_at is not None:
            return "paused"
        if self.phase == "finished":
            return "finished"
        v = self.destination
        if self.phase == "scheduled":
            if t < self.departure_s:
                return "wait"
            self.phase = "travelling"
            self.emit("journey_started", t, place=v.place, purpose=v.purpose)
        if self.phase == "travelling":
            if np.linalg.norm(np.asarray(xy) - v.xy) > arrival_radius or speed > 0.08:
                return "travel"
            self.phase = "queueing"
            self.emit("arrived", t, place=v.place)
        if self.phase == "queueing":
            if not occupancy.acquire(v.place, self.actor, v.capacity):
                return "wait"
            self.phase = "servicing"
            self.service_started_s = t
            self.emit("visit_started", t, place=v.place, purpose=v.purpose)
        if self.phase == "servicing":
            # A pushed/displaced actor does not keep accumulating service time.
            if np.linalg.norm(np.asarray(xy) - v.xy) > arrival_radius or speed > 0.08:
                occupancy.release(v.place, self.actor)
                self.service_started_s = None
                self.phase = "travelling"
                self.emit("visit_interrupted", t, place=v.place)
                return "travel"
            if t - self.service_started_s < v.dwell_s:
                return "wait"
            occupancy.release(v.place, self.actor)
            self.emit("visit_completed", t, place=v.place, purpose=v.purpose)
            self.index += 1
            if self.index == len(self.visits):
                self.phase = "finished"
                self.emit("itinerary_completed", t)
                return "finished"
            self.phase = "scheduled"
            self.departure_s = t
            return "next_destination"
        return "wait"

    def summary(self):
        return dict(
            actor=self.actor,
            role=self.role,
            phase=self.phase,
            completed_visits=self.index,
            planned_visits=len(self.visits),
            events=self.events,
        )
