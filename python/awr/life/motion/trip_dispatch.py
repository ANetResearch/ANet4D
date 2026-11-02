"""Event-driven finite vehicle trips over an authoritative shared backend.

Activity policies submit destinations; this layer never invents an itinerary,
return trip, dwell deadline or pose. Completed cars remain in physical occupancy.
The caller owns the one world tick and calls observe only after it is accepted.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Trip:
    id: str
    vehicle: str
    source: str
    target: str
    not_before_s: float
    purpose: str


class TripDispatcher:
    def __init__(self, driving, parked):
        self.driving = driving
        self.parked = dict(parked)
        if not self.parked or set(self.parked) - set(driving.current.names):
            raise ValueError("Known authoritative fleet required")
        if driving.active:
            raise ValueError("Dispatcher must start with parked cars")
        for name, node in self.parked.items():
            if node not in driving.graph.nodes or driving.graph.nodes[node]["kind"] != "berth":
                raise ValueError("Initial mapped berth required")
            row = driving.current.row(name)
            if not driving._at(row[[1, 2, 8]], driving.graph.nodes[node]["pose"], abs(row[11])):
                raise ValueError("Initial parked state must be physically observed")
        self.pending = {}
        self.active = {}
        self.used = set()
        self.completed = []
        self.events = []
        self.epoch = 0
        self.attempted = {}
        self.last_time = float(driving.current.time_s)

    def _clock(self):
        now = float(self.driving.current.time_s)
        if not math.isfinite(now) or now < self.last_time:
            raise ValueError("Monotonic authoritative clock required")
        self.last_time = now
        return now

    def submit(self, trip):
        if (
            not isinstance(trip, Trip)
            or any(
                not isinstance(v, str) or not v for v in (trip.id, trip.vehicle, trip.source, trip.target, trip.purpose)
            )
            or not math.isfinite(trip.not_before_s)
            or trip.id in self.used
            or trip.vehicle not in self.parked
            or trip.vehicle in self.pending
            or trip.vehicle in self.active
            or trip.source != self.parked[trip.vehicle]
            or trip.source == trip.target
            or trip.target not in self.driving.graph.nodes
            or self.driving.graph.nodes[trip.target]["kind"] != "berth"
        ):
            raise ValueError("Unique finite trip from an actually parked vehicle required")
        now = self._clock()
        self.used.add(trip.id)
        self.pending[trip.vehicle] = trip
        self.events.append(
            dict(
                kind="trip_requested",
                request=trip.id,
                vehicle=trip.vehicle,
                t_s=now,
                source=trip.source,
                target=trip.target,
                purpose=trip.purpose,
                not_before_s=trip.not_before_s,
            )
        )

    def occupancy_changed(self):
        """Caller signals a committed external parking/occupancy change."""
        self.epoch += 1

    def dispatch(self):
        now = self._clock()
        if self.driving.active != {n: e for n, (_, e) in self.active.items()}:
            raise ValueError("Another owner changed active fleet trips")
        ordered = sorted(self.pending.values(), key=lambda r: (r.not_before_s, r.vehicle, r.id))
        for trip in ordered:
            if trip.not_before_s > now + 1e-8 or self.attempted.get(trip.id) == self.epoch:
                continue
            self.attempted[trip.id] = self.epoch
            route = self.driving.plan(trip.vehicle, trip.source, trip.target)
            row = self.driving.current.row(trip.vehicle)
            if route is None or not self.driving.acquire(route.evidence, row[[1, 2, 8]], abs(row[11])):
                self.events.append(
                    dict(
                        kind="trip_wait",
                        request=trip.id,
                        vehicle=trip.vehicle,
                        t_s=now,
                        reason="no_currently_admissible_route_or_reservation",
                    )
                )
                continue
            self.active[trip.vehicle] = (trip, route.evidence)
            del self.pending[trip.vehicle]
            del self.parked[trip.vehicle]
            self.events.append(
                dict(
                    kind="trip_departed",
                    request=trip.id,
                    vehicle=trip.vehicle,
                    t_s=now,
                    source=trip.source,
                    target=trip.target,
                    evidence=route.evidence,
                    wait_s=max(0.0, now - trip.not_before_s),
                )
            )
        return dict(self.driving.active)

    def observe(self):
        now = self._clock()
        arrived = []
        if self.driving.active != {n: e for n, (_, e) in self.active.items()}:
            raise ValueError("Another owner changed active fleet trips")
        for name, (trip, evidence) in sorted(list(self.active.items())):
            row = self.driving.current.row(name)
            if not self.driving.can_arrive(evidence, row[[1, 2, 8]], abs(row[11]), require_executed=True):
                continue
            if not self.driving.arrive(evidence, row[[1, 2, 8]], abs(row[11])):
                raise RuntimeError("Observed trip completion lost its transaction")
            self.parked[name] = trip.target
            del self.active[name]
            self.attempted.pop(trip.id, None)
            event = dict(
                kind="trip_parked",
                request=trip.id,
                vehicle=name,
                t_s=now,
                target=trip.target,
                purpose=trip.purpose,
                evidence=evidence,
            )
            self.events.append(event)
            self.completed.append(event)
            arrived.append(trip)
            self.epoch += 1
        return arrived
