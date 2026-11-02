"""Deterministic departure arbitration for already admitted vehicle routes.

Reservations cover the complete swept journey. This conservative first policy
serialises intersecting journeys, but independent corridors remain concurrent.
It is not a lane-following capacity model and never replaces live collision,
ground, driver, destination-capacity or actual-arrival checks in the backend.
"""

import math

from .curb_allocation import swept


class DepartureReservations:
    def __init__(self, routes):
        """routes maps stable vehicle IDs to (admitted geometry, actual size)."""
        self.corridors = {key: swept(route, size) for key, (route, size) in routes.items()}
        self.conflicts = {key: set() for key in self.corridors}
        keys = sorted(self.corridors)
        for i, key in enumerate(keys):
            for other in keys[i + 1 :]:
                if self.corridors[key].intersects(self.corridors[other]):
                    self.conflicts[key].add(other)
                    self.conflicts[other].add(key)
        self.active = set()
        self.pending = {}
        self.last_time = -math.inf

    def request(self, vehicle, now):
        if vehicle not in self.corridors:
            raise ValueError("Departure needs registered admitted geometry")
        if not math.isfinite(now) or now < self.last_time:
            raise ValueError("Traffic admission time must be finite and monotonic")
        self.last_time = now
        if vehicle in self.active:
            return True
        self.pending.setdefault(vehicle, (now, vehicle))
        conflict = self.conflicts[vehicle]
        if conflict & self.active:
            return False
        # An older conflicting waiter keeps its place when a reservation ends;
        # unrelated journeys are not held behind the head of a global queue.
        if any(other in conflict and order < self.pending[vehicle] for other, order in self.pending.items()):
            return False
        self.active.add(vehicle)
        del self.pending[vehicle]
        return True

    def release(self, vehicle):
        """Backend calls only after measured parking, never on nominal elapsed time."""
        if vehicle not in self.active:
            raise ValueError("Vehicle has no traffic reservation to release")
        self.active.remove(vehicle)
