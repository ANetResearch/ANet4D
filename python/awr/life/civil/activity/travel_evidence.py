"""Join observed car journeys to completed driver activities.

This is semantic evidence in addition to tape geometry and movement audits. A
parking label alone is insufficient. Legacy logs without transaction IDs fail
closed when a scenario requires this evidence. No event is generated here.
"""

import math
from collections import defaultdict


class TravelEvidence:
    def __init__(self, events, places, duration_s, *, tolerance_s=0.2, arrival_radius_m=0.6):
        self.duration = float(duration_s)
        self.tolerance = float(tolerance_s)
        self.radius = float(arrival_radius_m)
        if not all(math.isfinite(x) and x > 0 for x in (self.duration, self.tolerance, self.radius)):
            raise ValueError("Invalid travel evidence bounds")
        self.places = {p["id"]: p for p in places}
        if len(self.places) != len(places):
            raise ValueError("Duplicate activity place")
        grouped = defaultdict(list)
        for event in events:
            if event.get("intent_id") is not None:
                grouped[event["intent_id"]].append(event)
        self.by_vehicle = defaultdict(list)
        for ident, rows in sorted(grouped.items()):
            chain = self._chain(ident, rows)
            if chain is not None:
                self.by_vehicle[chain["vehicle"]].append(chain)

    def _chain(self, ident, rows):
        selected = [e for e in rows if e["kind"] == "activity_selected"]
        finished = [e for e in rows if e["kind"] == "activity_completed"]
        if len(selected) != 1 or len(finished) != 1:
            return None
        start, end = selected[0], finished[0]
        place = self.places.get(start.get("place"))
        berth = self.places.get(start.get("destination_berth"))
        if (
            start.get("mode") != "car"
            or not start.get("vehicle")
            or not place
            or not berth
            or place.get("kind") in ("berth", "parking", "parking_lot")
            or berth.get("kind") != "berth"
            or berth["id"] not in place.get("parking", ())
        ):
            return None
        if any(any(e.get(k) != start.get(k) for k in ("actor", "purpose", "place")) for e in rows):
            return None
        steps = start.get("steps", [])
        kinds = [s["kind"] for s in steps]
        if kinds != ["walk", "board", "drive", "alight", "walk", "activity"]:
            return None
        done = sorted((e for e in rows if e["kind"] == "step_completed"), key=lambda e: e["step_index"])
        if [e["step_index"] for e in done] != list(range(len(steps))):
            return None
        times = [start["t_s"], *[e["t_s"] for e in done], end["t_s"]]
        if (
            not all(math.isfinite(t) for t in times)
            or times[0] < 0
            or times[-1] >= self.duration
            or any(b < a for a, b in zip(times, times[1:], strict=False))
            or abs(end["t_s"] - done[-1]["t_s"]) > self.tolerance
        ):
            return None
        for i, (step, event) in enumerate(zip(steps, done, strict=False)):
            if event.get("step") != step["kind"] or event.get("entity") != step["entity"]:
                return None
            speed = event.get("speed_mps")
            if (
                speed is None
                or not math.isfinite(speed)
                or abs(speed) > 0.08
                or not self._near(event.get("xy"), step["xy"])
            ):
                return None
            if step["kind"] in ("board", "drive", "alight"):
                car_speed = event.get("vehicle_speed_mps")
                if (
                    car_speed is None
                    or not math.isfinite(car_speed)
                    or abs(car_speed) > 0.08
                    or not self._near(event.get("vehicle_xy"), step.get("partner_xy") or step["xy"])
                ):
                    return None
            service = event.get("service_started_s")
            dwell = step.get("duration_s", 0.0)
            if (
                service is None
                or not math.isfinite(service)
                or not math.isfinite(dwell)
                or dwell < 0
                or service < times[i] - self.tolerance
                or event["t_s"] - service + self.tolerance < dwell
            ):
                return None
        if (
            steps[2]["entity"] != start["vehicle"]
            or any(steps[i]["entity"] != start["actor"] for i in (0, 1, 3, 4, 5))
            or steps[-1]["duration_s"] <= 0
            or not self._near(steps[-1]["xy"], place["xy"])
            or not self._near(steps[2]["xy"], berth["xy"])
        ):
            return None
        return dict(
            intent_id=ident,
            vehicle=start["vehicle"],
            driver=start["actor"],
            place=place["id"],
            destination_group=place.get("group") or place["id"],
            purpose=start["purpose"],
            board_s=done[1]["t_s"],
            park_s=done[2]["t_s"],
            park_xy=done[2].get("vehicle_xy"),
            berth_xy=berth["xy"],
            alight_s=done[3]["t_s"],
            activity_started_s=done[-1]["service_started_s"],
            activity_completed_s=end["t_s"],
        )

    def _near(self, a, b):
        return (
            isinstance(a, (tuple, list))
            and isinstance(b, (tuple, list))
            and len(a) == len(b) == 2
            and all(math.isfinite(x) for x in (*a, *b))
            and math.dist(a, b) <= self.radius
        )

    def match(self, trip, *, next_departure_s=math.inf):
        """Exactly one completed service must follow this measured parking event."""
        end = trip.get("t1_s")
        if end is None:
            return None
        candidates = [
            c
            for c in self.by_vehicle.get(trip["vehicle"], ())
            if c["board_s"] - self.tolerance <= trip["t0_s"] < c["park_s"]
            and abs(c["park_s"] - end) <= self.tolerance
            and c["activity_completed_s"] <= next_departure_s + self.tolerance
            and self._near(c["park_xy"], trip.get("end_xy_m"))
            and self._near(c["berth_xy"], trip.get("end_xy_m"))
        ]
        return candidates[0] if len(candidates) == 1 else None
