"""Measured recurring travel, independent of population labels and tape horizon.

This module observes a completed recording; it never schedules departures.
An effective departure alone does not establish a completed journey. Parking
events and measured endpoint positions are required, alongside physical QA.
"""

from __future__ import annotations

import math

import numpy as np


def _driver_observations(by_vehicle, activity, duration):
    """Report causal journeys independently of optional recurrence thresholds.

    Short legal trips can be real activities. The legacy effective-distance and
    minimum-fleet gates must not decide whether their evidence is inspected.
    This joins recorded movement to Director observations; it is not a camera,
    collision, support or publication certificate.
    """
    rows = []
    for aid, trips in by_vehicle.items():
        ordered = sorted(trips, key=lambda t: t["t0_s"])
        completed, matched = [], []
        for number, trip in enumerate(ordered):
            end = trip.get("t1_s")
            point = np.asarray(trip.get("end_xy_m", []), float)
            if not (
                trip.get("status") == "completed"
                and end is not None
                and 0 <= trip["t0_s"] < end < duration
                and trip.get("legal") is True
                and trip.get("moving_s", 0.0) > 0
                and point.shape == (2,)
                and np.isfinite(point).all()
            ):
                continue
            completed.append(trip)
            chain = (
                activity.match(
                    trip, next_departure_s=ordered[number + 1]["t0_s"] if number + 1 < len(ordered) else math.inf
                )
                if activity is not None
                else None
            )
            if chain is not None:
                matched.append(chain)
        rows.append(
            dict(
                vehicle=aid,
                completed_legal_moving_trips=len(completed),
                completed_driver_activities=len(matched) if activity is not None else None,
                trips_without_completed_activity=len(completed) - len(matched) if activity is not None else None,
                evidence=matched,
            )
        )
    return dict(
        schema="awr.life.driver_activity_observations.v1",
        status="checked" if activity is not None else "unavailable",
        completed_legal_moving_trips=sum(r["completed_legal_moving_trips"] for r in rows),
        completed_driver_activities=sum(r["completed_driver_activities"] for r in rows)
        if activity is not None
        else None,
        vehicles_with_completed_activity=sum(bool(r["evidence"]) for r in rows) if activity is not None else None,
        vehicles=rows,
        admitted=False,
        scope="Fully recorded walk/board/drive/alight/walk/service chains matched to measured legal trips. "
        "No minimum distance, fleet size or recurrence threshold. Missing activity logs are unknown, not zero. "
        "Warm-up, partial chains and unfinished service do not qualify; physical and camera checks are separate.",
    )


def recurring_fleet(
    trips,
    vehicle_ids,
    moving,
    *,
    tick_s,
    min_trips=2,
    min_destinations=2,
    destination_separation_m=20.0,
    require_activity_chain=False,
    activity_events=(),
    activity_places=(),
):
    moving = np.asarray(moving, bool)
    vehicle_ids = list(vehicle_ids)
    if (
        moving.ndim != 2
        or moving.shape[1] != len(vehicle_ids)
        or not isinstance(require_activity_chain, bool)
        or len(set(vehicle_ids)) != len(vehicle_ids)
        or not len(moving)
        or not math.isfinite(tick_s)
        or tick_s <= 0
        or not isinstance(min_trips, int)
        or min_trips < 2
        or not isinstance(min_destinations, int)
        or min_destinations < 2
        or not math.isfinite(destination_separation_m)
        or destination_separation_m <= 0
    ):
        raise ValueError("Invalid recurring fleet audit inputs")
    duration = len(moving) * tick_s
    from .activity.travel_evidence import TravelEvidence

    # Observe supplied activity evidence even when recurrence is not a gate.
    # Keep historical gate semantics separate from this descriptive evidence.
    observed_activity = (
        TravelEvidence(activity_events, activity_places, duration) if activity_events and activity_places else None
    )
    activity = (
        (observed_activity or TravelEvidence(activity_events, activity_places, duration))
        if require_activity_chain
        else None
    )
    by_vehicle = {aid: [] for aid in vehicle_ids}
    for trip in trips:
        if trip["vehicle"] not in by_vehicle:
            raise ValueError("Trip refers to an unknown vehicle")
        by_vehicle[trip["vehicle"]].append(trip)
    rows = []
    for col, aid in enumerate(vehicle_ids):
        complete, destinations, groups, chains, rejected = [], [], set(), [], []
        ordered = sorted(by_vehicle[aid], key=lambda t: t["t0_s"])
        for number, tr in enumerate(ordered):
            # Only fully observed, purposeful, legal journeys count. Warm-up
            # arrivals and a trip unfinished at the final frame cannot qualify.
            end = tr.get("t1_s")
            point = np.asarray(tr.get("end_xy_m", []), float)
            if not (
                tr.get("effective")
                and tr.get("status") == "completed"
                and end is not None
                and 0 <= tr["t0_s"] < end < duration
                and tr.get("legal") is True
                and tr.get("end_purpose")
                and tr.get("moving_s", 0.0) > 0
                and point.shape == (2,)
                and np.isfinite(point).all()
            ):
                continue
            chain = (
                activity.match(
                    tr, next_departure_s=ordered[number + 1]["t0_s"] if number + 1 < len(ordered) else math.inf
                )
                if activity
                else None
            )
            if activity and chain is None:
                rejected.append(dict(t0_s=tr["t0_s"], t1_s=end, reason="missing_completed_driver_activity_chain"))
                continue
            complete.append(tr)
            if chain:
                chains.append(chain)
            # Conservative spatial deduplication in chronological order: a new
            # berth a few metres along the same frontage is not a new destination.
            if (chain is None or chain["destination_group"] not in groups) and all(
                np.linalg.norm(point - p) >= destination_separation_m for p in destinations
            ):
                destinations.append(point)
                if chain:
                    groups.add(chain["destination_group"])
        rows.append(
            dict(
                vehicle=aid,
                completed_effective_legal_trips=len(complete),
                distinct_destinations=len(destinations),
                destination_xy_m=[p.tolist() for p in destinations],
                qualified=len(complete) >= min_trips and len(destinations) >= min_destinations,
                moving_s=round(float(moving[:, col].sum() * tick_s), 3),
                activity_chain_evidence=chains,
                activity_chain_rejections=rejected,
                completed_trip_evidence=[
                    dict(t0_s=t["t0_s"], t1_s=t["t1_s"], end_xy_m=t["end_xy_m"], purpose=t["end_purpose"])
                    for t in complete
                ],
            )
        )
    idle = ~moving.any(axis=1)
    edges = np.flatnonzero(np.diff(np.r_[False, idle, False]))
    longest = max((b - a for a, b in zip(edges[::2], edges[1::2], strict=False)), default=0) * tick_s
    minutes = []
    journeys = [t for ts in by_vehicle.values() for t in ts]
    for minute in range(math.ceil(duration / 60.0)):
        lo, hi = minute * 60.0, min((minute + 1) * 60.0, duration)
        block = moving[int(math.ceil(lo / tick_s)) : int(math.ceil(hi / tick_s))]
        minutes.append(
            dict(
                minute=minute,
                unique_moving_vehicles=int(block.any(axis=0).sum()),
                mean_moving_vehicles=round(float(block.sum(axis=1).mean()), 3) if len(block) else 0.0,
                departures=sum(lo <= t["t0_s"] < hi for t in journeys),
                arrivals=sum(t.get("t1_s") is not None and lo <= t["t1_s"] < hi for t in journeys),
            )
        )
    return dict(
        schema="awr.life.recurring_fleet.v1",
        duration_s=duration,
        rule=dict(
            min_completed_trips=min_trips,
            min_distinct_destinations=min_destinations,
            destination_separation_m=destination_separation_m,
            require_completed_driver_activity_chain=bool(require_activity_chain),
            window="fully observed journeys only; warm-up and unfinished trips excluded",
        ),
        qualified_vehicles=sum(r["qualified"] for r in rows),
        driver_activities=_driver_observations(by_vehicle, observed_activity, duration),
        longest_no_driving_s=round(float(longest), 3),
        per_minute=minutes,
        vehicles=rows,
    )
