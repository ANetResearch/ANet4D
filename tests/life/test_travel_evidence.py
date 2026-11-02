"""Recorded causal evidence, not physical or real-map driving admission."""

import math
from copy import deepcopy
from dataclasses import asdict

import numpy as np

from awr.life.civil.activity import Activity, Actor, Director, Place, Route, Vehicle
from awr.life.civil.activity.travel_evidence import TravelEvidence
from awr.life.civil.fleet_audit import recurring_fleet


def completed_roundtrip():
    places = [
        Place("home", (0.0, 0.0), "home", ("home-slot",), parking=("home-bay",)),
        Place("shop", (100.0, 0.0), "shop", ("shop-slot",), parking=("shop-bay",)),
        Place("home-bay", (10.0, 0.0), "berth", ("hb",), access_xy=(10.0, 2.0)),
        Place("shop-bay", (90.0, 0.0), "berth", ("sb",), access_xy=(90.0, 2.0)),
    ]

    def route(mode, a, b, entity):
        length = math.dist(a, b)
        return Route(length, length / (4 if mode == "drive" else 1), "test-only")

    d = Director(
        places,
        [Activity("shopping", ("shop",), (10.0, 10.0), 200.0, modes=("car",))],
        route,
        home_dwell_s=(10.0, 10.0),
        errands_before_home=1,
    )
    d.add_actor(Actor("person", "home", vehicle="car", completed={"shopping": -1000.0}))
    d.add_vehicle(Vehicle("car", "person", "home-bay"))
    t, xy, trips = 1.0, (0.0, 0.0), []
    for _ in range(2):
        intent = d.choose("person", t, xy)
        assert intent.destination_berth
        trip = dict(vehicle="car", effective=True, legal=True, status="completed", moving_s=20.0)
        while "person" in d.active:
            step = intent.step
            t += step.route.duration_s if step.route else 1.0
            xy = step.xy
            vehicle_xy = step.xy if step.kind == "drive" else step.partner_xy
            kwargs = dict(vehicle_xy=vehicle_xy, vehicle_speed=0.0) if vehicle_xy else {}
            if step.duration_s:
                assert not d.acknowledge("person", t, xy, 0.0, **kwargs)
                t += step.duration_s
            assert d.acknowledge("person", t, xy, 0.0, **kwargs)
            if step.kind == "board":
                trip["t0_s"] = t + 0.1
            if step.kind == "drive":
                trip.update(t1_s=t, end_xy_m=list(step.xy), end_purpose=intent.purpose)
        trips.append(trip)
        t += 1.0
    return d.events, [asdict(p) for p in places], trips


def strict(events, places, trips):
    return recurring_fleet(
        trips,
        ["car"],
        np.ones((300, 1), bool),
        tick_s=1.0,
        require_activity_chain=True,
        activity_events=events,
        activity_places=places,
    )


def test_actual_director_roundtrip_has_distinct_transactions_and_completed_activities():
    events, places, trips = completed_roundtrip()
    out = strict(events, places, trips)
    assert out["qualified_vehicles"] == 1
    chains = out["vehicles"][0]["activity_chain_evidence"]
    assert len({c["intent_id"] for c in chains}) == 2
    assert [c["purpose"] for c in chains] == ["shopping", "return_home"]
    assert all(c["activity_completed_s"] > c["alight_s"] >= c["park_s"] for c in chains)


def test_park_labels_without_driver_and_service_evidence_do_not_qualify():
    events, places, trips = completed_roundtrip()
    assert strict([], places, trips)["qualified_vehicles"] == 0
    assert strict(events, [], trips)["qualified_vehicles"] == 0
    for kind in ("activity_selected", "activity_completed"):
        assert strict([e for e in events if e["kind"] != kind], places, trips)["qualified_vehicles"] == 0
    for step in ("board", "drive", "alight", "activity"):
        assert strict([e for e in events if e.get("step") != step], places, trips)["qualified_vehicles"] == 0


def test_mismatched_driver_destination_park_pose_and_short_service_are_rejected():
    original, places, trips = completed_roundtrip()
    changes = [
        dict(actor="wrong-person"),
        dict(place="wrong-shop"),
        dict(vehicle_xy=[999.0, 999.0]),
        dict(service_started_s=999.0),
        dict(xy=[999.0, 999.0]),
        dict(vehicle_speed_mps=3.0),
    ]
    for change in changes:
        events = deepcopy(original)
        next(e for e in events if e.get("step") == "drive").update(change)
        assert strict(events, places, trips)["qualified_vehicles"] == 0
    events = deepcopy(original)
    service = next(e for e in events if e.get("step") == "activity")
    service["service_started_s"] = service["t_s"] - 0.1
    assert strict(events, places, trips)["qualified_vehicles"] == 0
    wrong = deepcopy(places)
    next(p for p in wrong if p["id"] == "shop")["parking"] = []
    assert strict(original, wrong, trips)["qualified_vehicles"] == 0


def test_far_apart_bays_of_one_facility_are_not_distinct_destinations():
    events, places, trips = completed_roundtrip()
    for p in places:
        if p["kind"] != "berth":
            p["group"] = "one-facility"
    out = strict(events, places, trips)
    assert out["vehicles"][0]["completed_effective_legal_trips"] == 2
    assert out["vehicles"][0]["distinct_destinations"] == 1
    assert out["qualified_vehicles"] == 0


def test_duplicate_or_ambiguous_event_chains_and_departure_before_service_fail_closed():
    events, places, trips = completed_roundtrip()
    duplicate = next(e for e in events if e.get("step") == "board")
    assert strict([*events, duplicate], places, trips)["qualified_vehicles"] == 0
    evidence = TravelEvidence(events, places, 300.0)
    assert evidence.match(trips[0], next_departure_s=trips[0]["t1_s"] + 1.0) is None
    assert TravelEvidence(events, places, trips[0]["t1_s"] + 1.0).match(trips[0]) is None


def test_observations_do_not_depend_on_recurring_fleet_gate_or_distance_threshold():
    events, places, trips = completed_roundtrip()
    for trip in trips:
        trip["effective"] = False  # e.g. a legitimate short journey in a small map
    report = recurring_fleet(
        trips, ["car"], np.ones((300, 1), bool), tick_s=1.0, activity_events=events, activity_places=places
    )
    assert report["qualified_vehicles"] == 0  # old gate is not weakened
    observed = report["driver_activities"]
    assert observed["status"] == "checked"
    assert observed["completed_legal_moving_trips"] == 2
    assert observed["completed_driver_activities"] == 2
    assert observed["vehicles_with_completed_activity"] == 1
    assert not observed["admitted"]
    broken = [e for e in events if e.get("step") != "alight"]
    observed = recurring_fleet(
        trips, ["car"], np.ones((300, 1), bool), tick_s=1.0, activity_events=broken, activity_places=places
    )["driver_activities"]
    assert observed["completed_legal_moving_trips"] == 2
    assert observed["completed_driver_activities"] == 0
    assert observed["vehicles"][0]["trips_without_completed_activity"] == 2


def test_missing_logs_are_unknown_and_park_only_or_warmup_cannot_be_a_completed_chain():
    events, places, trips = completed_roundtrip()

    def observe(rows, ev=events, loc=places):
        return recurring_fleet(
            rows, ["car"], np.ones((300, 1), bool), tick_s=1.0, activity_events=ev, activity_places=loc
        )["driver_activities"]

    unknown = observe(trips, ev=[])
    assert unknown["status"] == "unavailable"
    assert unknown["completed_driver_activities"] is None
    for changed in (dict(t0_s=-1), dict(t1_s=None, status="under_way"), dict(moving_s=0), dict(legal=False)):
        assert observe([{**trips[0], **changed}])["completed_driver_activities"] == 0
    assert observe([])["completed_driver_activities"] == 0
