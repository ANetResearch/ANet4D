"""Tours ask the backend for a tour route and are taken by car, under either selection policy."""

import math

import pytest

from awr.life.civil.activity import Activity, Actor, Director, Place, Route, Vehicle


def town(selection=None, *, rules=None, seed=17, calls=None):
    def route(mode, start, end, entity):
        if calls is not None:
            calls.append(mode)
        length = math.dist(start, end)
        return Route(length, length / (4 if mode in ("drive", "tour") else 1), f"test-{mode}")

    places = [
        Place("h", (0.0, 0.0), "home", ("home-slot",), parking=("h-bay",)),
        Place("shop", (100.0, 0.0), "shop", ("shop-slot",), parking=("shop-bay",)),
        Place("h-bay", (10.0, 0.0), "berth", ("h-bay-slot",), access_xy=(10.0, 2.0)),
        Place("shop-bay", (90.0, 0.0), "berth", ("shop-bay-slot",), access_xy=(90.0, 2.0)),
    ]
    rules = rules or [Activity("shopping", ("shop",), (10.0, 10.0), 200.0, modes=("walk", "car"))]
    d = Director(places, rules, route, seed=seed, home_dwell_s=(10.0, 10.0), errands_before_home=1, selection=selection)
    a = Actor("p", "h", vehicle="car")
    d.add_actor(a)
    a.completed.update({r.purpose: -1000.0 for r in rules})
    d.add_vehicle(Vehicle("car", "p", "h-bay"))
    return d


def mode(intent):
    return "car" if intent.destination_berth else "walk"


def test_a_tour_activity_asks_for_a_tour_route_and_takes_the_car():
    calls = []
    rules = [Activity("drive_around", ("shop",), (10.0, 10.0), 60.0, modes=("car",), tour=True)]
    for selection in (None, dict(kind="diverse_feasible_v1", mode_weights=dict(walk=1.0, car=1.0))):
        calls.clear()
        intent = town(selection, rules=rules, calls=calls).choose("p", 0.0, (0.0, 0.0))
        assert intent.purpose == "drive_around" and mode(intent) == "car" and "tour" in calls
    with pytest.raises(ValueError):
        town(rules=[Activity("drive_around", ("shop",), (10.0, 10.0), 60.0, modes=("walk", "car"), tour=True)])


def test_a_car_owner_can_become_a_cruising_driver():
    rules = [
        Activity("shopping", ("shop",), (10.0, 10.0), 200.0, modes=("walk", "car")),
        Activity("drive_around", ("shop",), (10.0, 10.0), 60.0, roles=("cruiser",), modes=("car",), tour=True),
    ]
    d = town(rules=rules)
    with pytest.raises(ValueError):
        d.assign_role("p", "cruiser", ("shopping",))
    d.assign_role("p", "cruiser", ("drive_around",), t=0.0)
    actor = d.actors["p"]
    assert actor.role == "cruiser" and actor.activities == ("drive_around",) and actor.completed["drive_around"] <= 0.0
    intent = d.choose("p", 60.0, (0.0, 0.0))
    assert intent.purpose == "drive_around" and mode(intent) == "car"


def test_cruisers_are_the_owners_whose_part_of_the_map_reaches_the_most_destinations():
    from awr.life.civil.grounded import choose_cruisers

    parts = {"p1": ["f", "g"], "p2": ["a", "b", "c", "d", "e", "lane"], "p3": ["h"], "p4": ["a", "b", "c", "d", "e"]}
    homes = {"p1": "f", "p2": "a", "p3": "h", "p4": "b"}
    facilities = {"a", "b", "c", "d", "e", "f", "g"}
    cruisers, reach = choose_cruisers(homes, facilities, 2, 2, parts.get)
    # p2 and p4 reach four other facility berths each with the other cars parked; p1 reaches one, p3 none.
    assert cruisers == ["p2", "p4"] and reach == {"p1": 1, "p2": 4, "p3": 0, "p4": 4}
    assert choose_cruisers(homes, facilities, 8, 2, parts.get)[0] == ["p2", "p4"]
    parts["p4"] = ["b", "c"]  # another car parked across p4's street
    assert choose_cruisers(homes, facilities, 2, 2, parts.get)[0] == ["p2"]


def test_no_car_ends_a_journey_at_the_home_berth_of_another_car_away_from_home():
    # Cruise5 r11f v10: two cruisers in turn parked at the home berth of a third car while it was away; its owner,
    # due home, had no berth to return to.
    def route(mode, start, end, entity):
        length = math.dist(start, end)
        return Route(length, length / (4 if mode in ("drive", "tour") else 1), f"test-{mode}")

    places = [
        Place("h", (0.0, 0.0), "home", ("home-slot",), parking=("h-bay",)),
        Place("g", (200.0, 0.0), "home", ("g-slot",), parking=("g-bay",)),
        Place("shop", (100.0, 0.0), "shop", ("shop-slot",), parking=("g-bay", "shop-bay")),
        Place("h-bay", (10.0, 0.0), "berth", ("h-bay-slot",), access_xy=(10.0, 2.0)),
        Place("g-bay", (110.0, 0.0), "berth", ("g-bay-slot",), access_xy=(110.0, 2.0)),
        Place("shop-bay", (90.0, 0.0), "berth", ("shop-bay-slot",), access_xy=(90.0, 2.0)),
        Place("far-bay", (300.0, 0.0), "berth", ("far-bay-slot",), access_xy=(300.0, 2.0)),
    ]
    rules = [Activity("shopping", ("shop",), (10.0, 10.0), 200.0, modes=("car",))]
    d = Director(places, rules, route, seed=17, home_dwell_s=(10.0, 10.0), errands_before_home=1)
    for actor, home, car, bay in (("p", "h", "car", "h-bay"), ("q", "g", "qcar", "g-bay")):
        d.add_actor(Actor(actor, home, vehicle=car))
        d.actors[actor].completed.update({r.purpose: -1000.0 for r in rules})
        d.add_vehicle(Vehicle(car, actor, bay))
    assert d.home_berths == {"h-bay": "car", "g-bay": "qcar"}
    # q's car is away from home: its berth is free, but kept for it.
    del d.holders["g-bay-slot"]
    d.vehicles["qcar"].berth = "far-bay"
    intent = d.choose("p", 0.0, (0.0, 0.0))
    assert intent.destination_berth == "shop-bay" and d.car_rejections.get("home_berth_of_another_car") == 1
