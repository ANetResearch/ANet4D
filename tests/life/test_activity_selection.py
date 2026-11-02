"""Demand diversity is not physical admission: test choices and unchanged contracts."""

import math
from collections import Counter

import pytest

from awr.life.civil.activity import Activity, Actor, Director, Place, Route, Vehicle
from awr.life.civil.activity.selection import Selection

POLICY = dict(kind="diverse_feasible_v1", mode_weights=dict(walk=1.0, car=1.0))


def world(seed=0, *, prune=True, drive=True, closed=False, duplicates=1, walk=True):
    calls = []

    def route(mode, start, end, entity):
        calls.append((mode, start, end, entity))
        if (mode == "drive" and not drive) or (mode == "walk" and not walk):
            return None
        distance = math.dist(start, end)
        # Deliberately slower than walking: feasible does not mean time optimal.
        return Route(distance, distance * (3.0 if mode == "drive" else 1.0), "fixture-route")

    places = [
        Place("home", (0.0, 0.0), "home", ("h",), parking=("home-bay",)),
        Place("home-bay", (5.0, 0.0), "berth", ("hb",), access_xy=(5.0, 2.0)),
        Place("shop-bay", (90.0, 0.0), "berth", ("sb",), access_xy=(90.0, 2.0)),
    ]
    for i in range(duplicates):
        places.append(
            Place(
                f"shop-{i}",
                (100.0, 0.0),
                "shop",
                (f"s{i}",),
                parking=("shop-bay",),
                group="shop",
                closes_s=20.0 if closed else math.inf,
            )
        )
    places.append(Place("park", (300.0, 0.0), "park", ("p",), group="park", parking=("shop-bay",)))
    rules = [Activity("visit", ("shop", "park"), (10.0, 10.0), 300.0, modes=("walk", "car"))]
    d = Director(
        places,
        rules,
        route,
        seed=seed,
        route_pruning=prune,
        selection=POLICY,
        home_dwell_s=(5.0, 5.0),
        errands_before_home=1,
    )
    d.add_actor(Actor("person", "home", vehicle="car", completed={"visit": -1000.0}))
    d.add_vehicle(Vehicle("car", "person", "home-bay"))
    return d, calls


def test_seeded_demand_reaches_both_modes_and_facilities_without_time_optimality():
    modes = Counter()
    facilities = Counter()
    for seed in range(100):
        d, _ = world(seed)
        selected = d.choose("person", 0.0, (0.0, 0.0))
        policy = next(r for r in d.events if r["kind"] == "activity_selection_policy")
        mode = "car" if selected.destination_berth else "walk"
        assert mode == policy["proposed_mode_order"][0]
        assert not policy["used_alternative"]
        modes[mode] += 1
        facilities[d.places[selected.place].group] += 1
        if mode == "car":
            assert any(s.kind == "board" for s in selected.steps)
            assert any(s.kind == "alight" for s in selected.steps)
    assert min(modes.values()) > 20 and set(modes) == {"walk", "car"}
    assert min(facilities.values()) > 20 and set(facilities) == {"shop", "park"}


def test_unavailable_preference_falls_back_without_reservation_leak():
    seed = next(s for s in range(100) if Selection(POLICY).mode_order(s, "person", 0, True)[0] == "car")
    d, calls = world(seed, drive=False)
    selected = d.choose("person", 0.0, (0.0, 0.0))
    assert selected.destination_berth is None
    evidence = next(r for r in d.events if r["kind"] == "activity_selection_policy")
    assert evidence["used_alternative"] and evidence["selected_mode"] == "walk"
    assert "sb" not in d.holders and d.holders["hb"] == "car"
    assert any(c[0] == "drive" for c in calls)
    d, _ = world(seed, walk=False)
    assert d.choose("person", 0.0, (0.0, 0.0)) is None
    assert d.holders == {"h": "person", "hb": "car"}


def test_pruned_and_exhaustive_demand_are_identical_and_do_not_rescan_on_observation():
    for seed in range(20):
        a, calls = world(seed, prune=True)
        b, all_calls = world(seed, prune=False)
        assert a.choose("person", 0.0, (0.0, 0.0)) == b.choose("person", 0.0, (0.0, 0.0))
        assert a._rng["person"].getstate() == b._rng["person"].getstate()
        n = len(calls)
        assert n <= len(all_calls)
        assert a.choose("person", 0.0, (0.0, 0.0)) is a.active["person"]
        assert len(calls) == n


def test_capacity_closing_return_and_group_replication_still_apply():
    for seed in range(20):
        a, _ = world(seed)
        b, _ = world(seed, duplicates=20)
        x = a.choose("person", 0.0, (0.0, 0.0))
        y = b.choose("person", 0.0, (0.0, 0.0))
        assert a.places[x.place].group == b.places[y.place].group
        assert bool(x.destination_berth) == bool(y.destination_berth)
        c, _ = world(seed, closed=True)
        assert c.choose("person", 0.0, (0.0, 0.0)).place == "park"
        d, _ = world(seed)
        d.holders.update(s0="other", p="other2")
        assert d.choose("person", 0.0, (0.0, 0.0)) is None
        e, _ = world(seed)
        e.actors["person"].visits_since_home = 1
        # The next desired facility cannot override the mandatory return leg.
        e.actors["person"].last_place = "shop-0"
        intent = e.choose("person", 0.0, (100.0, 0.0))
        assert intent.purpose == "return_home"


def test_selection_is_horizon_and_call_order_independent_with_configurable_demand():
    p = Selection(POLICY)
    prefix = [p.mode_order(8, "p", i, True) for i in range(100)]
    assert prefix == [p.mode_order(8, "p", i, True) for i in range(200)][:100]
    assert dict((i, p.mode_order(8, "p", i, True)) for i in reversed(range(100))) == dict(enumerate(prefix))
    q = Selection(dict(kind="diverse_feasible_v1", mode_weights=dict(walk=1, car=9)))
    assert sum(q.mode_order(8, "p", i, True)[0] == "car" for i in range(1000)) > 800
    assert p.mode_order(8, "p", 0, False) == ("walk",)


def test_grounded_adapter_consumes_policy_without_granting_driving_capability():
    from types import SimpleNamespace

    import numpy as np

    from awr.life.civil.activity.grounded import GroundedActivities

    g = GroundedActivities(
        SimpleNamespace(),
        dict(sites=[]),
        np.ones((2, 2), bool),
        dict(activity_engine=dict(modes=["walk"], selection=POLICY)),
        17,
    )
    assert g.director.selection.kind == "diverse_feasible_v1"
    assert g.modes == ("walk",) and not g.director.vehicles
    assert g.director.selection.mode_order(17, "p", 0, False) == ("walk",)


@pytest.mark.parametrize(
    "value",
    [
        {},
        dict(kind="other", mode_weights={"walk": 1, "car": 1}),
        dict(kind="diverse_feasible_v1", mode_weights={"walk": 1, "car": 0}),
        dict(kind="diverse_feasible_v1", mode_weights={"walk": True, "car": 1}),
        dict(kind="diverse_feasible_v1", mode_weights={"walk": 1, "car": float("nan")}),
    ],
)
def test_invalid_policy_fails_before_routes(value):
    with pytest.raises(ValueError):
        Director([], [], lambda *a: None, selection=value)
