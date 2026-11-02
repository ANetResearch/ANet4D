import math

import pytest

from awr.life.civil.activity import Activity, Actor, Director, Place, Route


def setup(*, busy=True, reachable=True, park_weight=1.0):
    places = [
        Place("home", (0.0, 0.0), "home", ("home-slot",)),
        Place("post", (100.0, 0.0), "post", ("post-slot",)),
        Place("park", (0.0, 100.0), "park", ("park-slot",)),
    ]

    def route(mode, start, end, actor):
        if not reachable:
            return None
        distance = math.dist(start, end)
        return Route(distance, distance, "measured:" + str((start, end)))

    d = Director(
        places,
        [
            Activity("post", ("post",), (10.0, 10.0), 600.0),
            Activity("rest", ("park",), (10.0, 10.0), 600.0, weight=park_weight),
        ],
        route,
        seed=17,
    )
    d.add_actor(Actor("resident", "home", completed={"post": -680.0, "rest": -132.0}))
    if busy:
        d.holders["post-slot"] = "other-resident"
    return d


def test_full_due_facility_allows_real_early_activity_without_erasing_pending_need():
    d = setup()
    before = d.actors["resident"].completed.copy()
    intent = d.choose("resident", 0.0, (0.0, 0.0))
    assert intent.purpose == "rest" and [s.kind for s in intent.steps] == ["walk", "activity"]
    assert d.actors["resident"].completed == before
    assert d.events[-1]["selection_reason"] == "available_early_activity_while_due_unavailable"
    assert d.events[-1]["deferred_ready_needs"] == {"post": 1.133}
    assert d.holders["post-slot"] == "other-resident" and d.holders["park-slot"] == "resident"
    assert not d.acknowledge("resident", 10.0, (0.0, 0.0), 0.0)  # no fake arrival
    assert d.acknowledge("resident", 100.0, (0.0, 100.0), 0.0)
    assert not d.acknowledge("resident", 100.0, (0.0, 100.0), 0.0)
    assert d.acknowledge("resident", 110.0, (0.0, 100.0), 0.0)
    assert d.actors["resident"].completed == {"post": -680.0, "rest": 110.0}


def test_available_due_activity_has_priority_even_over_high_weight_early_need():
    d = setup(busy=False, park_weight=1000.0)
    intent = d.choose("resident", 0.0, (0.0, 0.0))
    assert intent.purpose == "post"
    assert d.events[-1]["selection_reason"] == "due_activity_or_home"
    assert d.events[-1]["deferred_ready_needs"] == {}


def test_no_demand_remains_a_home_activity_not_compulsory_circulation():
    d = setup(busy=False)
    d.actors["resident"].completed = {"post": -1.0, "rest": -1.0}
    intent = d.choose("resident", 0.0, (0.0, 0.0))
    assert intent.purpose == "pause_at_home_street"
    assert [s.kind for s in intent.steps] == ["activity"]


def test_unreachable_or_full_early_activity_still_fails_instead_of_relabelling_wait():
    for blocked in ("route", "slot"):
        d = setup(reachable=blocked != "route")
        if blocked == "slot":
            d.holders["park-slot"] = "third-resident"
        assert d.choose("resident", 0.0, (0.0, 0.0)) is None
        assert d.events[-1]["early_fallback_attempted"]
        assert d.actors["resident"].completed == {"post": -680.0, "rest": -132.0}
        with pytest.raises(RuntimeError, match="no feasible continuation"):
            d.choose("resident", 135.0, (0.0, 0.0))
        assert not d.active


def test_actor_profile_still_limits_early_activities():
    d = setup()
    d.actors["resident"].activities = ("post",)
    assert d.choose("resident", 0.0, (0.0, 0.0)) is None
    assert "park-slot" not in d.holders
