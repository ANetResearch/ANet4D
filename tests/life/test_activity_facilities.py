"""Facility choice should not grow with duplicate standing-space samples."""

import math

from awr.life.civil.activity import Activity, Actor, Director, Place, Route


def choose(slots, *, unavailable=(), unreachable=(), pruning=True, speed_bound=True):
    home = Place("home", (0.0, 0.0), "home", ("home-slot",))
    facilities = [Place(f"park-{i:04}", (100.0 + i, 0.0), "park", (f"p-{i}",), group="park") for i in range(slots)]
    facilities += [Place("cafe", (110.0, 0.0), "cafe", ("cafe-slot",), group="cafe")]
    calls = []

    def route(mode, start, end, actor):
        calls.append(end)
        if end in unreachable:
            return None
        distance = math.dist(start, end)
        return Route(distance, distance, f"physical-proof:{end}")

    director = Director(
        [home, *facilities],
        [Activity("visit", ("park", "cafe"), (10.0, 10.0), 100.0)],
        route,
        seed=3,
        max_speed_mps={"walk": 1.0} if speed_bound else None,
        route_pruning=pruning,
        available=lambda p, a: p.id not in unavailable,
    )
    director.add_actor(Actor("resident", "home", completed={"visit": -1000.0}))
    intent = director.choose("resident", 0.0, (0.0, 0.0))
    return director, intent, calls


def test_dense_standing_samples_do_not_change_facility_choice_rng_or_routing_work():
    small, s, a = choose(1)
    dense, d, b = choose(1000)
    assert s == d
    assert small._rng["resident"].getstate() == dense._rng["resident"].getstate()
    assert a == b
    assert len(b) <= 2  # two facilities, not a thousand route proofs
    assert dense.events[-1]["choice_policy"] == dense.choice_policy


def test_unreachable_nearest_access_does_not_hide_a_farther_feasible_one():
    director, intent, calls = choose(
        100, unavailable=("cafe",), unreachable=tuple((float(x), 0.0) for x in range(100, 150))
    )
    assert intent.place == "park-0050"
    assert calls[-1] == (150.0, 0.0) and len(calls) == 51
    assert director.holders["p-50"] == "resident"
    assert all(f"p-{i}" not in director.holders for i in range(50))


def test_full_access_is_excluded_before_selecting_and_reserving_a_facility():
    director, intent, calls = choose(3, unavailable=("cafe", "park-0000", "park-0001"))
    assert intent.place == "park-0002" and calls == [(102.0, 0.0)]
    assert director.holders["p-2"] == "resident"


def test_nearest_access_policy_does_not_require_an_optional_speed_bound():
    _, intent, calls = choose(100, unavailable=("cafe",), speed_bound=False)
    assert intent.place == "park-0000" and calls == [(100.0, 0.0)]


def test_facility_pruning_matches_exhaustive_access_policy():
    d, a, _ = choose(100, unreachable=((100.0, 0.0),), pruning=True)
    e, b, _ = choose(100, unreachable=((100.0, 0.0),), pruning=False)
    assert a == b
    assert d._rng["resident"].getstate() == e._rng["resident"].getstate()
