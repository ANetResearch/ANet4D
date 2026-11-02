"""Resident drivers from a driving map: usable berths, home berths by measured walks, places and owned cars."""

import life_toytown as toy
import numpy as np
from shapely.geometry import box

from awr.life.civil.activity.director import Activity, Actor, Director, Place
from awr.life.civil.activity.driving_adapter import DrivingAdapter, strongly_connected_berths
from awr.life.civil.curb_allocation import footprint
from awr.life.civil.driving_graph import DrivingGraph, route_arrays

PERSON = np.r_[0.6, 0.6, 1.7]


def world():
    w = toy.build(drivers=3, walkers=0)
    carriageway = box(0.0, 0.0, 1.0, 1.0)  # anchors are compared with it only to prefer the kerb side
    adapter = DrivingAdapter(w.graph, toy.CAR, "toy", carriageway, w.walking, w.surface, 0.45, home_walk_m=40.0)
    return w, adapter


def test_berth_with_a_body_on_it_or_closing_a_passage_is_not_usable():
    w, adapter = world()
    berths = sorted(n for n, d in w.graph.nodes.items() if d["kind"] == "berth")
    assert strongly_connected_berths(w.graph) == set(berths)
    free = np.asarray(w.walking.free)
    usable = adapter.usable_berths([], free, None, lambda xy, yaw, size: True)
    assert sorted(usable) == berths
    pose = w.graph.nodes["berth:home1"]["pose"]
    parked = footprint(dict(xy=pose[:2], yaw=pose[2]), toy.CAR)
    w2, adapter2 = world()
    usable = adapter2.usable_berths([parked], free, None, lambda xy, yaw, size: True)
    assert "berth:home1" not in usable and adapter2.report["body_clear"] == len(berths) - 1
    w3, adapter3 = world()
    assert adapter3.usable_berths([], free, None, lambda xy, yaw, size: False) == {}


def test_home_berths_need_a_measured_walk_and_an_admitted_car():
    w, adapter = world()
    usable = adapter.usable_berths([], np.asarray(w.walking.free), None, lambda xy, yaw, size: True)
    homes = [(f"person-{k}", Place(f"h{k}", h["xy"], "home", (f"h{k}",))) for k, h in enumerate(w.homes)]
    far = ("person-far", Place("hfar", (5.0, 50.0), "home", ("hfar",)))

    def measured(a, b):
        return float(np.linalg.norm(np.subtract(a, b))) * 1.2

    placed = []
    chosen = adapter.assign(homes + [far], usable, measured, 10, lambda actor, berth: placed.append(berth) or True)
    assert [c[2] for c in chosen] == ["berth:home0", "berth:home1", "berth:home2"] == placed
    assert adapter.report["assignment"]["without_berth_in_reach"] == 1
    # No measured walk, or a car that does not fit: the resident stays a pedestrian.
    w, adapter = world()
    assert adapter.assign(homes, usable, lambda a, b: None, 10, lambda *_: True) == []
    assert adapter.report["assignment"]["without_walk"] == 3
    w, adapter = world()
    assert adapter.assign(homes, usable, measured, 10, lambda *_: False) == []
    assert adapter.report["assignment"]["car_not_admitted"] == 3


def test_bound_places_name_their_berths_and_owners_their_cars():
    w, adapter = world()
    usable = adapter.usable_berths([], np.asarray(w.walking.free), None, lambda xy, yaw, size: True)
    homes = [Place(f"h{k}", h["xy"], "home", (f"h{k}",)) for k, h in enumerate(w.homes)]
    cafe = Place("cafe", w.facilities[0]["xy"], "cafe", ("cafe",))
    director = Director(
        homes + [cafe], [Activity("coffee", ("cafe",), (60.0, 60.0), 600.0, modes=("walk", "car"))], lambda *a: None
    )
    for k, h in enumerate(homes):
        director.add_actor(Actor(f"person-{k}", h.id))
    chosen = [(f"person-{k}", h, f"berth:home{k}") for k, h in enumerate(homes)]
    adapter.bind(director, usable, chosen, [f"vehicle-{k}" for k in range(3)])
    # The home berth first, then nearby berths a returning driver can use when another car stands on it.
    assert director.places["h1"].parking[0] == "berth:home1" and set(director.places["h1"].parking) <= set(usable)
    assert len(set(director.places["h1"].parking)) == len(director.places["h1"].parking)
    assert director.places["h1"].parking == adapter.home_parking(homes[1], "berth:home1", usable)
    assert "berth:fac0" in director.places["cafe"].parking
    assert director.places["berth:fac0"].access_xy == usable["berth:fac0"][1]
    assert director.vehicles["vehicle-2"].owner == "person-2" and director.actors["person-2"].vehicle == "vehicle-2"
    assert director.vehicles["vehicle-2"].berth == "berth:home2"


def test_parking_rule_rejection_is_separate_from_geometric_berth_availability():
    w, adapter = world()
    for value in (False, None):
        assert (
            adapter.usable_berths([], np.asarray(w.walking.free), None, lambda *a: True, lambda *a: dict(legal=value))
            == {}
        )
        assert adapter.report["parking_rule_rejected"] > 0
    usable = adapter.usable_berths([], np.asarray(w.walking.free), None, lambda *a: True, lambda *a: dict(legal=True))
    assert usable and adapter.report["parking_rule_checked"]
    assert all(w.graph.nodes[b]["parking_legality"] == dict(legal=True, source="scene_parking_rules") for b in usable)


def test_facility_candidates_preserve_each_connected_road_component():
    """The closest lot cannot consume all candidates and hide another component.

    This only tests candidate enumeration, not parking admission or reachability
    with actual occupied bodies. Those remain the driving backend's checks.
    """
    graph = DrivingGraph()
    positions = {
        "lot-a": (1.0, 0.0),
        "lot-b": (2.0, 0.0),
        "road-a": (10.0, 0.0),
        "road-b": (11.0, 0.0),
        "far-a": (50.0, 0.0),
        "far-b": (51.0, 0.0),
    }
    for name, xy in positions.items():
        graph.add_node(name, (*xy, 0.0), kind="berth")
    for a, b in [("lot-a", "lot-b"), ("road-a", "road-b"), ("far-a", "far-b")]:
        for source, target in [(a, b), (b, a)]:
            points = np.array([positions[source], positions[target]])
            gear = np.array([1 if points[1, 0] > points[0, 0] else -1])
            graph.add_edge(source + "-" + target, source, target, route_arrays(points, np.zeros(2), gear), kind="lane")
    adapter = DrivingAdapter(
        graph, toy.CAR, "toy", box(-1, -1, 60, 10), None, None, 0.45, facility_walk_m=20.0, facility_berths=1
    )
    usable = {name: (graph.nodes[name]["pose"], xy) for name, xy in positions.items()}
    cafe = Place("cafe", (0.0, 0.0), "cafe", ("cafe",))
    assert adapter.parking_candidates(cafe, usable) == ("lot-a", "road-a")
    director = Director([cafe], [], lambda *args: None)
    adapter.bind(director, dict(reversed(list(usable.items()))), [], [])
    assert director.places["cafe"].parking == ("lot-a", "road-a")
    assert len(graph.components()) == 3  # no invented connection between lots


def test_in_vehicle_time_counts_with_its_factor_only_when_journeys_are_compared():
    import pytest

    from awr.life.civil.activity.director import Route, Step

    steps = [
        Step("walk", "p", (0.0, 0.0), route=Route(60.0, 60.0, "w")),
        Step("drive", "v", (0.0, 0.0), route=Route(600.0, 100.0, "d")),
        Step("walk", "p", (0.0, 0.0), route=Route(30.0, 30.0, "w")),
    ]
    plain = Director([], [], lambda *a: None)
    assert plain._perceived(steps, 198.0) == 198.0
    weighted = Director([], [], lambda *a: None, in_vehicle_time_factor=0.5)
    assert weighted._perceived(steps, 198.0) == pytest.approx(148.0)
    assert weighted._perceived(steps[:1], 60.0) == 60.0  # walking journeys are unchanged
    for bad in (0.0, 1.5, float("nan")):
        with pytest.raises(ValueError):
            Director([], [], lambda *a: None, in_vehicle_time_factor=bad)


def test_a_drive_shorter_than_the_minimum_is_not_offered():
    import math

    import pytest

    from awr.life.civil.activity.director import Route, Vehicle

    def route(mode, start, end, entity):
        d = math.dist(start, end)
        return Route(d, d / (6.0 if mode == "drive" else 1.0), mode) if d > 0 else None

    def director(min_drive_m):
        places = [
            Place("h", (0.0, 0.0), "home", ("h",), parking=("b0",)),
            Place("cafe", (40.0, 0.0), "cafe", ("cafe",), parking=("b1",)),
            Place("b0", (0.0, 2.0), "berth", ("b0",), access_xy=(0.0, 1.0)),
            Place("b1", (40.0, 2.0), "berth", ("b1",), access_xy=(40.0, 1.0)),
        ]
        d = Director(
            places,
            [Activity("coffee", ("cafe",), (60.0, 60.0), 600.0, modes=("walk", "car"))],
            route,
            min_drive_m=min_drive_m,
        )
        d.add_actor(Actor("p", "h", vehicle="v"))
        d.add_vehicle(Vehicle("v", "p", "b0"))
        return d

    near = director(60.0)
    assert near._journey(near.actors["p"], (0.0, 0.0), near.places["cafe"], "car", 0.0, 60.0) is None
    assert near.car_rejections == {"drive_shorter_than_minimum": 1}
    far = director(30.0)
    steps, bid, _ = far._journey(far.actors["p"], (0.0, 0.0), far.places["cafe"], "car", 0.0, 60.0)
    assert bid == "b1" and [s.kind for s in steps] == ["walk", "board", "drive", "alight", "walk"]
    for bad in (-1.0, float("inf")):
        with pytest.raises(ValueError):
            Director([], [], route, min_drive_m=bad)


def test_a_returning_driver_parks_near_home_when_its_berth_is_taken():
    import math

    from awr.life.civil.activity.director import Route, Vehicle

    def route(mode, start, end, entity):
        d = math.dist(start, end)
        return Route(d, d / (6.0 if mode == "drive" else 1.0), mode) if d > 0 else None

    places = [
        Place("h", (0.0, 0.0), "home", ("h",), parking=("b0", "b2")),
        Place("g", (0.0, 30.0), "home", ("g",), parking=("b3",)),
        Place("cafe", (200.0, 0.0), "cafe", ("cafe",), parking=("b1",)),
        Place("b0", (0.0, 2.0), "berth", ("b0",), access_xy=(0.0, 1.0)),
        Place("b1", (200.0, 2.0), "berth", ("b1",), access_xy=(200.0, 1.0)),
        Place("b2", (12.0, 2.0), "berth", ("b2",), access_xy=(12.0, 1.0)),
        Place("b3", (0.0, 32.0), "berth", ("b3",), access_xy=(0.0, 31.0)),
    ]
    d = Director(places, [Activity("coffee", ("cafe",), (60.0, 60.0), 600.0, modes=("walk", "car"))], route)
    d.add_actor(Actor("p", "h", vehicle="v"))
    d.add_actor(Actor("q", "g", vehicle="w"))
    d.add_vehicle(Vehicle("v", "p", "b1"))
    d.add_vehicle(Vehicle("w", "q", "b0"))  # q's car stands on p's berth
    steps, bid, _ = d._journey(d.actors["p"], (200.0, 0.0), d.places["h"], "car", 0.0, 60.0)
    assert bid == "b2" and steps[-1].xy == (0.0, 0.0)


def test_a_car_journey_is_not_offered_where_parking_closes_another_cars_way_home():
    import math

    from awr.life.civil.activity.director import Intent, Route, Step, Vehicle

    def route(mode, start, end, entity):
        d = math.dist(start, end)
        return Route(d, d / (6.0 if mode == "drive" else 1.0), mode) if d > 0 else None

    places = [
        Place("h", (0.0, 0.0), "home", ("h",), parking=("b0",)),
        Place("g", (0.0, 30.0), "home", ("g",), parking=("b3",)),
        Place("cafe", (200.0, 0.0), "cafe", ("cafe",), parking=("b1",)),
        Place("b0", (0.0, 2.0), "berth", ("b0",), access_xy=(0.0, 1.0)),
        Place("b1", (200.0, 2.0), "berth", ("b1",), access_xy=(200.0, 1.0)),
        Place("b2", (150.0, 2.0), "berth", ("b2",), access_xy=(150.0, 1.0)),
        Place("b3", (0.0, 32.0), "berth", ("b3",), access_xy=(0.0, 31.0)),
    ]
    seen = []

    def keeps_ways(vehicle, berth, needs):
        seen.append((vehicle, berth, needs))
        return berth != "b1"

    d = Director(
        places,
        [Activity("coffee", ("cafe",), (60.0, 60.0), 600.0, modes=("walk", "car"))],
        route,
        keeps_ways=keeps_ways,
    )
    d.add_actor(Actor("p", "h", vehicle="v"))
    d.add_actor(Actor("q", "g", vehicle="w"))
    d.add_vehicle(Vehicle("v", "p", "b0"))
    d.add_vehicle(Vehicle("w", "q", "b2"))  # q's car stands away from home
    assert d._journey(d.actors["p"], (0.0, 0.0), d.places["cafe"], "car", 0.0, 60.0) is None
    assert d.car_rejections["would_close_a_way_home"] == 1
    assert seen == [("v", "b1", {"w": (("b2", ("b3",)),)})]
    # An admitted journey that has not left needs its departure as well as, away from home, its way home.
    d.active["q"] = Intent("q", "cafe", "coffee", "cafe", [Step("drive", "w", (200.0, 2.0))], destination_berth="b1")
    assert d.ways_home("v") == {"w": (("b2", ("b1",)), ("b1", ("b3",)))}
    d.vehicles["w"].in_trip = True
    assert d.ways_home("v") == {"w": (("b1", ("b3",)),)}
    d.active["q"] = Intent("q", "g", "return_home", "g", [Step("drive", "w", (0.0, 32.0))], destination_berth="b3")
    d.vehicles["w"].in_trip = False
    assert d.ways_home("v") == {"w": (("b2", ("b3",)),)}  # going home: only its departure


def test_a_car_journey_is_not_offered_where_parking_blocks_a_road():
    import math

    from awr.life.civil.activity.director import Route, Vehicle

    def route(mode, start, end, entity):
        d = math.dist(start, end)
        return Route(d, d / (6.0 if mode == "drive" else 1.0), mode) if d > 0 else None

    places = [
        Place("h", (0.0, 0.0), "home", ("h",), parking=("b0",)),
        Place("cafe", (200.0, 0.0), "cafe", ("cafe",), parking=("b1", "b2")),
        Place("b0", (0.0, 2.0), "berth", ("b0",), access_xy=(0.0, 1.0)),
        Place("b1", (200.0, 2.0), "berth", ("b1",), access_xy=(200.0, 1.0)),
        Place("b2", (205.0, 2.0), "berth", ("b2",), access_xy=(205.0, 1.0)),
    ]
    d = Director(places, [Activity("coffee", ("cafe",), (60.0, 60.0), 600.0, modes=("walk", "car"))], route)
    d.add_actor(Actor("p", "h", vehicle="v"))
    d.add_vehicle(Vehicle("v", "p", "b0"))
    asked = []

    def keeps_roads(vehicle, berth):
        asked.append((vehicle, berth))
        return berth != "b1"

    d.keeps_roads = keeps_roads
    steps, berth, _ = d._journey(d.actors["p"], (0.0, 0.0), d.places["cafe"], "car", 0.0, 60.0)
    assert berth == "b2" and d.car_rejections["would_block_a_road"] == 1 and asked == [("v", "b1"), ("v", "b2")]
    d.keeps_roads = lambda vehicle, berth: False
    assert d._journey(d.actors["p"], (0.0, 0.0), d.places["cafe"], "car", 0.0, 60.0) is None


def test_a_car_parked_at_home_keeps_a_way_out_to_the_main_roads():
    # Cruise r11f v4/v5: vehicle-049 parked at its destination where, with two parked cars, it shut vehicle-047 in at
    # its owner's home; vehicle-047's owner then had no drive for 135 s. A car at home with no admitted journey needs
    # a way to one of the main road network's berths (way_out), as a car away needs its way home.
    import math

    from awr.life.civil.activity.director import Intent, Route, Step, Vehicle

    def route(mode, start, end, entity):
        d = math.dist(start, end)
        return Route(d, d / (6.0 if mode == "drive" else 1.0), mode) if d > 0 else None

    places = [
        Place("h", (0.0, 0.0), "home", ("h",), parking=("b0",)),
        Place("g", (0.0, 30.0), "home", ("g",), parking=("b3",)),
        Place("cafe", (200.0, 0.0), "cafe", ("cafe",), parking=("b1",)),
        Place("b0", (0.0, 2.0), "berth", ("b0",), access_xy=(0.0, 1.0)),
        Place("b1", (200.0, 2.0), "berth", ("b1",), access_xy=(200.0, 1.0)),
        Place("b3", (0.0, 32.0), "berth", ("b3",), access_xy=(0.0, 31.0)),
    ]
    d = Director(places, [Activity("coffee", ("cafe",), (60.0, 60.0), 600.0, modes=("walk", "car"))], route)
    d.add_actor(Actor("p", "h", vehicle="v"))
    d.add_actor(Actor("q", "g", vehicle="w"))
    d.add_vehicle(Vehicle("v", "p", "b0"))
    d.add_vehicle(Vehicle("w", "q", "b3"))  # both at home
    assert d.ways_home("v") == {}  # not required unless the backend names the main roads
    d.way_out = ("b1", "b3")
    assert d.ways_home("v") == {"w": (("b3", ("b1",)),)}  # its own berth is not a way out
    d.active["q"] = Intent("q", "cafe", "coffee", "cafe", [Step("drive", "w", (200.0, 2.0))], destination_berth="b1")
    assert d.ways_home("v") == {"w": (("b3", ("b1",)), ("b1", ("b3",)))}  # its journey's legs instead
    d.active.pop("q")
    assert d.ways_home("v", way_out=False) == {}  # a car coming home parks at home whatever it shuts in


def test_only_a_car_coming_home_may_shut_a_car_at_home_in_and_only_where_its_home_parking_has_no_other_place():
    # Cruise r11f v7: vehicle-048 drove around to a school whose parking holds a berth of its owner's home parking,
    # 6 m from its home berth; parked there, it shut vehicle-047 in at home. That journey did not bring the car home:
    # it keeps every way. A car coming home takes a place of its home parking that keeps the way out where one does.
    import math

    from awr.life.civil.activity.director import Route, Vehicle

    def route(mode, start, end, entity):
        d = math.dist(start, end)
        return Route(d, d / (6.0 if mode == "drive" else 1.0), mode) if d > 0 else None

    places = [
        Place("h", (0.0, 0.0), "home", ("h",), parking=("b0", "b1", "b2")),
        Place("g", (0.0, 30.0), "home", ("g",), parking=("b3",)),
        Place("school", (90.0, 0.0), "school", ("school",), parking=("b1",)),
        Place("b0", (200.0, 2.0), "berth", ("b0",), access_xy=(200.0, 1.0)),
        Place("b1", (60.0, 2.0), "berth", ("b1",), access_xy=(60.0, 1.0)),
        Place("b2", (-60.0, 2.0), "berth", ("b2",), access_xy=(-60.0, 1.0)),
        Place("b3", (0.0, 32.0), "berth", ("b3",), access_xy=(0.0, 31.0)),
        Place("main", (500.0, 0.0), "berth", ("main",), access_xy=(500.0, 1.0)),
    ]
    shut = {"b1"}  # parked at b1, a car shuts w in at b3

    def keeps_ways(vehicle, berth, needs):
        return not (berth in shut and any(len(legs) and legs[-1][1] == ("main",) for legs in needs.values()))

    d = Director(
        places,
        [Activity("lessons", ("school",), (60.0, 60.0), 600.0, modes=("walk", "car"))],
        route,
        keeps_ways=keeps_ways,
    )
    d.add_actor(Actor("p", "h", vehicle="v"))
    d.add_actor(Actor("q", "g", vehicle="w"))
    d.add_vehicle(Vehicle("v", "p", "b0"))
    d.add_vehicle(Vehicle("w", "q", "b3"))
    d.way_out = ("main",)
    assert d._journey(d.actors["p"], (0.0, 0.0), d.places["school"], "car", 0.0, 60.0) is None
    assert d.car_rejections["would_close_a_way_home"] == 1
    # Coming home: b1 is the nearest place of its home parking, but b2 keeps w's way out.
    steps, bid, cost = d._journey(d.actors["p"], (200.0, 0.0), d.places["h"], "car", 0.0, 60.0)
    assert bid == "b2"
    shut.add("b2")
    steps, bid, cost = d._journey(d.actors["p"], (200.0, 0.0), d.places["h"], "car", 0.0, 60.0)
    assert bid == "b1"  # no place of its home parking keeps the way out: it parks at home all the same
