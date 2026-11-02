"""Policy/ownership tests, not native walking or vehicle physics admission."""

from test_activity_engine import route, world


def after_shopping(router=route):
    d = world(car=True, router=router)
    d.errands_before_home = 2
    d.choose("p", 0.0, (0.0, 0.0))
    assert d.acknowledge("p", 12.0, (10.0, 2.0), 0.0)
    assert not d.acknowledge("p", 13.0, (10.0, 2.0), 0.0, vehicle_xy=(10.0, 0.0))
    assert d.acknowledge("p", 17.0, (10.0, 2.0), 0.0, vehicle_xy=(10.0, 0.0))
    d.observe_departure("p", (17.0, 0.0), vehicle_xy=(17.0, 0.0))
    assert d.acknowledge("p", 42.0, (90.0, 0.0), 0.0, vehicle_xy=(90.0, 0.0))
    assert not d.acknowledge("p", 43.0, (90.0, 2.0), 0.0, vehicle_xy=(90.0, 0.0))
    assert d.acknowledge("p", 47.0, (90.0, 2.0), 0.0, vehicle_xy=(90.0, 0.0))
    assert d.acknowledge("p", 60.0, (100.0, 0.0), 0.0)
    assert not d.acknowledge("p", 60.0, (100.0, 0.0), 0.0)
    assert d.acknowledge("p", 70.0, (100.0, 0.0), 0.0)
    d.actors["p"].completed["rest"] = -1000
    return d


def test_park_once_walk_to_second_activity_then_recover_the_same_car():
    d = after_shopping()
    intent = d.choose("p", 71.0, (100.0, 0.0))
    assert intent.purpose == "rest"
    assert [s.kind for s in intent.steps] == ["walk", "activity"]
    assert intent.destination_berth is None
    assert d.vehicles["car"].berth == "shop-bay" and d.vehicles["car"].occupant is None
    assert d.holders["shop-bay-slot"] == "car"
    d.observe_departure("p", (102.0, 0.0))
    assert d.holders["shop-bay-slot"] == "car"  # pedestrian departure is not car departure
    assert d.acknowledge("p", 171.0, (200.0, 0.0), 0.0)
    assert not d.acknowledge("p", 171.0, (200.0, 0.0), 0.0)
    assert d.acknowledge("p", 191.0, (200.0, 0.0), 0.0)
    home = d.choose("p", 192.0, (200.0, 0.0))
    assert home.purpose == "return_home" and home.destination_berth == "h-bay"
    assert home.steps[0].xy == (90.0, 2.0)  # fetch the car at the unchanged real berth
    assert [s.kind for s in home.steps] == ["walk", "board", "drive", "alight", "walk", "activity"]
    assert d.holders["shop-bay-slot"] == "car"


def test_one_way_walk_that_loses_access_to_the_car_is_not_selected():
    def router(mode, start, end, entity):
        if mode == "walk" and start == (200.0, 0.0) and end == (90.0, 2.0):
            return None
        return route(mode, start, end, entity)

    d = after_shopping(router)
    assert d.choose("p", 71.0, (100.0, 0.0)).purpose == "return_home"
    assert d.holders["shop-bay-slot"] == "car"


def test_park_and_walk_does_not_allow_walking_home_and_abandoning_the_car():
    d = after_shopping()
    a = d.actors["p"]
    assert d._journey(a, (100.0, 0.0), d.places["h"], "walk", 71.0, 10.0) is None
    d.vehicles["car"].in_trip = True
    assert d._journey(a, (100.0, 0.0), d.places["park"], "walk", 71.0, 10.0) is None


def test_second_activity_must_leave_time_to_reach_and_board_car_before_bay_closes():
    from dataclasses import replace

    d = after_shopping()
    d.places["shop-bay"] = replace(d.places["shop-bay"], closes_s=200.0)
    assert d.choose("p", 71.0, (100.0, 0.0)).purpose == "return_home"
    assert d.holders["shop-bay-slot"] == "car"
