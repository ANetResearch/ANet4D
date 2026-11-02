"""Measured-route policy contracts; acknowledgements here are not 3D validation."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box
from test_activity_engine import route
from test_activity_park_once import after_shopping

from awr.life.civil.activity.director import Place
from awr.life.civil.activity.driving_adapter import DrivingAdapter


def returning_owner(damage=None):
    # The nearer bay belongs to a disconnected road component. The last bay is
    # outside the public home parking search radius. No query or actor ID enters
    # the adapter's candidate policy.
    coords = {"h-bay": (10.0, 0.0), "alternative": (20.0, 0.0), "disconnected": (5.0, 0.0), "far": (300.0, 0.0)}
    access = {name: (x, y + 2.0) for name, (x, y) in coords.items()}

    def measured(mode, start, end, entity):
        if mode == "drive" and (start == coords["disconnected"] or end == coords["disconnected"]):
            return None
        if damage == "unwalkable" and mode == "walk" and start == access["alternative"] and end == (0.0, 0.0):
            return None
        return route(mode, start, end, entity)

    d = after_shopping(measured)
    d.errands_before_home = 1
    graph = SimpleNamespace(components=lambda: [["h-bay", "alternative", "far"], ["disconnected"]])
    adapter = DrivingAdapter(graph, (4.0, 2.0, 1.5), "car", box(-1, -1, 400, 10), None, None, 0.45, home_walk_m=40.0)
    usable = {name: ((*xy, 0.0), access[name]) for name, xy in coords.items()}
    for name in ("alternative", "disconnected", "far"):
        d.places[name] = Place(name, coords[name], "berth", (name,), access_xy=access[name])
    home = d.places["h"]
    candidates = adapter.home_parking(home, "h-bay", usable)
    assert candidates[0] == "h-bay" and set(candidates) == {"h-bay", "alternative", "disconnected"}
    d.places["h"] = replace(home, parking=candidates)
    # Another owner has reserved the public original berth while this car was
    # shopping. It must not be evicted or have its reservation overwritten.
    d.holders["h-bay-slot"] = "other-car"
    if damage == "occupied":
        d.holders["alternative"] = "third-car"
    if damage == "closed":
        d.places["alternative"] = replace(d.places["alternative"], closes_s=80.0)
    return d


def test_return_uses_available_berth_then_walks_home_without_abandoning_car():
    d = returning_owner()
    home = d.choose("p", 71.0, (100.0, 0.0))
    assert home.purpose == "return_home" and home.destination_berth == "alternative"
    assert [s.kind for s in home.steps] == ["walk", "board", "drive", "alight", "walk", "activity"]
    assert home.steps[0].xy == (90.0, 2.0) and home.steps[4].xy == (0.0, 0.0)
    assert d.holders["alternative"] == "car" and d.holders["h-bay-slot"] == "other-car"
    assert d.acknowledge("p", 90.0, (90.0, 2.0), 0.0)
    assert not d.acknowledge("p", 91.0, (90.0, 2.0), 0.0, vehicle_xy=(90.0, 0.0))
    assert d.acknowledge("p", 95.0, (90.0, 2.0), 0.0, vehicle_xy=(90.0, 0.0))
    d.observe_departure("p", (80.0, 0.0), vehicle_xy=(80.0, 0.0))
    assert "shop-bay-slot" not in d.holders
    # A report at the original occupied berth cannot complete the drive.
    assert not d.acknowledge("p", 120.0, (10.0, 0.0), 0.0, vehicle_xy=(10.0, 0.0))
    assert d.acknowledge("p", 121.0, (20.0, 0.0), 0.0, vehicle_xy=(20.0, 0.0))
    assert not d.acknowledge("p", 122.0, (20.0, 2.0), 0.0, vehicle_xy=(20.0, 0.0))
    assert d.acknowledge("p", 126.0, (20.0, 2.0), 0.0, vehicle_xy=(20.0, 0.0))
    assert d.acknowledge("p", 150.0, (0.0, 0.0), 0.0)
    assert not d.acknowledge("p", 151.0, (0.0, 0.0), 0.0)
    assert d.acknowledge("p", 161.0, (0.0, 0.0), 0.0)
    assert d.vehicles["car"].berth == "alternative" and d.vehicles["car"].occupant is None
    assert d.actors["p"].last_place == "h" and "p" not in d.active
    assert d.holders["alternative"] == "car" and d.holders["h-bay-slot"] == "other-car"


@pytest.mark.parametrize("damage", ["unwalkable", "occupied", "closed"])
def test_nearby_bays_do_not_override_access_capacity_or_opening_constraints(damage):
    d = returning_owner(damage)
    before = dict(d.holders)
    assert d.choose("p", 71.0, (100.0, 0.0)) is None
    assert d.holders == before and "p" not in d.active
    assert d.vehicles["car"].berth == "shop-bay"  # No forced return/teleport.


def test_an_owner_who_cannot_bring_the_car_home_at_the_last_attempt_walks_home_without_it():
    # cruise3-r11f v2, person-005 at 1352.6 s: every home berth was refused for 135 s and the run stopped. At its last
    # attempt before max_wait_s the owner leaves the car parked where it stands and walks home.
    d = returning_owner("occupied")
    for t in np.arange(71.0, 71.0 + d.max_wait_s - d.retry_s + 1e-6, d.retry_s):
        assert d.choose("p", float(t), (100.0, 0.0)) is None  # still waiting for a way to bring the car home
    home = d.choose("p", 71.0 + d.max_wait_s, (100.0, 0.0))
    assert home is not None and home.place == "h" and home.destination_berth is None
    assert [s.kind for s in home.steps] == ["walk", "activity"]
    assert d.vehicles["car"].berth == "shop-bay" and d.holders["shop-bay-slot"] == "car"
    left = [e for e in d.events if e["kind"] == "car_left_away"]
    assert len(left) == 1 and left[0]["berth"] == "shop-bay" and left[0]["waited_s"] == d.max_wait_s
