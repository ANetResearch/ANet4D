"""Passing/crossing permission must not create long-term activity capacity."""

from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import LineString, Point

from awr.life.civil.itineraries import life_pools
from awr.life.civil.outdoor_places import compile_places
from awr.life.civil.surface import SurfaceMap
from awr.life.civil.walking import WalkNetwork


@pytest.mark.parametrize("shared", [None, {"carriageway_cost": 2.0}])
def test_crossing_remains_walkable_but_never_becomes_home_or_facility_stay(shared):
    surface = SurfaceMap.__new__(SurfaceMap)
    surface.x0 = surface.y0 = 0.0
    surface.nx = 80
    surface.ny = 60
    surface.cell_m = 1.0
    surface.width_m = 80
    surface.height_m = 60
    surface.z = np.zeros((60, 80))
    surface.walk = {"person": np.ones((60, 80), bool)}
    sem = dict(
        roads=[dict(line=[[-10, 30], [90, 30]], width_m=6)],
        crossings=[dict(ends=[[35, 27], [35, 33]], width_m=5)],
        buildings=[],
        pois=[dict(id="post-office", kind="post_office", xy=[35, 30])],
    )
    # SUMO's pedestrian overlay can overlap a motor area at a crossing.
    lanes = SimpleNamespace(
        walking=LineString([[35, 27], [35, 33]]).buffer(2.5), driving=LineString([[-10, 30], [90, 30]]).buffer(3)
    )
    w = WalkNetwork.from_scene(surface, sem, 0.35, turn_radius=0.72, lane_network=lanes, shared_streets=shared)
    crossing = tuple(w.cell([35.25, 30.25]))
    assert w.free[crossing] and not w.standing_free[crossing]
    if shared:
        assert not w.carriageway[crossing]  # cheap passage is still cheap
    assert w.plan([10.25, 25.25], [60.25, 34.75]) is not None
    before = w.free.copy()
    candidate = compile_places(w, sem, clearance_m=1.0)
    assert candidate["places"]
    pools = life_pools(w, sem, candidate["places"], w.free, np.random.default_rng(3))
    assert pools["sites"] and pools["street"]
    protected = lanes.driving.union(lanes.walking)
    points = [slot["xy"] for s in pools["sites"] for slot in s["slots"]] + [s["xy"] for s in pools["street"]]
    for xy in points:
        assert w.standing_free[tuple(w.cell(xy))]
        assert protected.distance(Point(xy)) >= 0.72
    np.testing.assert_array_equal(w.free, before)
    assert w.summary()["standing_area_m2"] < w.summary()["free_area_m2"]
