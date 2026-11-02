"""Public-area capacity comes from mapped interior geometry, not POI radii."""

import numpy as np
import shapely
from scipy.spatial.distance import pdist
from shapely.geometry import Polygon
from test_life_place_spacing import network

from awr.life.civil.itineraries import life_pools


def test_region_stays_respect_boundary_body_room_and_disconnected_access():
    net, free = network()
    free[:, 18:22] = False
    net.turn_free = free.copy()
    net.turn_body = (1.0, 1.0)
    area = dict(id="park", kind="park", xy=[10.0, 10.0], polygon=[[4.0, 4.0], [16.0, 4.0], [16.0, 16.0], [4.0, 16.0]])
    places = [dict(id="old-entry", poi="park", kind="park", xy=[3.75, 10.0], function="public_open_space")]
    cfg = dict(
        open_region_activities=True, open_region_clearance_m=1.5, slot_spacing_m=2.0, max_open_component_slots=50
    )
    pools = life_pools(net, {"open_areas": [area]}, places, free, np.random.default_rng(1), cfg)
    sites = pools["sites"]
    assert len(sites) == 2 and len({s["comp"] for s in sites}) == 2
    assert all(s["facility"] == "park" and "old-entry" not in s["id"] for s in sites)
    points = np.array([slot["xy"] for s in sites for slot in s["slots"]])
    interior = Polygon(area["polygon"]).buffer(-np.sqrt(0.5))
    assert shapely.contains_xy(interior, points[:, 0], points[:, 1]).all()
    assert pdist(points).min() >= 2.0
    assert len(points) > 6  # not restricted to six positions around one entrance
    for point in points:
        assert net.waiting_clearance[tuple(net.cell(point))] >= 1.5
    again = life_pools(net, {"open_areas": [area]}, list(reversed(places)), free, np.random.default_rng(99), cfg)
    np.testing.assert_array_equal(points, [slot["xy"] for s in again["sites"] for slot in s["slots"]])
    assert places[0]["id"] == "old-entry"  # input semantics are not rewritten


def test_open_region_preserves_its_activity_kind_and_has_bounded_capacity():
    net, free = network()
    area = dict(id="pitch", kind="pitch", xy=[10.0, 10.0], polygon=[[2.0, 2.0], [18.0, 2.0], [18.0, 18.0], [2.0, 18.0]])
    result = life_pools(
        net,
        {"open_areas": [area]},
        [],
        free,
        np.random.default_rng(1),
        dict(open_region_activities=True, max_open_component_slots=3),
    )
    assert len(result["sites"]) == 1
    site = result["sites"][0]
    assert site["kind"] == "pitch" and len(site["slots"]) == 3
    assert site["capacity_evidence"]["source_region"] == "pitch"
    assert site["capacity_evidence"]["slot_budget"] == 3


def test_public_approach_preserves_usage_without_area_compilation():
    net, free = network()
    places = [dict(id="sports-approach", poi="sports", kind="pitch", xy=[10.0, 10.0], function="public_open_space")]
    pools = life_pools(net, {}, places, free, np.random.default_rng(1))
    assert len(pools["sites"]) == 1
    assert pools["sites"][0]["kind"] == "pitch"
    assert pools["sites"][0]["affordances"] == ("outdoor_stay",)
    assert places[0]["kind"] == "pitch"


def test_unknown_open_space_cannot_invent_park_capacity():
    for kind in (None, "unclassified_open_area"):
        net, free = network()
        places = [dict(id="unknown", kind=kind, xy=[10.0, 10.0], function="public_open_space")]
        assert life_pools(net, {}, places, free, np.random.default_rng(1))["sites"] == []
