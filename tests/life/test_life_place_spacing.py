"""Physical facility capacity is shared across overlapping mapped approaches."""

from types import SimpleNamespace

import numpy as np
from scipy.spatial.distance import cdist, pdist

from awr.life.civil.itineraries import life_pools


def network():
    rows, cols = np.mgrid[:40, :40]
    nodes = np.column_stack((rows.ravel(), cols.ravel()))
    free = np.ones((40, 40), bool)
    return SimpleNamespace(
        step=0.5,
        nodes=nodes,
        xy=(nodes[:, ::-1] + 0.5) * 0.5,
        carriageway=np.zeros_like(free),
        turn_free=free.copy(),
        cell=lambda q: np.floor(np.asarray(q)[::-1] / 0.5).astype(int),
    ), free


def test_overlapping_approaches_share_physical_room_across_all_sites():
    net, free = network()
    places = [
        dict(id="pitch-a", kind="pitch", xy=[9.25, 9.25], poi="pitch"),
        dict(id="pitch-b", kind="pitch", xy=[9.75, 9.75], poi="pitch"),
        dict(id="park", kind="park", xy=[10.25, 9.25], poi="park"),
    ]
    pools = life_pools(
        net,
        {},
        places,
        free,
        np.random.default_rng(1),
        dict(slot_radius_m={"default": 5.0}, slot_spacing_m=2.0, max_slots=6),
    )
    assert len(pools["sites"]) == 3
    assert all(len(s["slots"]) == 6 for s in pools["sites"])
    points = np.array([sl["xy"] for site in pools["sites"] for sl in site["slots"]])
    assert pdist(points).min() >= 2.0
    homes = np.array([s["xy"] for s in pools["street"]])
    assert cdist(homes, points).min() >= 3.0
    # Same inputs choose the same physical slots, independent of population RNG.
    again = life_pools(
        net,
        {},
        places,
        free,
        np.random.default_rng(99),
        dict(slot_radius_m={"default": 5.0}, slot_spacing_m=2.0, max_slots=6),
    )
    other = np.array([sl["xy"] for s in again["sites"] for sl in s["slots"]])
    np.testing.assert_array_equal(points, other)


def test_close_distinct_points_do_not_manufacture_another_facility_slot():
    net, free = network()
    places = [dict(id="a", kind="pitch", xy=[9.25, 9.25]), dict(id="b", kind="pitch", xy=[9.75, 9.25])]
    pools = life_pools(
        net,
        {},
        places,
        free,
        np.random.default_rng(1),
        dict(slot_radius_m={"default": 0.1}, slot_spacing_m=2.0, max_slots=1),
    )
    assert [s["id"] for s in pools["sites"]] == ["a"]
    np.testing.assert_array_equal(pools["sites"][0]["slots"][0]["xy"], [9.25, 9.25])


def test_rejected_candidate_does_not_suppress_free_space_further_out():
    net, free = network()
    places = [dict(id="a", kind="pitch", xy=[9.25, 9.25]), dict(id="b", kind="park", xy=[9.75, 9.25])]
    pools = life_pools(
        net,
        {},
        places,
        free,
        np.random.default_rng(1),
        dict(slot_radius_m={"pitch": 0.1, "park": 1.6}, slot_spacing_m=2.0, max_slots=1),
    )
    assert [s["id"] for s in pools["sites"]] == ["a", "b"]
    a, b = [s["slots"][0]["xy"] for s in pools["sites"]]
    assert np.linalg.norm(a - b) >= 2.0
    assert np.linalg.norm(b - places[1]["xy"]) <= 1.6


def test_facility_waiting_does_not_take_narrow_boundary_fringe():
    net, free = network()
    free[:8] = False
    net.turn_free = free.copy()
    places = [dict(id="office-approach", kind="townhall", xy=[9.25, 4.25])]
    cfg = dict(slot_radius_m={"default": 4.0}, max_slots=6)
    old = life_pools(net, {}, places, free, np.random.default_rng(1), cfg)
    assert net.waiting_clearance[tuple(net.cell(old["sites"][0]["slots"][0]["xy"]))] == 0.5
    result = life_pools(
        net, {}, places, free, np.random.default_rng(1), dict(cfg, facility_circulation_clearance_m=1.5)
    )
    assert result["sites"]
    for site in result["sites"]:
        assert site["capacity_evidence"]["circulation_clearance_m"] == 1.5
        for slot in site["slots"]:
            assert net.waiting_clearance[tuple(net.cell(slot["xy"]))] >= 1.5


def test_facility_with_no_circulation_room_is_not_invented():
    net, free = network()
    free[:] = False
    free[8, 4:30] = True
    net.turn_free = free.copy()
    result = life_pools(
        net,
        {},
        [dict(id="narrow", kind="townhall", xy=[9.25, 4.25])],
        free,
        np.random.default_rng(1),
        dict(facility_circulation_clearance_m=1.5),
    )
    assert result["sites"] == []


def test_only_explicit_outdoor_semantics_grant_stay_affordance():
    net, free = network()
    point = dict(id="office", kind="townhall", xy=[9.25, 9.25])
    for function, expected in (
        (None, ()),
        ("indoor_service", ()),
        ("outdoor_approach", ("outdoor_stay",)),
        ("public_open_space", ("outdoor_stay",)),
    ):
        pools = life_pools(net, {}, [dict(point, function=function)], free, np.random.default_rng(1))
        assert pools["sites"][0]["affordances"] == expected
