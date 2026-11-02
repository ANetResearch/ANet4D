"""Indexed planning must preserve the exact reference routes and live safety."""

import math

import numpy as np
import pytest
from test_activity_driving import (  # noqa: F401 (contacts: autouse fixture)  # synthetic contact fixture
    SIZE,
    contacts,
    fixture,
)

from awr.life.civil.activity.driving import Body


def pair(**kwargs):
    indexed, starts, ends = fixture(**kwargs)
    reference, _, _ = fixture(**kwargs)
    reference.index_occupancy = False
    # Exercise occupancy reuse independently of the whole-plan cache.
    indexed.cache_plan_results = reference.cache_plan_results = False
    return indexed, reference, starts, ends


def test_index_matches_exact_intersections_under_pose_size_presence_updates():
    indexed, reference, starts, _ = pair()
    rng = np.random.default_rng(4107)
    bodies = [None, Body((0.0, 0.0, 0.0), SIZE), Body(starts["b"], SIZE)]
    bodies += [
        Body(
            (float(rng.uniform(-10, 10)), float(rng.uniform(-6, 6)), float(rng.uniform(-math.pi, math.pi))),
            (float(rng.uniform(2, 7)), float(rng.uniform(1, 3)), 1.5),
        )
        for _ in range(50)
    ]
    for body in bodies:
        occupancy = dict(a=Body(starts["a"], SIZE), b=body)
        for routes in (indexed, reference):
            routes.update_occupancy(occupancy)
        expected = reference.plan("a", "a0", "a1")
        assert indexed.plan("a", "a0", "a1") == expected
        builds = indexed.stats["occupancy_index_builds"]
        for _ in range(3):
            indexed.update_occupancy(dict(occupancy))
            assert indexed.plan("a", "a0", "a1") == expected
        assert indexed.stats["occupancy_index_builds"] == builds
    assert indexed.stats["occupancy_index_hits"] > 0
    assert indexed.stats["occupancy_index_builds"] == len(bodies)
    assert reference.stats["occupancy_index_builds"] == 0


@pytest.mark.parametrize("destination_blocks", [False, True])
def test_active_destination_changes_invalidate_index_without_pose_change(destination_blocks):
    indexed, reference, starts, ends = pair(b_end=(0.0, 0.0, math.pi / 2) if destination_blocks else None)
    a0 = indexed.plan("a", "a0", "a1")
    assert a0 == reference.plan("a", "a0", "a1") and a0 is not None
    before = indexed.stats["occupancy_index_builds"]
    for routes in (indexed, reference):
        b = routes.plan("b", "b0", "b1")
        assert routes.acquire(b.evidence, starts["b"], 0.0)
    a = indexed.plan("a", "a0", "a1")
    assert a == reference.plan("a", "a0", "a1")
    assert (a is None) == destination_blocks
    assert indexed.stats["occupancy_index_builds"] == before + 1
    if not destination_blocks:
        # Planning can use the future parked pose; a departing car still waits
        # before the crossing where an executing car's current body stands.
        for routes in (indexed, reference):
            routes.update_occupancy(dict(a=Body(starts["a"], SIZE), b=Body((0.0, 0.0, math.pi / 2), SIZE)))
            assert routes.plan("a", "a0", "a1") == a
            assert routes.acquire(a.evidence, starts["a"], 0.0)
            run = routes.traffic.layout(a.evidence).runs[0]
            assert routes.traffic_limit("a", run.stop - 1.0, 0.0) == run.stop
    for routes in (indexed, reference):
        routes.update_occupancy(dict(a=Body(starts["a"], SIZE), b=Body(ends["b"], SIZE)))
        assert routes.arrive(routes.active["b"], ends["b"], 0.0)
    if not destination_blocks:
        assert indexed.traffic_limit("a", 8.0, 0.0) is None and reference.traffic_limit("a", 8.0, 0.0) is None
    else:
        assert indexed.plan("a", "a0", "a1") == reference.plan("a", "a0", "a1")


def test_one_index_keeps_self_exclusion_specific_to_the_requesting_vehicle():
    indexed, reference, starts, _ = pair(crossing=False)
    for vehicle in ("a", "b", "a", "b"):
        result = indexed.plan(vehicle, vehicle + "0", vehicle + "1")
        assert result is not None and result == reference.plan(vehicle, vehicle + "0", vehicle + "1")
    assert indexed.stats["occupancy_index_builds"] == 1
    assert indexed.stats["occupancy_index_queries"] == 2


def test_active_motion_reuses_future_plan_but_departure_checks_live_body():
    indexed, reference, starts, ends = pair(crossing=False)
    indexed.cache_plan_results = True
    for routes in (indexed, reference):
        b = routes.plan("b", "b0", "b1")
        assert routes.acquire(b.evidence, starts["b"], 0.0)
    original = indexed.plan("a", "a0", "a1")
    assert original is not None
    assert original == reference.plan("a", "a0", "a1")
    builds = indexed.stats["occupancy_index_builds"]
    hits = indexed.stats["snapshot_plan_hits"]
    fixed_body = indexed.bodies["a"]
    # Put the active car at a's departure: future berth-based planning remains
    # valid, but its current physical body must prevent actual departure.
    for routes in (indexed, reference):
        routes.update_occupancy(dict(a=Body(starts["a"], SIZE), b=Body(starts["a"], SIZE)))
    assert indexed.plan("a", "a0", "a1") is original
    assert original == reference.plan("a", "a0", "a1")
    assert indexed.stats["snapshot_plan_hits"] == hits + 1
    assert indexed.stats["occupancy_index_builds"] == builds
    assert indexed.bodies["a"] is fixed_body
    assert indexed.bodies["b"].equals(Body(starts["a"], SIZE).polygon())
    assert not indexed.acquire(original.evidence, starts["a"], 0.0)
    # Once parked, b is an actual stationary obstacle again. A later observation
    # must invalidate cached plans, including cached unreachability.
    for routes in (indexed, reference):
        routes.update_occupancy(dict(a=Body(starts["a"], SIZE), b=Body(ends["b"], SIZE)))
        assert routes.arrive(routes.active["b"], ends["b"], 0.0)
    assert indexed.plan("a", "a0", "a1") == reference.plan("a", "a0", "a1")
    for routes in (indexed, reference):
        routes.update_occupancy(dict(a=Body(starts["a"], SIZE), b=Body(starts["a"], SIZE)))
    assert indexed.plan("a", "a0", "a1") == reference.plan("a", "a0", "a1") is None
    for routes in (indexed, reference):
        routes.update_occupancy(dict(a=Body(starts["a"], SIZE), b=None))
    assert indexed.plan("a", "a0", "a1") == reference.plan("a", "a0", "a1") == original
