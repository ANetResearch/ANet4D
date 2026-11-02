from types import SimpleNamespace

import numpy as np

from awr.life.civil.activity import Place
from awr.life.motion.spatial_capacity import SpatialActivityCapacity


def test_separate_facility_ids_cannot_reserve_overlapping_space():
    a = Place("court-a", (0.0, 0.0), "courthouse", ("a",), group="court")
    b = Place("pitch-b", (0.5, 0.0), "pitch", ("b",), group="pitch")
    c = Place("pitch-c", (3.0, 0.0), "pitch", ("c",), group="pitch")
    guard = SpatialActivityCapacity([a, b, c], spacing_m=2.0)
    director = SimpleNamespace(holders={"a": "alice"})
    assert not guard.available(director, b, "bob")
    assert guard.available(director, c, "bob")
    assert guard.available(director, a, "alice")
    director.holders.clear()
    assert guard.available(director, b, "bob")
    assert guard.stats["close_place_pairs"] == 1


def test_physical_occupant_blocks_unreserved_place_until_departure():
    p = Place("p", (0.0, 0.0), "park", ("p",))
    guard = SpatialActivityCapacity([p], spacing_m=2.0)
    director = SimpleNamespace(holders={})
    xy = np.array([[0.0, 0.0], [4.0, 0.0]])
    guard.observe(["alice", "bob"], xy, [0.5, 0.5], [True, True])
    xy[0] = [10.0, 0.0]  # Caller mutation cannot rewrite the accepted observation.
    assert not guard.available(director, p, "bob")
    guard.observe(["alice", "bob"], xy, [0.5, 0.5], [True, True])
    assert guard.available(director, p, "bob")
