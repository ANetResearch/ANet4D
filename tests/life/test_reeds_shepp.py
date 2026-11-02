import math
from types import SimpleNamespace as Segment

import numpy as np
import pytest

from awr.life.civil.reeds_shepp import candidates, sample_segments
from awr.life.civil.vehicle_connections import angle_delta


def test_cusp_is_sampled_even_when_segment_shorter_than_step():
    route = sample_segments(
        [0, 0, 0],
        [
            Segment(type="straight", direction=1, length=0.1, turn_radius=4.0),
            Segment(type="straight", direction=-1, length=0.08, turn_radius=4.0),
            Segment(type="left", direction=1, length=0.07, turn_radius=4.0),
        ],
    )
    np.testing.assert_allclose(route["points"][:3], [[0, 0], [0.1, 0], [0.02, 0]], atol=1e-10)
    np.testing.assert_allclose(route["required_stops_m"], [0.1, 0.18])
    assert route["gear"].tolist() == [1, -1, 1]


def test_general_reeds_shepp_endpoints_and_reverse_kinematics():
    pytest.importorskip("rsplan")  # Optional compiler dependency, not simulator runtime.
    rng = np.random.default_rng(481)
    cases = [([0, 0, 0], [10, 8, math.pi / 2]), ([0, 0, 0], [-3, 0, 0])]
    cases += [
        (
            np.r_[rng.uniform(-15, 15, 2), rng.uniform(-math.pi, math.pi)],
            np.r_[rng.uniform(-15, 15, 2), rng.uniform(-math.pi, math.pi)],
        )
        for _ in range(30)
    ]
    saw_cusps = False
    for start, end in cases:
        routes = candidates(start, end, [4.2, 1.8, 1.5])
        assert routes
        for route in routes:
            np.testing.assert_allclose(route["points"][0], start[:2], atol=1e-7)
            np.testing.assert_allclose(route["points"][-1], end[:2], atol=1e-7)
            assert abs(angle_delta(route["yaw"][-1], end[2])) < 1e-7
            ds = np.diff(route["arc"])
            mid = (route["yaw"][1:] + route["yaw"][:-1]) / 2
            assert (ds > 0).all() and (ds <= 0.250001).all()
            assert (abs(np.diff(route["yaw"])) / ds <= 0.2501).all()
            np.testing.assert_allclose(
                np.diff(route["points"], axis=0) / ds[:, None] * route["gear"][:, None],
                np.column_stack([np.cos(mid), np.sin(mid)]),
                atol=1e-7,
            )
            np.testing.assert_allclose(
                route["required_stops_m"], route["arc"][np.flatnonzero(np.diff(route["gear"])) + 1]
            )
            saw_cusps |= len(route["required_stops_m"]) >= 2
    assert saw_cusps


def test_reject_invalid_segments():
    with pytest.raises(ValueError):
        sample_segments([0, 0, 0], [Segment(type="left", direction=0, length=2.0, turn_radius=4.0)])


def test_bay_runways_clear_body_before_turning_and_reverse_into_berth():
    pytest.importorskip("rsplan")
    routes = candidates([0, 0, 0], [12, 9, math.pi / 2], [4.2, 1.8, 1.5], departure_m=4.0, arrival_m=-4.0)
    assert routes
    for route in routes:
        departure = route["arc"] <= 4.0 + 1e-8
        np.testing.assert_allclose(route["points"][departure, 1], 0.0, atol=1e-8)
        arrival = route["arc"] >= route["arc"][-1] - 4.0 - 1e-8
        np.testing.assert_allclose(route["points"][arrival, 0], 12.0, atol=1e-8)
        np.testing.assert_allclose(route["points"][-1], [12, 9], atol=1e-7)
        assert route["gear"][0] == 1 and route["gear"][-1] == -1
