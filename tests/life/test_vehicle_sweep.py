"""Independent dense body checks for the continuous interpolation envelope."""

import math

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from awr.life.civil.collision import rectangle
from awr.life.civil.curb_allocation import swept
from awr.life.civil.vehicle_connections import candidates

SIZE = np.array([4.2, 1.8, 1.5])


def test_straight_motion_keeps_clearance_without_padding_the_road():
    route = dict(points=[[0.0, 0.0], [30.0, 0.0]], yaw=[0.0, 0.0])
    envelope = swept(route, SIZE)
    assert envelope.equals(box(-2.5, -1.3, 32.5, 1.3))
    assert not envelope.intersects(box(14, 1.31, 15, 1.4))
    assert envelope.intersects(box(14, 1.29, 15, 1.4))
    # Long sparse segments cannot leave holes between endpoint bodies.
    assert envelope.covers(box(14, -0.9, 16, 0.9))


def assert_dense_body_containment(route, margin=0.4):
    area = swept(route, SIZE, margin_m=margin).buffer(1e-9)
    for a, b, ya, yb in zip(
        route["points"][:-1], route["points"][1:], route["yaw"][:-1], route["yaw"][1:], strict=False
    ):
        for u in np.linspace(0.0, 1.0, 31):
            xy = np.asarray(a) * (1 - u) + np.asarray(b) * u
            body = Polygon(rectangle(xy, ya * (1 - u) + yb * u, SIZE, margin=margin))
            assert area.covers(body)


def test_turn_reverse_and_wrapped_heading_sweeps_cover_intermediate_bodies():
    routes = candidates([0, 0, 0], [12, 8, math.pi / 2], SIZE, reverse_arrival_m=(4.0,), max_length_m=50.0)
    assert routes
    for route in routes[:: max(1, len(routes) // 5)]:
        assert_dense_body_containment(route)
    # Even a supplied long rotation is covered; admission rejects its steering
    # separately. The envelope must not silently choose the shorter rotation.
    assert_dense_body_containment(dict(points=[[0.0, 0.0], [0.1, 0.0]], yaw=[3.0, -3.0]))


def test_random_interpolation_sweeps_include_all_four_corners():
    rng = np.random.default_rng(822)
    for _ in range(30):
        points = rng.normal(size=(2, 2)) * 4
        yaw = rng.uniform(-math.pi, math.pi, 2)
        assert_dense_body_containment(dict(points=points, yaw=yaw))


def test_physical_body_diagnostic_does_not_silently_change_default_clearance():
    route = dict(points=[[0.0, 0.0], [30.0, 0.0]], yaw=[0.0, 0.0])
    assert swept(route, SIZE, margin_m=0.0).equals(box(-2.1, -0.9, 32.1, 0.9))
    assert swept(route, SIZE).equals(box(-2.5, -1.3, 32.5, 1.3))
    for r in candidates([0, 0, 0], [12, 8, math.pi / 2], SIZE, max_length_m=50.0)[::4]:
        assert_dense_body_containment(r, 0.0)
    for bad in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            swept(route, SIZE, margin_m=bad)


@pytest.mark.parametrize(
    "route", [dict(points=[[0, 0]], yaw=[0]), dict(points=[[0, 0], [1, 0]], yaw=[0, float("nan")])]
)
def test_bad_sweep_input_fails(route):
    with pytest.raises(ValueError):
        swept(route, SIZE)
