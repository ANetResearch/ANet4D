import math

import numpy as np
import pytest
from shapely.geometry import box

from awr.life.civil.vehicle_connections import admit, angle_delta, candidates, tangent_paths, three_point_path


class Ground:
    def __init__(self, hole=None):
        self.hole = hole

    def supported(self, xy, profile):
        xy = np.asarray(xy)
        if self.hole is None:
            return np.ones(len(xy), bool)
        lo, hi = self.hole
        return ~((xy[:, 0] >= lo) & (xy[:, 0] <= hi))


SIZE = np.array([4.2, 1.8, 1.5])


@pytest.fixture(autouse=True)
def geometry_contacts(monkeypatch):
    # Synthetic car for unit geometry only; scene probes load real asset tyres.
    monkeypatch.setattr(
        "awr.life.civil.contacts.vehicle_contacts",
        lambda size: np.array([[-1.4, -0.7], [-1.4, 0.7], [1.4, -0.7], [1.4, 0.7]]),
    )


def assert_kinematics(route, start, end, radius):
    np.testing.assert_allclose(route["points"][0], start[:2], atol=1e-7)
    np.testing.assert_allclose(route["points"][-1], end[:2], atol=1e-7)
    assert abs(angle_delta(route["yaw"][0], start[2])) < 1e-7
    assert abs(angle_delta(route["yaw"][-1], end[2])) < 1e-7
    length = np.diff(route["arc"])
    assert (length > 0).all()
    assert (length <= 0.251).all()
    assert (np.abs(np.diff(route["yaw"])) / length <= 1 / radius + 1e-4).all()
    tangent = np.diff(route["points"], axis=0) / length[:, None]
    mid = (route["yaw"][:-1] + route["yaw"][1:]) / 2
    np.testing.assert_allclose(tangent * route["gear"][:, None], np.column_stack([np.cos(mid), np.sin(mid)]), atol=1e-7)
    np.testing.assert_allclose(route["required_stops_m"], route["arc"][np.flatnonzero(np.diff(route["gear"])) + 1])


def test_all_tangent_families_have_exact_endpoints_and_bounded_curvature():
    rng = np.random.default_rng(43)
    for _ in range(12):
        start = np.r_[rng.uniform(-10, 10, 2), rng.uniform(-math.pi, math.pi)]
        end = np.r_[rng.uniform(-10, 10, 2), rng.uniform(-math.pi, math.pi)]
        routes = tangent_paths(start, end, 4.0)
        assert routes
        for route in routes:
            assert_kinematics(route, start, end, 4.0)


def test_straight_path_is_not_a_full_circle_and_cannot_cross_missing_ground():
    route = candidates([0, 0, 0], [30, 0, 0], SIZE)[0]
    assert route["arc"][-1] == pytest.approx(30.0)
    assert admit(Ground(), route, SIZE, allowed_area=box(-4, -4, 34, 4)) == (True, None)
    assert admit(Ground(hole=(14, 16)), route, SIZE)[0] is False
    assert admit(Ground(), route, SIZE, obstacles=[box(14, -1, 16, 1)])[0] is False
    assert admit(Ground(), route, SIZE, allowed_area=box(-4, -0.8, 34, 0.8))[0] is False


def test_road_containment_and_obstacle_clearance_have_separate_meanings():
    route = candidates([0, 0, 0], [30, 0, 0], SIZE)[0]
    # Physical width 1.8 m fits this 2.0 m strip. The extra safety envelope
    # may extend into clear space, but the body may not cross the strip edge.
    area = box(-3, -1, 33, 1)
    assert admit(Ground(), route, SIZE, allowed_area=area) == (True, None)
    assert admit(Ground(), route, SIZE, allowed_area=box(-3, -0.89, 33, 1))[1] == "outside_permitted_manoeuvre_area"
    # An obstacle outside the semantic strip is still within .4 m clearance.
    assert admit(Ground(), route, SIZE, allowed_area=area, obstacles=[box(14, 1.1, 16, 1.2)])[1] == "static_obstacle"
    assert admit(Ground(hole=(14, 16)), route, SIZE, allowed_area=area)[1] == "unsupported_or_steering"


def test_return_turn_needs_real_turning_space_and_never_rotates_in_place():
    start, end = [0, 0, 0], [0, 4, math.pi]
    routes = candidates(start, end, SIZE, max_length_m=70.0)
    assert routes
    assert not any(admit(Ground(), r, SIZE, allowed_area=box(-3, -2, 30, 6))[0] for r in routes)
    assert any(admit(Ground(), r, SIZE, allowed_area=box(-30, -30, 30, 30))[0] for r in routes)
    for route in routes:
        assert_kinematics(route, start, end, 4.0)


def test_reverse_parking_keeps_body_heading_and_declares_each_gear_change():
    start, end = [0, 0, 0], [20, 8, math.pi / 2]
    routes = candidates(start, end, SIZE, reverse_arrival_m=(5.0,), max_length_m=90.0)
    assert routes
    for route in routes:
        assert_kinematics(route, start, end, 4.0)
        assert route["gear"][-1] == -1
        assert len(route["required_stops_m"]) == 1
    # Driving requests do not silently gain a reverse option.
    road = candidates([0, 0, 0], [-4, 0, 0], SIZE, max_length_m=5.0)
    assert not road
    park = candidates([0, 0, 0], [-4, 0, 0], SIZE, reverse_departure_m=(0.0, 5.0), max_length_m=5.0)
    assert len(park) == 1 and (park[0]["gear"] == -1).all()
    assert not candidates([0, 0, 0], [-8, 0, 0], SIZE, reverse_departure_m=(0.0, 5.0), max_length_m=9.0)


def test_finite_budgets_and_bad_input_fail_explicitly():
    assert not candidates([0, 0, 0], [100, 0, 0], SIZE, max_length_m=20.0)
    with pytest.raises(ValueError):
        candidates([0, 0, float("nan")], [0, 0, 0], SIZE)


def test_three_point_return_has_two_full_stop_cusps_without_pose_jumps():
    for yaw in (-2.0, 0.0, 1.0):
        start = np.array([13.0, -5.0, yaw])
        for side in (-1, 1):
            goal = start.copy()
            goal[:2] += side * 3.2 * np.array([-math.sin(yaw), math.cos(yaw)])
            goal[2] += math.pi
            route = three_point_path(start, goal, 4.0)
            assert_kinematics(route, start, goal, 4.0)
            assert len(route["required_stops_m"]) == 2
            assert np.sum(np.diff(route["arc"])[route["gear"] == -1]) > 0
    assert three_point_path([0, 0, 0], [0, 20, math.pi], 4.0) is None
    assert three_point_path([0, 0, 0], [1, 3.2, math.pi], 4.0) is None
