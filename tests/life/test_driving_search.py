import numpy as np
import pytest
from shapely.geometry import box

from awr.life.civil.driving_search import connect, primitive
from awr.life.civil.vehicle_connections import admit, candidates

SIZE = np.array([4.2, 1.8, 1.5])


class Ground:
    def supported(self, xy, profile):
        return np.ones(len(xy), bool)


@pytest.fixture(autouse=True)
def contacts(monkeypatch):
    monkeypatch.setattr(
        "awr.life.civil.contacts.vehicle_contacts",
        lambda size: np.array([[-1.4, -0.7], [-1.4, 0.7], [1.4, -0.7], [1.4, 0.7]]),
    )


def test_exact_pose_primitives_have_bounded_curvature():
    start = np.array([13.0, 4.0, 0.7])
    for turn in [0.0, -0.2, 0.2]:
        r = primitive(start, turn)
        np.testing.assert_allclose(r["points"][0], start[:2])
        assert np.max(np.abs(np.diff(r["yaw"])) / np.diff(r["arc"])) <= abs(turn) + 1e-4
        assert (r["gear"] == 1).all()


def test_more_than_two_bends_around_a_parked_obstacle():
    start = [0, 0, 0]
    end = [23, 0, 0]
    domain = box(-3, -6, 26, 6)
    obstacles = [box(10, -0.8, 12, 0.8)]
    assert not any(
        admit(Ground(), r, SIZE, allowed_area=domain, obstacles=obstacles)[0]
        for r in candidates(start, end, SIZE, max_length_m=50.0)
    )
    r, stats = connect(
        Ground(), start, end, SIZE, allowed_area=domain, obstacles=obstacles, max_expansions=2000, max_length_m=50.0
    )
    assert r is not None, stats
    assert admit(Ground(), r, SIZE, allowed_area=domain, obstacles=obstacles)[0]
    np.testing.assert_allclose(r["points"][-1], end[:2], atol=1e-6)
    assert r["yaw"][-1] == pytest.approx(0.0, abs=1e-6)


def test_budget_failure_does_not_move_endpoints_or_ignore_wall():
    domain = box(-3, -2, 26, 2)
    wall = box(10, -3, 12, 3)
    r, stats = connect(
        Ground(),
        [0, 0, 0],
        [23, 0, 0],
        SIZE,
        allowed_area=domain,
        obstacles=[wall],
        max_expansions=10,
        max_length_m=50.0,
    )
    assert r is None
    assert stats.get("budget_exhausted", 0) + stats.get("search_exhausted", 0) == 1
    with pytest.raises(ValueError):
        connect(Ground(), [0, 0, 0], [23, 0, 0], SIZE, allowed_area=domain, max_expansions=0)


def test_already_at_goal_does_not_invent_a_loop_or_rotate_in_place():
    r, stats = connect(Ground(), [0, 0, 0.7], [0, 0, 0.7], SIZE, allowed_area=box(-10, -10, 10, 10))
    assert r is None and stats == {"already_at_goal": 1}


def test_finer_search_keeps_the_same_physical_checks():
    domain = box(-3, -6, 26, 6)
    wall = box(10, -0.8, 12, 0.8)
    r, stats = connect(
        Ground(),
        [0, 0, 0],
        [23, 0, 0],
        SIZE,
        allowed_area=domain,
        obstacles=[wall],
        max_expansions=4000,
        max_length_m=50.0,
        resolution_m=0.25,
        heading_bins=144,
        primitive_length_m=0.75,
    )
    assert r is not None, stats
    assert admit(Ground(), r, SIZE, allowed_area=domain, obstacles=[wall])[0]
    np.testing.assert_allclose(r["points"][-1], [23, 0], atol=1e-6)
    for length in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            connect(Ground(), [0, 0, 0], [23, 0, 0], SIZE, allowed_area=domain, primitive_length_m=length)


@pytest.mark.parametrize("side", ["start", "goal"])
def test_invalid_endpoint_rejected_before_search_without_snapping(side):
    start, goal = np.array([0.0, 0.0, 0.0]), np.array([23.0, 0.0, 0.0])
    # Centre lies inside, but the actual body does not.
    domain = box(-1 if side == "start" else -3, -3, 24 if side == "goal" else 26, 3)
    route, stats = connect(Ground(), start, goal, SIZE, allowed_area=domain)
    assert route is None and stats == {side + "_outside_permitted_area": 1}
    np.testing.assert_array_equal(start, [0, 0, 0])
    np.testing.assert_array_equal(goal, [23, 0, 0])


def test_blocked_and_unsupported_endpoints_do_not_consume_search_budget():
    domain = box(-3, -3, 26, 3)
    route, stats = connect(
        Ground(), [0, 0, 0], [23, 0, 0], SIZE, allowed_area=domain, obstacles=[box(22, -0.1, 24, 0.1)]
    )
    assert route is None and stats == {"goal_static_obstacle": 1}

    class MissingGoalGround:
        def supported(self, xy, profile):
            return np.asarray(xy)[:, 0] < 20.0

    route, stats = connect(MissingGoalGround(), [0, 0, 0], [23, 0, 0], SIZE, allowed_area=domain)
    assert route is None and stats == {"goal_unsupported": 1}


def test_endpoint_admission_matches_body_boundary_and_keeps_obstacle_margin():
    domain = box(-2.2, -1.0, 25.2, 1.0)
    route, stats = connect(Ground(), [0, 0, 0], [23, 0, 0], SIZE, allowed_area=domain)
    assert route is not None, stats
    assert admit(Ground(), route, SIZE, allowed_area=domain) == (True, None)
    route, stats = connect(
        Ground(), [0, 0, 0], [23, 0, 0], SIZE, allowed_area=domain, obstacles=[box(-0.1, 1.1, 0.1, 1.2)]
    )
    assert route is None and stats == {"start_static_obstacle": 1}


def test_prepared_permission_preserves_concave_boundary_holes_and_tolerance():
    from shapely.ops import unary_union

    from awr.life.civil.vehicle_connections import PreparedPermission

    area = unary_union([box(-10, -3, 30, 3), box(10, -10, 16, 10)]).difference(box(12, -1, 14, 1))
    prepared = PreparedPermission(area)
    for x in np.linspace(-9, 29, 21):
        for y in (-6.0, -1.0, 0.0, 2.8, 6.0):
            r = primitive([x, y, 0.2], 0.1)
            expected = admit(Ground(), r, SIZE, allowed_area=area)
            assert admit(Ground(), r, SIZE, allowed_area=prepared) == expected
    with pytest.raises(ValueError):
        PreparedPermission(box(0, 0, 0, 0))


def test_static_route_guide_finds_safe_detour_with_bounded_search_effort():
    start, goal = [0, 0, 0], [23, 0, 0]
    area, obstacles = box(-3, -6, 26, 6), [box(10, -0.8, 12, 0.8)]
    plain, baseline = connect(
        Ground(), start, goal, SIZE, allowed_area=area, obstacles=obstacles, max_expansions=32, max_length_m=50.0
    )
    assert plain is None and baseline["budget_exhausted"] == 1
    guide = [[0, 0], [7, -3], [15, -3], [23, 0]]
    route, stats = connect(
        Ground(),
        start,
        goal,
        SIZE,
        allowed_area=area,
        obstacles=obstacles,
        max_expansions=32,
        max_length_m=50.0,
        guide_points=guide,
    )
    assert route is not None and stats["expanded"] < 32 and stats["guide_queries"] > 0
    assert admit(Ground(), route, SIZE, allowed_area=area, obstacles=obstacles) == (True, None)
    np.testing.assert_allclose(route["points"][[0, -1]], np.array([start, goal])[:, :2], atol=1e-6)
    np.testing.assert_allclose(route["yaw"][[0, -1]], [0, 0], atol=1e-6)
    again, repeated = connect(
        Ground(),
        start,
        goal,
        SIZE,
        allowed_area=area,
        obstacles=obstacles,
        max_expansions=32,
        max_length_m=50.0,
        guide_points=guide,
    )
    assert repeated == stats
    np.testing.assert_array_equal(again["points"], route["points"])


def test_guide_cannot_authorize_wall_crossing_or_move_endpoints():
    area = box(-3, -2, 26, 2)
    wall = box(10, -3, 12, 3)
    route, stats = connect(
        Ground(),
        [0, 0, 0],
        [23, 0, 0],
        SIZE,
        allowed_area=area,
        obstacles=[wall],
        max_expansions=32,
        max_length_m=50.0,
        guide_points=[[0, 0], [23, 0]],
    )
    assert route is None and stats.get("static_obstacle", 0) > 0
    with pytest.raises(ValueError, match="exact XY endpoints"):
        connect(Ground(), [0, 0, 0], [23, 0, 0], SIZE, allowed_area=area, guide_points=[[1, 0], [23, 0]])
    with pytest.raises(ValueError, match="duplicate"):
        connect(Ground(), [0, 0, 0], [23, 0, 0], SIZE, allowed_area=area, guide_points=[[0, 0], [0, 0], [23, 0]])


def test_long_guide_does_not_prune_a_short_physically_valid_route():
    route, stats = connect(
        Ground(),
        [0, 0, 0],
        [23, 0, 0],
        SIZE,
        allowed_area=box(-3, -2, 26, 2),
        max_length_m=25.0,
        guide_points=[[0, 0], [0, 100], [23, 100], [23, 0]],
    )
    assert route is not None and route["arc"][-1] <= 25.0
    assert stats["expanded"] == 1
