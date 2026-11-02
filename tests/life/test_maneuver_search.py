import math

import numpy as np
import pytest
from shapely.geometry import box

from awr.life.civil.driving_search import primitive
from awr.life.civil.maneuver_search import connect
from awr.life.civil.reeds_shepp import candidates
from awr.life.civil.vehicle_connections import admit

SIZE = [4.2, 1.8, 1.5]
START, GOAL = [0, -1, 0], [0, 1, math.pi]
AREA = box(-12, -3.2, 12, 3.2)


class Ground:
    def supported(self, xy, profile):
        return np.ones(len(xy), bool)


@pytest.fixture(autouse=True)
def contacts(monkeypatch):
    monkeypatch.setattr(
        "awr.life.civil.contacts.vehicle_contacts",
        lambda size: np.array([[-1.4, -0.7], [-1.4, 0.7], [1.4, -0.7], [1.4, 0.7]]),
    )


def test_reverse_primitive_respects_heading_and_signed_motion():
    for curvature in (0.0, -0.2, 0.2):
        route = primitive([13, 4, 0.7], curvature, gear=-1)
        ds = np.diff(route["arc"])
        middle_yaw = (route["yaw"][1:] + route["yaw"][:-1]) / 2
        np.testing.assert_allclose(
            np.diff(route["points"], axis=0) / ds[:, None],
            -np.column_stack([np.cos(middle_yaw), np.sin(middle_yaw)]),
            atol=1e-8,
        )
        assert (route["gear"] == -1).all()
        assert route["yaw"][-1] == pytest.approx(0.7 - curvature * 1.5)


def test_multi_point_turn_where_analytic_families_leave_road():
    pytest.importorskip("rsplan")
    assert not any(
        admit(Ground(), r, SIZE, allowed_area=AREA)[0] for r in candidates(START, GOAL, SIZE, max_length_m=50.0)
    )
    route, stats = connect(Ground(), START, GOAL, SIZE, maneuver_area=AREA, max_expansions=100, max_length_m=50.0)
    assert route is not None, stats
    assert admit(Ground(), route, SIZE, allowed_area=AREA) == (True, None)
    np.testing.assert_allclose(route["points"][[0, -1]], np.array([START, GOAL])[:, :2], atol=1e-7)
    assert abs(abs(route["yaw"][-1]) - math.pi) < 1e-7
    assert 3 < len(route["required_stops_m"]) <= 6
    np.testing.assert_allclose(route["required_stops_m"], route["arc"][np.flatnonzero(np.diff(route["gear"])) + 1])
    replay, repeated = connect(Ground(), START, GOAL, SIZE, maneuver_area=AREA, max_expansions=100, max_length_m=50.0)
    from awr.life.civil.driving_motion import Schedule

    motion = Schedule(route)
    assert len(motion.gear_stops) == len(route["required_stops_m"])
    for a, b, arc in motion.gear_stops:
        stopped = motion.at((a + b) / 2)
        assert stopped["speed_mps"] == 0 and stopped["gear"] == 0
        assert stopped["arc_m"] == pytest.approx(arc)
    assert repeated == stats
    np.testing.assert_array_equal(replay["points"], route["points"])


def test_gear_and_expansion_budgets_are_not_silently_relaxed():
    pytest.importorskip("rsplan")
    for options in (dict(max_gear_changes=0, max_expansions=100), dict(max_expansions=1)):
        route, stats = connect(Ground(), START, GOAL, SIZE, maneuver_area=AREA, **options)
        assert route is None
        assert stats.get("budget_exhausted", 0) + stats.get("search_exhausted", 0) == 1


def test_obstacle_and_unsupported_interior_cannot_be_crossed():
    pytest.importorskip("rsplan")
    # Endpoints supported and unblocked; every connecting path must cross x=0.
    wall = box(-0.04, -4, 0.04, 4)
    route, stats = connect(
        Ground(), [-6, 0, 0], [6, 0, 0], SIZE, maneuver_area=AREA, obstacles=[wall], max_expansions=100
    )
    assert route is None and stats.get("static_obstacle", 0) + stats.get("finish_static_obstacle", 0) > 0

    # A whole-footprint gap avoids relying on a sample accidentally hitting a
    # tiny stripe. Endpoints are clear; the car cannot leap the centre section.
    class Gap:
        def supported(self, xy, profile):
            return np.abs(np.asarray(xy)[:, 0]) > 1.0

    route, stats = connect(Gap(), [-6, 0, 0], [6, 0, 0], SIZE, maneuver_area=AREA, max_expansions=100)
    assert (
        route is None and stats.get("unsupported_or_steering", 0) + stats.get("finish_unsupported_or_steering", 0) > 0
    )
    assert not any(k.endswith("_unsupported") and k.startswith(("start", "goal")) for k in stats)


def test_explicit_domain_and_valid_settings_are_required():
    for options in (dict(maneuver_area=None), dict(max_gear_changes=-1), dict(gear_change_cost_m=-1)):
        args = dict(maneuver_area=AREA)
        args.update(options)
        with pytest.raises(ValueError):
            connect(Ground(), START, GOAL, SIZE, **args)


def test_backing_is_confined_to_parking_but_forward_travel_can_leave():
    pytest.importorskip("rsplan")
    parking = box(-2.2, -2, 2.2, 2)
    # Both the primitive and analytic finish must enforce the complete body,
    # even if the centre itself would still be inside the parking polygon.
    route, stats = connect(
        Ground(),
        [0, 0, 0],
        [-1, 0, 0],
        SIZE,
        maneuver_area=AREA,
        reverse_area=parking,
        max_length_m=1.01,
        max_expansions=10,
        primitive_length_m=0.5,
    )
    assert route is None and stats.get("finish_reverse_outside_permitted_area", 0) > 0
    assert stats.get("reverse_outside_permitted_area", 0) > 0
    route, stats = connect(
        Ground(),
        [0, 0, 0],
        [4, 0, 0],
        SIZE,
        maneuver_area=AREA,
        reverse_area=parking,
        max_length_m=4.01,
        max_expansions=10,
    )
    assert route is not None and (route["gear"] == 1).all(), stats
    # The same backing route is possible when its actual body fits the area.
    route, stats = connect(
        Ground(),
        [0, 0, 0],
        [-1, 0, 0],
        SIZE,
        maneuver_area=AREA,
        reverse_area=box(-4, -2, 3, 2),
        max_length_m=1.01,
        max_expansions=10,
    )
    assert route is not None and (route["gear"] == -1).all(), stats


def test_reverse_domain_cannot_enlarge_global_permission_or_skip_cusps():
    from awr.life.civil.driving_graph import route_arrays
    from awr.life.civil.maneuver_search import reverse_contained
    from awr.life.civil.vehicle_connections import PreparedPermission

    with pytest.raises(ValueError, match="inside"):
        connect(Ground(), START, GOAL, SIZE, maneuver_area=AREA, reverse_area=box(-20, -20, 20, 20))
    r = route_arrays([[0, 0], [1, 0], [0, 0], [2, 0]], [0, 0, 0, 0], [1, -1, 1])
    assert not reverse_contained(r, SIZE, PreparedPermission(box(-2.2, -2, 2.2, 2)))
    assert reverse_contained(r, SIZE, PreparedPermission(box(-2.2, -2, 3.2, 2)))


def test_contiguous_backing_limit_crosses_primitive_and_finish_seams():
    from awr.life.civil.driving_graph import route_arrays
    from awr.life.civil.maneuver_search import reverse_run

    last = route_arrays([[0, 0], [-2, 0], [-3, 0]], [0, 0, 0], [-1, -1])
    assert reverse_run(last, initial_m=3.0, limit_m=6.0) == (True, 6.0)
    assert reverse_run(last, initial_m=3.01, limit_m=6.0)[0] is False
    # A genuine forward leg ends the preceding reverse run.
    reset = route_arrays([[0, 0], [1, 0], [-2, 0]], [0, 0, 0], [1, -1])
    assert reverse_run(reset, initial_m=5.0, limit_m=6.0) == (True, 3.0)


def test_search_rejects_overlong_backing_before_returning_a_candidate():
    route, stats = connect(
        Ground(),
        [0, 0, 0],
        [-4, 0, 0],
        SIZE,
        maneuver_area=AREA,
        max_length_m=4.01,
        max_expansions=100,
        allowed_gears=(-1,),
        max_reverse_leg_m=3.0,
    )
    assert route is None
    assert stats["finish_reverse_leg_budget"] > 0 and stats["reverse_leg_budget"] > 0
    route, stats = connect(
        Ground(),
        [0, 0, 0],
        [-4, 0, 0],
        SIZE,
        maneuver_area=AREA,
        max_length_m=4.01,
        max_expansions=100,
        allowed_gears=(-1,),
        max_reverse_leg_m=4.0,
    )
    assert route is not None and route["arc"][-1] == pytest.approx(4.0)


def test_final_alignment_reserve_cannot_extend_a_reverse_leg_beyond_limit():
    kwargs = dict(maneuver_area=AREA, max_length_m=3.01, max_expansions=40, allowed_gears=(-1,), max_reverse_leg_m=4.0)
    route, stats = connect(Ground(), [0, 0, 0], [-3, 0, 0], SIZE, terminal_reverse_reserve_m=2.0, **kwargs)
    assert route is None and stats["finish_reverse_leg_budget"] > 0
    route, stats = connect(Ground(), [0, 0, 0], [-3, 0, 0], SIZE, terminal_reverse_reserve_m=1.0, **kwargs)
    assert route is not None
    route, stats = connect(
        Ground(),
        [0, 0, 0],
        [6, 0, 0],
        SIZE,
        maneuver_area=AREA,
        max_length_m=6.01,
        max_reverse_leg_m=1.0,
        allowed_gears=(1,),
    )
    assert route is not None and (route["gear"] == 1).all()
    for options in (
        dict(max_reverse_leg_m=0),
        dict(max_reverse_leg_m=float("inf")),
        dict(terminal_reverse_reserve_m=1.0),
        dict(max_reverse_leg_m=2.0, terminal_reverse_reserve_m=3.0),
    ):
        with pytest.raises(ValueError, match="reverse"):
            connect(Ground(), [0, 0, 0], [-3, 0, 0], SIZE, maneuver_area=AREA, **options)


def test_search_finds_another_turn_instead_of_returning_a_late_rejected_one():
    def longest_reverse(route):
        breaks = np.r_[0, np.flatnonzero(np.diff(route["gear"])) + 1, len(route["gear"])]
        return max(
            route["arc"][b] - route["arc"][a]
            for a, b in zip(breaks[:-1], breaks[1:], strict=False)
            if route["gear"][a] < 0
        )

    options = dict(maneuver_area=AREA, max_expansions=200, max_length_m=50.0)
    old, _ = connect(Ground(), START, GOAL, SIZE, **options)
    assert old is not None and longest_reverse(old) > 3.0
    new, stats = connect(Ground(), START, GOAL, SIZE, max_reverse_leg_m=3.0, **options)
    assert new is not None and longest_reverse(new) <= 3.0
    assert stats["finish_reverse_leg_budget"] > 0 and stats["expanded"] <= 200
    assert admit(Ground(), new, SIZE, allowed_area=AREA) == (True, None)
    repeated, again = connect(Ground(), START, GOAL, SIZE, max_reverse_leg_m=3.0, **options)
    assert again == stats
    np.testing.assert_array_equal(new["points"], repeated["points"])
