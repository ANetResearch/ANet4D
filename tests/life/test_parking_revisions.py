"""Parking-induced replanning at a world-step boundary, using real geometry.

Flat synthetic roads isolate transaction safety; they do not certify the town.
"""

import numpy as np
import pytest
from test_traffic_runs import SIZE, contacts, detour  # noqa: F401

from awr.life.civil.activity.driving import Body
from awr.life.civil.curb_allocation import footprint
from awr.life.motion.traffic_evidence import snapshot
from awr.life.motion.vehicles import LiveDrive


def active():
    routes, starts = detour()
    for v in ("y", "x"):
        plan = routes.plan(v, v + "0", v + "1")
        assert routes.acquire(plan.evidence, starts[v], 0.0)
        routes.executions[v] = LiveDrive(routes.plans[plan.evidence]["schedule"].route)
    return routes, starts


def obstruction(routes):
    p = routes.graph.nodes["x1"]["pose"]
    return footprint(dict(xy=p[:2], yaw=p[2]), SIZE)


def test_execution_refusal_leaves_all_reservations_and_controllers_unchanged(monkeypatch):
    routes, _ = active()
    before = snapshot(routes)
    controller = routes.executions["y"]

    def refuse(*args):
        raise ValueError("unsupported route transition")

    monkeypatch.setattr(controller, "prepare_reroute", refuse)
    assert not routes.revise("y", obstruction(routes), reason="test-parking")
    after = snapshot(routes)
    for key in ("trips", "occupancy", "executions", "resources"):
        assert before[key] == after[key]
    assert routes.executions["y"] is controller and not routes.revisions


def test_reservation_refusal_rolls_back_prior_layout_updates(monkeypatch):
    routes, _ = active()
    before = snapshot(routes)
    old = dict(routes.active)
    controllers = dict(routes.executions)
    refresh = routes.traffic._refresh

    def refuse_last(vehicle, *, check=False):
        # The other trip is refreshed first; refusal must restore both trips.
        return False if vehicle == "y" and check else refresh(vehicle, check=check)

    monkeypatch.setattr(routes.traffic, "_refresh", refuse_last)
    assert not routes.revise("y", obstruction(routes), reason="test-parking")
    after = snapshot(routes)
    for key in ("trips", "occupancy", "executions", "resources"):
        assert before[key] == after[key]
    assert routes.active == old and not routes.revisions
    assert all(routes.executions[v] is controllers[v] for v in controllers)


def test_failed_traffic_retries_reuse_geometry_but_recheck_reservations(monkeypatch):
    routes, _ = active()
    monkeypatch.setattr(routes.traffic, "revise", lambda *a: False)
    blocked = obstruction(routes)
    assert not routes.revise("y", blocked, reason="test-parking")
    searches = routes.stats["path_searches"]
    assert not routes.revise("y", blocked, reason="test-parking")
    assert routes.stats["path_searches"] == searches
    assert routes.stats["revision_path_hits"] > 0


def test_a_slowing_car_reconsiders_a_splice_on_the_same_edge():
    routes, _ = active()
    executor = routes.executions["y"]
    executor.arc_m, executor.speed_mps = 28.0, 3.0
    blocked = obstruction(routes)
    assert not routes.revise("y", blocked, reason="test-parking")
    executor.speed_mps = 0.0
    assert routes.revise("y", blocked, reason="test-parking")
    new = routes.executions["y"]
    assert new is not executor
    assert new.arc_m == executor.arc_m and new.time_s == executor.time_s
    assert np.array_equal(new.state()["xy"], executor.state()["xy"])


def test_replanning_cache_is_invalidated_by_a_parked_body_move(monkeypatch):
    routes, starts = detour()
    plan = routes.plan("y", "y0", "y1")
    assert routes.acquire(plan.evidence, starts["y"], 0.0)
    routes.executions["y"] = LiveDrive(routes.plans[plan.evidence]["schedule"].route)
    monkeypatch.setattr(routes.traffic, "revise", lambda *a: False)
    assert not routes.revise("y", obstruction(routes), reason="test-parking")
    searches = routes.stats["path_searches"]
    assert routes._revision_paths
    routes.update_occupancy(dict(routes.occupancy, x=Body((0.0, -29.0, np.pi / 2), SIZE)))
    assert not routes._revision_paths
    assert not routes.revise("y", obstruction(routes), reason="test-parking")
    assert routes.stats["path_searches"] > searches


def test_motion_limits_never_revise_another_car_mid_batch(monkeypatch):
    routes, _ = active()
    last = routes.traffic.layout("x").runs[-1]

    def forbidden(*args, **kwargs):
        raise AssertionError("replan inside motion proposal")

    monkeypatch.setattr(routes, "revise", forbidden)
    assert routes.traffic_limit("x", last.stop - 1.0, 0.0) == last.stop


def test_live_replanned_cars_reach_their_original_berths_without_body_contact():
    routes, starts = active()
    poses = {v: np.asarray(p) for v, p in starts.items()}
    speeds = dict.fromkeys(starts, 0.0)
    done = set()
    for _ in range(1500):
        routes.prepare_revisions()
        assert all(h in t["runs"] for t in routes.traffic.trips.values() for h in t["held"])
        for v in sorted(routes.active):
            evidence = routes.active[v]
            state = routes.advance_motion(evidence, poses[v], speeds[v], 0.1)
            poses[v] = np.r_[state["xy"], state["yaw"]]
            speeds[v] = state["speed_mps"]
        routes.update_occupancy({v: Body(tuple(p), SIZE) for v, p in poses.items()})
        # Actual bodies, not a nominal reservation label.
        assert not routes.occupancy["x"].polygon().intersects(routes.occupancy["y"].polygon())
        for v in list(routes.active):
            if routes.executions[v].arrived:
                assert routes.can_arrive(routes.active[v], poses[v], speeds[v], require_executed=True)
                assert routes.arrive(routes.active[v], poses[v], speeds[v])
                done.add(v)
        if len(done) == 2:
            break
    assert done == {"x", "y"}
    assert routes.stats.get("revisions", 0) >= 1
    assert len(routes.completed_motion) == 2


def test_reroute_checks_unsampled_prefix_and_gears_without_mutating_controller():
    points = np.column_stack([np.arange(61.0), np.zeros(61)])
    route = dict(points=points, yaw=np.zeros(61))
    controller = LiveDrive(route)
    altered = points.copy()
    altered[1, 1] = 0.01
    with pytest.raises(ValueError, match="follow the present"):
        controller.prepare_reroute(dict(points=altered, yaw=np.zeros(61)), 30.0)
    gear = np.ones(60, dtype=int)
    gear[1] = -1
    with pytest.raises(ValueError, match="follow the present"):
        controller.prepare_reroute(dict(route, gear=gear), 30.0)
    assert np.array_equal(controller.route["points"], points)
    assert controller.arc_m == controller.time_s == 0.0
