import math

import numpy as np
import pytest
from shapely.geometry import box

from awr.life.civil.activity.driving import Body, DrivingRoutes
from awr.life.civil.driving_graph import DrivingGraph, route_arrays

SIZE = (4.2, 1.8, 1.5)


class Ground:
    def supported(self, xy, profile):
        return np.ones(len(xy), bool)


@pytest.fixture(autouse=True)
def contacts(monkeypatch):
    monkeypatch.setattr(
        "awr.life.civil.contacts.vehicle_contacts",
        lambda size: np.array([[-1.4, -0.7], [-1.4, 0.7], [1.4, -0.7], [1.4, 0.7]]),
    )


def fixture(*, crossing=True, resource=False, surface=None, b_end=None, extra=None):
    graph = DrivingGraph()
    starts = dict(a=(-12, 0, 0), b=(0, -12, math.pi / 2) if crossing else (-12, 10, 0))
    ends = dict(a=(12, 0, 0), b=(0, 12, math.pi / 2) if crossing else (12, 10, 0))
    if b_end is not None:
        ends["b"] = b_end
    areas = {}
    for vehicle in starts:
        a, b = starts[vehicle], ends[vehicle]
        graph.add_node(vehicle + "0", a, kind="berth")
        graph.add_node(vehicle + "1", b, kind="berth")
        graph.add_edge(
            vehicle,
            vehicle + "0",
            vehicle + "1",
            route_arrays(np.linspace(a[:2], b[:2], 100), np.full(100, a[2])),
            kind="travel",
            blockers=["b"] if vehicle == "a" else [],
            traffic_resource="shared" if resource else None,
        )
        areas[vehicle] = box(-15, -15, 15, 15)
    occupancy = {v: Body(p, SIZE) for v, p in {**starts, **(extra or {})}.items()}
    return DrivingRoutes(graph, surface or Ground(), SIZE, areas, occupancy), starts, ends


def drive(routes, cars, starts, ticks=600):
    """Advance the given trips together by live motion proposals; returns the per-tick traffic holds."""
    poses = {v: np.asarray(starts[v], float) for v in starts}
    speeds = {v: 0.0 for v in starts}
    holds = []
    for _ in range(ticks):
        tick = {}
        for v, evidence in sorted(cars.items()):
            if v not in routes.active:
                continue
            state = routes.executions[v].state() if v in routes.executions else None
            if state is not None and state["arrived"]:
                continue
            tick[v] = routes.traffic_limit(v, state["arc_m"] if state else 0.0, speeds[v]) if state else None
            result = routes.advance_motion(evidence, poses[v], speeds[v], 0.1)
            poses[v] = np.r_[result["xy"], result["yaw"]]
            speeds[v] = result["speed_mps"]
        routes.update_occupancy({v: Body(tuple(poses[v]), SIZE) for v in starts})
        bodies = [routes.occupancy[v].polygon() for v in cars]
        tick["gap"] = min((p.distance(q) for i, p in enumerate(bodies) for q in bodies[i + 1 :]), default=math.inf)
        holds.append(tick)
        if all(routes.executions[v].state()["arrived"] for v in cars if v in routes.executions):
            break
    return poses, holds


def test_crossing_cars_take_the_crossing_in_turn():
    routes, starts, ends = fixture()
    a, b = (routes.plan(v, v + "0", v + "1") for v in ("a", "b"))
    assert routes.acquire(a.evidence, starts["a"], 0) and routes.acquire(b.evidence, starts["b"], 0)
    runs = {v: routes.traffic.layout(r.evidence).runs for v, r in (("a", a), ("b", b))}
    assert [len(r) for r in runs.values()] == [1, 1] and 0 < runs["a"][0].stop < 12
    poses, holds = drive(routes, dict(a=a.evidence, b=b.evidence), starts)
    held = [t for t in holds if any(h is not None for v, h in t.items() if v != "gap")]
    assert held, "one car waits before the crossing"
    assert min(t["gap"] for t in holds) > 0.0
    assert all(routes.executions[v].state()["arrived"] for v in "ab")
    # One car is refused the crossing, or, holding it, stops before it to let the other through (a yield).
    assert routes.stats["traffic_rejections"] == 0
    assert routes.traffic.stats["refused_conflict"] + routes.traffic.stats.get("yields", 0) > 0


def test_arrival_is_observed_not_predicted():
    routes, starts, ends = fixture()
    a = routes.plan("a", "a0", "a1")
    assert routes.acquire(a.evidence, starts["a"], 0)
    assert routes.nominal_at(a.evidence, a.duration_s + 1000)["arrived"]
    # Predicted arrival/time passage is not observed arrival or resource release.
    assert not routes.arrive(a.evidence, ends["a"], 1.0)
    assert not routes.arrive(a.evidence, (12, 0, math.pi), 0.0)
    assert not routes.arrive(a.evidence, (11, 0, 0), 0.0)
    assert not routes.arrive(a.evidence, ends["a"], 0.0)  # snapshot still at source
    with pytest.raises(ValueError, match="disappear"):
        routes.update_occupancy(dict(a=None, b=Body(starts["b"], SIZE)))
    routes.update_occupancy(dict(a=Body(ends["a"], SIZE), b=Body(starts["b"], SIZE)))
    assert routes.arrive(a.evidence, ends["a"], 0.0)
    with pytest.raises(ValueError, match="admission"):
        routes.nominal_at(a.evidence, 0)


def test_separate_roads_move_concurrently_whatever_their_labels():
    for shared in (False, True):
        routes, starts, ends = fixture(crossing=False, resource=shared)
        a, b = (routes.plan(v, v + "0", v + "1") for v in ("a", "b"))
        assert routes.acquire(a.evidence, starts["a"], 0) and routes.acquire(b.evidence, starts["b"], 0)
        assert not routes.traffic.layout(a.evidence).runs
        poses, holds = drive(routes, dict(a=a.evidence, b=b.evidence), starts)
        assert all(h is None for t in holds for v, h in t.items() if v != "gap")


def test_current_bodies_replace_old_blocker_labels_without_erasing_population():
    routes, starts, ends = fixture()
    initial = routes.plan("a", "a0", "a1")
    routes.update_occupancy(dict(a=Body(starts["a"], SIZE), b=Body((0, 0, 0), SIZE)))
    assert routes.plan("a", "a0", "a1") is None
    assert not routes.acquire(initial.evidence, starts["a"], 0)
    assert routes.active == {}
    routes.update_occupancy(dict(a=Body(starts["a"], SIZE), b=Body(starts["b"], SIZE)))
    again = routes.plan("a", "a0", "a1")
    assert again == initial
    assert routes.stats["edge_checks"] == 1  # frozen geometry is not rebuilt
    with pytest.raises(ValueError, match="complete"):
        routes.update_occupancy(dict(a=Body(starts["a"], SIZE)))


def test_compiled_plan_reuse_still_rechecks_new_occupancy_and_limits(monkeypatch):
    import awr.life.civil.activity.driving as driving

    calls = []
    original = driving.swept

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(driving, "swept", counted)
    routes, starts, _ = fixture()
    routes.cache_plan_results = False  # independently exercise compiled geometry reuse
    first = routes.plan("a", "a0", "a1")
    compiled = len(calls)
    for _ in range(5):
        assert routes.plan("a", "a0", "a1") == first
    assert len(calls) == compiled
    assert routes.stats["plan_cache_hits"] == 5
    # Geometry reuse must not reuse an earlier occupancy decision.
    routes.update_occupancy(dict(a=Body(starts["a"], SIZE), b=Body((0, 0, 0), SIZE)))
    assert routes.plan("a", "a0", "a1") is None
    assert not routes.acquire(first.evidence, starts["a"], 0.0)
    routes.update_occupancy(dict(a=Body(starts["a"], SIZE), b=Body(starts["b"], SIZE)))
    assert routes.plan("a", "a0", "a1") == first
    routes.limits = driving.Limits(forward_mps=2.0)
    slower = routes.plan("a", "a0", "a1")
    assert slower.duration_s > first.duration_s and slower.evidence != first.evidence
    assert len(calls) == compiled + 1


def test_snapshot_planning_matches_uncached_and_invalidates_negative_results(monkeypatch):
    cached, starts, _ = fixture()
    reference, _, _ = fixture()
    reference.cache_plan_results = False
    calls = []
    path = cached.graph.path

    def counted(*args, **kwargs):
        calls.append(1)
        return path(*args, **kwargs)

    monkeypatch.setattr(cached.graph, "path", counted)
    for other in (Body(starts["b"], SIZE), Body((0.0, 0.0, 0.0), SIZE), None, Body(starts["b"], SIZE)):
        occupancy = dict(a=Body(starts["a"], SIZE), b=other)
        for backend in (cached, reference):
            backend.update_occupancy(occupancy)
        expected = reference.plan("a", "a0", "a1")
        assert cached.plan("a", "a0", "a1") == expected
        before = len(calls)
        for _ in range(4):
            # Identical observations must not erase useful cache entries.
            cached.update_occupancy(dict(occupancy))
            assert cached.plan("a", "a0", "a1") == expected
        assert len(calls) == before
    assert cached.stats["snapshot_negative_hits"] == 4
    assert cached.stats["snapshot_plan_hits"] == 16


def test_snapshot_results_change_with_limits_and_reservations():
    from awr.life.civil.driving_motion import Limits

    routes, starts, _ = fixture(b_end=(0.0, 0.0, math.pi / 2))
    initial = routes.plan("a", "a0", "a1")
    assert routes.plan("a", "a0", "a1") == initial
    routes.limits = Limits(forward_mps=2.0)
    slow = routes.plan("a", "a0", "a1")
    assert slow.duration_s > initial.duration_s
    b = routes.plan("b", "b0", "b1")
    assert routes.acquire(b.evidence, starts["b"], 0.0)
    # No observed pose changed, but the newly reserved destination blocks a.
    assert routes.plan("a", "a0", "a1") is None


def test_each_route_is_physically_readmitted_for_actual_body_and_own_domain():
    class Gap:
        def supported(self, xy, profile):
            return np.abs(np.asarray(xy)[:, 0]) > 1

    routes, starts, ends = fixture(surface=Gap())
    assert routes.plan("a", "a0", "a1") is None
    routes, starts, ends = fixture()
    routes.areas["a"] = box(-15, -0.8, 15, 0.8)
    assert routes.plan("a", "a0", "a1") is None  # centre fits; actual car does not
    routes, starts, ends = fixture()
    routes.update_occupancy(dict(a=Body(starts["a"], (6, 2, 2)), b=Body(starts["b"], SIZE)))
    with pytest.raises(ValueError, match="body differs"):
        routes.plan("a", "a0", "a1")


def test_departure_requires_exact_stopped_vehicle_and_unchanged_body():
    routes, starts, ends = fixture()
    a = routes.plan("a", "a0", "a1")
    assert not routes.acquire(a.evidence, starts["a"], 1)
    assert not routes.acquire(a.evidence, (-11, 0, 0), 0)
    routes.update_occupancy(dict(a=Body(starts["a"], (6, 2, 2)), b=Body(starts["b"], SIZE)))
    assert not routes.acquire(a.evidence, starts["a"], 0)
    assert routes.active == {}


def test_invalid_map_or_unknown_berth_fails_closed():
    routes, _, _ = fixture()
    with pytest.raises(ValueError, match="known map blockers"):
        DrivingRoutes(routes.graph, Ground(), SIZE, routes.areas, dict(a=None))
    with pytest.raises(ValueError, match="permitted area"):
        DrivingRoutes(routes.graph, Ground(), SIZE, {}, routes.occupancy)
    with pytest.raises(ValueError, match="exact mapped berths"):
        routes.plan("a", "a0", "nearby")


def test_director_callback_uses_exact_berths_without_snapping():
    routes, starts, ends = fixture()
    expected = routes.plan("a", "a0", "a1")
    assert routes.route("drive", starts["a"][:2], ends["a"][:2], "a") == expected
    assert routes.route("walk", starts["a"][:2], ends["a"][:2], "a") is None
    assert routes.route("drive", (-11.99, 0), ends["a"][:2], "a") is None
    import copy

    graph = copy.deepcopy(routes.graph)
    graph.add_node("other-heading", [-12, 0, math.pi], kind="berth")
    ambiguous = DrivingRoutes(graph, Ground(), SIZE, routes.areas, routes.occupancy)
    assert ambiguous.route("drive", starts["a"][:2], ends["a"][:2], "a") is None


def test_backend_geometry_is_frozen_including_equal_size_edits():
    import copy

    routes, _, _ = fixture()
    with pytest.raises(ValueError, match="Frozen"):
        routes.graph.add_node("new", (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="Frozen"):
        routes.graph.edges = {}
    with pytest.raises(TypeError):
        routes.graph.edges["a"]["target"] = "b1"
    with pytest.raises(ValueError):
        routes.graph.edges["a"]["route"]["arc"][-1] = 1.0
    with pytest.raises(TypeError):
        routes.graph.links["a0"] = ("b",)
    with pytest.raises(ValueError):
        routes.graph.nodes["a0"]["pose"][0] = 100.0
    # Recompilation can edit a NEW graph without invalidating a live backend.
    editable = copy.deepcopy(routes.graph)
    editable.nodes["a0"]["pose"][0] = 100.0
    assert routes.graph.nodes["a0"]["pose"][0] == -12.0


def test_bounded_static_trees_match_reference_paths_and_evict():
    import life_toytown as toy

    w = toy.build(drivers=3, reverse_exit_m=2.5)
    routes = w.routes
    routes.static_tree_capacity = 2
    nodes = sorted(routes.graph.nodes)
    for source in nodes[::3]:
        for target in nodes:
            assert routes._static_path(source, target) == routes.graph.path(source, target)
        assert len(routes._static_trees) <= 2
    assert routes.stats["static_tree_hits"] > 0
    assert routes.stats["static_tree_evictions"] > 0
    routes.static_tree_capacity = 0
    assert routes._static_path(nodes[0], nodes[-1]) == routes.graph.path(nodes[0], nodes[-1])
    isolated, _, _ = fixture(crossing=False)
    assert isolated._static_path("a0", "b1") is None
    with pytest.raises(ValueError, match="Unknown"):
        isolated._static_path("unknown", "a1")


def test_static_tree_equal_cost_ties_and_blocked_path_fallback(monkeypatch):
    import copy

    original, starts, _ = fixture(crossing=False)
    graph = copy.deepcopy(original.graph)
    graph.add_edge("z-alternate", "a0", "a1", copy.deepcopy(graph.edges["a"]["route"]), kind="travel")
    areas = dict(original.areas, **{"z-alternate": original.areas["a"]})
    routes = DrivingRoutes(graph, Ground(), SIZE, areas, original.occupancy)
    assert routes._static_path("a0", "a1") == graph.path("a0", "a1") == ["a"]
    edge = routes._edge
    # A static choice is only a candidate: failed physical admission forces
    # an exact blocked-edge search, retaining the alternate admitted path.
    monkeypatch.setattr(routes, "_edge", lambda e: (False, "unsupported", None) if e == "a" else edge(e))
    calls = []
    path = routes.graph.path

    def searched(*args, **kwargs):
        calls.append(set(kwargs.get("blocked_edges", ())))
        return path(*args, **kwargs)

    monkeypatch.setattr(routes.graph, "path", searched)
    result = routes.plan("a", "a0", "a1")
    assert calls == [{"a"}]
    assert routes.plans[result.evidence]["edges"] == ("z-alternate",)
    # Real observed occupancy can still reject the alternate and stale plan.
    routes.update_occupancy(dict(a=Body(starts["a"], SIZE), b=Body((0.0, 0.0, 0.0), SIZE)))
    assert routes.plan("a", "a0", "a1") is None
    assert not routes.acquire(result.evidence, starts["a"], 0.0)


def test_tree_and_uncached_backend_match_through_live_changes():
    cached, starts, _ = fixture()
    reference, _, _ = fixture()
    reference.static_tree_capacity = 0
    reference.cache_plan_results = False
    for b in (Body(starts["b"], SIZE), Body((0.0, 0.0, 0.0), SIZE), None, Body(starts["b"], SIZE)):
        for backend in (cached, reference):
            backend.update_occupancy(dict(a=Body(starts["a"], SIZE), b=b))
        assert cached.plan("a", "a0", "a1") == reference.plan("a", "a0", "a1")
    assert cached.stats["static_tree_builds"] == 1


def test_live_motion_stays_reserved_through_braking_until_observed_arrival():
    routes, starts, ends = fixture()
    a = routes.plan("a", "a0", "a1")
    b = routes.plan("b", "b0", "b1")
    assert routes.acquire(a.evidence, starts["a"], 0)
    pose = np.asarray(starts["a"])
    speed = 0.0
    obstruction = None
    for k in range(300):
        if k == 20:
            e = routes.executions["a"]
            obstruction = e.arc_m + e.speed_mps**2 / (2 * routes.limits.braking_mps2) + 0.2
        result = routes.advance_motion(
            a.evidence, pose, speed, 0.1, stop_before_arc_m=obstruction if 20 <= k < 60 else None
        )
        pose = np.r_[result["xy"], result["yaw"]]
        speed = result["speed_mps"]
        routes.update_occupancy(dict(a=Body(pose, SIZE), b=Body(starts["b"], SIZE)))
        if result["arrived"]:
            break
    assert result["arrived"] and routes.arrive(a.evidence, pose, speed)
    assert routes.completed_motion[-1]["brake_count"] == 1
    assert routes.completed_motion[-1]["elapsed_s"] > a.duration_s
    assert routes.acquire(b.evidence, starts["b"], 0)


def test_refused_pose_cannot_be_hidden_by_advancing_or_claiming_arrival():
    routes, starts, ends = fixture()
    a = routes.plan("a", "a0", "a1")
    with pytest.raises(ValueError, match="traffic admission"):
        routes.advance_motion(a.evidence, starts["a"], 0, 0.1)
    assert routes.acquire(a.evidence, starts["a"], 0)
    result = routes.advance_motion(a.evidence, starts["a"], 0, 0.1)
    # A collision hold that kept the body at the source invalidates further
    # motion; it is not a paused clock that can later jump back into its path.
    with pytest.raises(ValueError, match="Measured motion differs"):
        routes.advance_motion(a.evidence, starts["a"], 0, 0.1)
    routes.update_occupancy(dict(a=Body(ends["a"], SIZE), b=Body(starts["b"], SIZE)))
    assert not routes.arrive(a.evidence, ends["a"], 0)
    pose = np.r_[result["xy"], result["yaw"]]
    routes.update_occupancy(dict(a=Body(pose, SIZE), b=Body(starts["b"], SIZE)))
    assert routes.advance_motion(a.evidence, pose, result["speed_mps"], 0.1)["arc_m"] > result["arc_m"]


def test_transient_traffic_does_not_erase_a_route_but_holds_the_crossing():
    routes, starts, ends = fixture()
    b = routes.plan("b", "b0", "b1")
    assert routes.acquire(b.evidence, starts["b"], 0.0)
    # The first car is presently crossing the second car's route.
    routes.update_occupancy(dict(a=Body(starts["a"], SIZE), b=Body((0.0, 0.0, math.pi / 2), SIZE)))
    a = routes.plan("a", "a0", "a1")
    assert a is not None and set(routes.active) == {"b"}
    assert routes.acquire(a.evidence, starts["a"], 0.0)
    run = routes.traffic.layout(a.evidence).runs[0]
    assert routes.traffic_limit("a", run.stop - 1.0, 0.0) == run.stop
    assert routes.occupancy["b"].pose == (0.0, 0.0, math.pi / 2)
    # Only a real observed arrival releases the crossing.
    routes.update_occupancy(dict(a=Body(starts["a"], SIZE), b=Body(ends["b"], SIZE)))
    assert routes.arrive(b.evidence, ends["b"], 0.0)
    assert routes.traffic_limit("a", run.stop - 1.0, 0.0) is None


def test_a_driving_car_final_berth_is_not_erased_from_route_feasibility():
    routes, starts, ends = fixture(b_end=(0.0, 0.0, math.pi / 2))
    assert routes.plan("a", "a0", "a1") is not None
    b = routes.plan("b", "b0", "b1")
    assert routes.acquire(b.evidence, starts["b"], 0.0)
    # Its actual starting body is clear, but its reserved final parking body
    # will obstruct this route: treating all active cars as absent is wrong.
    assert routes.plan("a", "a0", "a1") is None
    assert routes.active == {"b": b.evidence}


def test_repeated_plans_are_reused_only_while_the_observed_state_is_unchanged():
    routes, starts, ends = fixture(crossing=False)
    first = routes.plan("a", "a0", "a1")
    assert routes.plan("a", "a0", "a1") is first and routes.stats["snapshot_plan_hits"] == 1
    checks = routes.stats["edge_checks"]
    # A body moved onto the route: the next request is planned again and refused.
    routes.update_occupancy(dict(a=Body(starts["a"], SIZE), b=Body((0, 0, 0), SIZE)))
    assert routes.plan("a", "a0", "a1") is None and routes.stats["snapshot_plan_hits"] == 1
    assert routes.stats["edge_cache_hits"] >= 1 and routes.stats["edge_checks"] == checks
    routes.update_occupancy(dict(a=Body(starts["a"], SIZE), b=Body(starts["b"], SIZE)))
    assert routes.plan("a", "a0", "a1").evidence == first.evidence


def test_a_berth_outside_the_reachable_map_is_refused_without_a_search():
    routes, starts, ends = fixture(crossing=False)
    assert routes.plan("a", "a0", "b1") is None
    assert routes.route("drive", starts["a"][:2], ends["b"][:2], "a") is None
    assert routes.route("drive", starts["a"][:2], (99.0, 99.0), "a") is None


def test_static_paths_from_the_kept_tree_equal_the_graph_search():
    routes, starts, ends = fixture(crossing=False)
    for source in routes.graph.nodes:
        for target in routes.graph.nodes:
            if source != target:
                assert routes._static_path(source, target) == routes.graph.path(source, target)
    import life_toytown as toy

    world = toy.build(drivers=3, reverse_exit_m=2.5)
    nodes = sorted(world.routes.graph.nodes)
    for source in nodes[::3]:
        for target in nodes[::2]:
            if source != target:
                assert world.routes._static_path(source, target) == world.routes.graph.path(source, target)


def test_plans_hold_while_only_a_car_in_an_active_trip_moves():
    routes, starts, ends = fixture(crossing=False)
    b = routes.plan("b", "b0", "b1")
    assert routes.acquire(b.evidence, starts["b"], 0.0)
    first = routes.plan("a", "a0", "a1")
    hits = routes.stats["snapshot_plan_hits"]
    # b drives on: it is planned at its reserved destination, so a's plan is reused, not planned again.
    moved = (starts["b"][0] + 1.0, starts["b"][1], starts["b"][2])
    routes.update_occupancy(dict(a=Body(starts["a"], SIZE), b=Body(moved, SIZE)))
    assert routes.plan("a", "a0", "a1") is first and routes.stats["snapshot_plan_hits"] == hits + 1
    assert routes.bodies["b"].equals(Body(moved, SIZE).polygon())
    # A parked body that moves changes the plans.
    routes2, starts2, _ = fixture(crossing=False)
    p = routes2.plan("a", "a0", "a1")
    routes2.update_occupancy(dict(a=Body(starts2["a"], SIZE), b=Body((0, 0, 0), SIZE)))
    assert routes2.plan("a", "a0", "a1") is not p


def test_a_pending_journey_occupies_its_destination_in_plans():
    """A car whose journey is admitted but has not departed will stand at its destination berth: plans that must hold
    later (a way back, a way home) do not cross that berth, and the car's own plans are not blocked by it."""
    graph = DrivingGraph()
    poses = dict(a0=(-12, 0, 0), a1=(12, 0, 0), b0=(-12, 10, 0), b1=(12, 10, 0), bx=(0, 0, 0))
    for name, pose in poses.items():
        graph.add_node(name, pose, kind="berth")
    for edge, (u, v) in dict(a=("a0", "a1"), b=("b0", "b1")).items():
        graph.add_edge(
            edge,
            u,
            v,
            route_arrays(np.linspace(poses[u][:2], poses[v][:2], 100), np.zeros(100)),
            kind="travel",
            blockers=[],
        )
    routes = DrivingRoutes(
        graph,
        Ground(),
        SIZE,
        {e: box(-15, -15, 15, 15) for e in graph.edges},
        dict(a=Body(poses["a0"], SIZE), b=Body(poses["b0"], SIZE)),
    )
    pending = {}
    routes.pending = lambda: pending
    assert routes.plan("a", "a0", "a1") is not None
    pending["b"] = "bx"  # b will park on a's way
    assert routes.plan("a", "a0", "a1") is None and routes.stats["blocked_by_bodies"] == 1
    assert routes.plan("b", "b0", "b1") is not None  # its own destination does not block b
    del pending["b"]
    assert routes.plan("a", "a0", "a1") is not None  # a new snapshot once the journey is gone


def test_a_car_does_not_park_where_it_closes_another_cars_way_home():
    graph = DrivingGraph()
    starts = dict(a=(-12, 0, 0), b=(0, -12, math.pi / 2))
    ends = dict(a=(12, 0, 0), b=(0, 12, math.pi / 2))
    for vehicle in starts:
        a, b = starts[vehicle], ends[vehicle]
        graph.add_node(vehicle + "0", a, kind="berth")
        graph.add_node(vehicle + "1", b, kind="berth")
        graph.add_edge(
            vehicle,
            vehicle + "0",
            vehicle + "1",
            route_arrays(np.linspace(a[:2], b[:2], 100), np.full(100, a[2])),
            kind="travel",
            blockers=["b"] if vehicle == "a" else [],
        )
    graph.add_node("x", (0, 5, math.pi / 2), kind="berth")  # a berth on b's way home
    routes = DrivingRoutes(
        graph,
        Ground(),
        SIZE,
        {e: box(-15, -15, 15, 15) for e in graph.edges},
        {v: Body(p, SIZE) for v, p in starts.items()},
    )
    needs = {"b": (("b0", ("b1",)),)}  # b stands away at b0; its home berth is b1
    assert routes.plan("b", "b0", "b1") is not None
    assert not routes.keeps_ways("a", "x", needs) and routes.stats["ways_home_closed"] == 1
    assert routes.keeps_ways("a", "a1", needs)
    assert routes.keeps_ways("a", "x", {}) and routes.keeps_ways("b", "x", needs)  # nobody else, or its own way
    # The same berth lies on b's departure to b1 (an admitted journey that has not left): it is refused as well.
    assert not routes.keeps_ways("a", "x", {"b": (("b0", ("b1",)), ("b1", ("b0",)))})


def test_known_blocked_edges_keep_the_shortest_clear_length(monkeypatch):
    """Removing every edge already known to fail in the snapshot gives paths of the same length as removing the
    failed edges of one path at a time (random eastbound lattice of lane changes with parked cars)."""
    rng = np.random.default_rng(11)
    n, gap = 6, 15.0
    s = np.linspace(0.0, 1.0, 60)
    step = 3 * s**2 - 2 * s**3
    slope = 6 * s - 6 * s**2
    searches = []
    for trial in range(6):
        graph = DrivingGraph()
        areas = {}
        for i in range(n):
            for j in range(n):
                graph.add_node(f"n{i}-{j}", (i * gap, j * gap, 0.0), kind="lane")
        for i in range(n - 1):
            for j in range(n):
                for k in (j - 1, j, j + 1):
                    if 0 <= k < n:
                        dy = (k - j) * gap
                        points = np.column_stack([i * gap + s * gap, j * gap + dy * step])
                        name = f"e{i}-{j}-{k}"
                        graph.add_edge(
                            name,
                            f"n{i}-{j}",
                            f"n{i + 1}-{k}",
                            route_arrays(points, np.arctan2(dy * slope, gap)),
                            kind="travel",
                        )
                        areas[name] = box(-20, -20, n * gap + 20, n * gap + 20)
        occupancy = {"me": Body((-100.0, -100.0, 0.0), SIZE)}
        for c in range(int(rng.integers(4, 10))):
            i, j = rng.integers(0, n - 1), rng.integers(0, n)
            occupancy[f"p{c}"] = Body((i * gap + gap / 2, j * gap + float(rng.choice([0.0, gap / 2])), 0.0), SIZE)
        fast = DrivingRoutes(graph, Ground(), SIZE, areas, occupancy)
        slow = DrivingRoutes(graph, Ground(), SIZE, areas, occupancy)
        monkeypatch.setattr(slow, "_known_invalid", lambda vehicle: set())

        def length(edges):
            return None if edges is None else round(sum(float(graph.edges[e]["route"]["arc"][-1]) for e in edges), 6)

        found = 0
        for _ in range(60):
            a, b = sorted(rng.choice(n, 2, replace=False))
            source, target = f"n{a}-{rng.integers(n)}", f"n{b}-{rng.integers(n)}"
            for vehicle in ("me", "p0"):
                expected = length(slow._path(vehicle, source, target))
                assert length(fast._path(vehicle, source, target)) == expected
                found += expected is not None
        assert found and fast.stats["path_searches"] <= slow.stats["path_searches"]
        searches.append((fast.stats["path_searches"], slow.stats["path_searches"]))
    assert sum(f for f, _ in searches) < sum(s for _, s in searches), searches  # blocked paths were searched


def test_a_tour_passes_drawn_lane_waypoints_and_is_pure_for_a_snapshot():
    graph = DrivingGraph()
    xs = [-40.0, -20.0, 0.0, 20.0, 40.0]
    graph.add_node("s", (-60.0, 0.0, 0.0), kind="berth")
    graph.add_node("t", (60.0, 0.0, 0.0), kind="berth")
    for k, x in enumerate(xs):
        graph.add_node(f"l{k}", (x, 0.0, 0.0), kind="lane", fragment=0, direction="forward")
    chain = ["s"] + [f"l{k}" for k in range(len(xs))] + ["t"]
    for a, b in zip(chain, chain[1:], strict=False):
        pa, pb = graph.nodes[a]["pose"][:2], graph.nodes[b]["pose"][:2]
        graph.add_edge(f"{a}-{b}", a, b, route_arrays(np.linspace(pa, pb, 40), np.zeros(40)), kind="travel")
    areas = {e: box(-70, -10, 70, 10) for e in graph.edges}
    routes = DrivingRoutes(
        graph, Ground(), SIZE, areas, dict(car=Body((-60.0, 0.0, 0.0), SIZE)), tour_legs=2, tour_spacing_m=0.0
    )
    tour = routes.plan_tour("car", "s", "t")
    assert tour is not None and routes.plan_tour("car", "s", "t") == tour
    assert abs(tour.length_m - 120.0) < 1e-6 and routes.plans[tour.evidence]["edges"] == tuple(
        f"{a}-{b}" for a, b in zip(chain, chain[1:], strict=False)
    )
    assert routes.route("tour", (-60.0, 0.0), (60.0, 0.0), "car") == tour
    assert routes.plan_tour("car", "t", "s") is None


def test_a_route_keeps_to_a_lane_of_its_own_direction_where_the_road_has_one():
    from awr.life.civil.driving_graph import SHARED_LANE_COST

    graph = DrivingGraph()
    graph.add_node("s", (-30.0, 0.0, 0.0), kind="berth")
    graph.add_node("t", (30.0, 0.0, 0.0), kind="berth")
    for name, mode, y in (("shared", "exclusive_bidirectional", 0.0), ("own", "separated_two_way", -1.6)):
        graph.add_node(name + "0", (-20.0, y, 0.0), kind="lane", traffic_mode=mode)
        graph.add_node(name + "1", (20.0, y, 0.0), kind="lane", traffic_mode=mode)
        graph.add_edge(
            name,
            name + "0",
            name + "1",
            route_arrays(np.linspace((-20.0, y), (20.0, y), 81), np.zeros(81)),
            kind="travel",
        )
        for a, b in (("s", name + "0"), (name + "1", "t")):
            pa, pb = graph.nodes[a]["pose"], graph.nodes[b]["pose"]
            graph.add_edge(a + "-" + b, a, b, route_arrays(np.linspace(pa[:2], pb[:2], 41), np.zeros(41)), kind="join")
    # Changes of lane along the shared lane (a waypoint between two of its layouts) are no way round its cost.
    graph.add_node("hop", (0.0, 0.0, 0.0), kind="lane", traffic_mode="exclusive_bidirectional")
    for a, b in (("shared0", "hop"), ("hop", "shared1")):
        pa, pb = graph.nodes[a]["pose"], graph.nodes[b]["pose"]
        graph.add_edge(a + "-" + b, a, b, route_arrays(np.linspace(pa[:2], pb[:2], 41), np.zeros(41)), kind="join")
    # The lane of its own direction is 0.4 m longer on its approaches; the shared lane costs more per metre.
    assert SHARED_LANE_COST > 1.0
    path = graph.path("s", "t")
    assert "own" in path and not any(e in path for e in ("shared", "shared0-hop", "hop-shared1"))
    routes = DrivingRoutes(
        graph, Ground(), SIZE, {e: box(-35, -10, 35, 10) for e in graph.edges}, dict(car=Body((-30.0, 0.0, 0.0), SIZE))
    )
    assert "own" in routes.plans[routes.plan("car", "s", "t").evidence]["edges"]


def test_tour_waypoints_keep_to_streets_a_car_passes_through(monkeypatch):
    # A one-way loop c0-c3 with a spur c1 -> d, turned back at d (kind 'return') to e -> c1. Tours draw their
    # waypoints from the loop, where a car passes through, not from the spur it would hold both ways (cruise r11f v3:
    # vehicle-043 drove down a single-lane road, turned back at its end and held it 300 m both ways). On a map whose
    # turn-back-free part is too small (THROUGH_MIN_M), every lane node is drawn.
    import awr.life.civil.activity.driving as D
    from awr.life.civil.driving_graph import TURN_BACK_COST_M

    graph = DrivingGraph()
    poses = dict(
        s=(-60.0, 0.0),
        c0=(-40.0, 0.0),
        c1=(0.0, 0.0),
        c2=(0.0, 40.0),
        c3=(-40.0, 40.0),
        d=(40.0, 0.0),
        e=(40.0, 4.0),
        t=(20.0, 60.0),
    )
    for name, xy in poses.items():
        graph.add_node(name, (*xy, 0.0), kind="berth" if name in "st" else "lane")

    def edge(a, b, kind):
        pa, pb = np.asarray(poses[a]), np.asarray(poses[b])
        # Headings are not checked here (waypoint draws only): every pose faces east.
        graph.add_edge(f"{a}-{b}", a, b, route_arrays(np.linspace(pa, pb, 20), np.zeros(20)), kind=kind)

    edge("s", "c0", "exit")
    edge("c2", "t", "entry")
    for a, b in (("c0", "c1"), ("c1", "c2"), ("c2", "c3"), ("c3", "c0"), ("c1", "d"), ("e", "c1")):
        edge(a, b, "travel")
    edge("d", "e", "return")
    assert graph.cost("d-e") == pytest.approx(4.0 + TURN_BACK_COST_M)
    drawn = []

    def draws(limit):
        monkeypatch.setattr(D, "THROUGH_MIN_M", limit)
        routes = DrivingRoutes(
            graph,
            Ground(),
            SIZE,
            {e: box(-70, -10, 70, 70) for e in graph.edges},
            dict(car=Body((-60.0, 0.0, 0.0), SIZE)),
            tour_legs=2,
            tour_spacing_m=0.0,
        )
        monkeypatch.setattr(routes, "_path", lambda vehicle, a, b, **kw: drawn.append(b))
        for k in range(40):
            routes.trips_done["car"] = k
            routes.plan_tour("car", "s", "t")
        return {n for n in drawn if n != "t"}

    through = draws(100.0)
    assert through and through <= {"c0", "c1", "c2", "c3"}
    drawn.clear()
    assert draws(1e9) & {"d", "e"}


def test_tour_legs_go_round_a_block_rather_than_turn_back(monkeypatch):
    # A one-way block a -> b -> c -> d -> a, and a turn back at b onto the street back to a (b -> e -> a). From the
    # waypoint b the turn back is the cheaper way to t, by less than TURN_BACK_COST_M; a tour leg goes round the block
    # (cruise5 r11f v2: vehicle-045 drove 150 m down a street, turned back at a junction and drove back along it).
    graph = DrivingGraph()
    poses = dict(
        s=(-60.0, 0.0), a=(-40.0, 0.0), b=(0.0, 0.0), c=(0.0, 40.0), d=(-40.0, 40.0), e=(0.0, 4.0), t=(-60.0, 4.0)
    )
    for name, xy in poses.items():
        graph.add_node(name, (*xy, 0.0), kind="berth" if name in "st" else "lane")

    def edge(a, b, kind):
        pa, pb = np.asarray(poses[a]), np.asarray(poses[b])
        graph.add_edge(f"{a}-{b}", a, b, route_arrays(np.linspace(pa, pb, 20), np.zeros(20)), kind=kind)

    edge("s", "a", "exit")
    edge("a", "t", "entry")
    edge("b", "e", "return")
    for x, y in (("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("e", "a")):
        edge(x, y, "travel")
    routes = DrivingRoutes(
        graph,
        Ground(),
        SIZE,
        {e: box(-70, -10, 10, 50) for e in graph.edges},
        dict(car=Body((-60.0, 0.0, 0.0), SIZE)),
        tour_legs=1,
        tour_spacing_m=0.0,
    )
    assert routes._path("car", "b", "t") == ["b-e", "e-a", "a-t"]
    assert routes._path("car", "b", "t", turn_backs=False) == ["b-c", "c-d", "d-a", "a-t"]
    monkeypatch.setattr(routes, "_through_nodes", lambda: frozenset({"b"}))
    monkeypatch.setattr(routes, "_compile", lambda vehicle, source, target, edges: tuple(edges))
    assert routes.plan_tour("car", "s", "t") == ("s-a", "a-b", "b-c", "c-d", "d-a", "a-t")
    # Where only a turn back leads on, a leg takes it.
    graph2 = DrivingGraph()
    for name in ("s", "a", "b", "e", "t"):
        graph2.add_node(name, (*poses[name], 0.0), kind="berth" if name in "st" else "lane")
    graph = graph2
    edge("s", "a", "exit")
    edge("a", "t", "entry")
    edge("b", "e", "return")
    edge("a", "b", "travel")
    edge("e", "a", "travel")
    routes = DrivingRoutes(
        graph,
        Ground(),
        SIZE,
        {e: box(-70, -10, 10, 50) for e in graph.edges},
        dict(car=Body((-60.0, 0.0, 0.0), SIZE)),
        tour_legs=1,
        tour_spacing_m=0.0,
    )
    monkeypatch.setattr(routes, "_through_nodes", lambda: frozenset({"b"}))
    monkeypatch.setattr(routes, "_compile", lambda vehicle, source, target, edges: tuple(edges))
    assert routes.plan_tour("car", "s", "t") == ("s-a", "a-b", "b-e", "e-a", "a-t")


def test_a_tour_does_not_drive_a_street_back_the_way_it_came(monkeypatch):
    # Cruise5 r11f v8: vehicle-043's waypoint lay down a street whose only way on was round a block at its end and back
    # along the street; the car held 250 m of it both ways. A street a -> b with a block b -> l -> c at its end and the
    # way back c -> d; a ring a -> w -> v -> d beside it. Waypoint l sends the tour back along the street, waypoint v
    # does not: of the draws, the tour takes v.
    graph = DrivingGraph()
    poses = dict(
        s=(-20.0, 0.0),
        a=(0.0, 0.0),
        b=(100.0, 0.0),
        l=(120.0, 20.0),
        c=(100.0, 4.0),
        d=(0.0, 4.0),
        t=(-20.0, 4.0),
        w=(20.0, -40.0),
        v=(60.0, -40.0),
    )
    roads = dict(
        a=("road:x", "forward"),
        b=("road:x", "forward"),
        c=("road:x", "reverse"),
        d=("road:x", "reverse"),
        l=("road:z", "forward"),
        w=("road:w", "forward"),
        v=("road:w", "forward"),
    )
    for name, xy in poses.items():
        extra = dict(traffic_resource=roads[name][0], direction=roads[name][1]) if name in roads else {}
        graph.add_node(name, (*xy, 0.0), kind="berth" if name in "st" else "lane", **extra)

    def edge(a, b, kind):
        pa, pb = np.asarray(poses[a]), np.asarray(poses[b])
        graph.add_edge(f"{a}-{b}", a, b, route_arrays(np.linspace(pa, pb, 20), np.zeros(20)), kind=kind)

    edge("s", "a", "exit")
    edge("d", "t", "entry")
    for x, y, kind in (
        ("a", "b", "travel"),
        ("b", "l", "join"),
        ("l", "c", "join"),
        ("c", "d", "travel"),
        ("a", "w", "join"),
        ("w", "v", "travel"),
        ("v", "d", "join"),
    ):
        edge(x, y, kind)
    routes = DrivingRoutes(
        graph,
        Ground(),
        SIZE,
        {e: box(-70, -60, 140, 50) for e in graph.edges},
        dict(car=Body((-20.0, 0.0, 0.0), SIZE)),
        tour_legs=1,
        tour_spacing_m=0.0,
    )
    monkeypatch.setattr(routes, "_through_nodes", lambda: frozenset({"l", "v"}))
    monkeypatch.setattr(routes, "_compile", lambda vehicle, source, target, edges: tuple(edges))
    assert routes._doubles_back([], ["a-b", "b-l", "l-c", "c-d"]) and not routes._doubles_back(
        [], ["a-w", "w-v", "v-d"]
    )
    assert routes._doubles_back(["a-b", "b-l", "l-c"], ["c-d"])
    # Cruise5 r11f v9: a tour ending near its start drives roads back further on; only nearby doubling back counts.
    monkeypatch.setattr("awr.life.civil.activity.driving.DOUBLE_BACK_M", 50.0)
    assert not routes._doubles_back(["a-b", "b-l", "l-c"], ["c-d"])
    monkeypatch.setattr("awr.life.civil.activity.driving.DOUBLE_BACK_M", 300.0)
    tours = set()
    for trips in range(6):
        routes.trips_done["car"] = trips
        tours.add(routes.plan_tour("car", "s", "t"))
    assert tours == {("s-a", "a-w", "w-v", "v-d", "d-t")} and routes.stats["tour_waypoints_doubling_back"] > 0
    # Where every waypoint drives a road back, a tour still goes.
    monkeypatch.setattr(routes, "_through_nodes", lambda: frozenset({"l"}))
    routes.trips_done["car"] = 7
    assert routes.plan_tour("car", "s", "t") == ("s-a", "a-b", "b-l", "l-c", "c-d", "d-t")


def test_a_path_search_without_a_way_records_the_bodies_in_it():
    routes, starts, ends = fixture(extra=dict(c=(0.0, 0.0, 0.0)))
    assert routes.plan("a", "a0", "a1") is None
    failure = routes.path_failures["a"]
    assert failure["source"] == "a0" and failure["target"] == "a1" and failure["blocking_bodies"] == ["c"]
    assert failure["removed_edges"] >= 1 and failure["unsupported_edges"] == 0


def test_way_out_nodes_spread_over_the_largest_strongly_connected_part():
    graph = DrivingGraph()
    poses = {f"n{k}": (20.0 * k, 0.0, 0.0) for k in range(5)}
    poses.update(lone0=(0.0, 50.0, 0.0), lone1=(20.0, 50.0, 0.0), park=(40.0, -4.0, 0.0))
    for name, pose in poses.items():
        graph.add_node(name, pose, kind="berth" if name == "park" else "lane")

    def edge(a, b):
        graph.add_edge(
            f"{a}-{b}", a, b, route_arrays(np.linspace(poses[a][:2], poses[b][:2], 20), np.zeros(20)), kind="travel"
        )

    for k in range(4):
        edge(f"n{k}", f"n{k + 1}")
        edge(f"n{k + 1}", f"n{k}")
    edge("n2", "park")
    edge("park", "n2")
    edge("lone0", "lone1")  # one way only: not strongly connected
    routes = DrivingRoutes(
        graph, Ground(), SIZE, {e: box(-10, -10, 100, 60) for e in graph.edges}, dict(car=Body((0.0, 0.0, 0.0), SIZE))
    )
    assert routes.way_out_nodes() == ("n2", "n0", "n4")  # lane nodes only: a berth may be taken by a parked car


def test_a_car_does_not_park_where_it_shuts_a_car_at_home_off_the_main_roads():
    # h is parked at its owner's home at the end of a street (h0 -> n0 -> n1 -> n2); the main roads begin at n2.
    # Parking across the street at x shuts h in (cruise r11f v4/v5: vehicle-047); parking at y, off the street, does not.
    graph = DrivingGraph()
    poses = dict(
        h0=(-20.0, 0.0, 0.0),
        n0=(0.0, 0.0, 0.0),
        n1=(10.0, 0.0, 0.0),
        n2=(30.0, 0.0, 0.0),
        x=(20.0, 0.0, 0.0),
        y=(20.0, 20.0, 0.0),
    )
    for name, pose in poses.items():
        graph.add_node(name, pose, kind="berth" if name in ("h0", "x", "y") else "lane")
    for a, b in (("h0", "n0"), ("n0", "n1"), ("n1", "n2")):
        graph.add_edge(
            f"{a}-{b}", a, b, route_arrays(np.linspace(poses[a][:2], poses[b][:2], 40), np.zeros(40)), kind="travel"
        )
    routes = DrivingRoutes(
        graph,
        Ground(),
        SIZE,
        {e: box(-30, -10, 40, 30) for e in graph.edges},
        dict(h=Body(poses["h0"], SIZE), other=Body((20.0, 20.0, 0.0), SIZE)),
    )
    needs = {"h": (("h0", ("n2",)),)}
    searches = []
    search = routes._path
    routes._path = lambda *args, **kwargs: searches.append(args[1:3]) or search(*args, **kwargs)
    assert not routes.keeps_ways("other", "x", needs) and routes.stats["ways_home_closed"] == 1
    assert routes.keeps_ways("other", "y", needs)
    # Within one planning snapshot h's way out is searched once, and a berth's verdict is kept for other parkers.
    assert not routes.keeps_ways("third", "x", needs) and routes.keeps_ways("third", "y", needs)
    assert searches == [("h0", "n2"), ("h0", "n2")]  # the way, then the way past the body parked at x
    routes.update_occupancy(dict(h=Body(poses["h0"], SIZE), other=Body((20.0, 21.0, 0.0), SIZE)))
    assert not routes.keeps_ways("other", "x", needs) and len(searches) == 4  # a new snapshot: searched afresh
