"""Progressive traffic admission: conflict runs, standing room, front and waiting order."""

import math

import numpy as np
from shapely.geometry import LineString, box
from test_activity_driving import SIZE, Ground, contacts, drive, fixture  # noqa: F401 (contacts: autouse fixture)

from awr.life.civil.activity.driving import Body, DrivingRoutes
from awr.life.civil.activity.traffic_runs import _pose, _slice
from awr.life.civil.driving_graph import DrivingGraph, route_arrays


def narrow(gap=0.6, a=(-30.0, 30.0), b=(30.0, -30.0)):
    """a drives east along y = 0; b drives west on a road that narrows to `gap` beside it in the middle (`a`, `b`: the
    x of their berths)."""
    graph = DrivingGraph()
    starts, ends = {}, {}
    a = np.linspace((a[0], 0.0), (a[1], 0.0), 241)
    x = np.linspace(b[0], b[1], 241)
    u = np.clip((np.abs(x) - 10.0) / 20.0, 0.0, 1.0)
    b = np.column_stack([x, gap + (6.0 - gap) * (1.0 - np.cos(np.pi * u)) / 2.0])
    for v, line in (("a", a), ("b", b)):
        d = np.diff(line, axis=0)
        yaw = np.arctan2(d[:, 1], d[:, 0])
        yaw = np.r_[yaw, yaw[-1]]
        graph.add_node(v + "0", (*line[0], yaw[0]), kind="berth")
        graph.add_node(v + "1", (*line[-1], yaw[-1]), kind="berth")
        starts[v], ends[v] = (*line[0], yaw[0]), (*line[-1], yaw[-1])
        graph.add_edge(v, v + "0", v + "1", route_arrays(line, yaw), kind="travel")
    areas = {v: box(-35, -10, 35, 15) for v in starts}
    return DrivingRoutes(graph, Ground(), SIZE, areas, {v: Body(p, SIZE) for v, p in starts.items()}), starts, ends


def test_opposite_cars_on_a_narrow_stretch_take_it_in_turn():
    routes, starts, ends = narrow()
    a, b = (routes.plan(v, v + "0", v + "1") for v in "ab")
    # Runs are taken against the traffic present: alone on the road, a car has no run to reserve.
    assert routes.acquire(a.evidence, starts["a"], 0.0)
    assert not routes.traffic.layout(a.evidence).runs
    assert routes.acquire(b.evidence, starts["b"], 0.0)
    runs = routes.traffic.layout(a.evidence).runs
    assert len(runs) == 1 and runs[0].stop > 0.0 and runs[0].room is not None
    poses, holds = drive(routes, dict(a=a.evidence, b=b.evidence), starts)
    assert any(h is not None for t in holds for v, h in t.items() if v != "gap")
    assert min(t["gap"] for t in holds) > 0.0
    assert all(routes.executions[v].state()["arrived"] for v in "ab")


def test_lanes_wide_apart_carry_opposite_traffic_together():
    routes, starts, ends = narrow(gap=3.5)
    a, b = (routes.plan(v, v + "0", v + "1") for v in "ab")
    assert routes.acquire(a.evidence, starts["a"], 0.0) and routes.acquire(b.evidence, starts["b"], 0.0)
    assert not routes.traffic.layout(a.evidence).runs


def stand(routes, vehicle, evidence, pose):
    """A car in a trip standing at `pose` (white box: its trip as the controller would have left it)."""
    routes.traffic.trips[vehicle] = dict(evidence=evidence, arc=0.0, speed=0.0, unsafe=None, runs=[], held=[])
    occupancy = dict(routes.occupancy)
    occupancy[vehicle] = Body(pose, SIZE)
    routes.update_occupancy(occupancy)


def test_a_run_needs_standing_room_after_it():
    routes, starts, ends = fixture(extra=dict(c=(30.0, 30.0, 0.0)))
    a, b = (routes.plan(v, v + "0", v + "1") for v in "ab")
    assert routes.acquire(a.evidence, starts["a"], 0.0) and routes.acquire(b.evidence, starts["b"], 0.0)
    run = routes.traffic.layout(a.evidence).runs[0]
    lo, hi, capacity, _ = run.room
    assert capacity == int((hi - lo) // (SIZE[0] + 1.0)) + 1 >= 1
    # A car standing in the stretch after the crossing takes its only place.
    stand(routes, "c", a.evidence, (-12.0 + hi, 0.0, 0.0))
    if capacity == 1:
        assert routes.traffic_limit("a", run.stop - 1.0, 0.0) == run.stop
        assert routes.traffic.waiting["a"]["reason"] == "room"
    routes.traffic.cancel("c")
    assert routes.traffic_limit("a", run.stop - 1.0, 0.0) is None


def test_the_front_car_of_a_stretch_reserves_first():
    routes, starts, ends = fixture(extra=dict(c=(30.0, 30.0, 0.0)))
    a, b = (routes.plan(v, v + "0", v + "1") for v in "ab")
    assert routes.acquire(a.evidence, starts["a"], 0.0) and routes.acquire(b.evidence, starts["b"], 0.0)
    run = routes.traffic.layout(a.evidence).runs[0]
    stand(routes, "c", a.evidence, (-12.0 + run.stop - 1.0, 0.0, 0.0))
    assert routes.traffic_limit("a", run.stop - 6.0, 0.0) == run.stop
    assert routes.traffic.waiting["a"]["reason"] == "front"


def test_a_car_does_not_reserve_a_run_another_car_waits_in_ahead_of_it():
    # c stands in a's run on a's flow, ahead of a, at a pose safe for c (cruise smoke v7: vehicle-041 in vehicle-040's
    # run), and waits there for a run of its own. Holding the run, a would stand behind c in it while c's run may
    # meet a's: a reserves only once c holds its run.
    from awr.life.civil.activity.traffic_runs import Run

    routes, starts, ends = fixture(extra=dict(c=(30.0, 30.0, 0.0)))
    a, b = (routes.plan(v, v + "0", v + "1") for v in "ab")
    assert routes.acquire(a.evidence, starts["a"], 0.0) and routes.acquire(b.evidence, starts["b"], 0.0)
    run = routes.traffic.layout(a.evidence).runs[0]
    mid = (run.s0 + run.s1) / 2.0
    stand(routes, "c", a.evidence, (-12.0 + mid, 0.0, 0.0))
    own = Run(mid + 0.5, mid + 3.0, mid, (), None, None)
    trip = routes.traffic.trips["c"]
    trip.update(arc=mid, unsafe=routes.traffic._unsafe("c"), runs=[own], held=[])
    assert routes.traffic_limit("a", run.stop - 6.0, 0.0) == run.stop
    assert routes.traffic.waiting["a"]["reason"] == "front"
    trip["held"] = [own]
    assert routes.traffic_limit("a", run.stop - 6.0, 0.0) is None


def test_a_car_does_not_reserve_a_run_while_a_car_waits_in_its_room_for_a_run_that_meets_it():
    # cq3p-d12-s3 v1: vehicle-041 reserved its run into a dead end while vehicle-050, ahead of it on the same way,
    # moved on to stand in the turn back after that run, waiting (behind 041 in turn) for its way out, which meets
    # 041's run: each then waited for the other. A car waiting at a stop in the standing room after this car's run,
    # for a run that meets this one, is a front car; one waiting for a run that does not meet it is not.
    from awr.life.civil.activity.traffic_runs import Run

    routes, starts, ends = fixture(extra=dict(c=(30.0, 30.0, 0.0)))
    a, b = (routes.plan(v, v + "0", v + "1") for v in "ab")
    assert routes.acquire(a.evidence, starts["a"], 0.0) and routes.acquire(b.evidence, starts["b"], 0.0)
    t = routes.traffic
    ra, rb = (t.layout(r.evidence).runs[0] for r in (a, b))
    assert ra.room is not None and t._conflicts(ra, rb)
    mid = (ra.room[0] + ra.room[1]) / 2.0
    stand(routes, "c", a.evidence, (-12.0 + mid, 0.0, 0.0))
    out = Run(mid + 0.5, mid + 3.0, mid, rb.parts, None, None)  # its way out meets a's run
    t.trips["c"].update(arc=mid, unsafe=t._unsafe("c"), runs=[out], held=[])
    t.waiting["c"] = dict(order=0, run=out, reason="order")
    assert t._refusal("a", ra, ra.stop) == "front"
    t.waiting["c"]["run"] = Run(mid + 0.5, mid + 3.0, mid, (), None, None)
    assert t._refusal("a", ra, ra.stop) is None


def test_a_longer_wait_for_a_conflicting_run_goes_first():
    routes, starts, ends = fixture()
    a, b = (routes.plan(v, v + "0", v + "1") for v in "ab")
    assert routes.acquire(a.evidence, starts["a"], 0.0) and routes.acquire(b.evidence, starts["b"], 0.0)
    ra, rb = (routes.traffic.layout(r.evidence).runs[0] for r in (a, b))
    routes.traffic.waiting["b"] = dict(order=0, run=rb, reason="conflict")
    assert routes.traffic_limit("a", ra.stop - 1.0, 0.0) == ra.stop
    assert routes.traffic.waiting["a"]["reason"] == "order"
    assert routes.traffic_limit("b", rb.stop - 1.0, 0.0) is None
    assert routes.traffic_limit("a", ra.stop - 1.0, 0.0) == ra.stop
    assert routes.traffic.waiting["a"]["reason"] == "conflict"


def test_a_reversing_berth_manoeuvre_is_another_flow_where_it_sweeps_back_over_the_lane():
    graph = DrivingGraph()
    for name, x in (("p", -20.0), ("m", 0.0), ("q", 20.0)):
        graph.add_node(name, (x, 0.0, 0.0), kind="lane", fragment=1, direction="forward")
    graph.add_node("berth", (-6.0, -2.5, 0.0), kind="berth")

    def line(a, b, n=60):
        return np.linspace(a, b, n)

    graph.add_edge("e", "p", "m", route_arrays(line((-20.0, 0.0), (0.0, 0.0)), np.zeros(60)), kind="travel")
    graph.add_edge("e2", "m", "q", route_arrays(line((0.0, 0.0), (20.0, 0.0)), np.zeros(60)), kind="travel")
    # Into the berth from m: reverse back along the kerb behind m.
    back = np.vstack([line((0.0, 0.0), (-6.0, -2.5), 60)])
    graph.add_edge("y", "m", "berth", route_arrays(back, np.zeros(60), -np.ones(59)), kind="entry")
    graph.add_edge("j", "m", "q", route_arrays(line((0.0, 0.0), (20.0, 0.0)), np.zeros(60)), kind="join")
    areas = {e: box(-25, -10, 25, 10) for e in graph.edges}
    routes = DrivingRoutes(graph, Ground(), SIZE, areas, {})
    from awr.life.civil.curb_allocation import footprint

    upstream = footprint(dict(xy=(-6.0, 0.0), yaw=0.0), SIZE)
    near = footprint(dict(xy=(-1.0, 0.0), yaw=0.0), SIZE)
    assert not routes.traffic._same_flow("e", "y", upstream)
    assert routes.traffic._same_flow("e", "j", near) and routes.traffic._same_flow("e", "e2", near)


def test_a_car_does_not_stand_on_a_pedestrian_crossing():
    from shapely.geometry import LineString

    graph = DrivingGraph()
    graph.add_node("s", (-30.0, 0.0, 0.0), kind="berth")
    graph.add_node("t", (30.0, 0.0, 0.0), kind="berth")
    graph.add_edge(
        "e", "s", "t", route_arrays(np.linspace((-30.0, 0.0), (30.0, 0.0), 121), np.zeros(121)), kind="travel"
    )
    crossing = LineString([(0.0, -4.0), (0.0, 4.0)]).buffer(2.0, cap_style="flat")
    routes = DrivingRoutes(
        graph,
        Ground(),
        SIZE,
        {"e": box(-35, -10, 35, 10)},
        dict(car=Body((-30.0, 0.0, 0.0), SIZE)),
        no_standing=[crossing],
    )
    (run,) = routes.traffic.layout(routes.plan("car", "s", "t").evidence).runs
    from awr.life.civil.curb_allocation import footprint

    assert not footprint(dict(xy=(run.stop - 30.0, 0.0), yaw=0.0), SIZE).intersects(crossing)
    assert footprint(dict(xy=(run.s0 - 30.0, 0.0), yaw=0.0), SIZE).intersects(crossing) and run.room is not None


def test_a_car_that_can_no_longer_stop_keeps_the_run_a_departure_creates():
    probe, starts, ends = fixture()
    pa, pb = (probe.plan(v, v + "0", v + "1") for v in "ab")
    assert probe.acquire(pa.evidence, starts["a"], 0.0) and probe.acquire(pb.evidence, starts["b"], 0.0)
    stop = probe.traffic.layout(pa.evidence).runs[0].stop
    routes, starts, ends = fixture()
    a, b = (routes.plan(v, v + "0", v + "1") for v in "ab")
    assert routes.acquire(a.evidence, starts["a"], 0.0)
    # a drives alone towards the crossing, too fast to stop before it, when b departs.
    assert routes.traffic_limit("a", stop - 1.0, 8.0) is None
    assert routes.acquire(b.evidence, starts["b"], 0.0)
    held = routes.traffic.trips["a"]["held"]
    assert len(held) == 1 and held[0].stop == stop and routes.traffic.stats["committed"] == 1
    rb = routes.traffic.layout(b.evidence).runs[0]
    assert routes.traffic_limit("b", rb.stop - 1.0, 0.0) == rb.stop
    assert routes.traffic.waiting["b"]["reason"] == "conflict"


def test_a_waiting_car_keeps_its_turn_while_its_run_is_laid_out_anew():
    from dataclasses import replace

    routes, starts, ends = fixture()
    a, b = (routes.plan(v, v + "0", v + "1") for v in "ab")
    assert routes.acquire(a.evidence, starts["a"], 0.0) and routes.acquire(b.evidence, starts["b"], 0.0)
    ra, rb = (routes.traffic.layout(r.evidence).runs[0] for r in (a, b))
    t = routes.traffic
    assert routes.traffic_limit("a", ra.stop - 1.0, 0.0) is None
    assert routes.traffic_limit("b", rb.stop - 1.0, 0.0) == rb.stop
    turn = t.waiting["b"]["order"]
    # The traffic changed and b's run was laid out anew from the same stop: b still waits there, as long as before.
    again = replace(rb, s1=rb.s1 - 0.5, tree=None)
    assert not t._try("b", again, rb.stop - 1.0)
    assert t.waiting["b"]["order"] == turn and t.waiting["b"]["run"] is again
    # A run from another stop is another wait.
    assert not t._try("b", replace(rb, stop=rb.stop - 2.0, tree=None), rb.stop - 3.0)
    assert t.waiting["b"]["order"] > turn


def test_a_longer_wait_for_room_cleared_by_cars_driving_on_keeps_its_turn():
    # cruise5 r11f v2: vehicle-043 waited at a junction for a conflicting run, was then refused for the standing room
    # after it, where vehicle-041 drove on ahead of it, and vehicle-045 took a conflicting run meanwhile: 043 stood
    # 60 s. The car waiting for its room keeps its turn while the cars in that room drive on; it loses it while one of
    # them waits, yields or stops (that car may be waiting for the car behind), and to a car in that room.
    routes, starts, ends = fixture(extra=dict(c=(30.0, 30.0, 0.0)))
    a, b = (routes.plan(v, v + "0", v + "1") for v in "ab")
    assert routes.acquire(a.evidence, starts["a"], 0.0) and routes.acquire(b.evidence, starts["b"], 0.0)
    ra, rb = (routes.traffic.layout(r.evidence).runs[0] for r in (a, b))
    t = routes.traffic
    t.waiting["b"] = dict(order=0, run=rb, reason="room", room_by=("c",))
    stand(routes, "c", a.evidence, (30.0, 30.0, 0.0))
    t.trips["c"]["speed"] = 4.0
    assert t._refusal("a", ra, ra.stop) == "order"
    t.trips["c"]["speed"] = 0.0
    assert t._refusal("a", ra, ra.stop) is None
    t.trips["c"]["speed"] = 4.0
    t.waiting["c"] = dict(order=1, run=None, reason="conflict")
    assert t._refusal("a", ra, ra.stop) is None
    del t.waiting["c"]
    t.yields["c"] = dict(waiter="a", stop=1.0)
    assert t._refusal("a", ra, ra.stop) is None
    del t.yields["c"]
    t.waiting["b"]["room_by"] = ("a", "c")
    assert t._refusal("a", ra, ra.stop) is None


def test_a_car_does_not_wait_behind_a_longer_wait_for_its_own_body_to_move():
    routes, starts, ends = fixture()
    a, b = (routes.plan(v, v + "0", v + "1") for v in "ab")
    assert routes.acquire(a.evidence, starts["a"], 0.0) and routes.acquire(b.evidence, starts["b"], 0.0)
    ra, rb = (routes.traffic.layout(r.evidence).runs[0] for r in (a, b))
    # b has waited longer for its run, but a's body stands in it: b waits for a, so a does not wait for b.
    routes.traffic.waiting["b"] = dict(order=0, run=rb, reason="conflict")
    routes.update_occupancy(dict(a=Body((-1.0, 0.0, 0.0), SIZE), b=Body(starts["b"], SIZE)))
    assert routes.traffic._occupies("a", rb)
    assert routes.traffic._refusal("a", ra, ra.stop) is None


def test_the_layouts_of_one_road_carry_one_flow_each_way():
    graph = DrivingGraph()

    def line(y, x0, x1):
        return np.linspace((x0, y), (x1, y), 41)

    # Two layouts of one road a lane's width apart (a centre lane and a lane beside parked cars), and the
    # centre lane's opposite direction.
    for name, fragment, direction, y, x0, x1, resource in (
        ("c", 1, "forward", 0.0, -20.0, 20.0, "road:r"),
        ("p", 2, "forward", 0.8, -20.0, 20.0, "road:r"),
        ("o", 1, "reverse", 0.0, 20.0, -20.0, "road:r"),
        ("q", 3, "forward", 0.8, -20.0, 20.0, "road:other"),
    ):
        pts = line(y, x0, x1)
        yaw = np.full(41, 0.0 if x1 > x0 else math.pi)
        graph.add_node(
            name + "0",
            (*pts[0], yaw[0]),
            kind="lane",
            fragment=fragment,
            direction=direction,
            traffic_resource=resource,
        )
        graph.add_node(
            name + "1",
            (*pts[-1], yaw[0]),
            kind="lane",
            fragment=fragment,
            direction=direction,
            traffic_resource=resource,
        )
        graph.add_edge(name, name + "0", name + "1", route_arrays(pts, yaw), kind="travel")
    routes = DrivingRoutes(graph, Ground(), SIZE, {e: box(-25, -10, 25, 10) for e in graph.edges}, {})
    from awr.life.civil.curb_allocation import footprint

    body = footprint(dict(xy=(0.0, 0.0), yaw=0.0), SIZE)
    assert routes.traffic._same_flow("c", "p", body) and routes.traffic._same_flow("p", "c", body)
    assert not routes.traffic._same_flow("c", "o", body)
    assert not routes.traffic._same_flow("c", "q", body)


def test_a_waiting_car_is_described_with_the_cars_it_waits_for():
    routes, starts, ends = fixture()
    a, b = (routes.plan(v, v + "0", v + "1") for v in "ab")
    assert routes.acquire(a.evidence, starts["a"], 0.0) and routes.acquire(b.evidence, starts["b"], 0.0)
    ra, rb = (routes.traffic.layout(r.evidence).runs[0] for r in (a, b))
    assert routes.traffic_limit("b", rb.stop - 1.0, 0.0) is None
    assert routes.traffic_limit("a", ra.stop - 1.0, 0.0) == ra.stop
    described = routes.traffic.describe("a")
    assert described["waiting"]["reason"] == "conflict" and described["waiting"]["holding"] == ["b"]
    assert routes.traffic.describe("b")["held"] and routes.traffic.describe("nobody") is None


def test_route_geometry_is_kept_for_trips_and_a_bounded_number_of_other_routes(monkeypatch):
    import awr.life.civil.activity.traffic_runs as module

    monkeypatch.setattr(module, "KEEP_COURSES", 1)
    routes, starts, ends = fixture()
    a, b = (routes.plan(v, v + "0", v + "1") for v in "ab")
    assert routes.acquire(a.evidence, starts["a"], 0.0)
    routes.traffic.layout(b.evidence)
    assert set(routes.traffic.courses) == {a.evidence, b.evidence}
    # Before another route's geometry is built, the oldest routes not in a trip go: the trip's geometry stays.
    routes.traffic._prune()
    assert a.evidence in routes.traffic.courses and b.evidence not in routes.traffic.courses


def test_a_change_between_layouts_of_one_road_keeps_its_flow():
    graph = DrivingGraph()

    def lane(name, x, y, fragment, direction="forward", resource="road:r"):
        graph.add_node(
            name,
            (x, y, 0.0 if direction == "forward" else math.pi),
            kind="lane",
            fragment=fragment,
            direction=direction,
            traffic_resource=resource,
        )

    def edge(name, a, b, kind):
        pa, pb = graph.nodes[a]["pose"], graph.nodes[b]["pose"]
        pts = np.linspace(pa[:2], pb[:2], 41)
        graph.add_edge(name, a, b, route_arrays(pts, np.full(41, pa[2])), kind=kind)

    lane("c0", -20.0, 0.0, 1)
    lane("c1", 20.0, 0.0, 1)
    lane("s0", -20.0, 0.8, 2)
    lane("s1", 20.0, 0.8, 2)
    lane("x1", 20.0, 0.8, 3, resource="road:other")
    edge("c", "c0", "c1", "travel")
    edge("s", "s0", "s1", "travel")
    edge("change", "c0", "s1", "join")
    edge("turn", "c0", "x1", "join")
    routes = DrivingRoutes(graph, Ground(), SIZE, {e: box(-25, -10, 25, 10) for e in graph.edges}, {})
    from awr.life.civil.curb_allocation import footprint

    body = footprint(dict(xy=(0.0, 0.4), yaw=0.0), SIZE)
    assert routes.traffic._lane("change") == routes.traffic._lane("s") == ("road", "road:r", "forward")
    assert routes.traffic._same_flow("s", "change", body)
    assert routes.traffic._lane("turn") is None and not routes.traffic._same_flow("s", "turn", body)


def detour():
    """y drives east along y = 0 (x -40 to 40), past a stretch with a detour to its north; x drives north to a berth
    whose parked body stands across that stretch."""
    graph = DrivingGraph()

    def line(name, a, b, kind, points=None, yaw=None):
        pa, pb = graph.nodes[a]["pose"], graph.nodes[b]["pose"]
        pts = np.linspace(pa[:2], pb[:2], 81) if points is None else points
        if yaw is None:
            d = np.diff(pts, axis=0)
            yaw = np.arctan2(d[:, 1], d[:, 0])
            yaw = np.r_[yaw, yaw[-1]]
        graph.add_edge(name, a, b, route_arrays(pts, yaw), kind=kind)

    graph.add_node("y0", (-40.0, 0.0, 0.0), kind="berth")
    graph.add_node("y1", (40.0, 0.0, 0.0), kind="berth")
    for name, x in (("n0", -30.0), ("n1", -10.0), ("n2", 10.0), ("n3", 30.0)):
        graph.add_node(name, (x, 0.0, 0.0), kind="lane")
    line("out", "y0", "n0", "exit")
    line("w", "n0", "n1", "travel")
    line("mid", "n1", "n2", "travel")
    u = np.linspace(-10.0, 10.0, 161)
    k = np.pi / 20.0
    line(
        "detour",
        "n1",
        "n2",
        "travel",
        np.column_stack([u, 4.0 * np.sin(k * (u + 10.0)) ** 2]),
        np.arctan(4.0 * k * np.sin(2.0 * k * (u + 10.0))),
    )
    line("e", "n2", "n3", "travel")
    line("in", "n3", "y1", "entry")
    graph.add_node("x0", (0.0, -30.0, math.pi / 2), kind="berth")
    graph.add_node("x1", (0.0, -2.0, math.pi / 2), kind="berth")
    line("x", "x0", "x1", "travel")
    starts = dict(y=(-40.0, 0.0, 0.0), x=(0.0, -30.0, math.pi / 2))
    routes = DrivingRoutes(
        graph,
        Ground(),
        SIZE,
        {e: box(-45, -35, 45, 15) for e in graph.edges},
        {v: Body(p, SIZE) for v, p in starts.items()},
    )
    return routes, starts


def test_a_car_parking_across_another_cars_way_takes_that_car_round_it():
    routes, starts = detour()
    y, x = routes.plan("y", "y0", "y1"), routes.plan("x", "x0", "x1")
    assert "mid" in routes.plans[y.evidence]["edges"]
    assert routes.acquire(y.evidence, starts["y"], 0.0) and routes.acquire(x.evidence, starts["x"], 0.0)
    last = routes.traffic.layout(x.evidence).runs[-1]
    assert last.room is None
    # x parked would stand across y's way: y, still able to change its way, is taken round by the detour at the start
    # of the tick, before any car looks ahead.
    routes.traffic.trips["x"].update(arc=last.stop - 1.0, speed=0.0)
    routes.prepare_revisions()
    assert routes.traffic_limit("x", last.stop - 1.0, 0.0) is None
    assert all(h in t["runs"] for t in routes.traffic.trips.values() for h in t["held"])
    new = routes.active["y"]
    assert new != y.evidence and routes.plans[new]["edges"] == ("out", "w", "detour", "e", "in")
    assert routes.revisions["y"] == [(y.evidence, new, 30.0, "parking:x")]
    assert routes.traffic.trips["y"]["evidence"] == new and routes.traffic.stats["revised"] == 1


def test_a_detour_round_a_parking_car_looks_first_for_a_way_without_a_turn_back(monkeypatch):
    # cruise5 r11f v4: a detour round a car parking took vehicle-043 down a street to a three-point turn and back.
    routes, starts = detour()
    y, x = routes.plan("y", "y0", "y1"), routes.plan("x", "x0", "x1")
    assert routes.acquire(y.evidence, starts["y"], 0.0) and routes.acquire(x.evidence, starts["x"], 0.0)
    last = routes.traffic.layout(x.evidence).runs[-1]
    routes.traffic.trips["x"].update(arc=last.stop - 1.0, speed=0.0)
    searches = []
    search = routes._path
    monkeypatch.setattr(routes, "_path", lambda *a, **k: searches.append(k.get("turn_backs", True)) or search(*a, **k))
    routes.prepare_revisions()
    assert searches[0] is False  # the way it replaces has no turn back
    assert routes.plans[routes.active["y"]]["edges"] == ("out", "w", "detour", "e", "in")


def test_a_car_that_can_no_longer_change_its_way_keeps_it_and_the_parking_waits():
    routes, starts = detour()
    y, x = routes.plan("y", "y0", "y1"), routes.plan("x", "x0", "x1")
    assert routes.acquire(y.evidence, starts["y"], 0.0) and routes.acquire(x.evidence, starts["x"], 0.0)
    # y drives at 3 m/s 2 m before the turn to the detour (stopping distance 2.25 m and a second of travel): it can
    # no longer take it, nor has it reached the crossing with x's way.
    routes.traffic.trips["y"].update(arc=28.0, speed=3.0)
    last = routes.traffic.layout(x.evidence).runs[-1]
    routes.traffic.trips["x"].update(arc=last.stop - 1.0, speed=0.0)
    routes.prepare_revisions()
    assert routes.traffic_limit("x", last.stop - 1.0, 0.0) == last.stop
    assert routes.traffic.waiting["x"]["reason"] == "parking"
    assert routes.active["y"] == y.evidence and not routes.revisions


def test_a_car_keeps_only_the_parts_of_its_run_it_has_not_left_behind(monkeypatch):
    # a holds a run over its whole way (as if a third car's way lay along it), crossing b's way in the middle (cruise
    # r11f v1: vehicle-041 waited 60 s for the far end of a 308 m run of vehicle-043, long past the stretch they share).
    # b may take the crossing once a's body has left it a car length behind, though a still holds the rest of its run.
    import awr.life.civil.activity.traffic_runs as T

    monkeypatch.setattr(T, "PART_M", 4.0)
    routes, starts, ends = fixture()
    a, b = (routes.plan(v, v + "0", v + "1") for v in "ab")
    assert routes.acquire(a.evidence, starts["a"], 0.0) and routes.acquire(b.evidence, starts["b"], 0.0)
    traffic = routes.traffic
    whole = traffic._runs(a.evidence, np.ones(len(traffic.course(a.evidence).arcs), bool))[0]
    assert len(whole.parts) == 6 and whole.ends[2] == 12.0
    run = traffic.layout(b.evidence).runs[0]
    for arc, refused in ((10.0, True), (16.0 + SIZE[0] - 0.5, True), (16.0 + SIZE[0] + 0.5, False)):
        routes.update_occupancy(dict(routes.occupancy, a=Body((-12.0 + arc, 0.0, 0.0), SIZE)))
        traffic.trips["a"].update(arc=arc, unsafe=traffic._unsafe("a"), runs=[whole], held=[whole])
        assert (traffic._refusal("b", run, run.stop - 1.0) == "conflict") == refused
    assert traffic.trips["a"]["held"] == [whole]


def test_a_car_that_can_still_stop_before_a_crossing_it_holds_lets_a_shorter_way_through():
    # a holds the crossing but is still well before it; b, at its stop, has less to drive to leave the crossing than
    # a: a yields - b reserves, a stops before the crossing until b has left it. Close to the crossing at speed, a
    # could not stop before it in time, and b waits.
    routes, starts, ends = fixture()
    a, b = (routes.plan(v, v + "0", v + "1") for v in "ab")
    assert routes.acquire(a.evidence, starts["a"], 0.0) and routes.acquire(b.evidence, starts["b"], 0.0)
    ra, rb = (routes.traffic.layout(p.evidence).runs[0] for p in (a, b))
    assert routes.traffic_limit("a", ra.stop - 4.0, 0.0) is None and routes.traffic.trips["a"]["held"] == [ra]
    for arc, speed, yields in ((ra.stop - 0.5, 3.0, False), (ra.stop - 4.0, 0.0, True)):
        routes.traffic.trips["a"].update(arc=arc, speed=speed)
        routes.traffic._yielded.clear()
        limit = routes.traffic_limit("b", rb.stop - 1.0, 0.0)
        assert (limit is None) == yields
        assert (routes.traffic.waiting.get("b", {}).get("reason") == "conflict") != yields
    stop = routes.traffic_limit("a", ra.stop - 4.0, 0.0)
    assert stop is not None and ra.stop - 1.0 < stop < ra.stop and routes.traffic.stats["yields"] == 1
    assert routes.traffic.yields["a"]["waiter"] == "b"
    # The front rule sees a car that yields as one that will wait there.
    assert routes.traffic._wait_pose("a") is not None
    # Once b has left the crossing, a drives on through it.
    routes.update_occupancy(dict(routes.occupancy, b=Body(ends["b"], SIZE)))
    assert routes.traffic_limit("b", rb.s1 + 0.5, 2.0) is None and not routes.traffic.trips["b"]["held"]
    assert routes.traffic_limit("a", ra.stop - 4.0, 0.0) is None and "a" not in routes.traffic.yields


def test_cars_crossing_with_yields_never_touch_and_both_arrive():
    routes, starts, ends = fixture()
    a, b = (routes.plan(v, v + "0", v + "1") for v in "ab")
    assert routes.acquire(a.evidence, starts["a"], 0.0) and routes.acquire(b.evidence, starts["b"], 0.0)
    poses, holds = drive(routes, dict(a=a.evidence, b=b.evidence), starts)
    assert min(t["gap"] for t in holds) > 0.0
    assert all(routes.executions[v].state()["arrived"] for v in "ab")


def test_a_join_swinging_out_beside_its_lane_is_a_merge():
    # Codex review AU (docs/audit/traffic-join-au on codex/autonomous-mixed-life): the map compiler's finite-curvature
    # joins from (0, 0, 0) to (20, 1, 1.2) all end 1 m from the lane they leave, but swing out up to 3.18 m on the way,
    # where a car on that lane fits beside the joining car. A join is the lane's flow only if no point of it lies a
    # body width from the lane; then a car on the lane cannot stand beside any pose of it.
    from types import SimpleNamespace

    from awr.life.civil.activity.traffic_runs import TrafficRuns
    from awr.life.civil.curb_allocation import footprint
    from awr.life.civil.vehicle_connections import candidates

    start, goal = (0.0, 0.0, 0.0), (20.0, 1.0, 1.2)
    kinds = []
    for route in candidates(start, goal, SIZE, max_length_m=50.0)[:3]:
        graph = DrivingGraph()
        for name, pose, fragment in (("start", start, 1), ("straight_end", (60.0, 0.0, 0.0), 1), ("join_end", goal, 2)):
            graph.add_node(
                name, pose, kind="lane", fragment=fragment, direction="forward", traffic_resource="road:example"
            )
        graph.add_edge(
            "straight",
            "start",
            "straight_end",
            route_arrays(np.array([[0.0, 0.0], [60.0, 0.0]]), np.zeros(2)),
            kind="travel",
        )
        graph.add_edge("join", "start", "join_end", route, kind="join")
        traffic = TrafficRuns(SimpleNamespace(graph=graph, size=SIZE))
        same = traffic._lane("join") == traffic._lane("straight")
        beside = any(
            not footprint(dict(xy=p, yaw=y), SIZE, margin_m=0.0).intersects(
                footprint(dict(xy=(p[0], 0.0), yaw=0.0), SIZE, margin_m=0.0)
            )
            for p, y in zip(route["points"][:, :2], route["yaw"], strict=False)
        )
        assert not (same and beside)
        kinds.append(same)
    assert kinds.count(False) >= 2


def test_a_car_passing_beside_the_standing_room_still_yields(monkeypatch):
    # b drives north along x = 3.2 across a's way; the standing room after b's crossing lies in b's lane. a turns north
    # after the crossing into the lane beside that room (x = 6, 2.8 m apart): it never stands where b may wait, so,
    # still well before the crossing, it may yield (cruise r6f v13: a band round the room's centre line took in the
    # lane beside it, and no car ever yielded).
    import awr.life.civil.activity.traffic_runs as T

    monkeypatch.setattr(T, "PART_M", 4.0)
    graph = DrivingGraph()
    u = np.linspace(-np.pi / 2, 0.0, 61)[1:]
    turn = np.column_stack([6.0 * np.cos(u), 6.0 + 6.0 * np.sin(u)])
    a = np.r_[np.linspace((-12.0, 0.0), (0.0, 0.0), 49), turn, np.linspace((6.0, 6.0), (6.0, 30.0), 97)[1:]]
    lines = dict(a=a, b=np.linspace((3.2, -12.0), (3.2, 40.0), 209))
    starts = {}
    for v, line in lines.items():
        d = np.diff(line, axis=0)
        yaw = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        yaw = np.r_[yaw, yaw[-1]]
        graph.add_node(v + "0", (*line[0], yaw[0]), kind="berth")
        graph.add_node(v + "1", (*line[-1], yaw[-1]), kind="berth")
        starts[v] = (*line[0], yaw[0])
        graph.add_edge(v, v + "0", v + "1", route_arrays(line, yaw), kind="travel")
    routes = DrivingRoutes(
        graph, Ground(), SIZE, {v: box(-20, -20, 20, 45) for v in lines}, {v: Body(p, SIZE) for v, p in starts.items()}
    )
    pa, pb = (routes.plan(v, v + "0", v + "1") for v in "ab")
    assert routes.acquire(pa.evidence, starts["a"], 0.0) and routes.acquire(pb.evidence, starts["b"], 0.0)
    traffic = routes.traffic
    rb = traffic.layout(pb.evidence).runs[0]
    assert rb.room is not None
    # a holds a run over its whole way (as if a third car's way lay along it), the lane beside b's room included.
    whole = traffic._runs(pa.evidence, np.ones(len(traffic.course(pa.evidence).arcs), bool))[0]
    traffic.trips["a"].update(arc=1.0, speed=0.0, unsafe=traffic._unsafe("a"), runs=[whole], held=[whole])
    way, room = traffic._zone("b", rb, rb.stop - 1.0)
    beside = [
        k
        for k, (_, q) in enumerate(whole.parts)
        if whole.ends[k] > 20.0
        and q.distance(LineString(_slice(traffic.course(pb.evidence).route, rb.room[0], rb.room[1])["points"]))
        < SIZE[1] / 2.0 + 1.0
    ]
    assert beside  # a's parts in the lane beside lie within the old band round the room's centre line
    assert not any(len(room.query(q, predicate="intersects")) for _, q in whole.parts)
    assert routes.traffic_limit("b", rb.stop - 1.0, 0.0) is None and traffic.yields["a"]["waiter"] == "b"


def test_reservation_changes_are_journalled_for_the_tape():
    routes, starts, ends = fixture()
    a, b = (routes.plan(v, v + "0", v + "1") for v in "ab")
    assert routes.acquire(a.evidence, starts["a"], 0.0) and routes.acquire(b.evidence, starts["b"], 0.0)
    ra, rb = (routes.traffic.layout(p.evidence).runs[0] for p in (a, b))
    routes.traffic.journal.clear()
    assert routes.traffic_limit("a", ra.stop - 1.0, 0.0) is None
    assert routes.traffic_limit("b", rb.stop - 0.5, 3.0) == rb.stop
    assert routes.traffic_limit("b", rb.stop - 0.5, 3.0) == rb.stop  # the same wait is journalled once
    kinds = [(row["kind"], row["vehicle"]) for row in routes.traffic.journal]
    assert kinds == [("traffic_reserved", "a"), ("traffic_wait", "b")]
    wait = routes.traffic.journal[1]
    assert wait["reason"] == "conflict" and wait["holding"] == ["a"]


def street():
    """The narrow street F0-F1 (road n forward, east) beside G1-G0 (road n reverse, west), and a way round the block
    (road r) from before the street to beyond it, a little longer. Car a drives east from a0 to a1, car b west from b0
    to b1."""
    graph = DrivingGraph()

    def s_curve(p, q, n=40):
        s = np.linspace(0.0, 1.0, n)
        dx, dy = q[0] - p[0], q[1] - p[1]
        return route_arrays(
            np.column_stack([p[0] + s * dx, p[1] + dy * (3 * s**2 - 2 * s**3)]), np.arctan2(dy * (6 * s - 6 * s**2), dx)
        )

    lanes = dict(
        E0=((-50.0, 0.0, 0.0), "road:n", "forward"),
        F0=((-30.0, 0.0, 0.0), "road:n", "forward"),
        F1=((30.0, 0.0, 0.0), "road:n", "forward"),
        K1=((50.0, 0.0, 0.0), "road:n", "forward"),
        G1=((30.0, 0.6, math.pi), "road:n", "reverse"),
        G0=((-30.0, 0.6, math.pi), "road:n", "reverse"),
        R0=((-10.0, -20.0, 0.0), "road:r", "forward"),
        R1=((10.0, -20.0, 0.0), "road:r", "forward"),
    )
    for name, (pose, road, direction) in lanes.items():
        graph.add_node(name, pose, kind="lane", traffic_resource=road, direction=direction)
    berths = dict(a0=(-70.0, -8.0, 0.0), a1=(70.0, -8.0, 0.0), b0=(70.0, 8.0, math.pi), b1=(-70.0, 8.0, math.pi))
    for name, pose in berths.items():
        graph.add_node(name, pose, kind="berth")
    for a, b, kind in (
        ("a0", "E0", "exit"),
        ("K1", "a1", "entry"),
        ("b0", "G1", "exit"),
        ("G0", "b1", "entry"),
        ("E0", "F0", "travel"),
        ("F0", "F1", "travel"),
        ("F1", "K1", "travel"),
        ("G1", "G0", "travel"),
        ("E0", "R0", "join"),
        ("R0", "R1", "travel"),
        ("R1", "K1", "join"),
    ):
        graph.add_edge(f"{a}-{b}", a, b, s_curve(graph.nodes[a]["pose"], graph.nodes[b]["pose"]), kind=kind)
    routes = DrivingRoutes(
        graph,
        Ground(),
        SIZE,
        {e: box(-80, -30, 80, 20) for e in graph.edges},
        dict(a=Body(berths["a0"], SIZE), b=Body(berths["b0"], SIZE)),
    )
    return routes, berths


def test_a_plan_goes_round_a_narrow_street_another_car_drives_the_other_way(monkeypatch):
    # Cruise r11f v9/v10/v15, cruise5 r11f v8: cars met on 130-320 m of a lane both directions share and the later one
    # waited 60 s for the far end of it. Car b drives west along the narrow street; car a wants to go east along it and
    # has a way round the block: it takes that.
    def routes():
        r, berths = street()
        assert r.acquire(r.plan("b", "b0", "b1").evidence, berths["b0"], 0.0)
        return r

    r = routes()
    assert r._meetings("a", ["F0-F1"])[1] > 50.0 and not r._meetings("b", ["G1-G0"])[0]  # its own way is no meeting
    assert not r._meetings("a", ["E0-R0", "R0-R1", "R1-K1"])[0]
    assert "R0-R1" in r.plans[r.plan("a", "a0", "a1").evidence]["edges"] and r.stats["meetings_avoided"] == 1
    # Round the block costs more than the plan drives to avoid a meeting: it keeps the street.
    monkeypatch.setattr("awr.life.civil.activity.driving.MEETING_DETOUR_M", 1.0)
    r = routes()
    assert "F0-F1" in r.plans[r.plan("a", "a0", "a1").evidence]["edges"] and r.stats["meetings_kept"] == 1


def test_a_car_expected_to_wait_long_for_a_run_goes_round_it(monkeypatch):
    # Cruise r11f v15: vehicle-045 stood over 60 s before a 136 m run vehicle-043 held on a lane both directions
    # share. Car b holds the narrow street westward; car a, driving east towards it with a way round the block still
    # open, is taken round when it would wait for b at least CONGESTION_WAIT_S, and keeps the street otherwise.
    monkeypatch.setattr("awr.life.civil.activity.driving.MEETING_DETOUR_M", 1.0)  # both plans keep the street

    def routes(into, speed, stuck=False):
        r, berths = street()
        b = r.plan("b", "b0", "b1").evidence
        assert r.acquire(b, berths["b0"], 0.0)
        a = r.plan("a", "a0", "a1").evidence
        assert "F0-F1" in r.plans[a]["edges"] and r.acquire(a, berths["a0"], 0.0)
        run = r.traffic.layout(b).runs[0]
        assert r.traffic_limit("b", run.stop - 1.0, 0.0) is None and r.traffic.trips["b"]["held"] == [run]
        arc = run.s0 + into * (run.s1 - run.s0)
        r.traffic.trips["b"].update(arc=arc, speed=speed)
        r.update_occupancy(dict(r.occupancy, b=Body(_pose(r.traffic.course(b).route, arc), SIZE)))
        if stuck:
            r.traffic.waiting["b"] = dict(order=1, run=run, reason="conflict")
        r.traffic.trips["a"].update(arc=0.0, speed=6.0)
        r.prepare_revisions(0.0)
        return r, a

    r, a = routes(0.1, 3.0)  # b needs about 20 s more in the street, a reaches it in 6 s
    new = r.active["a"]
    assert new != a and "R0-R1" in r.plans[new]["edges"] and "F0-F1" not in r.plans[new]["edges"]
    assert r.revisions["a"][0][3] == "congestion:b" and r.traffic.stats["congestion_revised"] == 1
    assert [e["kind"] for e in r.traffic.journal][-1] == "traffic_congestion_reroute"
    r, a = routes(0.9, 6.0)  # b is nearly through
    assert r.active["a"] == a and "congestion_searches" not in r.traffic.stats
    r, a = routes(0.9, 6.0, stuck=True)  # b waits in the street for a run of its own
    assert r.active["a"] != a
    # A way round that costs more than allowed is not taken; the car waits for the run.
    monkeypatch.setattr("awr.life.civil.activity.traffic_runs.CONGESTION_DETOUR_M", 1.0)
    r, a = routes(0.9, 6.0, stuck=True)
    assert r.active["a"] == a and r.stats["revisions_too_long"] >= 1


def test_a_way_round_the_check_refuses_leaves_every_trip_as_it_was(monkeypatch):
    # cq3p v3: vehicle-050 was taken round vehicle-053's run into the dead end vehicle-053 was driving to and stood 60 s.
    # A congestion revision must shorten the expected wait (TrafficRuns._relieve passes that check): a way round the
    # check refuses changes nothing.
    monkeypatch.setattr("awr.life.civil.activity.driving.MEETING_DETOUR_M", 1.0)
    monkeypatch.setattr("awr.life.civil.activity.traffic_runs.CONGESTION_WAIT_S", 1e9)  # no revision of its own

    def routes():
        r, berths = street()
        b = r.plan("b", "b0", "b1").evidence
        assert r.acquire(b, berths["b0"], 0.0)
        a = r.plan("a", "a0", "a1").evidence
        assert r.acquire(a, berths["a0"], 0.0)
        run = r.traffic.layout(b).runs[0]
        assert r.traffic_limit("b", run.stop - 1.0, 0.0) is None
        arc = run.s0 + 0.1 * (run.s1 - run.s0)
        r.traffic.trips["b"].update(arc=arc, speed=3.0)
        r.update_occupancy(dict(r.occupancy, b=Body(_pose(r.traffic.course(b).route, arc), SIZE)))
        r.traffic.trips["a"].update(arc=0.0, speed=6.0)
        r.prepare_revisions(0.0)
        return r, a

    r, a = routes()
    trips = {v: dict(t) for v, t in r.traffic.trips.items()}
    calls = []
    assert not r.revise("a", r.occupancy["b"].polygon(), reason="congestion:b", check=lambda: calls.append(1))
    assert calls == [1] and r.active["a"] == a and r.traffic.stats["revision_not_better"] == 1
    assert {v: t["evidence"] for v, t in r.traffic.trips.items()} == {v: t["evidence"] for v, t in trips.items()}
    r, a = routes()
    assert r.revise("a", r.occupancy["b"].polygon(), reason="congestion:b", check=lambda: True)
    assert r.active["a"] != a and "F0-F1" not in r.plans[r.active["a"]]["edges"]


def test_a_departure_waits_while_its_way_would_soon_meet_an_oncoming_car(monkeypatch):
    # Cruise5 r11f v12: vehicle-050 set out into a street vehicle-043 was driving the other way along 317 m of one
    # shared lane and stood 60 s in its trip. A departure whose way would meet a car in a trip coming the other way
    # within HOLD_WINDOW_S waits parked at its berth instead, HOLD_MAX_S at most, and sets out once the other car is through.
    from awr.life.civil.activity import driving

    monkeypatch.setattr("awr.life.civil.activity.driving.MEETING_DETOUR_M", 1.0)  # a keeps the street

    def routes():
        r, berths = street()
        b = r.plan("b", "b0", "b1").evidence
        assert r.acquire(b, berths["b0"], 0.0)
        return r, berths, b, r.plan("a", "a0", "a1").evidence

    r, berths, b, a = routes()
    assert "F0-F1" in r.plans[a]["edges"]
    assert r.acquire(a, berths["a0"], 0.0)  # without the tick time no departure waits
    r, berths, b, a = routes()
    r.prepare_revisions(0.0)
    assert not r.acquire(a, berths["a0"], 0.0) and r.stats["departures_held"] == 1 and "a" not in r.active
    r.prepare_revisions(driving.HOLD_MAX_S)
    assert r.acquire(a, berths["a0"], 0.0)  # it has waited long enough: it sets out and takes its turn
    r, berths, b, a = routes()
    r.prepare_revisions(0.0)
    assert not r.acquire(a, berths["a0"], 0.0)
    r.update_occupancy(dict(r.occupancy, b=Body(berths["b1"], SIZE)))  # b has parked at the far end of the street
    r.arrive(b, berths["b1"], 0.0)
    r.prepare_revisions(1.0)
    assert r.acquire(a, berths["a0"], 0.0) and r.stats["departure_hold_ticks"] == 1


def test_a_departure_whose_first_run_begins_at_the_berth_waits_parked_for_it():
    # cq3p v1: vehicle-045 pulled 3.5 m out of its kerb berth to the stop of its first run and stood there 60 s in its
    # trip. A first run beginning within a car length and gap of the berth is reserved before the car sets out.
    def cars():
        routes, starts, ends = narrow(a=(-22.0, 30.0))  # a's berth 0.5 m before its run through the narrows
        a, b = (routes.plan(v, v + "0", v + "1").evidence for v in "ab")
        assert routes.acquire(b, starts["b"], 0.0)
        return routes, starts, ends, a, b

    routes, starts, ends, a, b = cars()
    assert routes.acquire(a, starts["a"], 0.0)  # b can still stop before the narrows: a sets out holding its first run
    run = routes.traffic.trips["a"]["held"][0]
    assert 0.0 < run.stop <= routes.traffic.length + 1.0 and run is routes.traffic.layout(a).runs[0]
    routes, starts, ends, a, b = cars()
    pose, speed = np.asarray(starts["b"], float), 0.0
    for tick in range(600):
        if tick == 60:
            assert abs(pose[0]) < 10.0  # b is in the narrows
            assert (
                not routes.acquire(a, starts["a"], 0.0) and "a" not in routes.active and "a" not in routes.traffic.trips
            )
            assert routes.traffic.waiting["a"]["reason"] == "conflict"
        if routes.executions.get("b") is not None and routes.executions["b"].state()["arrived"]:
            break
        result = routes.advance_motion(b, pose, speed, 0.1)
        pose, speed = np.r_[result["xy"], result["yaw"]], result["speed_mps"]
        routes.update_occupancy(dict(routes.occupancy, b=Body(tuple(pose), SIZE)))
    assert routes.arrive(b, tuple(pose), speed)
    assert routes.acquire(a, starts["a"], 0.0)  # b has passed


def test_cars_on_consecutive_edges_of_one_route_follow_each_other(monkeypatch):
    """cq3p v7: vehicle-053 followed vehicle-040 along the same edges of a road whose route alternates between two
    layouts by joins shifted more than a body width; the corridors of a join and the travel edges before and after
    it meet, so each such pair counted as another flow, and vehicle-053 waited 60 s for vehicle-040's whole 204 m run.
    A car on the next edge of a route both take, or on the edge another route takes after a common edge, follows or
    leads; a car merging from another edge still meets the run."""
    from types import SimpleNamespace

    from awr.life.civil.activity.traffic_runs import Run, TrafficRuns

    # edges a: A -> N, b: N -> B, c: N -> C, z: Z -> N
    traffic = TrafficRuns(SimpleNamespace(graph=None, size=SIZE))
    monkeypatch.setattr(traffic, "_same_flow", lambda e, f, body: e == f)  # every corridor met is another flow
    near = box(-3.0, -1.0, 3.0, 1.0)  # the parts meet around the node N

    def run(*parts):
        return Run(0.0, 1.0, 0.0, tuple((e, near) for e, _ in parts), None, None, prev=tuple(p for _, p in parts))

    follower, leader = run(("a", None)), run(("b", "a"))
    assert not traffic._conflicts(follower, leader) and not traffic._conflicts(leader, follower)
    assert not traffic._conflicts(run(("b", "a")), run(("c", "a")))  # parted after the common edge a
    assert traffic._conflicts(follower, run(("z", None)))  # merging into N from another edge
    assert traffic._conflicts(follower, run(("b", "z")))  # ahead on b, but came from z
    assert traffic._conflicts(
        Run(0.0, 1.0, 0.0, (("a", near),), None, None), Run(0.0, 1.0, 0.0, (("b", near),), None, None)
    )
