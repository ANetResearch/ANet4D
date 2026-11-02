"""Parked cars that never move keep the driving map's roads passable (awr.life.civil.activity.connected_parking)."""

import math

import numpy as np

from awr.life.civil.activity.connected_parking import ConnectedRoads
from awr.life.civil.curb_allocation import footprint
from awr.life.civil.driving_graph import DrivingGraph, route_arrays

SIZE = (4.2, 1.8, 1.5)
L = 60.0


def s_curve(a, b, n=80):
    """From pose a to pose b, both heading along x (east or west): x linear, y smoothstep."""
    s = np.linspace(0.0, 1.0, n)
    dx, dy = b[0] - a[0], b[1] - a[1]
    pts = np.column_stack([a[0] + s * dx, a[1] + dy * (3 * s**2 - 2 * s**3)])
    return route_arrays(pts, np.arctan2(dy * (6 * s - 6 * s**2), dx))


def u_turn(a, b, n=80):
    """Half circle from a to b (opposite headings)."""
    c = (np.asarray(a[:2]) + np.asarray(b[:2])) / 2
    r = float(np.linalg.norm(np.asarray(b[:2]) - np.asarray(a[:2]))) / 2
    th = np.linspace(0.0, math.pi, n)
    sign = 1.0 if math.cos(a[2]) > 0 else -1.0
    start = math.atan2(a[1] - c[1], a[0] - c[0])
    ang = start + sign * th
    pts = np.column_stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)])
    return route_arrays(pts, a[2] + sign * th)


def town(bypass=False):
    g = DrivingGraph()
    poses = dict(
        A0=(0.0, 0.0, 0.0),
        A1=(L, 0.0, 0.0),
        B1=(L, 10.0, math.pi),
        B0=(0.0, 10.0, math.pi),
        P=(L / 2 + 8.0, 14.0, math.pi),
        Q=(L / 2 - 8.0, 14.0, math.pi),
    )
    for name, pose in poses.items():
        g.add_node(name, pose, kind="berth" if name in "PQ" else "lane")
    g.add_edge(
        "east",
        "A0",
        "A1",
        route_arrays(np.column_stack([np.linspace(0.0, L, 120), np.zeros(120)]), np.zeros(120)),
        kind="travel",
    )
    if bypass:
        up, down = s_curve(poses["A0"], (L / 2, 5.0, 0.0)), s_curve((L / 2, 5.0, 0.0), poses["A1"])
        g.add_edge(
            "east-bypass",
            "A0",
            "A1",
            route_arrays(np.vstack([up["points"], down["points"][1:]]), np.concatenate([up["yaw"], down["yaw"][1:]])),
            kind="travel",
        )
    g.add_edge("turn-east", "A1", "B1", u_turn(poses["A1"], poses["B1"]), kind="join")
    g.add_edge(
        "west",
        "B1",
        "B0",
        route_arrays(np.column_stack([np.linspace(L, 0.0, 120), np.full(120, 10.0)]), np.full(120, math.pi)),
        kind="travel",
    )
    g.add_edge("turn-west", "B0", "A0", u_turn(poses["B0"], poses["A0"]), kind="join")
    for b in "PQ":  # kerb berths beside the westbound lane
        g.add_edge(f"to-{b}", "B1", b, s_curve(poses["B1"], poses[b]), kind="curb_access")
        g.add_edge(f"from-{b}", b, "B0", s_curve(poses[b], poses["B0"]), kind="curb_access")
    return g


def body(x, y, yaw=0.0):
    return footprint(dict(xy=np.array([x, y]), yaw=yaw), np.array(SIZE))


def test_a_parked_car_may_not_cut_the_only_road():
    roads = ConnectedRoads(town(), SIZE)
    assert not roads.allows(body(L / 2, 0.0))  # across the eastbound lane: A0 no longer reaches A1
    assert roads.report["refused"] == 1


def test_a_parked_car_may_block_a_lane_with_a_way_round_or_a_berth_access():
    roads = ConnectedRoads(town(bypass=True), SIZE)
    assert roads.allows(body(L / 2, 0.0))  # the bypass keeps A0 -> A1
    roads.reserve(body(L / 2, 0.0))
    assert not roads.allows(body(L / 2, 5.0))  # now the bypass is the only way east
    berth_only = ConnectedRoads(town(), SIZE)
    p = town().nodes["P"]["pose"]
    assert berth_only.allows(body(p[0], p[1] + 0.2, math.pi))  # on the berth: its access is closed, the road is not


def test_standing_bodies_define_the_roads_that_are_kept():
    turn = body(5.0, 5.0, math.pi / 2)  # on the western turn, the only way from B0 back to A0
    assert not ConnectedRoads(town(), SIZE).allows(turn)
    roads = ConnectedRoads(town(), SIZE, [body(L / 2, 0.0)])  # the eastbound lane is already blocked: the loop is open
    assert roads.allows(turn)  # no mutually reachable lanes are left to cut


def test_blocked_long_edges_do_not_exempt_their_unoccupied_endpoints():
    # Ported from codex DK (review of 947c03d). A two-node road component: parking halfway blocks both travel
    # directions, but neither endpoint is occupied. The stretch the body closes holds both endpoints, so no lane is left
    # outside it to fall apart: the part is closed as a whole.
    g = DrivingGraph()
    for name, pose in dict(A=(0.0, 0.0, 0.0), B=(60.0, 0.0, 0.0), P=(0.0, 10.0, 0.0), Q=(60.0, 10.0, 0.0)).items():
        g.add_node(name, pose, kind="berth" if name in "PQ" else "lane")

    def link(name, source, target):
        points = np.linspace(g.nodes[source]["pose"][:2], g.nodes[target]["pose"][:2], 60)
        g.add_edge(name, source, target, route_arrays(points, np.zeros(60)), kind="travel")

    for a, b in [("A", "B"), ("B", "A"), ("P", "A"), ("A", "P"), ("Q", "B"), ("B", "Q")]:
        link(a + b, a, b)
    roads = ConnectedRoads(g, SIZE)
    assert not roads.allows(body(30.0, 0.0))
    assert not roads.keeps(body(30.0, 0.0), roads.standing([]))


def test_a_whole_component_cannot_disappear_as_occupied_nodes():
    # Ported from codex DK: a body over a whole part closes it.
    from shapely.geometry import box

    roads = ConnectedRoads(town(), SIZE)
    assert not roads.allows(box(-20.0, -20.0, 80.0, 30.0))


def test_a_car_beside_a_passing_lane_may_close_the_stretch_it_stands_on():
    # A kerb lane split by a waypoint (K) beside the eastbound lane: a car across the kerb lane closes K's way on,
    # while the eastbound lane still passes it; A0, A1, B0 and B1 still reach each other.
    g = town()
    g.add_node("K", (L / 2, 3.0, 0.0), kind="lane")
    g.add_edge("kerb-in", "A0", "K", s_curve(g.nodes["A0"]["pose"], g.nodes["K"]["pose"]), kind="join")
    g.add_edge("kerb-out", "K", "A1", s_curve(g.nodes["K"]["pose"], g.nodes["A1"]["pose"]), kind="join")
    roads = ConnectedRoads(g, SIZE)
    assert roads.allows(body(L / 2 + 3.0, 3.3))
    assert roads.allows(body(L / 2, 0.0))  # across the eastbound lane only: the kerb lane passes it
    assert not roads.allows(body(L / 2, 1.5, math.pi / 2))  # across both lanes: A0 no longer reaches A1


def test_a_car_parking_during_the_run_keeps_the_roads_the_standing_cars_leave(monkeypatch):
    # Cruise5 r11f v5: vehicle-046 parked at a kerb berth whose body cut the only way along a street; vehicle-043 was
    # sent round by a turn back that held the street both ways for 265 m. A car parking during the run cuts no road
    # the bodies standing then (the planning bodies other than its own) leave, as the cars parked before the run.
    from shapely.geometry import box
    from test_activity_driving import Ground

    from awr.life.civil.activity.driving import Body, DrivingRoutes

    monkeypatch.setattr(
        "awr.life.civil.contacts.vehicle_contacts",
        lambda size: np.array([[-1.4, -0.7], [-1.4, 0.7], [1.4, -0.7], [1.4, 0.7]]),
    )
    g = town()
    for name, x in (("R", L / 2), ("S", L / 2 + 10.0)):  # berths standing on the eastbound lane
        g.add_node(name, (x, 0.0, 0.0), kind="berth")
        g.add_edge(
            f"to-{name}",
            "A0",
            name,
            route_arrays(np.column_stack([np.linspace(0.0, x, 60), np.zeros(60)]), np.zeros(60)),
            kind="curb_access",
        )
        g.add_edge(
            f"from-{name}",
            name,
            "A1",
            route_arrays(np.column_stack([np.linspace(x, L, 60), np.zeros(60)]), np.zeros(60)),
            kind="curb_access",
        )
    roads = ConnectedRoads(g, SIZE)
    q, p = tuple(g.nodes["Q"]["pose"]), tuple(g.nodes["P"]["pose"])

    def routes(other):
        r = DrivingRoutes(
            g,
            Ground(),
            np.array(SIZE),
            {e: box(-50, -50, 150, 150) for e in g.edges},
            {"car": Body(q, SIZE), "other": Body(other, SIZE)},
        )
        r.roads = roads
        return r

    free = routes(p)
    assert not free.keeps_roads("car", "R") and not free.keeps_roads("car", "S")  # A0 no longer reaches A1
    assert free.keeps_roads("car", "P") and free.stats["roads_refused"] == 2
    assert not free.keeps_roads("car", "R") and free.stats["roads_checked"] == 3  # the verdict is kept
    cut = routes(tuple(g.nodes["R"]["pose"]))  # the other car already stands across the eastbound lane
    assert cut.keeps_roads("car", "S")
    assert not cut.keeps_roads("other", "S")  # its own body leaves R: the road it would cut is open then
    assert roads.report["checked"] == 0  # the run's checks leave the bodies recorded before it untouched


def test_the_stretch_a_parked_car_closes_takes_the_lane_nodes_whose_ways_all_lead_into_it():
    # Cruise5 r11f v6/v7: a kerb lane joins the lane beside it at waypoints, so the lane node just before a parked car
    # keeps one way on, into the stretch the car closes, and none out of it. It is part of that stretch: the eastbound
    # lane still passes the car (a cruiser found no destination at all while such nodes counted as roads cut).
    g = town()
    for name, x in (("K0", L / 2 - 20.0), ("K1", L / 2 - 8.0), ("K2", L / 2 + 8.0), ("K3", L / 2 + 20.0)):
        g.add_node(name, (x, 3.0, 0.0), kind="lane", fragment=7, direction="forward")
    g.add_edge("kerb-in", "A0", "K0", s_curve(g.nodes["A0"]["pose"], g.nodes["K0"]["pose"]), kind="join")
    for a, b in (("K0", "K1"), ("K1", "K2"), ("K2", "K3")):
        xa, xb = g.nodes[a]["pose"][0], g.nodes[b]["pose"][0]
        g.add_edge(
            f"kerb-{a}",
            a,
            b,
            route_arrays(np.column_stack([np.linspace(xa, xb, 30), np.full(30, 3.0)]), np.zeros(30)),
            kind="travel",
        )
    g.add_edge("kerb-out", "K3", "A1", s_curve(g.nodes["K3"]["pose"], g.nodes["A1"]["pose"]), kind="join")
    roads = ConnectedRoads(g, SIZE)
    assert roads._hits(body(L / 2, 3.3)) & {"kerb-K0", "kerb-K1", "kerb-K2", "east"} == {"kerb-K1"}
    assert roads.allows(body(L / 2, 3.3))  # across K1 -> K2: K0 leads only into it, K3 is reached only from it
    assert not roads.allows(body(L / 2, 1.5, math.pi / 2))  # across both lanes: A0 no longer reaches A1
