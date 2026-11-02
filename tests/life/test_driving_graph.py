import numpy as np
import pytest

from awr.life.civil.driving_graph import DrivingGraph, from_probes, route_arrays


def graph():
    g = DrivingGraph()
    for key, x, kind in (
        ("home", 0.0, "berth"),
        ("shop", 10.0, "lane"),
        ("park", 20.0, "berth"),
        ("nearby_disconnected", 20.01, "berth"),
    ):
        g.add_node(key, [x, 0, 0], kind=kind)
    for key, a, b, x0, x1 in (
        ("out", "home", "shop", 0.0, 10.0),
        ("next", "shop", "park", 10.0, 20.0),
        ("return", "park", "home", 20.0, 0.0),
    ):
        points = np.column_stack([np.linspace(x0, x1, 81), np.zeros(81)])
        g.add_edge(key, a, b, route_arrays(points, np.zeros(81), np.full(80, 1 if x1 > x0 else -1)), kind="test")
    return g


def test_directed_reachability_never_snaps_to_a_nearby_destination():
    g = graph()
    assert g.path("home", "park") == ["out", "next"]
    assert g.path("home", "nearby_disconnected") is None
    assert g.path("home", "park", blocked_edges={"next"}) is None


def test_a_berth_is_a_terminal_never_a_way_along_the_road():
    g = DrivingGraph()
    for key, x, kind in (("a", 0.0, "lane"), ("p", 5.0, "berth"), ("b", 10.0, "lane")):
        g.add_node(key, [x, 0, 0], kind=kind)
    for key, s, t, x0, x1 in (
        ("a_p", "a", "p", 0.0, 5.0),
        ("p_b", "p", "b", 5.0, 10.0),
        ("a_b", "a", "b", 0.0, 10.0),
        ("b_a", "b", "a", 10.0, 0.0),
    ):
        points = np.column_stack([np.linspace(x0, x1, 41), np.zeros(41)])
        g.add_edge(key, s, t, route_arrays(points, np.zeros(41), np.full(40, 1 if x1 > x0 else -1)), kind="test")
    assert g.path("a", "b") == ["a_b"]
    # Into the parking place and out again is no way from a to b.
    assert g.path("a", "b", blocked_edges={"a_b"}) is None
    assert g.path("a", "p") == ["a_p"] and g.path("p", "a") == ["p_b", "b_a"]
    assert g.components()[0] == ["a", "b", "p"]
    # Without the way back, the berth links no part: it does not join a to b.
    assert sorted(map(len, g.components(blocked_edges={"b_a"}))) == [1, 1, 1]


def test_concatenation_preserves_heading_and_requires_stop_at_gear_change():
    g = graph()
    with pytest.raises(ValueError, match="terminal berth"):
        g.concatenate(["next", "return"])
    g.nodes["park"]["kind"] = "lane"  # an actual road node may carry a gear change (codex 2fe87cd)
    route = g.concatenate(["next", "return"])
    np.testing.assert_allclose(route["points"][0], [10.0, 0.0])
    np.testing.assert_allclose(route["points"][-1], [0.0, 0.0])
    assert route["arc"][-1] == pytest.approx(30.0)
    assert route["required_stops_m"] == [10.0]
    assert len(route["gear"]) == len(route["points"]) - 1
    with pytest.raises(ValueError, match="disconnected"):
        g.concatenate(["return", "next"])


def test_mismatching_pose_and_rotation_in_place_are_rejected():
    g = graph()
    with pytest.raises(ValueError, match="exact node pose"):
        g.add_edge("bad", "home", "shop", route_arrays([[0, 0], [9.9, 0]], [0, 0]), kind="test")
    with pytest.raises(ValueError, match="jump in place"):
        route_arrays([[0, 0], [0, 0]], [0, 1.0])
    with pytest.raises(ValueError, match="Conflicting poses"):
        g.add_node("home", [0, 0, 1.0])


def test_import_keeps_alternate_bay_exits_when_one_edge_is_blocked():
    road = dict(
        asset="fixture",
        size_m=[4, 2, 1.5],
        return_connections=[],
        two_way_fragments=[
            dict(
                forward_xy=[[0, 0], [1, 0], [2, 0]],
                forward_yaw=[0, 0, 0],
                reverse_xy=[[2, 4], [1, 4], [0, 4]],
                reverse_yaw=[np.pi] * 3,
            )
        ],
    )
    exits = [
        dict(
            lane_node=f"lane:0:forward:{k}",
            points=[[-1, 0], [k, 0]],
            yaw=[0, 0],
            gear=[1],
            blocked_by_existing_allocation=[],
        )
        for k in (0, 2)
    ]
    parking = dict(
        asset="fixture",
        size_m=[4, 2, 1.5],
        lane_nodes=[],
        bays=[
            dict(
                bay="b",
                region="lot",
                slot=dict(xy=[-1, 0], yaw=0),
                connections=dict(exit=exits[0]),
                connection_alternatives=dict(exit=exits),
            )
        ],
    )
    g = from_probes(road, parking)
    assert g.path("berth:b", "lane:0:forward:2", blocked_edges={"berth:b:exit:0"}) == ["berth:b:exit:1"]
    with pytest.raises(ValueError, match="Unknown blocked"):
        g.path("berth:b", "lane:0:forward:2", blocked_edges={"misspelled-edge"})


def test_round_trip_summary_changes_when_a_real_return_edge_is_added():
    from awr.life.civil.driving_graph import summarize_berths

    g = DrivingGraph()
    g.add_node("home", [0, 0, 0], kind="berth", region="north")
    g.add_node("service", [100, 0, 0], kind="berth", region="south")
    g.add_edge("out", "home", "service", route_arrays([[0, 0], [100, 0]], [0, 0]), kind="fixture")
    assert summarize_berths(g) == []
    g.add_edge("back", "service", "home", route_arrays([[100, 0], [0, 0]], [0, 0], [-1]), kind="fixture")
    summary = summarize_berths(g)
    assert len(summary) == 1 and summary[0]["qualifying_origin_berths"] == 2
    assert summary[0]["regions"] == {"north": 1, "south": 1}


@pytest.mark.parametrize("points", [[[0, 0], [5e-10, 0], [1, 0]], [[0, 0], [1, 0], [1 + 5e-10, 0]]])
def test_curve_producer_and_serialized_graph_share_sample_contract(points):
    import json

    from awr.life.civil.driving_graph import load_candidate
    from awr.life.civil.vehicle_connections import _join

    route = _join([(np.asarray(points, float), np.zeros(3), 1)])
    g = DrivingGraph()
    g.add_node("a", [*points[0], 0], kind="lane")
    g.add_node("b", [*points[-1], 0], kind="lane")
    g.add_edge("curve", "a", "b", route, kind="join")
    doc = json.loads(
        json.dumps(
            dict(schema="awr.life.driving_map_candidate.v1", nodes=g.nodes, edges=g.edges), default=lambda v: v.tolist()
        )
    )
    restored = load_candidate(doc).edges["curve"]["route"]
    np.testing.assert_array_equal(restored["points"], route["points"])
    np.testing.assert_array_equal(restored["arc"], route["arc"])
    assert restored["required_stops_m"] == route["required_stops_m"]


@pytest.mark.parametrize("distance", [0.0, 5e-11])
def test_both_route_boundaries_reject_unusable_sample(distance):
    from awr.life.civil.vehicle_connections import _join

    points = np.array([[0.0, 0.0], [distance, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="Zero-distance"):
        _join([(points, np.zeros(3), 1)])
    with pytest.raises(ValueError, match="jump in place"):
        route_arrays(points, np.zeros(3))
