from copy import deepcopy

import numpy as np
import pytest

from awr.life.civil.driving_graph import DrivingGraph
from awr.life.motion.access_composition import compose
from awr.life.motion.maneuver import Maneuver
from awr.life.motion.road_guides import body_route


def document(g, templates):
    return dict(
        schema="awr.life.driving_map_candidate.v1",
        candidate_only=True,
        nodes=g.nodes,
        edges=g.edges,
        rear_axle_templates={
            k: dict(points=m.points, yaw=m.yaw, gear=m.gear, rear_x=m.rear_x) for k, m in templates.items()
        },
    )


def fixture():
    g = DrivingGraph()
    for name, x in [("a", 1.0), ("b", 11.0)]:
        g.add_node(name, [x, 0.0, 0.0], kind="lane", road="road", direction="forward", traffic_resource="road:road")
    m = Maneuver([[0.0, 0.0], [10.0, 0.0]], [0.0, 0.0], [1], -1.0)
    g.add_edge("whole", "a", "b", body_route(m), kind="travel")
    source = deepcopy(document(g, {"whole": m}))
    parent = deepcopy(source)
    c = DrivingGraph()
    for n, v in g.nodes.items():
        c.add_node(n, v["pose"], **{k: x for k, x in v.items() if k != "pose"})
    c.add_node("port", [6.0, 0.0, 0.0], kind="lane", road="road", direction="forward", traffic_resource="road:road")
    c.add_node("berth", [7.0, 0.0, 0.0], kind="berth", region="curb:road", parking_legality={"legal": True})
    t = {}
    for key, a, b, points, kind in [
        ("first", "a", "port", [[0.0, 0.0], [5.0, 0.0]], "travel"),
        ("second", "port", "b", [[5.0, 0.0], [10.0, 0.0]], "travel"),
        ("access", "port", "berth", [[5.0, 0.0], [6.0, 0.0]], "entry"),
    ]:
        t[key] = Maneuver(points, [0.0, 0.0], [1], -1.0)
        c.add_edge(key, a, b, body_route(t[key]), kind=kind)
    return parent, source, c, t


def test_composition_preserves_existing_route_and_connects_new_ports():
    args = fixture()
    g, t, r = compose(*args)
    assert "whole" in g.edges and r["parent_edges_removed"] == 0
    assert g.path("a", "berth") == ["first", "access"]
    assert set(r["added_edges"]) == {"first", "second", "access"}
    np.testing.assert_array_equal(t["whole"].points, args[0]["rear_axle_templates"]["whole"]["points"])
    assert set(args[0]["nodes"]) == {"a", "b"}
    # Repeating the same source compile does not accumulate duplicated roads.
    again, _, rr = compose(document(g, t), args[1], args[2], args[3])
    assert not rr["added_edges"] and len(again.edges) == len(g.edges)


@pytest.mark.parametrize(
    "field,value", [("road", "other"), ("traffic_resource", "different"), ("direction", "reverse")]
)
def test_matching_position_does_not_allow_source_authority_to_change(field, value):
    args = fixture()
    args[2].nodes["a"][field] = value
    with pytest.raises(ValueError, match="identity"):
        compose(*args)


def test_conflicting_original_pose_cannot_be_overwritten():
    args = fixture()
    args[2].nodes["a"]["pose"] = np.array([1.01, 0.0, 0.0])
    with pytest.raises(ValueError, match="identity"):
        compose(*args)


def test_existing_edge_traffic_reservation_cannot_be_dropped():
    args = fixture()
    g, t, _ = compose(*args)
    parent = document(g, t)
    args[2].edges["access"]["traffic_resources"] = ()
    with pytest.raises(ValueError, match="authority"):
        compose(parent, args[1], args[2], args[3])
