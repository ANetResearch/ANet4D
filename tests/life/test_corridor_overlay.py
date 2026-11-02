from copy import deepcopy

import numpy as np
import pytest

from awr.life.civil.driving_graph import DrivingGraph
from awr.life.motion.corridor_overlay import connect
from awr.life.motion.maneuver import Maneuver
from awr.life.motion.road_guides import body_route


def document(g, t):
    return dict(
        schema="awr.life.driving_map_candidate.v1",
        candidate_only=True,
        nodes=g.nodes,
        edges=g.edges,
        rear_axle_templates={k: dict(points=m.points, yaw=m.yaw, gear=m.gear, rear_x=m.rear_x) for k, m in t.items()},
    )


def fixture():
    g = DrivingGraph()
    templates = {}
    for name, x in [("start", 0.0), ("old_exit", 4.0), ("new_entry", 8.0), ("end", 12.0)]:
        g.add_node(
            name,
            [x, 0.0, 0.0],
            kind="lane",
            road="road",
            direction="forward",
            guide_offset_m=0.0,
            traffic_mode="exclusive_bidirectional",
            traffic_resource="road:road",
        )
    for name, a, b in [("old", "old_exit", "end"), ("new", "start", "new_entry")]:
        x = np.arange(g.nodes[a]["pose"][0], g.nodes[b]["pose"][0] + 1)
        m = Maneuver(np.column_stack((x - 1, np.zeros(len(x)))), np.zeros(len(x)), np.ones(len(x) - 1), -1.0)
        g.add_edge(name, a, b, body_route(m), kind="travel", blockers=("neighbour",), traffic_resources=("road:road",))
        templates[name] = m
    return document(g, templates)


def test_independently_compiled_midroad_ports_become_reachable_without_new_geometry():
    doc = fixture()
    g, t, r = connect(doc)
    path = g.path("old_exit", "new_entry")
    assert path
    assert set(doc["edges"]) <= set(g.edges)
    route = g.concatenate(path)
    np.testing.assert_array_equal(route["points"], doc["edges"]["old"]["route"]["points"][:5])
    for edge in r["added_edges"]:
        e = g.edges[edge["edge"]]
        assert e["blockers"] == ("neighbour",) and e["traffic_resources"] == ("road:road",)
    again, _, rr = connect(document(g, t))
    assert not rr["added_edges"]
    assert len(again.edges) == len(g.edges)


@pytest.mark.parametrize(
    "field,value",
    [
        ("road", "other"),
        ("direction", "reverse"),
        ("guide_offset_m", 1.0),
        ("traffic_mode", "oneway"),
        ("traffic_resource", "other"),
    ],
)
def test_nearby_port_in_different_corridor_is_not_a_connection(field, value):
    doc = fixture()
    doc["nodes"]["new_entry"][field] = value
    g, _, _ = connect(doc)
    assert g.path("old_exit", "new_entry") is None


def test_coincident_alias_ports_share_real_nonzero_slices():
    doc = fixture()
    doc["nodes"]["alias"] = deepcopy(doc["nodes"]["old_exit"])
    g, _, _ = connect(doc)
    assert g.path("alias", "new_entry")
    assert all(e["route"]["arc"][-1] > 0 for e in g.edges.values())


def test_budget_failure_does_not_mutate_input():
    doc = fixture()
    with pytest.raises(ValueError, match="budget"):
        connect(doc, max_added_edges=1)
    assert set(doc["edges"]) == {"old", "new"}
