import numpy as np
import pytest
from shapely.geometry import box

from awr.life.civil.driving_graph import DrivingGraph, route_arrays
from awr.life.civil.lane_ports import add_lane_ports


@pytest.fixture(autouse=True)
def synthetic_contacts(monkeypatch):
    # Unit geometry only; the real scene compiler still loads authored tyres.
    monkeypatch.setattr(
        "awr.life.civil.contacts.vehicle_contacts",
        lambda size: np.array([[-1.3, -0.7], [-1.3, 0.7], [1.3, -0.7], [1.3, 0.7]]),
    )


class Ground:
    def supported(self, xy, profile="vehicle"):
        return np.ones(len(np.asarray(xy).reshape(-1, 2)), bool)


def scene():
    points = np.column_stack([np.arange(11.0), np.zeros(11)])
    road = dict(
        road="r",
        forward_xy=points.tolist(),
        forward_yaw=[0.0] * 11,
        source_oneway=True,
        traffic_mode="oneway",
        traffic_resource="road:r",
    )
    sem = dict(roads=[dict(id="r", width_m=8, line=[[0, 0], [10, 0]])])
    src = dict(
        features=[
            dict(
                properties=dict(
                    feature="edge", edge_id="r", oneway=True, agents=["vehicle"], from_node="a", to_node="b"
                ),
                geometry=dict(coordinates=[[0, 0], [10, 0]]),
            )
        ]
    )
    graph = DrivingGraph()
    for i in (0, 10):
        graph.add_node(
            f"lane:0:forward:{i}",
            [i, 0, 0],
            kind="lane",
            fragment=0,
            direction="forward",
            sample=i,
            traffic_resource="road:r",
        )
    graph.add_edge(
        "original",
        "lane:0:forward:0",
        "lane:0:forward:10",
        route_arrays(points, np.zeros(11)),
        kind="travel",
        blockers=["old-occupant"],
    )
    return graph, dict(driving_fragments=[road]), sem, src


def run(graph, probe, sem, src, **kwargs):
    return add_lane_ports(
        graph,
        probe,
        sem,
        src,
        Ground(),
        [4.0, 2.0, 1.5],
        kwargs.pop("area", box(-3, -5, 13, 5)),
        kwargs.pop("obstacles", {}),
        road_ids=["r"],
        **kwargs,
    )


def test_exact_source_samples_and_all_continuations_preserve_parent():
    graph, probe, sem, src = scene()
    output, audit = run(graph, probe, sem, src, spacing_m=2.0, obstacles={"parked": box(4.8, -1, 5.2, 1)})
    assert len(graph.nodes) == 2 and list(graph.edges) == ["original"]
    assert len(audit["added_nodes"]) == 4 and len(audit["added_edges"]) == 5
    assert output.edges["original"]["blockers"] == ("old-occupant",)
    for i in (2, 4, 6, 8):
        np.testing.assert_array_equal(output.nodes[f"lane:0:forward:{i}"]["pose"], [i, 0, 0])
    assert any("parked" in output.edges[e]["blockers"] for e in audit["added_edges"])
    assert all(output.edges[e]["traffic_resources"] == ("road:r",) for e in audit["added_edges"])
    assert all((output.edges[e]["route"]["gear"] == 1).all() for e in audit["added_edges"])
    again, new = run(output, probe, sem, src, spacing_m=2.0)
    assert not new["added_nodes"] and not new["added_edges"]


def test_permission_hole_cannot_be_bridged_by_new_travel():
    graph, probe, sem, src = scene()
    area = box(-3, -5, 13, 5).difference(box(4.9, -5, 5.1, 5))
    output, audit = run(graph, probe, sem, src, area=area)
    assert audit["counts"]["outside_permission_pose"] > 0
    assert any(k.startswith("travel_") for k in audit["counts"])
    for name in audit["added_edges"]:
        p = output.edges[name]["route"]["points"]
        assert not (p[0, 0] < 5 < p[-1, 0])


def test_source_direction_and_existing_pose_are_not_inferred():
    graph, probe, sem, src = scene()
    probe["driving_fragments"][0].update(reverse_xy=[[10, 0], [0, 0]], reverse_yaw=[np.pi, np.pi])
    with pytest.raises(ValueError, match="one-way"):
        run(graph, probe, sem, src)
    graph, probe, sem, src = scene()
    graph.nodes["lane:0:forward:10"]["pose"][0] += 0.1
    with pytest.raises(ValueError, match="source sample"):
        run(graph, probe, sem, src)


def test_invalid_spacing_fails_before_mutation():
    graph, probe, sem, src = scene()
    for spacing in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            run(graph, probe, sem, src, spacing_m=spacing)
    assert len(graph.nodes) == 2


def test_unknown_parent_fragment_rejected_even_outside_selected_roads():
    graph, probe, sem, src = scene()
    graph.add_node("stale", [0, 20, 0], kind="lane", fragment=1, direction="forward", sample=0)
    with pytest.raises(ValueError, match="exact source fragment roster"):
        run(graph, probe, sem, src)
    assert len(graph.nodes) == 3 and list(graph.edges) == ["original"]
