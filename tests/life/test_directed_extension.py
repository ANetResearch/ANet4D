import numpy as np
import pytest
from shapely.geometry import box

from awr.life.civil.directed_extension import additions, extend
from awr.life.civil.driving_graph import DrivingGraph, route_arrays

SIZE = np.array([4.2, 1.8, 1.5])


class Ground:
    def supported(self, xy, profile):
        return np.ones(len(xy), bool)


def fragment(lo, hi, road="street"):
    points = np.c_[np.arange(lo, hi + 0.01), np.zeros(hi - lo + 1)]
    return dict(
        road=road,
        traffic_mode="separated_two_way",
        traffic_resource="road:" + road,
        station_m=[lo, hi],
        forward_xy=points.tolist(),
        forward_yaw=[0.0] * len(points),
        source_oneway=False,
    )


def parent():
    graph = DrivingGraph()
    fragments = [fragment(0, 5), fragment(15, 20)]
    for i, f in enumerate(fragments):
        names = [f"old:{i}:start", f"old:{i}:end"]
        for name, k in zip(names, [0, len(f["forward_xy"]) - 1], strict=False):
            graph.add_node(
                name,
                [*f["forward_xy"][k], 0.0],
                kind="lane",
                fragment=i,
                direction="forward",
                sample=k,
                traffic_resource=f["traffic_resource"],
            )
        graph.add_edge(
            f"old:{i}",
            *names,
            route_arrays(f["forward_xy"], f["forward_yaw"]),
            kind="travel",
            traffic_resource=f["traffic_resource"],
        )
    return graph, fragments


@pytest.fixture(autouse=True)
def contacts(monkeypatch):
    monkeypatch.setattr(
        "awr.life.civil.contacts.vehicle_contacts",
        lambda size: np.array([[-1.4, -0.7], [-1.4, 0.7], [1.4, -0.7], [1.4, 0.7]]),
    )


def test_only_uncovered_directional_intervals_extend_the_parent():
    _, old = parent()
    assert additions(old, [fragment(0, 5), fragment(15, 20)]) == []
    assert len(additions(old, [fragment(0, 20)])) == 1
    assert len(additions(old, [fragment(0, 5, road="different")])) == 1


def test_exact_overlap_preserves_old_graph_ids_and_reconnects_supported_lane():
    graph, old = parent()
    nodes = set(graph.nodes)
    edges = set(graph.edges)
    assert graph.path("old:0:start", "old:1:end") is None
    report = extend(graph, old, [fragment(0, 20)], Ground(), SIZE, box(-10, -10, 30, 10))
    assert nodes <= set(graph.nodes) and edges <= set(graph.edges)
    assert report["reused_exact_anchors"] == 4
    route = graph.concatenate(graph.path("old:0:start", "old:1:end"))
    assert route["arc"][-1] == 20.0
    assert graph.path("old:1:end", "old:0:start") is None


def test_support_holes_and_permission_boundaries_cannot_be_bridged():
    class Gap(Ground):
        def supported(self, xy, profile):
            return np.abs(np.asarray(xy)[:, 0] - 10.0) > 1.0

    for ground, permission in [
        (Gap(), box(-10, -10, 30, 10)),
        (Ground(), box(-10, -10, 8, 10).union(box(12, -10, 30, 10))),
    ]:
        graph, old = parent()
        report = extend(graph, old, [fragment(0, 20)], ground, SIZE, permission)
        assert graph.path("old:0:start", "old:1:end") is None
        assert any(k.startswith("rejected:") for k in report)


def test_parked_bodies_remain_explicit_route_blockers():
    graph, old = parent()
    extend(graph, old, [fragment(0, 20)], Ground(), SIZE, box(-10, -10, 30, 10), [("parked-car", box(9, -1, 11, 1))])
    blocked = [k for k, e in graph.edges.items() if "parked-car" in e["blockers"]]
    assert blocked
    assert graph.path("old:0:start", "old:1:end", blocked_edges=blocked) is None


def test_nearby_pose_is_not_snapped_or_silently_reoriented():
    graph, old = parent()
    extra = fragment(0, 20)
    extra["forward_xy"] = [[x, 0.05] for x, y in extra["forward_xy"]]
    extend(graph, old, [extra], Ground(), SIZE, box(-10, -10, 30, 10))
    assert graph.path("old:0:start", "old:1:end") is None
    graph, old = parent()
    old[0]["forward_xy"][0][1] += 0.1
    with pytest.raises(ValueError, match="exact poses"):
        extend(graph, old, [fragment(0, 20)], Ground(), SIZE, box(-10, -10, 30, 10))
