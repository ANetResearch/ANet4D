import numpy as np
import pytest

from awr.life.civil.driving_graph import from_probes, load_candidate
from awr.life.civil.road_corridors import directions, measured_fragments, source_roads


class Ground:
    def supported(self, xy, profile):
        return np.abs(np.asarray(xy)[:, 1]) < 1.5


def source(oneway=False):
    road = dict(id="street", width_m=3.2 if oneway else 6.4, line=[[0.0, 0.0], [20.0, 0.0]])
    feature = dict(
        properties=dict(
            feature="edge",
            edge_id="street",
            oneway=oneway,
            agents=["vehicle"],
            from_node="a",
            to_node="b",
            osm_id="way/1",
        ),
        geometry=dict(coordinates=road["line"]),
    )
    return dict(roads=[road]), dict(features=[feature])


@pytest.fixture(autouse=True)
def contacts(monkeypatch):
    monkeypatch.setattr(
        "awr.life.civil.contacts.vehicle_contacts",
        lambda size: np.array([[-1.4, -0.7], [-1.4, 0.7], [1.4, -0.7], [1.4, 0.7]]),
    )


def test_narrow_measured_road_is_exclusive_not_two_full_lanes():
    sem, graph = source()
    roads, counts = measured_fragments(Ground(), sem, graph, [4.2, 1.8, 1.5])
    assert len(roads) == 1 and roads[0]["traffic_mode"] == "exclusive_bidirectional"
    assert roads[0]["reservation_required"]
    np.testing.assert_allclose(roads[0]["forward_xy"], roads[0]["reverse_xy"][::-1])
    assert roads[0]["length_m"] == 20.0  # Previously discarded by the 30 m cutoff.


def test_oneway_is_preserved_through_graph_compilation_and_reload():
    sem, source_graph = source(True)
    roads, _ = measured_fragments(Ground(), sem, source_graph, [4.2, 1.8, 1.5])
    assert len(roads) == 1 and directions(roads[0]) == ("forward",)
    probe = dict(asset="fixture", size_m=[4.2, 1.8, 1.5], driving_fragments=roads, return_connections=[])
    parking = dict(asset="fixture", size_m=[4.2, 1.8, 1.5], lane_nodes=[], bays=[])
    graph = from_probes(probe, parking)
    assert all(node["direction"] == "forward" for node in graph.nodes.values())
    assert all(edge["traffic_resource"] == "road:street" for edge in graph.edges.values())
    again = load_candidate(dict(schema="awr.life.driving_map_candidate.v1", nodes=graph.nodes, edges=graph.edges))
    assert next(iter(again.edges.values()))["traffic_resource"] == "road:street"
    assert next(iter(again.edges.values()))["traffic_resources"] == ("road:street",)
    names = list(graph.nodes)
    assert graph.path(names[-1], names[0]) is None


def test_missing_or_reversed_source_direction_fails_closed():
    sem, graph = source(True)
    sem["roads"][0]["line"] = sem["roads"][0]["line"][::-1]
    with pytest.raises(ValueError, match="orientation"):
        source_roads(sem, graph)
    sem, graph = source(True)
    del graph["features"][0]["properties"]["oneway"]
    with pytest.raises(ValueError, match="direction"):
        source_roads(sem, graph)
    with pytest.raises(ValueError, match="reverse lane"):
        directions(dict(source_oneway=True, forward_xy=[], forward_yaw=[], reverse_xy=[], reverse_yaw=[]))


def test_inter_corridor_join_claims_both_resources():
    from awr.life.civil.driving_graph import DrivingGraph, route_arrays

    graph = DrivingGraph()
    graph.add_node("a", [0, 0, 0], traffic_resource="road:a")
    graph.add_node("b", [10, 0, 0], traffic_resource="road:b")
    graph.add_edge("join", "a", "b", route_arrays([[0, 0], [10, 0]], [0, 0]), kind="join")
    assert graph.edges["join"]["traffic_resources"] == ("road:a", "road:b")


def test_damaged_opposite_lane_does_not_erase_supported_direction():
    class HalfRoad:
        def supported(self, xy, profile):
            return np.asarray(xy)[:, 1] < -0.1

    sem, graph = source()
    # poses_at's positive offset is to the right (negative y on this road).
    legacy, _ = measured_fragments(HalfRoad(), sem, graph, [4.2, 1.8, 1.5])
    assert not legacy
    roads, _ = measured_fragments(HalfRoad(), sem, graph, [4.2, 1.8, 1.5], independent_directions=True)
    assert len(roads) == 1
    assert directions(roads[0]) == ("forward",)
    assert roads[0]["traffic_mode"] == "separated_two_way"
    assert not roads[0]["source_oneway"]
    assert roads[0]["length_m"] == 20.0


def test_independent_lane_survey_does_not_invent_against_oneway():
    sem, graph = source(True)
    roads, _ = measured_fragments(Ground(), sem, graph, [4.2, 1.8, 1.5], independent_directions=True)
    assert len(roads) == 1 and directions(roads[0]) == ("forward",)


def test_lanes_beside_a_curb_parking_strip_keep_clear_of_the_parked_body():
    from shapely.geometry import LineString

    from awr.life.civil.curb import road_layout
    from awr.life.civil.curb_allocation import footprint

    class Wide:
        def supported(self, xy, profile):
            return np.abs(np.asarray(xy)[:, 1]) < 3.2

    size = [4.2, 1.8, 1.5]
    sem, graph = source()
    legacy, _ = measured_fragments(Wide(), sem, graph, size)
    roads, counts = measured_fragments(Wide(), sem, graph, size, beside_parking=True)
    assert len(roads) == len(legacy) + 2 and counts["beside_parking"] == 2
    assert all("beside_parking" not in r for r in legacy)
    strip = road_layout(6.4, size)
    for side, sign in (("right", 1.0), ("left", -1.0)):
        # Parked on the right of the source direction lies at negative y for a road along +x.
        parked = footprint(dict(xy=[10.0, -sign * strip["parking_offset_m"]], yaw=0.0), np.asarray(size))
        lane = next(r for r in roads if r.get("beside_parking") == side)
        for direction in directions(lane):
            corridor = LineString(lane[direction + "_xy"]).buffer(size[1] / 2 + 0.4)
            assert not corridor.intersects(parked)
        # Each earlier layout has a lane the parked car blocks on this narrow street.
        for other in legacy:
            assert any(
                LineString(other[d + "_xy"]).buffer(size[1] / 2 + 0.4).intersects(parked) for d in directions(other)
            )


def test_parking_alternatives_preserve_oneway_and_shared_resource_on_reload():
    class Wide:
        def supported(self, xy, profile):
            return np.abs(np.asarray(xy)[:, 1]) < 3.2

    sem, source_graph = source(True)
    sem["roads"][0]["width_m"] = 6.4
    size = [4.2, 1.8, 1.5]
    roads, _ = measured_fragments(Wide(), sem, source_graph, size, beside_parking=True, independent_directions=True)
    assert any(r.get("beside_parking") == "right" for r in roads)
    assert not any(r.get("beside_parking") == "left" for r in roads)
    assert all(directions(r) == ("forward",) for r in roads)
    graph = from_probes(
        dict(asset="fixture", size_m=size, driving_fragments=roads, return_connections=[]),
        dict(asset="fixture", size_m=size, lane_nodes=[], bays=[]),
    )
    loaded = load_candidate(dict(schema="awr.life.driving_map_candidate.v1", nodes=graph.nodes, edges=graph.edges))
    assert all(e["traffic_resources"] == ("road:street",) for e in loaded.edges.values())


def test_parking_strip_does_not_create_support_or_extra_width():
    sem, graph = source(True)  # 3.2 m is insufficient for parking and a travel lane.
    roads, _ = measured_fragments(Ground(), sem, graph, [4.2, 1.8, 1.5], beside_parking=True)
    assert roads and not any("beside_parking" in r for r in roads)
    sem, graph = source()  # Labelled 6.4 m, actual support is only the centre strip.
    roads, _ = measured_fragments(Ground(), sem, graph, [4.2, 1.8, 1.5], beside_parking=True)
    assert roads and not any("beside_parking" in r for r in roads)


def test_lateral_paths_keep_to_free_cells_and_prefer_the_best_scored_offset():
    from awr.life.civil.road_corridors import lateral_paths

    free = np.zeros((9, 30), bool)
    free[2:7, :] = True
    free[4:7, 10:16] = False  # an obstacle on one side: the path must hold cells 2-3 there
    free[:, 25] = False  # a gap: the path ends and another begins after it
    score = -np.abs(np.arange(9) - 5.0)[:, None] * np.ones((1, 30))  # prefer cell 5
    runs = lateral_paths(free, score)
    assert [(a, b) for a, b, _ in runs] == [(0, 25), (26, 30)]
    cells = runs[0][2]
    assert all(free[c, k] for k, c in enumerate(cells)) and np.all(np.abs(np.diff(cells)) <= 1)
    assert cells[0] == 5 and cells[12] <= 3 and cells[24] == 5  # back to the preferred cell beyond the obstacle


class OffsetCarriageway:
    """A 40 m street mapped 6.4 m wide whose measured carriageway lies 0.8 m to the left of the centre line, with an
    obstacle on its left half between x = 18 and 21 m."""

    def supported(self, xy, profile):
        xy = np.asarray(xy)
        ok = (xy[:, 1] > -1.4) & (xy[:, 1] < 2.6)
        return ok & ~((xy[:, 0] > 18.0) & (xy[:, 0] < 21.0) & (xy[:, 1] > 0.9))


def street(length=40.0):
    sem, graph = source()
    sem["roads"][0]["line"] = graph["features"][0]["geometry"]["coordinates"] = [[0.0, 0.0], [length, 0.0]]
    return sem, graph


def test_fixed_lanes_break_at_a_one_sided_obstacle_but_a_lane_following_the_carriageway_passes_it():
    sem, graph = street()
    fixed, _ = measured_fragments(OffsetCarriageway(), sem, graph, [4.2, 1.8, 1.5])
    centre = [f for f in fixed if f["traffic_mode"] == "exclusive_bidirectional"]
    assert len(centre) == 2 and not any(f["station_m"][0] < 19.0 < f["station_m"][1] for f in centre)
    roads, counts = measured_fragments(OffsetCarriageway(), sem, graph, [4.2, 1.8, 1.5], follow_carriageway=True)
    assert [f["traffic_mode"] for f in roads] == ["exclusive_bidirectional"] and roads[0]["follows_carriageway"]
    lane = np.asarray(roads[0]["forward_xy"])
    assert lane[0, 0] < 3.0 and lane[-1, 0] > 37.0
    beside = lane[np.abs(lane[:, 0] - 19.5) < 0.5]
    assert len(beside) and np.all(beside[:, 1] < 0.9 - 1.02)  # the padded body clears the obstacle
    np.testing.assert_allclose(roads[0]["reverse_xy"], lane[::-1])
    assert "separated_two_way" not in {f["traffic_mode"] for f in roads}  # too narrow for two cars abreast


def test_a_lane_following_the_carriageway_stays_inside_the_permitted_area():
    from shapely.geometry import box

    sem, graph = street()
    # The permitted area ends at x = 30 m: the lane ends where the body would leave it.
    roads, _ = measured_fragments(
        OffsetCarriageway(),
        sem,
        graph,
        [4.2, 1.8, 1.5],
        follow_carriageway=True,
        permitted_area=box(-5.0, -5.0, 30.0, 5.0),
    )
    assert len(roads) == 1 and 27.0 < roads[0]["station_m"][1] <= 30.0 - 2.1


def test_waypoint_lane_nodes_divide_each_lane():
    sem, graph = street()
    roads, _ = measured_fragments(OffsetCarriageway(), sem, graph, [4.2, 1.8, 1.5], follow_carriageway=True)
    probe = dict(asset="fixture", size_m=[4.2, 1.8, 1.5], driving_fragments=roads, return_connections=[])
    parking = dict(asset="fixture", size_m=[4.2, 1.8, 1.5], lane_nodes=[], bays=[])
    plain = from_probes(probe, parking)
    spaced = from_probes(probe, parking, node_spacing_m=10.0)
    assert len(plain.nodes) == 4
    waypoints = [n for n, v in spaced.nodes.items() if v.get("waypoint")]
    assert len(waypoints) == 2 * 3 and len(spaced.nodes) == 4 + 6
    forward = sorted((v["pose"][0], n) for n, v in spaced.nodes.items() if v["direction"] == "forward")
    assert np.all(np.diff([x for x, _ in forward]) < 10.5)
    assert spaced.path(forward[0][1], forward[-1][1]) is not None


def test_a_waypoint_is_not_placed_beside_another_lane_node():
    sem, graph = street()
    roads, _ = measured_fragments(OffsetCarriageway(), sem, graph, [4.2, 1.8, 1.5], follow_carriageway=True)
    probe = dict(asset="fixture", size_m=[4.2, 1.8, 1.5], driving_fragments=roads, return_connections=[])
    plain = from_probes(
        probe, dict(asset="fixture", size_m=[4.2, 1.8, 1.5], lane_nodes=[], bays=[]), node_spacing_m=10.0
    )
    # A parking connection's lane node 1 m before the second waypoint of the forward lane takes its place.
    fragment, direction = 0, "forward"
    xy = np.asarray(roads[fragment][direction + "_xy"], float)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    sample = int(np.searchsorted(arc, 19.0))
    node = dict(
        fragment=fragment,
        direction=direction,
        sample=sample,
        pose=[*xy[sample], roads[fragment][direction + "_yaw"][sample]],
    )
    spaced = from_probes(
        probe, dict(asset="fixture", size_m=[4.2, 1.8, 1.5], lane_nodes=[node], bays=[]), node_spacing_m=10.0
    )

    def marks(g):
        return sorted(
            v["sample"] for v in g.nodes.values() if v["direction"] == direction and v["fragment"] == fragment
        )

    assert len(marks(spaced)) == len(marks(plain)) and sample in marks(spaced)
