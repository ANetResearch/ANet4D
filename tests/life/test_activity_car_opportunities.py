import numpy as np

from awr.life.civil.activity.director import Place
from awr.life.civil.activity.opportunities import screen_car_activities
from awr.life.civil.driving_graph import DrivingGraph, route_arrays


def fixture(same_facility=False):
    graph = DrivingGraph()
    for name, x in [("a", 0), ("b", 10), ("c", 70), ("d", 140)]:
        graph.add_node(name, [x, 0, 0], kind="berth")
    for a in graph.nodes:
        for b in graph.nodes:
            if a == b:
                continue
            start = graph.nodes[a]["pose"][:2]
            end = graph.nodes[b]["pose"][:2]
            graph.add_edge(
                a + b,
                a,
                b,
                route_arrays([start, end], [0, 0], [1 if end[0] > start[0] else -1]),
                kind="test",
                blockers=["parked-car"] if "d" in (a, b) else [],
            )
    places = [
        Place(n, tuple(d["pose"][:2]), "berth", (n,), access_xy=(d["pose"][0], 3)) for n, d in graph.nodes.items()
    ]
    places += [
        Place("real-home", (5, 10), "home", ("h",), parking=("a", "b")),
        Place("facility-c", (70, 10), "shop", ("c-slot",), parking=("c",), group="site1"),
        Place(
            "facility-d", (140, 10), "park", ("d-slot",), parking=("d",), group="site1" if same_facility else "site2"
        ),
    ]
    return places, graph


def test_multiple_berths_do_not_multiply_one_actual_resident_home():
    places, graph = fixture()
    result = screen_car_activities(places, graph)
    assert result["qualifying_berths"] == 2
    assert result["distinct_home_berth_matching"] == 1
    assert not result["release_evidence"]
    for row in result["rows"]:
        assert len({v["group"] for v in row["service_pair"]}) == 2
        assert set(b for v in row["service_pair"] for b in v["original_allocation_blockers"]) == {"parked-car"}


def test_two_approaches_to_same_facility_are_not_two_activities():
    places, graph = fixture(same_facility=True)
    result = screen_car_activities(places, graph)
    assert result["qualifying_berths"] == result["distinct_home_berth_matching"] == 0


def test_known_parked_car_cannot_disappear_from_blocked_screen():
    places, graph = fixture()
    result = screen_car_activities(places, graph, blocked_edges=[e for e, v in graph.edges.items() if v["blockers"]])
    assert result["distinct_home_berth_matching"] == 0


def test_two_different_home_ids_allow_two_driver_candidates():
    places, graph = fixture()
    places.append(Place("other-home", (10, 10), "home", ("h2",), parking=("b",)))
    result = screen_car_activities(places, graph)
    assert result["distinct_home_berth_matching"] == 2
    assert len({m["berth"] for m in result["matching"]}) == 2


def test_route_trees_preserve_exact_graph_paths():
    places, graph = fixture()
    result = screen_car_activities(places, graph)
    for row in result["rows"]:
        for service in row["service_pair"]:
            assert service["outward_edges"] == graph.path(row["home_berth"], service["berth"])
            assert service["return_edges"] == graph.path(service["berth"], row["home_berth"])
            assert service["outward_m"] == np.linalg.norm(
                graph.nodes[row["home_berth"]]["pose"][:2] - graph.nodes[service["berth"]]["pose"][:2]
            )
