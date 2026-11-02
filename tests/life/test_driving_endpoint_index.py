"""An acceleration structure must preserve exact endpoint/ambiguity semantics."""

import copy
import math

import numpy as np
import pytest
from test_activity_driving import SIZE, Ground, fixture

from awr.life.civil.activity.driving import DrivingRoutes


def test_index_matches_scan_at_boundary_and_duplicate_headings():
    original, _, _ = fixture()
    graph = copy.deepcopy(original.graph)
    graph.add_node("duplicate", (-12.0, 0.0, math.pi), kind="berth")
    graph.add_node("large-coordinate", (10000.0, -8000.0, 0.0), kind="berth")
    routes = DrivingRoutes(graph, Ground(), SIZE, original.areas, original.occupancy)
    points = []
    for p in [np.asarray(d["pose"][:2]) for d in graph.nodes.values()]:
        for r in [0.0, np.nextafter(1e-7, 0.0), 1e-7, np.nextafter(1e-7, math.inf), 1.00001e-7, 1e-5]:
            for a in np.linspace(0, 2 * math.pi, 17):
                points.append(p + r * np.array([math.cos(a), math.sin(a)]))
    points.extend(np.random.default_rng(7).uniform(-20, 20, (100, 2)))
    for xy in points:
        expected = sorted(
            n for n, d in graph.nodes.items() if d["kind"] == "berth" and np.linalg.norm(d["pose"][:2] - xy) <= 1e-7
        )
        assert routes._berths_at(xy) == expected
    assert routes.route("drive", (-12.0, 0.0), (12.0, 0.0), "a") is None


def test_no_berths_and_nonfinite_points_do_not_produce_routes():
    from awr.life.civil.driving_graph import DrivingGraph

    graph = DrivingGraph()
    graph.add_node("road", (0.0, 0.0, 0.0), kind="lane")
    routes = DrivingRoutes(graph, Ground(), SIZE, {}, {})
    assert routes.route("drive", (0.0, 0.0), (1.0, 0.0), "absent") is None
    for point in [(math.nan, 0.0), (math.inf, 0.0), (0.0,), [0.0, 0.0, 0.0]]:
        with pytest.raises(ValueError, match="Finite exact"):
            routes.route("drive", point, (0.0, 0.0), "absent")
