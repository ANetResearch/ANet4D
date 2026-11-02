import numpy as np
import pytest

from awr.life.civil.route_diagnostics import footprint_evidence, route_evidence
from awr.life.civil.vehicle_connections import candidates

SIZE = np.array([4.2, 1.8, 1.5])


class Ground:
    def __init__(self, hole=None):
        self.hole = hole

    def supported(self, xy, profile):
        p = np.asarray(xy)
        return np.ones(len(p), bool) if self.hole is None else ~((p[:, 0] >= self.hole[0]) & (p[:, 0] <= self.hole[1]))


@pytest.fixture(autouse=True)
def contacts(monkeypatch):
    monkeypatch.setattr(
        "awr.life.civil.contacts.vehicle_contacts",
        lambda size: np.array([[-1.4, -0.7], [-1.4, 0.7], [1.4, -0.7], [1.4, 0.7]]),
    )


def test_missing_support_is_distinct_from_steering():
    route = candidates([0, 0, 0], [30, 0, 0], SIZE)[0]
    good = route_evidence(Ground(), route, SIZE)
    bad = route_evidence(Ground((14, 16)), route, SIZE)
    assert good["valid"] and not good["reasons"]
    assert set(bad["reasons"]) == {"sampled_pose_support", "between_pose_support"}
    assert bad["endpoints_supported"] == [True, True]
    assert len(bad["pose_support"]["examples"]) <= 8


def test_subsample_hole_is_retained_and_endpoints_are_not_blame():
    # Coarse spacing used only to expose the interpolation distinction.
    r = dict(points=np.array([[0.0, 0.0], [10.0, 0.0]]), yaw=np.zeros(2), arc=np.array([0.0, 10.0]))
    report = route_evidence(Ground((4.9, 5.1)), r, SIZE)
    assert report["endpoints_supported"] == [True, True]
    assert report["reasons"] == ["between_pose_support"]


def test_actual_arc_and_legacy_spacing_are_separately_identified():
    r = dict(points=np.array([[0.0, 0.0], [0.1, 0.0]]), yaw=np.array([0.0, 0.04]), arc=np.array([0.0, 0.1]))
    report = route_evidence(Ground(), r, SIZE)
    assert report["reasons"] == ["actual_arc_steering"]
    r["arc"][1] = 1.0
    r["yaw"][1] = 0.1
    assert route_evidence(Ground(), r, SIZE)["reasons"] == ["legacy_spacing_steering"]


def test_independent_node_poses_do_not_create_fictitious_segments_or_turns():
    valid, report = footprint_evidence(Ground((40, 60)), [[0, 0], [100, 0]], [0, 3.0], SIZE)
    assert valid.all() and report["failed_poses"] == 0


def test_proof_backend_is_not_silently_explained_using_grid_rules():
    g = Ground()
    g.vehicle_route_supported = lambda *a: True
    with pytest.raises(ValueError, match="ordinary"):
        route_evidence(g, candidates([0, 0, 0], [10, 0, 0], SIZE)[0], SIZE)


def test_diagnostic_drift_fails_closed(monkeypatch):
    monkeypatch.setattr("awr.life.civil.curb.valid_route", lambda *a: False)
    with pytest.raises(ValueError, match="differs"):
        route_evidence(Ground(), candidates([0, 0, 0], [10, 0, 0], SIZE)[0], SIZE)
