import numpy as np
import pytest

from awr.life.civil.surface import SurfaceMap
from awr.life.civil.surface_diagnostics import support_evidence


def grid():
    g = SurfaceMap.__new__(SurfaceMap)
    g.x0, g.y0, g.cell_m, g.nx, g.ny = -1.0, -2.0, 0.5, 12, 10
    g.z = np.zeros((g.ny, g.nx))
    g.walk = {"vehicle": np.ones_like(g.z, bool)}
    return g


def test_mask_and_missing_stencil_are_distinct_and_not_capacity_claims():
    g = grid()
    g.walk["vehicle"][2, 2] = False
    g.z[5, 5] = np.nan
    g.walk["vehicle"][5, 5] = False
    # Identical mask cells counted once, repeated queries retained.
    points = [[0.1, -0.9], [0.1, -0.9], [1.75, 0.75], [3.0, 1.5], [-1.0, -2.0], [np.nan, 0]]
    result = support_evidence(g, points)
    assert result["counts"] == dict(
        nonfinite=1, outside_height_stencil=1, missing_height=1, measured_mask_denied=2, supported=1
    )
    assert result["unique_measured_mask_denied_cells"] == 1
    assert result["missing_height_and_mask_denied"] == 1
    assert sum(result["counts"].values()) == len(points)
    assert "not a mesh" in result["scope"]


def test_mask_allowed_does_not_override_missing_neighbour_height():
    g = grid()
    g.z[2, 3] = np.nan
    r = support_evidence(g, [[0.3, -0.9]])
    assert r["counts"]["missing_height"] == 1
    assert r["examples"][0]["mask_allowed"]


def test_boundaries_and_random_points_agree_with_runtime():
    g = grid()
    rng = np.random.default_rng(7)
    g.walk["vehicle"][:] = rng.random(g.z.shape) > 0.3
    g.z[rng.random(g.z.shape) < 0.1] = np.nan
    p = np.concatenate([rng.uniform([-2, -3], [6, 5], (4000, 2)), [[-0.75, -1.75], [4.75, 2.75], [-0.7500001, -1.75]]])
    r = support_evidence(g, p, examples=2)
    assert r["counts"]["supported"] == int(g.supported(p, "vehicle").sum())
    assert sum(r["counts"].values()) == len(p)
    assert len(r["examples"]) <= 8


def test_drift_fails_closed():
    g = grid()
    g.supported = lambda p, profile: np.zeros(len(p), bool)
    with pytest.raises(ValueError, match="differs"):
        support_evidence(g, [[0.0, 0.0]])


def test_empty_points_and_no_mutation():
    g = grid()
    before = g.z.copy()
    r = support_evidence(g, np.empty((0, 2)))
    assert not sum(r["counts"].values()) and not r["examples"]
    np.testing.assert_array_equal(g.z, before)


def test_full_footprint_and_between_samples_are_classified(monkeypatch):
    from awr.life.civil.route_diagnostics import route_evidence

    monkeypatch.setattr("awr.life.civil.contacts.vehicle_contacts", lambda size: np.array([[0.0, 0.0]]))
    g = grid()
    g.walk["vehicle"][4, 4] = False
    route = dict(points=np.array([[0.0, 0.0], [2.0, 0.0]]), yaw=np.zeros(2), arc=np.array([0.0, 2.0]))
    r = route_evidence(g, route, [0.3, 0.3, 0.3], explain_grid=True)
    assert r["pose_support"]["grid"]["counts"]["measured_mask_denied"] == 0
    assert r["between_pose_support"]["grid"]["counts"]["measured_mask_denied"] > 0
    assert r["reasons"] == ["between_pose_support"]
