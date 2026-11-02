"""A selected continuous path survives node-raster aliasing, never new obstacles."""

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from awr.life.civil.activity.director import Director, Intent, Route, Step
from awr.life.civil.walking import WalkNetwork
from awr.life.motion.engine import LifeEngine
from awr.life.motion.route import LifeRouter


def handoff():
    # The body's inflated raster closes all three node rows. The continuous
    # bottom passage still clears the exact car and the declared free cells.
    # No node, body, margin or standing coordinate is changed to open it.
    network = WalkNetwork(np.ones((3, 20), bool), [0.0, 0.0], step=1.0)
    free = network.obstacle_mask([[8.0, 1.0]], [[4.0, 2.0, 1.5]], 0.3, [0.0])
    router = LifeRouter(network, free, landmarks=0)
    points = np.array([[3.0, 0.0], [5.0, -0.48], [11.0, -0.48], [15.0, 0.0]])
    assert all(network.clear_segment(a, b, free) for a, b in zip(points[:-1], points[1:], strict=False))
    assert router.plan(points[0], points[-1]) is None
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    proof = Route(length, length, "backend-generated-path", points=points)
    intent = Intent("walker", "place", "visit", "slot", [Step("walk", "walker", tuple(points[-1]), route=proof)])
    engine = LifeEngine.__new__(LifeEngine)
    engine.ids = ["walker"]
    engine.xy = points[:1].copy()
    engine.stats = dict(route_plans=0, route_cache_hits=0)
    engine.events = []
    engine.agendas = {0: SimpleNamespace(director=SimpleNamespace(active={"walker": intent}))}
    engine.route_cache = {}
    engine.router = router
    engine.route_policy = None
    return engine, network, free, intent, points


def test_selected_path_survives_raster_disconnect_and_is_detached():
    e, n, free, intent, points = handoff()
    result = e.route(0, points[-1], t=0.0)
    np.testing.assert_allclose(result, points)
    assert e.stats["walk_proofs_reused"] == 1
    assert e.stats.get("route_plans", 0) == 0
    assert e.events[-1]["activity_route_evidence"] == intent.step.route.evidence
    result[:] = 100.0
    np.testing.assert_allclose(intent.step.route.points, points)


def test_changed_parked_geometry_rejects_previous_path_and_replans(monkeypatch):
    e, n, free, intent, points = handoff()
    changed = n.obstacle_mask([[8.0, 1.0], [12.0, 0.0]], [[4.0, 2.0, 1.5], [2.0, 1.0, 1.0]], 0.3, [0.0, 0.0])
    e.router = LifeRouter(n, changed, landmarks=0)
    attempts = []
    monkeypatch.setattr(e.router, "plan", lambda a, b: attempts.append((a.copy(), b.copy())))
    assert e.route(0, points[-1]) is None
    assert len(attempts) == 1
    assert e.stats["walk_proofs_invalidated"] == 1
    np.testing.assert_allclose(intent.step.route.points, points)


def test_changed_static_permission_also_invalidates_previous_path():
    e, n, free, intent, points = handoff()
    mask = np.array(n.free, copy=True)
    mask[:, 12] = False
    newer = WalkNetwork(mask, [0.0, 0.0], step=1.0)
    e.router = LifeRouter(newer, newer.obstacle_mask([[8.0, 1.0]], [[4.0, 2.0, 1.5]], 0.3, [0.0]), landmarks=0)
    assert e.route(0, points[-1]) is None


def test_reconnection_requires_local_clearance_and_does_not_snap():
    e, n, free, intent, points = handoff()
    e.xy[0] = [3.1, 0.1]
    result = e.route(0, points[-1])
    np.testing.assert_array_equal(result[0], e.xy[0])
    assert all(n.clear_segment(a, b, free) for a, b in zip(result[:-1], result[1:], strict=False))
    e.xy[0] = [8.0, 1.0]  # actual car body; no evidence permits an escape teleport
    assert e._activity_walk(0, points[-1]) is None
    e.xy[0] = [1.0, 0.0]  # beyond the bounded connector to the selected start
    assert e._activity_walk(0, points[-1]) is None


@pytest.mark.parametrize("change", [dict(entity="another"), dict(kind="drive"), dict(xy=(16.0, 0.0))])
def test_evidence_belongs_to_the_current_actor_step_and_destination(change):
    e, n, free, intent, points = handoff()
    intent.steps[0] = replace(intent.steps[0], **change)
    assert e._activity_walk(0, points[-1]) is None


def test_route_geometry_is_immutable_and_checked_against_measurements():
    points = np.array([[0.0, 0.0], [3.0, 4.0]])
    proof = Route(5.0, 5.0, "proof", points=points)
    points[:] = 99.0
    assert proof.points == ((0.0, 0.0), (3.0, 4.0))
    d = Director([], [], lambda *args: proof)
    assert d._checked_route(proof, "walk", (0.0, 0.0), (3.0, 4.0)) is proof
    for invalid in [replace(proof, length_m=6.0), replace(proof, points=((1.0, 0.0), (3.0, 4.0)))]:
        with pytest.raises(ValueError, match="geometry differs"):
            d._checked_route(invalid, "walk", (0.0, 0.0), (3.0, 4.0))
    with pytest.raises(ValueError, match="finite XY"):
        Route(5.0, 5.0, "proof", points=((0.0, 0.0), (math.nan, 4.0)))
