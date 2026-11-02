import unittest

import numpy as np
from scipy.spatial import cKDTree

from awr.life.civil.collision import actor_blocked
from awr.life.civil.grounded import at
from awr.life.civil.prediction import blocked_disks, blocked_many, route_at_many


class Prediction(unittest.TestCase):
    def test_pedestrian_disk_clearance_and_long_vehicle_sidewalk(self):
        xy = np.array([[0.0, 0.0], [10.0, 10.0], [30.0, 10.0]])
        sizes = np.array([[1.0, 0.6, 1.8], [1.0, 0.6, 1.8], [12.0, 2.5, 3.0]])
        circles = np.array([0.6, 0.6, np.nan])
        radii = np.linalg.norm(sizes[:, :2], axis=1) / 2
        points = np.array([[11.3, 10.0], [11.27, 10.0], [30.0, 12.0], [30.0, 11.0]])
        expected = [False, True, False, True]
        np.testing.assert_array_equal(
            blocked_disks(0, points, 0.6, xy, np.zeros(3), sizes, cKDTree(xy), radii, circles), expected
        )

    def test_interpolation_is_identical_at_ends_and_corners(self):
        for route in [
            dict(points=np.array([[0, 0], [1, 0], [1, 2.0]]), arc=np.array([0, 1, 3.0])),
            dict(points=np.array([[0, 0], [0.35, 0.1], [0.7, 0.2]]), yaw=np.array([3.0, -3.0, -2.8])),
        ]:
            d = np.linspace(-1, 5, 100)
            p, y = route_at_many(route, d)
            for i, x in enumerate(d):
                a, b = at(route, x)
                np.testing.assert_allclose(p[i], a, atol=1e-12)
                self.assertAlmostEqual(y[i], b)

    def test_full_body_queries_match_scalar_at_every_sample(self):
        rng = np.random.default_rng(3)
        xy = rng.normal(size=(30, 2)) * 5
        yaw = rng.uniform(-3, 3, 30)
        sizes = rng.uniform(0.5, 4, (30, 3))
        radii = np.linalg.norm(sizes[:, :2], axis=1) / 2
        tree = cKDTree(xy)
        pp = rng.normal(size=(100, 2)) * 5
        aa = rng.uniform(-3, 3, 100)
        expect = [
            actor_blocked(0, p, a, xy, yaw, sizes, tree, radii, body_size=sizes[0])
            for p, a in zip(pp, aa, strict=False)
        ]
        np.testing.assert_array_equal(blocked_many(0, pp, aa, sizes[0], xy, yaw, sizes, tree, radii), expect)


def test_blocked_ahead_ignores_a_close_follower_behind():
    import numpy as np
    from scipy.spatial import cKDTree

    from awr.life.civil.prediction import blocked_ahead, blocked_many

    body = np.array([1.2, 0.72, 1.7])
    positions = np.array([[0.0, 0.0], [-1.1, 0.0]])  # walker at origin, follower 1.1 m behind
    yaws = np.zeros(2)
    sizes = np.array([body, body])
    radii = np.linalg.norm(sizes[:, :2], axis=1) / 2
    tree = cKDTree(positions)
    d = np.arange(0.025, 1.5, 0.025)
    pts = np.column_stack([d, np.zeros_like(d)])
    ang = np.zeros(len(d))
    inflated = body + np.array([0.25, 0.25, 0.0])
    assert blocked_many(0, pts, ang, inflated, positions, yaws, sizes, tree, radii).any()
    assert not blocked_ahead(0, pts, ang, inflated, positions, yaws, sizes, tree, radii).any()
    # a body in front still blocks
    positions2 = np.array([[0.0, 0.0], [1.6, 0.0]])
    tree2 = cKDTree(positions2)
    assert blocked_ahead(0, pts, ang, inflated, positions2, yaws, sizes, tree2, radii).any()


def test_resting_neighbour_is_tested_against_the_own_body():
    # A walker passes a standing person whose box is 0.44 m from the walker's
    # centre line. The 0.25 m inflation (for a neighbour turning in the tick)
    # refuses it; at rest the own body plus 0.1 m is used, which passes.
    from awr.life.civil.prediction import blocked_ahead

    pts = np.stack([np.linspace(0, 4, 81), np.zeros(81)], -1)
    ang = np.zeros(81)
    me = np.array([1.2, 0.7, 1.7])
    positions = np.array([[0.0, 0.0], [2.0, 0.44 + 0.35]])
    yaws = np.zeros(2)
    sizes = np.array([me, me])
    tree = cKDTree(positions)
    radii = np.linalg.norm(sizes[:, :2], axis=1) / 2
    inflated = me + np.array([0.25, 0.25, 0.0])
    assert blocked_ahead(0, pts, ang, inflated, positions, yaws, sizes, tree, radii).any()
    assert not blocked_ahead(
        0,
        pts,
        ang,
        inflated,
        positions,
        yaws,
        sizes,
        tree,
        radii,
        still=np.array([False, True]),
        still_body=me + np.array([0.1, 0.1, 0.0]),
    ).any()
    assert blocked_ahead(
        0,
        pts,
        ang,
        inflated,
        positions,
        yaws,
        sizes,
        tree,
        radii,
        still=np.array([False, False]),
        still_body=me + np.array([0.1, 0.1, 0.0]),
    ).any()

    hit, who = blocked_ahead(0, pts, ang, inflated, positions, yaws, sizes, tree, radii, return_blocker=True)
    assert hit.any() and who == 1


def test_lagging_heading_of_the_own_body_is_tested():
    # A walker on a left bend reaches each sample with the heading of a step that
    # began earlier (here 0.3 rad to the right of the tangent). Its front corner
    # then reaches a standing body that the tangent pose clears by 0.03 m.
    from awr.life.civil.prediction import blocked_ahead

    pts = np.stack([np.linspace(0, 3, 121), np.zeros(121)], -1)
    ang = np.zeros(121)
    me = np.array([1.0, 0.6, 1.7])
    other = np.array([0.4, 0.6, 1.7])
    positions = np.array([[0.0, 0.0], [2.0, -0.68]])
    yaws = np.zeros(2)
    sizes = np.array([me, other])
    tree = cKDTree(positions)
    radii = np.linalg.norm(sizes[:, :2], axis=1) / 2
    kw = dict(still=np.array([False, True]), still_body=me + np.array([0.1, 0.1, 0.0]))
    inflated = me + np.array([0.25, 0.25, 0.0])
    assert not blocked_ahead(0, pts, ang, inflated, positions, yaws, sizes, tree, radii, **kw).any()
    lagged = np.full((1, len(pts)), -0.3)
    hit, who = blocked_ahead(
        0,
        pts,
        ang,
        inflated,
        positions,
        yaws,
        sizes,
        tree,
        radii,
        return_blocker=True,
        headings=(pts[None], lagged),
        **kw,
    )
    assert hit.any() and who == 1
    # samples without a further heading (NaN) are tested at the tangent only
    assert not blocked_ahead(
        0,
        pts,
        ang,
        inflated,
        positions,
        yaws,
        sizes,
        tree,
        radii,
        headings=(pts[None], np.full((1, len(pts)), np.nan)),
        **kw,
    ).any()


def test_a_resting_body_does_not_start_into_a_passer_by_behind_it():
    # A standing person (index 0, facing +x) is about to start. A walker passes
    # behind it across its heading; its braking corridor overlaps the standing
    # person's walking box. A follower would be exempt (it brakes for the body
    # ahead), but the standing person relies on nothing of the kind from a
    # passer-by: it waits. A body at rest behind it stays exempt.
    from awr.life.civil.prediction import blocked_ahead

    pts = np.stack([np.linspace(0, 2, 81), np.zeros(81)], -1)
    ang = np.zeros(81)
    me = np.array([1.0, 0.6, 1.7])
    positions = np.array([[0.0, 0.0], [-0.8, 0.3]])
    yaws = np.array([0.0, -np.pi / 2])
    sizes = np.array([me, me + np.array([0.35, 0, 0])])
    tree = cKDTree(positions)
    radii = np.linalg.norm(sizes[:, :2], axis=1) / 2
    inflated = me + np.array([0.25, 0.25, 0.0])
    still_body = me + np.array([0.1, 0.1, 0.0])
    moving = np.array([True, False])
    assert not blocked_ahead(
        0, pts, ang, inflated, positions, yaws, sizes, tree, radii, still=moving, still_body=still_body
    ).any()
    assert blocked_ahead(
        0, pts, ang, inflated, positions, yaws, sizes, tree, radii, still=moving, still_body=still_body, starting=True
    ).any()
    resting = np.array([True, True])
    from awr.life.civil.prediction import blocked_many

    assert blocked_many(0, pts[:1], ang[:1], still_body, positions, yaws, sizes, tree, radii).all()  # overlaps behind
    assert not blocked_ahead(
        0, pts, ang, inflated, positions, yaws, sizes, tree, radii, still=resting, still_body=still_body, starting=True
    ).any()
