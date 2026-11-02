import unittest

import numpy as np

from awr.life.civil.road_surface import fill_gaps, lane_profile, plane_fit


class RoadSurface(unittest.TestCase):
    def test_gap_filling_is_bounded_and_interior_only(self):
        v = np.array(
            [
                0.0,
                1.0,
                np.nan,
                np.nan,
                4.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                14.0,
                np.nan,
            ]
        )
        z, filled = fill_gaps(v, 1.0, 5.0)
        self.assertTrue(np.allclose(z[2:4], [2.0, 3.0]))
        self.assertTrue(filled[2:4].all())
        self.assertTrue(np.isnan(z[5:14]).all(), "a gap longer than max_gap stays unknown")
        self.assertTrue(np.isnan(z[-1]), "no measured section after the end: stays unknown")

    def test_profile_grade_and_cross_slope(self):
        s = np.arange(0, 50, 0.5)
        zc = 0.1 * s
        zl, zr = zc - 0.08, zc + 0.08
        zc_noisy = zc.copy()
        zc_noisy[40] += 1.5  # a baked object in one sample
        zc_noisy[60:70] = np.nan  # a 5 m canopy gap on every track
        zl, zr = zl.copy(), zr.copy()
        zl[60:70] = np.nan
        zr[60:70] = np.nan
        prof = lane_profile(zc_noisy, zl, zr, 1.0, 0.5)
        self.assertTrue(np.isfinite(prof["z"]).all())
        self.assertTrue(prof["filled"][60:70].all())
        self.assertLess(abs(prof["z"][40] - zc[40]), 0.2, "the robust profile rejects the spike")
        self.assertTrue(np.allclose(prof["grade"][5:-5], 0.1, atol=0.02))
        self.assertTrue(np.allclose(prof["cross"], 0.08, atol=1e-6))

    def test_plane_fit_recovers_a_plane_inside_the_weight_mask(self):
        yy, xx = np.mgrid[:40, :40]
        Z = (0.1 * xx + 0.05 * yy).astype(np.float32)
        w = np.zeros_like(Z)
        w[10:30, 10:30] = 1.0
        Z[5, 5] = 100.0  # outside the mask: ignored
        fitted, slope, n = plane_fit(w, Z, 1.0, 5)
        self.assertTrue(np.allclose(fitted[12:28, 12:28], Z[12:28, 12:28], atol=1e-3))
        self.assertTrue(np.allclose(slope[12:28, 12:28], np.hypot(0.1, 0.05), atol=1e-3))
        self.assertTrue(np.isnan(fitted[0, 0]))


if __name__ == "__main__":
    unittest.main()


def test_graded_vehicle_ground_takes_reconstruction_noise_as_the_road_but_not_a_step():
    import numpy as np

    from awr.life.civil.surface import flat_vehicle_ground, graded_vehicle_ground

    rng = np.random.default_rng(0)
    cell = 0.5
    y, x = np.mgrid[:60, :80] * cell
    # A road climbing at 18 % with single-cell reconstruction spikes of 9 cm on 5 % of its cells.
    z = 0.18 * x + np.where(rng.random(x.shape) < 0.05, rng.choice([-0.09, 0.09], x.shape), 0.0)
    person = np.ones(z.shape, bool)
    flat = flat_vehicle_ground(z, person, cell)
    graded = graded_vehicle_ground(z, person, cell)
    inner = (slice(4, -4), slice(4, -4))
    assert graded[inner].all() and flat[inner].mean() < 0.8
    # A 30 cm step across the road: neither criterion lets a car stand at it.
    stepped = z + np.where(x >= 20.0, 0.3, 0.0)
    edge = (slice(4, -4), slice(39, 41))
    assert not graded_vehicle_ground(stepped, person, cell)[edge].any()
    assert not flat_vehicle_ground(stepped, person, cell)[edge].any()
    # A 25 cm object on the road, a smooth 30 % grade and unmeasured ground are not vehicle ground.
    lump = z.copy()
    lump[30, 40] += 0.25
    assert not graded_vehicle_ground(lump, person, cell)[30, 40]
    assert not graded_vehicle_ground(0.3 * x, person, cell)[inner].any()
    holed = z.copy()
    holed[25:35, 35:45] = np.nan
    assert not graded_vehicle_ground(holed, person, cell)[25:35, 35:45].any()
    # Cells not clear for a person stay closed.
    closed = person.copy()
    closed[10, 10] = False
    assert not graded_vehicle_ground(z, closed, cell)[10, 10]
