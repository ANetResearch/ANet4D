import numpy as np
from scipy.ndimage import binary_erosion

from awr.life.civil.surface import flat_vehicle_ground
from awr.life.envpack.local_registration import DisplacementField, cross_sections, densify, smooth


class Strip:
    """Vehicle ground between two kerbs at y = lo and y = hi."""

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def supported(self, xy, profile="vehicle"):
        y = np.asarray(xy, float).reshape(-1, 2)[:, 1]
        return (y >= self.lo) & (y <= self.hi)


def test_flat_vehicle_ground_stops_at_a_step_and_matches_the_support_rule():
    z = np.zeros((20, 30))
    z[:, 18:] = 0.3
    z[0, 0] = np.nan
    person = np.ones_like(z, bool)
    person[5, 5] = False
    ground = flat_vehicle_ground(z, person, 0.5)
    assert ground[10, 10] and ground[10, 24] and not ground[10, 17] and not ground[10, 18] and not ground[5, 5]
    dy, dx = np.gradient(z, 0.5)
    slope = np.hypot(dx, dy)
    valid = np.isfinite(z) & np.isfinite(slope)
    assert np.array_equal(ground, binary_erosion(valid & (slope <= 0.25), iterations=1) & person)


def test_a_carriageway_beside_its_centreline_is_measured_as_an_offset():
    road = dict(id="r", line=[[0.0, 0.0], [60.0, 0.0]], width_m=6.0)
    rows, counts = cross_sections(Strip(-1.5, 4.5), [road])
    assert counts["measured"] == counts["stations"] > 20
    assert all(abs(m["offset_m"] - 1.5) < 0.11 and abs(m["width_m"] - 6.0) < 0.21 for m in rows)
    assert all(m["normal"] == [-0.0, 1.0] or np.allclose(m["normal"], [0.0, 1.0]) for m in rows)


def test_open_or_out_of_width_ground_gives_no_measurement():
    road = dict(id="r", line=[[0.0, 0.0], [60.0, 0.0]], width_m=6.0)
    assert cross_sections(Strip(-50.0, 50.0), [road])[1]["unbounded"] > 0
    rows, counts = cross_sections(Strip(-0.5, 0.5), [road])
    assert not rows and counts["width_out_of_bounds"] == counts["stations"]


def test_outliers_are_dropped_against_the_running_median():
    rows = [dict(road="r", s_m=float(s), offset_m=1.0 if s != 10 else 3.0) for s in range(0, 40, 2)]
    kept, dropped = smooth(rows)
    assert dropped == 1 and all(m["offset_m"] == 1.0 for m in kept)


def test_field_follows_the_normal_offset_and_vanishes_away_from_measurements():
    road = dict(id="r", line=[[0.0, 0.0], [200.0, 0.0]], width_m=6.0)
    rows, _ = cross_sections(Strip(-1.0, 5.0), [road])
    field = DisplacementField(rows, sigma_m=6.0, prior=0.5, max_m=5.0)
    d = field.displace([[100.0, 0.0], [100.0, 40.0]])
    assert abs(d[0, 0]) < 1e-6 and 1.85 < d[0, 1] <= 2.0 and np.allclose(d[1], 0.0)
    assert (
        np.hypot(*DisplacementField(rows, sigma_m=6.0, prior=0.5, max_m=1.0).displace([[100.0, 0.0]])[0]) <= 1.0 + 1e-9
    )


def test_densify_keeps_vertices_and_bounds_segments():
    out = np.asarray(densify([[0.0, 0.0], [5.0, 0.0], [5.0, 1.0]], 2.0))
    assert out[0].tolist() == [0.0, 0.0] and [5.0, 0.0] in out.tolist() and out[-1].tolist() == [5.0, 1.0]
    assert np.max(np.linalg.norm(np.diff(out, axis=0), axis=1)) <= 2.0 + 1e-9
