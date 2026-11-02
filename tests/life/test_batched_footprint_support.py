import numpy as np

from awr.life.civil.grounded import footprint_support, support_mask
from awr.life.civil.surface import SurfaceMap


def terrain():
    surface = object.__new__(SurfaceMap)
    surface.x0 = -8.0
    surface.y0 = -8.0
    surface.cell_m = 0.5
    surface.nx = surface.ny = 32
    rng = np.random.default_rng(91)
    surface.z = np.zeros((32, 32))
    surface.z[4:6, 18:20] = np.nan
    surface.walk = {p: rng.random((32, 32)) > 0.08 for p in ("person", "vehicle")}
    return surface


def test_batching_retains_bit_exact_probe_coordinates_and_all_pointwise_failures():
    rng = np.random.default_rng(17)
    pp = rng.uniform(-10, 10, (8197, 2))
    yaw = rng.uniform(-4, 4, len(pp))
    offsets = rng.uniform(-2, 2, (17, 2))
    c, s = np.cos(yaw), np.sin(yaw)
    original = [pp + np.column_stack([dx * c - dy * s, dx * s + dy * c]) for dx, dy in offsets]
    surface = terrain()
    record = []

    class Recording:
        def supported(self, xy, profile):
            record.append(xy.copy())
            return surface.supported(xy, profile)

    for profile in ("person", "vehicle"):
        record.clear()
        expected = np.logical_and.reduce([surface.supported(q, profile) for q in original])
        actual = footprint_support(Recording(), pp, c, s, offsets, profile)
        assert np.array_equal(actual, expected)
        assert np.array_equal(np.concatenate(record).reshape(len(pp), len(offsets), 2), np.stack(original, axis=1))
        assert len(record) == 3 and max(map(len, record)) <= 4096 * 17
        assert actual.any() and not actual.all()


def test_vehicle_intermediate_hole_is_still_rejected_and_steering_is_unchanged(monkeypatch):
    monkeypatch.setattr("awr.life.civil.contacts.vehicle_contacts", lambda size: np.array([[0.0, 0.0]]))

    class NarrowHole:
        def supported(self, xy, profile):
            return ~((np.abs(xy[:, 0] - 0.175) < 0.01) & (np.abs(xy[:, 1]) < 0.01))

    # Both endpoint centres are supported, but an interpolated centre hits the hole.
    pp = np.array([[0.0, 0.0], [0.35, 0.0]])
    assert NarrowHole().supported(pp, "vehicle").all()
    assert not support_mask(NarrowHole(), pp, np.zeros(2), [4.2, 1.8, 1.5], "vehicle").any()

    class Everywhere:
        def supported(self, xy, profile):
            return np.ones(len(xy), bool)

    assert support_mask(Everywhere(), pp, np.zeros(2), [4.2, 1.8, 1.5], "vehicle").all()
    assert not support_mask(Everywhere(), pp, np.array([0.0, 1.0]), [4.2, 1.8, 1.5], "vehicle")[1]
