import numpy as np
import shapely

from awr.life.civil.measured_support import compile_grid, domains


def semantics():
    return dict(roads=[dict(line=[[0, 10], [20, 10]], width_m=6)], buildings=[], paths=[], open_areas=[])


def compile(sem=None, ground=None, clear=None):
    return compile_grid(
        sem or semantics(),
        [0, 0, 20, 20],
        1.0,
        lambda p: np.ones(len(p)) * 100,
        ground or (lambda p, h: (np.zeros(len(p)), np.ones(len(p), bool))),
        clear or (lambda p, z: np.ones(len(p), bool)),
        batch=61,
    )


def test_finite_hint_never_creates_missing_ground():
    z, p, v, info = compile(ground=lambda p, h: (h, np.zeros(len(p), bool)))
    assert np.isnan(z).all() and not p.any() and not v.any()
    assert info["mesh_hit_cells"] == 0


def test_actual_intersection_not_hint_is_recorded_and_holes_remain_blocked():
    z, p, v, _ = compile(ground=lambda xy, h: (np.ones(len(xy)) * 7, xy[:, 0] < 10))
    assert np.all(z[np.isfinite(z)] == 7)
    assert np.isnan(z[:, 10:]).all() and not p[:, 9:].any() and not v[:, 9:].any()


def test_buildings_and_clearance_reject_otherwise_flat_ground():
    sem = semantics()
    sem["buildings"] = [dict(polygon=[[4, 4], [8, 4], [8, 8], [4, 8]])]
    z, p, v, _ = compile(sem, clear=lambda xy, h: ~((xy[:, 0] > 12) & (h > 1.4)))
    assert np.isnan(z[4:8, 4:8]).all()
    assert not p[:, 12:].any() and not v[:, 12:].any()
    assert p[10, 10] and v[10, 10]


def test_road_candidate_does_not_allow_vehicles_on_steep_ground():
    z, p, v, _ = compile(ground=lambda xy, h: (0.4 * xy[:, 0], np.ones(len(xy), bool)))
    assert p.any() and not v.any()


def test_parking_requires_declared_source_evidence_and_buildings_stay_excluded():
    region = dict(polygon=[[30, 30], [40, 30], [40, 40], [30, 40]], verified=True)
    sem = semantics()
    sem["verified_parking_regions"] = [region]
    _, drive = domains(sem, [0, 0, 50, 50])
    assert not shapely.intersects_xy(drive, 35, 35)
    region["evidence"] = "mapped OSM parking polygon with visual review"
    _, drive = domains(sem, [0, 0, 50, 50])
    assert shapely.intersects_xy(drive, 35, 35)
    sem["buildings"] = [dict(polygon=[[34, 34], [36, 34], [36, 36], [34, 36]])]
    _, drive = domains(sem, [0, 0, 50, 50])
    assert not shapely.intersects_xy(drive, 35, 35)
