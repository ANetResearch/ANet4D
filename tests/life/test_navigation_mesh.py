from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from awr.life.civil.navigation_mesh import support_mesh


def surface():
    return SimpleNamespace(
        x0=0.0,
        y0=0.0,
        cell_m=1.0,
        nx=5,
        ny=5,
        z=np.zeros((5, 5)),
        walk={"vehicle": np.ones((5, 5), bool), "person": np.ones((5, 5), bool)},
    )


def faces(v, t):
    return [Polygon(v[row][:, [0, 2]] * [1.0, -1.0]) for row in t]


def test_permission_hole_between_valid_corners_cannot_be_bridged():
    s = surface()
    area = box(0, 0, 5, 5).difference(box(0.9, 0.9, 1.1, 1.1))
    # All four vertices of the lower-left quad are permitted, its interior isn't.
    v, t = support_mesh(s, area, "vehicle")
    assert len(t) == 30
    assert all(area.covers(f) for f in faces(v, t))
    assert all(np.cross(v[row[1]] - v[row[0]], v[row[2]] - v[row[0]])[1] > 0 for row in t)


def test_unknown_and_profile_invalid_cells_stay_absent_and_chunks_are_identical():
    s = surface()
    s.z[2, 2] = np.nan
    s.walk["vehicle"][0, 0] = False
    a = box(0, 0, 5, 5)
    v, t = support_mesh(s, a, "vehicle", chunk_rows=1)
    assert len(t) == 22  # four quads at the unknown centre and one at invalid corner
    assert np.isfinite(v).all()
    vv, tt = support_mesh(s, a, "vehicle", chunk_rows=32)
    np.testing.assert_array_equal(v, vv)
    np.testing.assert_array_equal(t, tt)
    assert len(support_mesh(s, a, "person")[1]) == 24


def test_bounds_clip_whole_quads_and_empty_regions_return_empty_meshes():
    s = surface()
    a = box(0, 0, 5, 5)
    v, t = support_mesh(s, a, "vehicle", bounds=[1, 1, 4, 4])
    assert len(t) == 8
    assert all(box(1, 1, 4, 4).covers(f) for f in faces(v, t))
    assert support_mesh(s, box(20, 20, 22, 22), "vehicle")[0].shape == (0, 3)
    for kw in ({"chunk_rows": 0}, {"bounds": [2, 1, 1, 2]}):
        with pytest.raises(ValueError):
            support_mesh(s, a, "vehicle", **kw)
    with pytest.raises(ValueError):
        support_mesh(s, None, "vehicle")


def test_export_binds_source_and_geometry_hashes_and_refuses_overwrite(tmp_path):
    import hashlib
    import json

    import shapely

    from awr.life.civil.navigation_mesh import export

    s = surface()
    root = tmp_path / "support"
    root.mkdir()
    (root / "meta.json").write_text(json.dumps(dict(schema="awr.life.support.v1", x0=0, y0=0, nx=5, ny=5, cell_m=1.0)))
    np.save(root / "ground.npy", s.z)
    for name in ("person", "vehicle"):
        np.save(root / f"walk_{name}.npy", s.walk[name])
    permission = tmp_path / "area.geojson"
    permission.write_text(shapely.to_geojson(box(0, 0, 5, 5)))
    out = tmp_path / "out"
    r = export(root, permission, "vehicle", out)
    assert r["release_evidence"] is False and r["triangles"] == 32
    for name, digest in r["outputs"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    before = (out / "positions.f32").read_bytes()
    with pytest.raises(FileExistsError):
        export(root, permission, "vehicle", out)
    assert (out / "positions.f32").read_bytes() == before
