import numpy as np

from awr.life.civil import body


def _fake_load(_asset):
    rest = np.array([[-0.15, -0.28, 0.0], [0.15, 0.28, 1.7], [0.0, 0.0, 0.9]])
    # walk cycle: front foot ahead, rear foot behind, arms slightly out
    poses = np.stack(
        [rest + [[-0.2, 0, 0], [0.4, 0.02, 0], [0, 0, 0]], rest + [[-0.35, -0.02, 0], [0.2, 0, 0], [0, 0, 0]]]
    )
    return {"dimensions_m": [0.3, 0.56, 1.7]}, {"p": rest, "poses": poses}, None


def test_extents_rest_inside_walk_and_monotone(monkeypatch):
    import awr.life.appearance.assets as assets

    monkeypatch.setattr(assets, "load", _fake_load)
    e = body.person_extents.__wrapped__("human-x", 1.7)
    rest, walk = e
    assert np.allclose(rest, [-0.17, 0.17, -0.30, 0.30])
    assert np.allclose(walk, [-0.52, 0.57, -0.32, 0.32])
    assert (walk[[0, 2]] <= rest[[0, 2]]).all() and (walk[[1, 3]] >= rest[[1, 3]]).all()
    sizes = [body.extents_at(e, v) for v in (0.0, 0.03, 0.1, 0.25, 0.5, 1.2)]
    widths = [s[1] - s[0] for s in sizes]
    assert widths == sorted(widths) and np.isclose(widths[0], 0.34) and np.isclose(widths[-1], 1.09)
    assert np.allclose(sizes[1], rest)  # at the rest threshold the rest pose is drawn


def test_boxes_offset_along_heading(monkeypatch):
    import awr.life.appearance.assets as assets

    monkeypatch.setattr(assets, "load", _fake_load)
    e = body.person_extents.__wrapped__("human-x", 1.7)
    ext = np.stack([e, e])
    centres, sizes = body.boxes(
        np.array([[0.0, 0.0], [5.0, 5.0]]), np.array([0.0, np.pi / 2]), np.array([1.0, 1.0]), ext
    )
    off = (-0.52 + 0.57) / 2
    assert np.allclose(centres, [[off, 0.0], [5.0, 5.0 + off]])
    assert np.allclose(sizes[:, :2], [[1.09, 0.64], [1.09, 0.64]])
    # rest turning disk: corners of the rest box about the origin
    assert np.isclose(body.turn_radius(e[0]), np.hypot(0.17, 0.30))


def test_centred_boxes_for_rigid_actors():
    ext = body.actor_extents(
        [{"class": "vehicle", "appearance": {"asset": "sedan@1"}, "size_mm": [4500, 1800, 1500]}],
        np.array([[4.5, 1.8, 1.5]]),
    )
    centres, sizes = body.boxes(np.array([[1.0, 2.0]]), np.array([0.3]), np.array([10.0]), ext)
    assert np.allclose(centres, [[1.0, 2.0]]) and np.allclose(sizes[0, :2], [4.5, 1.8])
