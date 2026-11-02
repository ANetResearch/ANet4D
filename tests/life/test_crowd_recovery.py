"""Short-retreat proposals at the actual rejected crowd frame and simple scenes.

The captured frame is a frozen-neighbour diagnostic, not a resumed simulation.
Its static segment answers came from the unchanged measured walking map.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from awr.life.civil import body
from awr.life.civil.crowd_recovery import choose_backsteps, retreat_corridor


def captured():
    root = Path(__file__).resolve().parent / "data/crowd-backsteps-v1"
    report = json.loads((root / "geometry-report.json").read_text())
    assert hashlib.sha256((root / "geometry.npz").read_bytes()).hexdigest() == report["geometry_sha256"]
    with np.load(root / "geometry.npz") as f:
        data = {k: f[k] for k in f.files}
    ids = data["ids"].tolist()
    rows = {ids.index(r["actor"]): r for r in report["rows"]}

    def clear(i, end):
        # Unknown directions are not licensed by a saved segment witness.
        return i in rows and any(
            o["static_clear"] and np.linalg.norm(np.asarray(o["endpoint"]) - end) < 1e-7 for o in rows[i]["options"]
        )

    return data, ids, rows, clear


def choose(data, eligible, clear, **kw):
    return choose_backsteps(
        *[data[k] for k in ("xy", "yaw", "speed", "ext", "centres", "sizes", "people")], eligible, clear, **kw
    )


def test_actual_blocked_yielder_does_not_trap_the_peer_with_available_space():
    data, ids, rows, clear = captured()
    eligible = np.zeros(len(ids), bool)
    a, b = map(ids.index, ("person-023", "person-217"))
    eligible[[a, b]] = True
    assert not any(o["static_clear"] for o in rows[b]["options"])
    selected = choose(data, eligible, clear)
    assert list(selected) == [a]
    assert selected[a]["priority_fallback"] and selected[a]["preferred_actor"] == b
    assert selected[a]["candidate_angle"] == 0.0


def test_actual_first_static_clear_retreat_can_hit_a_third_person():
    data, ids, rows, clear = captured()
    eligible = np.zeros(len(ids), bool)
    a, b = map(ids.index, ("person-196", "person-232"))
    eligible[[a, b]] = True
    first = rows[a]["options"][0]
    assert first["static_clear"] and first["blockers"] == ["person-102"]
    selected = choose(data, eligible, clear)
    assert a in selected and selected[a]["candidate_angle"] != 0.0
    chosen = next(o for o in rows[a]["options"] if o["angle"] == selected[a]["candidate_angle"])
    assert chosen["static_clear"] and not chosen["blockers"]


def test_selected_retreats_have_disjoint_reserved_body_sweeps():
    data, ids, rows, clear = captured()
    eligible = np.zeros(len(ids), bool)
    for i in rows:
        eligible[i] = data["speed"][i] <= 0.05
    before = {k: v.copy() for k, v in data.items()}
    selected = choose(data, eligible, clear)
    corridors = []
    assert len(selected) > 1
    for i, row in selected.items():
        corridor = retreat_corridor(
            data["xy"][i], data["yaw"][i], data["ext"][i], row["direction"], row["distance_m"], 0.2
        )
        assert not any(corridor.intersects(other) for other in corridors)
        corridors.append(corridor)
    for k in data:
        np.testing.assert_array_equal(data[k], before[k])


def simple():
    xy = np.array([[0.0, 0.0], [0.8, 0.0], [1.5, 0.0]])
    ext = np.tile(body.centred([0.4, 0.4, 1.7]), (3, 1, 1))
    yaw = np.zeros(3)
    speed = np.zeros(3)
    centres, sizes = body.boxes(xy, yaw, speed, ext)
    return dict(xy=xy, yaw=yaw, speed=speed, ext=ext, centres=centres, sizes=sizes, people=np.ones(3, bool))


def test_other_body_exclusion_and_current_retreat_are_respected():
    data = simple()
    eligible = np.array([True, True, False])
    selected = choose(data, eligible, lambda i, end: True)
    assert list(selected) == [1] and abs(selected[1]["candidate_angle"]) > 0.0
    assert not choose(data, eligible, lambda i, end: False)
    assert not choose(data, eligible, lambda i, end: True, active={1: (np.array([1.0, 0.0]), 0.6)})
    data["speed"][1] = 0.3
    eligible[1] = False
    assert not choose(data, eligible, lambda i, end: True)


def test_bad_common_frames_and_unbounded_retreat_parameters_rejected():
    data = simple()
    eligible = np.array([True, True, False])
    with pytest.raises(ValueError):
        choose(data, eligible, lambda i, end: True, dt=1.0)
    with pytest.raises(ValueError):
        choose(data, eligible, lambda i, end: True, max_speed=3.0)
    data["xy"][0, 0] = np.nan
    with pytest.raises(ValueError):
        choose(data, eligible, lambda i, end: True)
