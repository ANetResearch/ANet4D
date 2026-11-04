"""Canonical Event Tape v2（drone4d.tape.v2）：29 B 记录（u32 容器、俯仰与横滚），与 v1 并存。

v2 由目标生活引擎 awr.life 写出；v1 磁带与 v1 写入路径不变。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zstandard")

from awr.sim.events import tape as tp
from awr.sim.events.errors import EventError

from .test_tape_codec import H64, header, walkers

ROOT = Path(__file__).resolve().parents[3]


def header_v2(n_ticks: int, n_person: int, n_vehicle: int) -> dict:
    h = header(n_ticks, 0, n_person + n_vehicle)
    h["schema"] = "drone4d.tape.v2"
    h["record"] = tp.record_spec(2)
    for a in h["actors"][n_person:]:
        a["class"], a["profile"], a["size_mm"] = "vehicle", "car", [4500, 1800, 1500]
        a["appearance"] = {"asset": "car-proxy@1"}
    return h


def frames_v2(n_ticks: int, n_person: int, n_vehicle: int) -> np.ndarray:
    v1 = walkers(n_ticks, n_person + n_vehicle, seed=3)
    f = np.zeros(v1.shape, tp.RECORD_V2)
    for name in v1.dtype.names:
        if name != "container":
            f[name] = v1[name]
    f["container"] = tp.NO_CONTAINER_V2
    f["pitch_cdeg"] = np.round(np.linspace(-300, 300, n_ticks))[:, None]
    f["roll_cdeg"] = 120
    # 第一个人在 100..199 tick 乘坐第一辆车
    f["state"][100:200, 0] = 3
    f["container"][100:200, 0] = n_person
    return f


@pytest.fixture()
def tape_v2(tmp_path: Path):
    n_t, n_p, n_v = 350, 6, 3
    hd, fr = header_v2(n_t, n_p, n_v), frames_v2(n_t, n_p, n_v)
    h = tp.write_tape(tmp_path / "tape.bin", hd, fr, n_scripted=0)
    m = tp.manifest(h, hd, provenance={"generator": "test", "generator_version": "0", "recast_navigation": "none", "wasm_sha256": H64,
                                       "node": "none", "host": "test", "wall_ms": 0, "rss_mb": 0.0, "created_at": "2026-11-04T00:00:00Z"})
    (tmp_path / "tape.json").write_text(json.dumps(m))
    return tmp_path, hd, fr


def test_roundtrip_v2(tape_v2):
    d, _hd, fr = tape_v2
    t = tp.open_tape(d)
    assert t.version == 2 and t.record == tp.RECORD_V2 and t.header["record"]["bytes"] == 29
    got = np.concatenate([t.block_frames(i) for i in range(len(t.blocks))])
    exp = fr.copy()
    exp["yaw_cdeg"] = tp.fold_yaw(exp["yaw_cdeg"])
    for name in tp.RECORD_V2.names:
        assert np.array_equal(got[name], exp[name]), name
    st = t.actor_states(150 * tp.TICK_NS)
    assert st.state[0] == 3 and st.container[0] == 6
    assert st.pitch_rad is not None and abs(float(st.pitch_rad[1]) - np.radians(fr["pitch_cdeg"][150, 1] / 100)) < 1e-6
    assert abs(float(st.roll_rad[2]) - np.radians(1.2)) < 1e-9
    # 半 tick 处俯仰为线性插值
    st2 = t.actor_states(150 * tp.TICK_NS + tp.TICK_NS // 2)
    mid = np.radians((fr["pitch_cdeg"][150, 1] + fr["pitch_cdeg"][151, 1]) / 200)
    assert abs(float(st2.pitch_rad[1]) - mid) < 1e-6


def test_v2_header_schema(tape_v2):
    jsonschema = pytest.importorskip("jsonschema")
    from referencing import Registry, Resource

    d, _hd, _fr = tape_v2
    ev = ROOT / "packages/contracts/event"
    reg = Registry()
    for name in ("tape.v1.schema.json", "tape.v2.schema.json", "tape_manifest.v1.schema.json"):
        s = json.loads((ev / name).read_text())
        reg = reg.with_resource(s["$id"], Resource.from_contents(s))
    jsonschema.Draft202012Validator(json.loads((ev / "tape.v2.schema.json").read_text()), registry=reg).validate(tp.open_tape(d).header)
    man = json.loads((d / "tape.json").read_text())
    jsonschema.Draft202012Validator(json.loads((ev / "tape_manifest.v1.schema.json").read_text()), registry=reg).validate(man)


def test_v1_still_writes_and_reads(tmp_path: Path):
    hd, fr = header(120, 2, 3), walkers(120, 5)
    tp.write_tape(tmp_path / "tape.bin", hd, fr, n_scripted=2)
    t = tp.TapeReader(tmp_path / "tape.bin")
    assert t.version == 1 and t.record == tp.RECORD
    st = t.actor_states(5 * tp.TICK_NS)
    assert st.pitch_rad is not None and float(np.abs(st.pitch_rad).max()) == 0.0


def test_v2_rejects_bad_container(tmp_path: Path):
    hd, fr = header_v2(120, 2, 1), frames_v2(120, 2, 1)
    bad = fr.copy()
    bad["container"][10, 0] = 1                                   # 容器是人，不是车
    bad["state"][10, 0] = 3
    with pytest.raises(EventError):
        tp.write_tape(tmp_path / "a.bin", hd, bad, n_scripted=0)
    bad = fr.copy()
    bad["state"][10, 1] = 3                                       # in_vehicle 但无容器
    with pytest.raises(EventError):
        tp.write_tape(tmp_path / "b.bin", hd, bad, n_scripted=0)
    bad_hd = dict(hd, record=tp.record_spec(1))                    # 记录布局与 schema 不符
    with pytest.raises(EventError):
        tp.write_tape(tmp_path / "c.bin", bad_hd, fr, n_scripted=0)
