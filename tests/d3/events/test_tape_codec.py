"""Canonical Event Tape（M23 §6.7、§6.8；AWR-05 §5.2.3）：往返、哈希不变量、损坏检测、插值规则、头契约与存储预算。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zstandard")

from awr.sim.events import tape as tp
from awr.sim.events.errors import EventError
from awr.sim.events.jcs import jcs

ROOT = Path(__file__).resolve().parents[3]
H64 = "0" * 64


def header(n_ticks: int, n_s: int, n_b: int) -> dict:
    actors = [{"idx": i, "id": f"p{i:02d}" if i < n_s else f"bg{i - n_s:04d}", "class": "person", "scripted": i < n_s, "profile": "person",
               "size_mm": [500, 300, 1750], "appearance": {"asset": "person-proxy@1"}} for i in range(n_s + n_b)]
    return {"schema": "drone4d.tape.v1", "event_id": "synth-walk", "event_version": 1, "event_sha256": H64, "mode": "exogenous", "seed": 7,
            "tick_hz": 10, "n_ticks": n_ticks, "duration_ms": n_ticks * 100, "record": tp.record_spec(),
            "codec": {"name": "col-delta-zstd@1", "block_ticks": 100}, "actors": actors,
            "world": {"id": "synth", "content_version": "v1", "raycast_sha256": H64, "navmesh_sha256": {"person": H64}, "anchors_sha256": H64},
            "environment": {}, "env_track": [], "comms": {}, "anim": None, "realized": None}


def walkers(n_ticks: int, n_actors: int, seed: int = 0) -> np.ndarray:
    """1.0–1.6 m/s 沿折线步行（约 20 s 一个拐角，1 s 内转完）、偶发停留（state 2）；与 NavMesh 路径同量级的统计特征。"""
    rng = np.random.default_rng(seed)
    f = np.zeros((n_ticks, n_actors), tp.RECORD)
    pos = rng.uniform(-400, 400, (n_actors, 2))
    hd = rng.uniform(-np.pi, np.pi, n_actors)
    sp = rng.uniform(1.0, 1.6, n_actors)
    wait = np.zeros(n_actors, int)
    turn = np.zeros(n_actors)
    for k in range(n_ticks):
        turn = np.where(rng.random(n_actors) < 0.005, rng.uniform(-0.2, 0.2, n_actors), np.where(rng.random(n_actors) < 0.1, 0.0, turn))
        hd += turn
        wait = np.where((wait == 0) & (rng.random(n_actors) < 0.003), rng.integers(20, 60, n_actors), np.maximum(wait - 1, 0))
        v = np.where(wait[:, None] > 0, 0.0, sp[:, None] * np.stack([np.cos(hd), np.sin(hd)], 1))
        pos += v * 0.1
        f["x_mm"][k], f["y_mm"][k] = np.round(pos[:, 0] * 1000), np.round(pos[:, 1] * 1000)
        f["z_mm"][k] = 12_000
        f["yaw_cdeg"][k] = np.round(np.degrees(hd) * 100)              # 不折叠：写出方负责折叠
        f["vx_cmps"][k], f["vy_cmps"][k] = np.round(v[:, 0] * 100), np.round(v[:, 1] * 100)
        f["state"][k] = np.where(wait > 0, 2, 1)
        f["container"][k] = tp.NO_CONTAINER
    return f


@pytest.fixture()
def tape(tmp_path: Path):
    n_t, n_s, n_b = 450, 8, 40
    hd, fr = header(n_t, n_s, n_b), walkers(n_t, n_s + n_b)
    h = tp.write_tape(tmp_path / "tape.bin", hd, fr, n_scripted=n_s)
    m = tp.manifest(h, hd, provenance={"generator": "test", "generator_version": "0", "recast_navigation": "none", "wasm_sha256": H64,
                                       "node": "none", "host": "test", "wall_ms": 0, "rss_mb": 0.0, "created_at": "2026-10-17T00:00:00Z"})
    (tmp_path / "tape.json").write_text(json.dumps(m))
    return tmp_path, hd, fr, h


def test_round_trip_and_hashes(tape) -> None:
    d, hd, fr, h = tape
    r = tp.open_tape(d)
    assert r.header_bytes == jcs(hd) and r.n_scripted == 8 and len(r.blocks) == 5 and r.blocks[-1].n_ticks == 50
    back = np.concatenate([r.block_frames(i) for i in range(len(r.blocks))])
    want = fr.copy()
    want["yaw_cdeg"] = tp.fold_yaw(fr["yaw_cdeg"])
    assert np.array_equal(back, want)
    assert r.verify(h["content_sha256"]) == {k: h[k] for k in ("content_sha256", "body_sha256", "header_sha256", "file_sha256")}
    assert int(back["yaw_cdeg"].min()) >= -18000 and int(back["yaw_cdeg"].max()) < 18000
    assert r.block_frames(0, background=False).shape == (100, 8)


def test_content_hash_ignores_compression_and_split(tmp_path: Path, tape) -> None:
    _d, hd, fr, h = tape
    h1 = tp.write_tape(tmp_path / "a.bin", hd, fr, n_scripted=8, level=1)
    h2 = tp.write_tape(tmp_path / "b.bin", hd, fr, n_scripted=48)                       # 不拆背景子块
    assert h1["content_sha256"] == h2["content_sha256"] == h["content_sha256"]
    assert h1["file_sha256"] != h["file_sha256"]
    alt = dict(hd, seed=8)
    assert tp.write_tape(tmp_path / "c.bin", alt, fr, n_scripted=8)["content_sha256"] != h["content_sha256"]


def test_corruption_and_world_mismatch(tape) -> None:
    d, _hd, _fr, h = tape
    b = bytearray((d / "tape.bin").read_bytes())
    b[-40] ^= 0x10                                                                         # 末块数据
    (d / "tape.bin").write_bytes(bytes(b))
    with pytest.raises(EventError) as e:
        tp.open_tape(d)
    assert e.value.code == 614
    r = tp.TapeReader(d / "tape.bin")
    r.actor_states(1_000_000_000, [0, 1])                                                  # 未触及的块仍可读
    with pytest.raises(EventError) as e:
        r.check_world("v2")
    assert e.value.code == 615 and e.value.reason == "TAPE_WORLD_MISMATCH"
    (d / "tape.bin").write_bytes(b"NOTATAPE" + bytes(b[8:]))
    with pytest.raises(EventError) as e:
        tp.TapeReader(d / "tape.bin")
    assert e.value.code == 620
    assert h["bytes"] == len(b)


def test_interpolation_rules(tmp_path: Path) -> None:
    hd = header(5, 3, 0)
    f = np.zeros((5, 3), tp.RECORD)
    f["container"] = tp.NO_CONTAINER
    f["state"][:, 0] = 1
    f["x_mm"][:, 0] = [0, 100, 200, 300, 400]                                              # 1 m/s 匀速
    f["vx_cmps"][:, 0] = 100
    f["yaw_cdeg"][:, 0] = [17900, -17900, -17700, -17500, -17300]                          # 跨 ±180°
    f["state"][:, 1] = [1, 1, 3, 3, 1]                                                     # 上车：阶跃
    f["x_mm"][:, 1] = [0, 0, 5000, 5000, 9000]
    f["container"][2:4, 1] = 2
    f["state"][:, 2] = [0, 1, 1, 5, 5]
    f["x_mm"][:, 2] = [0, 1000, 2000, 2000, 2000]
    tp.write_tape(tmp_path / "t.bin", hd, f, n_scripted=3)
    r = tp.TapeReader(tmp_path / "t.bin")
    s = r.actor_states(150_000_000)
    assert s.pos_enu_m[0, 0] == pytest.approx(0.15, abs=1e-9) and s.vel_enu_mps[0, 0] == pytest.approx(1.0)
    assert np.degrees(s.yaw_rad[0]) == pytest.approx(-178.0, abs=1e-6)
    assert abs(np.degrees(r.actor_states(50_000_000).yaw_rad[0])) == pytest.approx(180.0, abs=1e-6)     # 最短弧跨 ±180°
    assert np.degrees(r.actor_states(25_000_000).yaw_rad[0]) == pytest.approx(179.5, abs=1e-6)
    assert s.pos_enu_m[1, 0] == 0.0 and s.state[1] == 1                                     # 1 → 3 阶跃：取第 k 帧
    assert s.state[2] == 1 and s.pos_enu_m[2, 0] == pytest.approx(1.5)                      # 线性段（速度为 0 的 Hermite 仍过端点）
    s = r.actor_states(50_000_000)
    assert s.state[2] == 0 and s.pos_enu_m[2, 0] == 0.0                                     # 0 → 1：阶跃
    s = r.actor_states(-1)
    assert (s.state == 0).all() and (s.vel_enu_mps == 0).all()
    s = r.actor_states(10_000_000_000)
    assert s.pos_enu_m[0, 0] == pytest.approx(0.4) and s.state.tolist() == [1, 1, 5] and s.vel_enu_mps[2, 0] == 0.0
    s = r.actor_states(200_000_000, [1])
    assert s.ids.tolist() == [1] and s.container.tolist() == [2]


def test_header_and_manifest_validate_against_contracts(tape) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    from referencing import Registry, Resource

    d, hd, _fr, _h = tape
    ev = ROOT / "packages/contracts/event"
    th = json.loads((ev / "tape.v1.schema.json").read_text())
    t2 = json.loads((ev / "tape.v2.schema.json").read_text())                      # 清单的 header 为 v1 或 v2
    tm = json.loads((ev / "tape_manifest.v1.schema.json").read_text())
    reg = Registry().with_resources([(s["$id"], Resource.from_contents(s)) for s in (th, t2, tm)])
    jsonschema.Draft202012Validator(th).validate(hd)
    jsonschema.Draft202012Validator(tm, registry=reg).validate(json.loads((d / "tape.json").read_text()))


def test_storage_budget_and_cli(tape, capsys: pytest.CaptureFixture[str]) -> None:
    from awr.sim.events.__main__ import main

    d, _hd, _fr, _h = tape
    assert main(["verify", str(d)]) == 0 and json.loads(capsys.readouterr().out)["checked_against_manifest"] is True
    assert main(["inspect", str(d / "tape.bin")]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["bytes_per_actor_second"] <= 25.0, info                                   # M23 §6.7.3 预算
    b = bytearray((d / "tape.bin").read_bytes())
    b[-40] ^= 0x10
    (d / "tape.bin").write_bytes(bytes(b))
    assert main(["verify", str(d)]) == 2 and json.loads(capsys.readouterr().err)["code"] == 614
