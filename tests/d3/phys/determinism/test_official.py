"""正式成绩配置两次运行逐字节对拍（D3-AC-17 前两项与背压一致性条款；ADR-162、ADR-177；P05）。

对拍工具 `tools/bench/d3/determinism_official.py`：同一 Run 规格、同一 `run_id` 与种子，按变体 a、b（两次相同运行）、m25（经 M25
会话、无注入）、bp1（M25 在途上限 1、每帧 50 ms）、bp2（bp1 + 写入器每条消息 10 ms、在途 8 条）各跑一次，比对 gz 位姿摘要、每 Agent 观测流与
动作流、逐帧摘要（platform_audit `frames.digest`）、`(uav, sensor, seq)` 集合与帧字节（交付与存档两种口径），摘要写
`determinism_official.json`。

三组用例：

- 本机（CPU 渲染替身 + kinematic 物理档 + 脚本策略，1 s 回合）：工具的运行、对拍与摘要；篡改一帧、一行观测摘要或位姿摘要时
  对拍判不一致并指出首个不同项；背压未发生时背压核对不通过；正式 Run 规格展开为 Realistic + L2（rgb1080-rs、LiDAR、IMU、GNSS）
  且各变体运行身份相同。
- dmax（d3-sensorsvc 宿主容器、绑定 GPU、挂载 docker 套接字，gz 为 d3-gz 兄弟容器）：正式配置（gz `physics` + M25 GPU 相机 + L2 +
  参考基线 track_share 含参考检测器）短回合 a、b、bp1 实跑并对拍（`AWR_D3_OFFICIAL_S`，缺省 12 s）。
- 正式产物核对（`AWR_D3_OFFICIAL_SUMMARY=<determinism_official.json>`）：完整 360 s 正式运行的摘要满足全部条款。

运行（dmax）：见 P05 记录中的 docker 命令（与 S03 Realistic 档同一宿主容器口径）。
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
D3 = HERE.parents[1]
ROOT = HERE.parents[3]
for _p in (D3 / "world", D3 / "events", ROOT / "tools" / "bench" / "d3"):
    sys.path.insert(0, str(_p))

import determinism_official as DO  # noqa: E402

FLEET_XY = ((-10.0, 50.0), (10.0, 50.0))
DURATION_S = 1.0
LOCAL_RUNS = ("a", "b", "bp1", "bp2")
FORMAL_SPEC = ROOT / "configs" / "bench.d3" / "crop-handoff-001.l2-3uav.run_spec.json"


# ---------------------------------------------------------------------------------------------------------------- 本机
@pytest.fixture(scope="module")
def synth_event(tmp_path_factory):
    pytest.importorskip("open3d")
    pytest.importorskip("zstandard")
    pytest.importorskip("mcap")
    from synthworld import build_synth_world
    from test_compile import EVENT, OVERLAY

    from awr.sim.events.compile import CompileOptions, compile_event

    tmp = tmp_path_factory.mktemp("p05ev")
    mp = pytest.MonkeyPatch()
    w = build_synth_world(tmp, mp)
    ev = copy.deepcopy(EVENT)
    ev["annotations"] = [{"id": "meet_ab", "type": "proximity_encounter", "participants": ["person_a", "person_b"], "window": [5, 25]}]
    ev["brief"] = {"public_anchors": ["ev_plaza", "ev_door"],
                   "targets": [{"actor": "person_a", "description_en": "Blue top", "description_zh": "蓝色上衣"}]}
    d = tmp / "ev"
    d.mkdir()
    (d / "anchors.json").write_text(json.dumps(OVERLAY))
    (d / "event.json").write_text(json.dumps(ev))
    # 与 tests/d3/deploy/test_backpressure.py 同一合成事件：门 3 以 lenient 编译（示意声明，夹具不核对语义）
    r = compile_event(d / "event.json", w, opts=CompileOptions(out=d / "v1", cache_dir=tmp / "cache", strict=False))
    assert r.status == "compiled", r.as_dict()
    yield w, d / "v1"
    mp.undo()


class FakeRaster:
    """EGL 光栅相机的 CPU 替身：按位姿生成确定性的渐变图（与 tests/d3/env/test_l2.py 同一口径）。"""

    def render(self, cm, R_ws, p_w, *, fx=None, rs_nodes=None, actors=None, linear=False):
        from awr.sensorsvc.camera.raster import RenderOut

        h, w = cm.height, cm.width
        v = int(abs(float(p_w[0]) * 7 + float(p_w[1]) * 3)) % 200
        rgb = np.broadcast_to((np.arange(w, dtype=np.uint16)[None, :, None] * 255 // w + v).astype(np.uint8) % 255, (h, w, 3)).copy()
        z16, z8 = np.zeros((h, w), np.uint16), np.zeros((h, w), np.uint8)
        return RenderOut(rgb, np.zeros((h, w), np.float32), z16, z8, 0.0, 0.0, 0, instance_unocc=z16)


def _l2(env):
    from awr.agentsdk.env.l2 import L2Config, L2Sensors

    return L2Sensors(world_dir=Path(env.spec.world_dir), tape=env.tape, uavs=env.uavs, seed=env.spec.seed, noise=env.spec.noise,
                     obs_level=env.spec.obs_level, nav=env.nav, mount_t_bs_m=env.cam.mount_t_bs_m, hfov_deg=env.cam.hfov_deg,
                     cfg=L2Config(camera_profile="rgb720-gs", cache_dir=Path(env.spec.world_dir).parent / "wq"), renderer=FakeRaster())


class _Scripted:
    """确定性脚本策略：goto 加云台指令，并发一条组内消息（动作流含导航、云台与通信三类条目）。"""

    def reset(self, config: dict) -> None:
        self.uav = config["uav_id"]
        self.peers = config["peers"]

    def step(self, obs: dict, inbox: list) -> dict:
        k = int(obs["k"])
        act = {"nav": {"kind": "goto", "target_enu": [0.0, 60.0, 30.0], "speed_mps": 4.0}, "gimbal": {"pitch_deg": -45.0, "yaw_deg": 10.0 + k}}
        if k % 3 == 0 and self.peers:
            act["msgs"] = [{"dst": self.peers[0], "topic": "track", "payload": f"{self.uav}:{k}".encode()}]
        return act


def _spec(synth_event, out: Path):
    from awr.agentsdk.env.spec import EnvSpec, FleetEntry

    w, ev = synth_event
    fleet = tuple(FleetEntry(f"uav-{i + 1:02d}", (x, y, 30.0), 0.0, airborne=True) for i, (x, y) in enumerate(FLEET_XY))
    return EnvSpec("t-p05", 3, w, ev, fleet, obs_level="L2", physics="kinematic", duration_s=DURATION_S, out_dir=out)


@pytest.fixture(scope="module")
def local_runs(synth_event, tmp_path_factory):
    out = tmp_path_factory.mktemp("p05")
    infos = {n: DO.run_one(_spec(synth_event, out / "runs" / n), "scripted", DO.VARIANTS[n], l2=_l2, agent_factory=lambda _u: _Scripted())
             for n in LOCAL_RUNS}
    return out, infos


def test_variants_bitwise_equal_and_summary(local_runs) -> None:
    out, infos = local_runs
    doc = DO.compare(out, LOCAL_RUNS)
    assert doc["schema"] == DO.SCHEMA and doc["acceptance"] == "D3-AC-17"
    assert doc["ok"], json.dumps({n: c for n, c in doc["comparisons"].items() if not c["ok"]}, indent=1)[:4000]
    assert set(doc["comparisons"]) == {"repeat", "bp_repeat", "bp_vs_free"}           # m25 未运行：其对拍对不出现
    n_ticks = round(DURATION_S * 10)
    for name, c in doc["comparisons"].items():
        assert c["pose_sha256"]["equal"] and c["run_hash_equal"] and c["digest_sha256_equal"] and c["n_digest_diffs"] == 0, name
        assert c["delivered_frames"]["equal"] and c["archived_frames"]["equal"], name
        assert c["delivered_frames"]["n_ref"] == 2 * (2 + 4 * n_ticks), name           # k = 0：相机与 GNSS；k ≥ 1 另有 LiDAR 与 IMU
        assert set(c["streams"]) == {"uav-01", "uav-02"} and all(x["obs"] and x["acts"] for x in c["streams"].values()), name
    for n, r in doc["runs"].items():
        assert all(r["checks"].values()), (n, r["checks"])
        assert r["streams"]["uav-01"]["n_obs"] == n_ticks + 1 and r["streams"]["uav-01"]["n_acts"] == n_ticks, n
    # 背压确实发生且只延长墙钟
    bp1, bp2, a = doc["runs"]["bp1"], doc["runs"]["bp2"], doc["runs"]["a"]
    assert a["sensorsvc"] is None and a["recorder"]["raw_blocked"] == 0
    for r in (bp1, bp2):
        sv = r["sensorsvc"]
        assert sv["inflight_cap"] == sv["max_inflight"] == 1 and sv["blocked_ticks"] == sv["ticks"] == n_ticks + 1 and sv["rejected_641"] > 0
        assert sv["wait_s"] >= 0.05 * sv["completed"] * DO.WAIT_FRAC and r["wall_s"] > a["wall_s"]
    assert bp2["recorder"]["raw_blocked"] > 0 and bp2["recorder"]["options"]["inject_delay_ms"] == DO.VARIANTS["bp2"].rec_delay_ms > 0
    assert bp2["recorder"]["options"]["queue_msgs"] == 8 and bp2["recorder"]["raw_max_depth"] <= 8
    assert bp1["recorder"]["options"]["inject_delay_ms"] == 0.0                         # 注入只作用于本变体，不泄漏到其后的运行
    # 产物：摘要、逐帧摘要副本与帧键清单
    saved = json.loads((out / DO.SUMMARY_NAME).read_text())
    assert saved["ok"] and saved["comparisons"].keys() == doc["comparisons"].keys()
    assert saved["config"]["run_id"] == "t-p05" and saved["config"]["physics"] == "kinematic" and saved["config"]["sensors"]["renderer"] == "FakeRaster"
    for n in LOCAL_RUNS:
        assert (out / "digests" / f"{n}.digest").read_text() == (out / "runs" / n / "frames.digest").read_text()
        keys = DO.read_keys(out / "runs" / n / DO.DELIVERED_KEYS)
        assert sorted(keys) == sorted(DO.read_keys(out / "runs" / n / DO.ARCHIVE_KEYS)) and len(keys) == 2 * (2 + 4 * n_ticks)
        assert {s for _u, s, _q in keys} == {"eo", "lidar0", "lidar0_imu", "gnss0"}
    # 帧字节确与交付一致：交付键的摘要即帧对象规范 JSON 的 sha256，各运行相同
    assert infos["a"]["delivered_keys_sha256"] == infos["bp2"]["delivered_keys_sha256"] == infos["b"]["delivered_keys_sha256"]
    assert infos["a"]["archive_keys_sha256"] == infos["bp2"]["archive_keys_sha256"]


def test_compare_reports_first_mismatch(local_runs, tmp_path: Path) -> None:
    src, _ = local_runs
    out = tmp_path / "t"
    for n in ("a", "b"):
        shutil.copytree(src / "runs" / n, out / "runs" / n)
    rb = out / "runs" / "b"
    # 1) 一帧字节不同（交付与存档两侧各改一帧）
    for name in (DO.DELIVERED_KEYS, DO.ARCHIVE_KEYS):
        lines = (rb / name).read_text().splitlines()
        u, s, q, h = lines[7].split(" ")
        lines[7] = " ".join((u, s, q, ("0" if h[0] != "0" else "1") + h[1:]))
        (rb / name).write_text("\n".join(lines) + "\n")
    # 2) uav-02 第 4 次观测的摘要不同（逐帧摘要中的 obs 流）
    dg = (rb / "frames.digest").read_text().splitlines()
    i = [j for j, ln in enumerate(dg) if ln.startswith("obs/uav-02 ")][3]
    s, k, h = dg[i].split(" ")
    dg[i] = f"{s} {k} {'f' if h[0] != 'f' else 'e'}{h[1:]}"
    (rb / "frames.digest").write_text("\n".join(dg) + "\n")
    doc = DO.compare(out, ("a", "b"))
    c = doc["comparisons"]["repeat"]
    assert not doc["ok"] and not c["ok"] and c["pose_sha256"]["equal"]
    assert c["delivered_frames"]["n_bytes_differ"] == 1 and c["archived_frames"]["n_bytes_differ"] == 1
    assert c["delivered_frames"]["n_missing"] == c["delivered_frames"]["n_extra"] == 0
    assert c["digest_diffs"] == [{"kind": "frame", "stream": "obs/uav-02", "index": 3, "k": int(k), "expected": h, "got": dg[i].split(" ")[2]}]
    # 3) gz 位姿摘要不同；缺一帧
    info = json.loads((rb / DO.RUN_INFO).read_text())
    info["pose_sha256"] = "0" * 64
    (rb / DO.RUN_INFO).write_text(json.dumps(info))
    lines = (rb / DO.DELIVERED_KEYS).read_text().splitlines()
    (rb / DO.DELIVERED_KEYS).write_text("\n".join(lines[:-1]) + "\n")
    doc = DO.compare(out, ("a", "b"))
    c = doc["comparisons"]["repeat"]
    assert not c["pose_sha256"]["equal"] and c["delivered_frames"]["n_missing"] == 1 and c["delivered_frames"]["missing"] == [
        [*lines[-1].split(" ")[:2], int(lines[-1].split(" ")[2])]]
    assert not doc["runs"]["b"]["checks"]["archive_equals_delivered"]                 # 存档帧集合与交付帧集合不再相同
    # 4) 运行失败：只有 error.txt 的变体记入 failed，结论不通过
    (out / "runs" / "bp1").mkdir()
    (out / "runs" / "bp1" / "error.txt").write_text("Traceback: gz instance exited (638)")
    doc = DO.compare(out, ("a", "bp1"))
    assert not doc["ok"] and "638" in doc["failed"]["bp1"] and doc["comparisons"] == {}


def test_backpressure_checks_require_injection() -> None:
    v = DO.VARIANTS["bp1"]
    base = {"variant": {**DO.asdict(v)}, "archive_keys_ok": True, "frames": {"delivered": 84}, "timeouts": {"uav-01": 0},
            "gz": {"restarts": 0}, "recorder": {"dropped": 0, "raw_blocked": 0, "writer_block_s": 0.0, "options": {"inject_delay_ms": 0.0}}}
    sv = {"mode": "benchmark", "inflight_cap": 1, "max_inflight": 1, "accepted": 84, "completed": 84, "rejected_641": 250, "ticks": 11,
          "blocked_ticks": 11, "delay_s": 0.05, "wait_s": 4.3}
    assert all(DO.run_checks({**base, "sensorsvc": sv}).values())
    c = DO.run_checks({**base, "sensorsvc": {**sv, "rejected_641": 0, "blocked_ticks": 0}})       # 无 641：背压未发生
    assert not c["m25_blocked_every_tick"]
    assert not DO.run_checks({**base, "sensorsvc": {**sv, "max_inflight": 2}})["m25_inflight_cap"]
    assert not DO.run_checks({**base, "sensorsvc": {**sv, "wait_s": 1.0}})["m25_delay_applied"]     # 等待 < 50 ms × 84 帧 × 95 %
    assert not DO.run_checks({**base, "sensorsvc": None})["m25_session"]
    c = DO.run_checks({**base, "variant": DO.asdict(DO.VARIANTS["bp2"]), "sensorsvc": sv})
    assert not c["recorder_blocked"]                                                             # 写入器注入未生效
    assert not DO.run_checks({**base, "sensorsvc": sv, "gz": {"restarts": 1}})["gz_restarts_0"]
    assert not DO.run_checks({**base, "sensorsvc": sv, "timeouts": {"uav-01": 2}})["agent_timeouts_0"]


def test_formal_run_spec_expands_to_realistic_l2(tmp_path: Path) -> None:
    from awr.agentsdk.env.l2 import L2Config

    job = {"spec": str(FORMAL_SPEC), "policy": "track_share", "seed": 1, "out": str(tmp_path / "o"), "worlds_root": str(tmp_path / "w"),
           "events_root": str(tmp_path / "e"), "gz": {"launcher": "docker", "image": "d3-gz:dev"},
           "images": {"gz": "sha256:" + "1" * 64, "sensorsvc": "sha256:" + "2" * 64}}
    specs = {n: DO._spec_for(job, n) for n in DO.DEFAULT_RUNS}
    s = specs["a"]
    assert s.physics == "realistic" and s.obs_level == "L2" and s.ranked and len(s.fleet) == 3 and s.duration_s == 360
    cfg = L2Config(**{k: v for k, v in (s.l2 or {}).items() if k in ("lidar", "imu", "gnss")})
    assert L2Config().camera_profile == "rgb1080-rs" and cfg.lidar and cfg.imu and cfg.gnss       # 正式相机档与全部 L2 传感器
    assert s.policy == {"kind": "reference", "name": "track_share"} and s.seed_commit
    assert s.versions["image.gz"] == "sha256:" + "1" * 64 and s.versions["image.sensorsvc"] == "sha256:" + "2" * 64
    # 各变体：同一 run_id、种子与运行身份（背压只改墙钟）；正式档是统一引擎，不建 gz 实例（ADR-226）
    assert len({x.run_id for x in specs.values()}) == 1 and len({x.run_hash() for x in specs.values()}) == 1
    assert all("name" not in x.gz and "run_dir" not in x.gz for x in specs.values())
    gzs = {n: DO._spec_for({**job, "physics": "gz"}, n) for n in DO.DEFAULT_RUNS}                  # gz 交叉验证档：实例名与运行目录各自独立
    assert all(x.physics == "gz" for x in gzs.values())
    assert len({x.gz["name"] for x in gzs.values()}) == len(gzs) and len({x.gz["run_dir"] for x in gzs.values()}) == len(gzs)
    assert {x.gz["export_root"] for x in gzs.values()} == {str(tmp_path / "o" / "export")}
    assert DO._spec_for({**job, "duration_s": 12}, "a").duration_s == 12
    assert DO.main(["run", "--events-root", str(tmp_path), "--out", str(tmp_path / "x"), "--runs", "a,zz"]) == 2


# ---------------------------------------------------------------------------------------------------------------- dmax：正式配置
def _formal_env() -> tuple[Path, Path]:
    for mod in ("moderngl", "open3d", "zstandard", "mcap", "PIL"):
        pytest.importorskip(mod)
    if shutil.which("docker") is None or not Path("/var/run/docker.sock").exists():
        pytest.skip("[platform] the formal configuration needs the host docker socket to start d3-gz sibling containers")
    from awr.phys.gz.instance import host_path

    if host_path(Path.cwd()) == str(Path.cwd().resolve()):
        pytest.skip("[platform] run under AWR_HOST_PATH_MAP so gz sibling containers can mount the run directories")
    worlds = Path(os.environ.get("AWR_WORLDS_DIR") or ROOT / "worlds")
    events = Path(os.environ.get("AWR_D3_EVENTS_ROOT") or ROOT / "d3data" / "events")
    if not (worlds / "oryahovo-crop1" / "visual/mesh/appearance/index.json").is_file():
        pytest.skip("[data] needs worlds/oryahovo-crop1 (make worlds)")
    if not (events / "crop-handoff-001" / "v1" / "tape.bin").is_file():
        pytest.skip("[data] needs the compiled event crop-handoff-001/v1 (AWR_D3_EVENTS_ROOT)")
    try:
        import moderngl

        moderngl.create_standalone_context(backend="egl").release()
    except Exception as e:  # pragma: no cover - 只在无 EGL 的环境
        pytest.skip(f"[gpu] EGL context unavailable: {e}")
    return worlds, events


@pytest.mark.slow
def test_official_config_gpu_short_episode(tmp_path_factory) -> None:
    """正式配置短回合（缺省 12 s）a、b、bp1：gz 位姿、观测与动作流、帧集合与字节逐一一致，背压确实发生。"""
    worlds, events = _formal_env()
    dur = float(os.environ.get("AWR_D3_OFFICIAL_S", "12"))
    out = Path(os.environ.get("AWR_D3_OFFICIAL_OUT") or tmp_path_factory.mktemp("p05-gpu"))
    cmd = [sys.executable, str(ROOT / "tools/bench/d3/determinism_official.py"), "run", "--spec", str(FORMAL_SPEC), "--worlds-root", str(worlds),
           "--events-root", str(events), "--out", str(out), "--duration-s", f"{dur:g}", "--runs", "a,b,bp1", "--jobs", "3",
           "--gz-image", os.environ.get("AWR_D3_GZ_IMAGE", "d3-gz:dev"), "--gz-prefix", f"d3-gz-p05t-{os.getpid()}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
    print(r.stdout[-4000:], r.stderr[-4000:])
    doc = json.loads((out / DO.SUMMARY_NAME).read_text())
    assert r.returncode == 0 and doc["ok"], json.dumps({"failed": doc["failed"], "cmp": doc["comparisons"]}, indent=1)[:6000]
    assert doc["config"]["physics"] == "realistic" and doc["config"]["tier"] == "l2" and doc["config"]["obs_level"] == "L2"   # ADR-226
    assert doc["config"]["sensors"]["renderer"] == "RasterCamera" and "NVIDIA" in str(doc["config"]["sensors"]["gl"])  # M25 GPU 相机（EGL）
    assert set(doc["comparisons"]) == {"repeat", "bp_vs_free"}
    n = round(dur * 10)
    for c in doc["comparisons"].values():
        assert c["delivered_frames"]["n_ref"] == 3 * (2 + 4 * n)                          # 3 机：k = 0 相机与 GNSS，k ≥ 1 四种传感器
    assert doc["runs"]["bp1"]["sensorsvc"]["blocked_ticks"] == n + 1


def test_official_summary_artifact() -> None:
    """完整正式运行（360 s，变体 a、b、m25、bp1、bp2）的摘要产物满足 D3-AC-17 的对拍与背压条款。"""
    p = os.environ.get("AWR_D3_OFFICIAL_SUMMARY")
    if not p:
        pytest.skip("[optin] set AWR_D3_OFFICIAL_SUMMARY=<determinism_official.json> of the formal 360 s run")
    doc = json.loads(Path(p).read_text())
    assert doc["schema"] == DO.SCHEMA and doc["ok"] and not doc["failed"]
    cfg = doc["config"]
    assert cfg["physics"] == "realistic" and cfg["tier"] == "l2" and cfg["obs_level"] == "L2" and cfg["duration_s"] == 360 and cfg["n_ticks"] == 3600
    assert cfg["policy"] in ("static_guard", "coverage_scan", "track_share") and {"image.gz", "image.sensorsvc"} <= set(cfg["versions"])
    assert cfg["sensors"]["renderer"] == "RasterCamera" and "NVIDIA" in str(cfg["sensors"]["gl"])
    assert set(doc["comparisons"]) == {p[0] for p in DO.PAIRS} and all(c["ok"] for c in doc["comparisons"].values())
    assert all(g["name"] for g in doc["platform"]["gpu"]) and doc["platform"]["cpu"]["isa_level"]
    for n, r in doc["runs"].items():
        assert all(r["checks"].values()), (n, r["checks"])
        assert r["frames"]["delivered"] == 3 * (2 + 4 * 3600), n
