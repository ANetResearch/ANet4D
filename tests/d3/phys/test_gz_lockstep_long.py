"""Realistic 档 3 机 360 s 锁步验收（D3-AC-34、D3-AC-47 的 gz 锁步项；tools/bench/d3/gz_lockstep.py）。

本机部分：`p600_bench` → d4d_quad 映射（质量、惯量、最大转速）与派生模型、run.sdf 改写；机动脚本只由 tick 决定；帧窗口终点与
`awr.sensorsvc.timing` 互逆；探针物理档对 StepDone 网格、tick 边界与定高误差的核对（以 PoseHistoryBuffer 驱动的替身后端）；
资源采样、`nvidia-smi` 快照解析与独占/共享标注；汇总判定（3 次取中位、同种子与有无 M25 负载的逐位一致、共享卡性能条款不计入）。

dmax 部分（d3-gz 容器内，`launcher = local`）：`test_three_uav_lockstep_env` 以 BenchEnv 驱动 3 架 p600_bench 映射的 x500 运行
`AWR_GZ_LOCKSTEP_S`（缺省 360）s，两次同种子运行逐位一致、每 tick 25 步、定高误差 ≤ 2 m、tick 差为 0、实例不挂 GPU、CPU ≤ 2 核、
内存 ≤ 2 GB；`test_report_acceptance` 校验 `AWR_GZ_LOCKSTEP_REPORT` 指向的完整报告（含 M25 负载 Run）。
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("gz_lockstep", ROOT / "tools" / "bench" / "d3" / "gz_lockstep.py")
gl = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("gz_lockstep", gl)
_spec.loader.exec_module(gl)

G = 9.80665
MODEL_SDF = ROOT / "native" / "gz" / "models" / "x500_d4d" / "model.sdf"


# ---------------------------------------------------------------------------------------------------------------- 机型映射
def test_p600_bench_params_follow_catalog_rule() -> None:
    import yaml

    q = gl.p600_bench_params()
    assert q.mass_kg == 3.935 and q.omega_max_rad_s == 1500.0 and q.k_f == 8.54858e-06
    # 同一缩放规则对 p600_mid360（3.825 kg）复现目录值 0.0598 / 0.1102（vehicles/p600/params.yaml）
    p600 = yaml.safe_load((ROOT / "vehicles/p600/params.yaml").read_text(encoding="utf-8"))
    s = (p600["mass_kg"]["value"] / 2.064) * (0.300 / 0.246) ** 2
    j = p600["inertia_kgm2"]["value"]
    assert abs(0.0217 * s - j["xx"]) < 1e-4 and abs(0.040 * s - j["zz"]) < 1e-4
    s_b = (3.935 / 2.064) * (0.300 / 0.246) ** 2
    assert q.inertia_kgm2 == pytest.approx((0.0217 * s_b, 0.0217 * s_b, 0.040 * s_b), abs=1e-6)
    assert q.thrust_to_weight == pytest.approx(1.99, abs=0.01)                    # AWR-20 §15.16.1 VH-1：推重比 1.99
    assert 4 * q.k_f * 1062.0 ** 2 == pytest.approx(q.mass_kg * G, rel=0.01)       # 悬停转速约 1062 rad/s，低于上限


def _links(root: ET.Element) -> dict[str, float]:
    return {ln.get("name"): float(ln.find("inertial/mass").text) for ln in root.iter("link") if ln.find("inertial/mass") is not None}


def test_derived_model_changes_only_base_inertial() -> None:
    base = MODEL_SDF.read_text(encoding="utf-8")
    q = gl.p600_bench_params()
    out = gl.derive_model_sdf(base, q)
    r0, r1 = ET.fromstring(base.encode()), ET.fromstring(out.encode())
    assert r1.find("model").get("name") == q.model
    m0, m1 = _links(r0), _links(r1)
    assert set(m0) == set(m1) and all(m0[k] == m1[k] for k in m0 if k != "base_link")
    assert sum(m1.values()) == pytest.approx(q.mass_kg, abs=1e-9)
    assert sum(m0.values()) == pytest.approx(gl.x500_params().mass_kg, abs=1e-9)
    bl = next(ln for ln in r1.iter("link") if ln.get("name") == "base_link")
    ii = {k: float(bl.find(f"inertial/inertia/{k}").text) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
    assert (ii["ixx"], ii["iyy"], ii["izz"]) == q.inertia_kgm2 and ii["ixy"] == ii["ixz"] == ii["iyz"] == 0.0
    # 碰撞、关节与桨叶不变
    for tag in ("collision", "joint"):
        assert [e.get("name") for e in r0.iter(tag)] == [e.get("name") for e in r1.iter(tag)]
    with pytest.raises(ValueError):
        gl.derive_model_sdf(base.replace('<link name="base_link">', '<link name="body">'), q)


def test_apply_vehicle_rewrites_includes_and_motor_limit(tmp_path: Path) -> None:
    from awr.phys.gz.compose import UavSpawn, compose_run

    spawns = [UavSpawn(f"uav-0{i}", (0.0, 4.0 * i, 0.3)) for i in range(1, 4)]
    sdf = compose_run(spawns, ground_plane=True)
    q = gl.p600_bench_params()
    out = gl.apply_vehicle(sdf, q, 3)
    assert out.count(f"<uri>model://{q.model}</uri>") == 3 and "model://x500_d4d<" not in out
    assert out.count("<maxRotVelocity>1500.0</maxRotVelocity>") == 12
    assert gl.sdf_gravity(out) == [0.0, 0.0, -G]
    ET.fromstring(out.encode())                                                    # 仍是合法 XML
    assert gl.apply_vehicle(sdf, gl.x500_params(), 3) == sdf                       # x500 原型不改写
    with pytest.raises(ValueError):
        gl.apply_vehicle(sdf, q, 2)
    d = gl.write_model(tmp_path, q, MODEL_SDF.read_text(encoding="utf-8"))
    assert (d / "model.sdf").is_file() and ET.fromstring((d / "model.config").read_bytes()).findtext("name") == q.model
    assert gl.write_model(tmp_path, gl.x500_params(), "") is None


# ---------------------------------------------------------------------------------------------------------------- 机动脚本
def _obs(k: int, pos=(76.0, -82.0, 31.3), **extra) -> dict:
    return {"k": k, "t_ns": k * gl.TICK_NS, "nav": {"pos_enu_m": list(pos)}, **extra}


def test_maneuver_policy_depends_only_on_tick() -> None:
    pytest.importorskip("jsonschema")
    from awr.agentsdk.schemas import bench_validator

    uavs = ["uav-01", "uav-02", "uav-03"]
    plan = gl.Maneuver()
    runs = []
    for noise in (0.0, 5.0):                                                       # 观测内容不同（k > 0 的位置、传感器），动作相同
        pol = gl.ManeuverPolicy(uavs, (76.0, -82.0), plan)
        seq = []
        for k in range(3600):
            for i, u in enumerate(uavs):
                pos = (76.0, -78.0 - 4 * i, 31.3) if k == 0 else (noise * k, -noise, 50.0 + noise)
                seq.append((u, pol.step(u, _obs(k, pos, sensors={"x": [noise]}))))
        runs.append(seq)
    assert runs[0] == runs[1]
    v = bench_validator("act")
    navs = [(u, a) for u, a in runs[0] if "nav" in a]
    assert all(not list(v.iter_errors(a)) for _u, a in runs[0][:30] + navs)
    k_legs = plan.leg_ticks()
    assert [a["k"] for u, a in navs if u == "uav-01"] == [0, *k_legs] and len(k_legs) == plan.n_legs + 1
    for i, u in enumerate(uavs):
        mine = [a["nav"] for uu, a in navs if uu == u]
        assert mine[0] == {"mode": "takeoff", "alt_agl_m": plan.alt_agl_m[i]}
        z = {n["pos_enu_m"][2] for n in mine[1:]}
        assert z == {31.3 + plan.alt_agl_m[i]}                                     # 航段保持起飞高度
        xy = [tuple(n["pos_enu_m"][:2]) for n in mine[1:-1]]
        assert all(abs(abs(x - 76.0) - 80.0) < 1e-9 and abs(abs(y + 82.0) - 80.0) < 1e-9 for x, y in xy)
        assert tuple(mine[-1]["pos_enu_m"][:2]) == (76.0, -78.0 - 4 * i) and mine[-1]["speed_mps"] == plan.return_speed
        assert max(n["speed_mps"] for n in mine[1:]) == 12.0
    # 同一时刻三机目标顶点互不相同（各机起始顶点错开）
    first = [tuple(a["nav"]["pos_enu_m"][:2]) for _u, a in navs if a["k"] == k_legs[0]]
    assert len(set(first)) == 3
    assert plan.alt_from_ns == round((plan.legs_from_s + plan.settle_s) * 10) * gl.TICK_NS
    assert plan.end_s() < 360.0
    with pytest.raises(ValueError):
        gl.ManeuverPolicy([*uavs, "uav-04"], (0.0, 0.0), plan)


# ---------------------------------------------------------------------------------------------------------------- 帧时间戳
def test_frame_window_end_inverts_sensor_timing() -> None:
    from awr.sensorsvc.timing import t_mid_ns

    k = 37
    t = k * gl.TICK_NS
    cam = {"seq": k, "t_mid_ns": t_mid_ns(t, 4_000_000, 20_000_000), "exposure_ns": 4_000_000, "readout_ns": 20_000_000}
    cam_odd = {"seq": k, "t_mid_ns": t_mid_ns(t, 3_333_333, 0), "exposure_ns": 3_333_333, "readout_ns": 0}
    lidar = {"seq": k - 1, "t_start_ns": (k - 1) * gl.TICK_NS}
    imu = {"seq": k - 1, "t0_ns": (k - 1) * gl.TICK_NS, "dt_ns": 5_000_000, "n": 20}
    gnss = {"seq": k, "t_ns": t}
    for sid, f in (("cam0", cam), ("cam0", cam_odd), ("lidar0", lidar), ("lidar0_imu", imu), ("gnss0", gnss)):
        assert gl.frame_window_end_ns(sid, f) == t, sid
    assert gl.frame_window_end_ns("x", {"seq": 1}) is None

    fc = gl.FrameCheck()
    fc.on_obs("uav-01", {"t_ns": 0, "sensors": {"cam0": [{"seq": 0, "t_mid_ns": 0, "exposure_ns": 4_000_000, "readout_ns": 0}]}})
    fc.on_obs("uav-01", {"t_ns": t, "sensors": {"cam0": [cam], "lidar0": [lidar], "lidar0_imu": [imu], "gnss0": [gnss], "detections": [{}]}})
    d = fc.as_dict()
    assert d["clamped_t0"] == 1 and d["frames"] == {"cam0": 1, "lidar0": 1, "lidar0_imu": 1, "gnss0": 1}
    assert set(d["max_abs_diff_ns"].values()) == {0} and not d["mismatches"]
    fc.on_obs("uav-02", {"t_ns": t + gl.TICK_NS, "sensors": {"gnss0": [gnss]}})                  # 晚交付一个 tick
    assert fc.as_dict()["max_abs_diff_ns"]["gnss0"] == gl.TICK_NS and fc.as_dict()["mismatches"][0]["uav"] == "uav-02"


# ---------------------------------------------------------------------------------------------------------------- 探针物理档
@dataclass
class _Tm:
    encode: int = 1_000
    physics: int = 2_000_000
    decode: int = 3_000
    append: int = 4_000


@dataclass
class _Res:
    k: int
    t_start_ns: int
    t_end_ns: int
    n_steps: int
    contacts: tuple
    timing: _Tm


class FakeBackend:
    """GzSimBackend 替身：按 tracker 的 goto 竖直分量一阶收敛写入 PoseHistoryBuffer（每 tick n_steps 个 4 ms 样本）。"""

    name = "gz"

    def __init__(self, uavs, n_steps: int = 25, z0: float = 30.0) -> None:
        from awr.phys.pose_history import PoseHistoryBuffer

        self.uavs = tuple(uavs)
        self.step_ns = gl.STEP_NS
        self.n_steps = n_steps
        self.history = PoseHistoryBuffer(self.uavs, hz=250, capacity_s=2.0)
        self.z = dict.fromkeys(self.uavs, z0)
        self.target = dict.fromkeys(self.uavs, z0)
        self.t = 0
        self.last_control_sha256 = b"\x00" * 32
        self.pending: list = []
        for u in self.uavs:
            self.history.append(u, 0, self._row(u))

    def _row(self, u):
        r = np.zeros(16)
        r[2], r[6], r[7] = self.z[u], 1.0, 9.0
        return r

    def open(self):
        return {"t0_gz_ns": 0}

    def submit(self, cmds):
        self.pending += list(cmds)
        return [True] * len(cmds)

    def advance(self, k):
        for c in [c for c in self.pending if c.apply_tick <= k]:
            if c.kind == "goto":
                self.target[c.uav] = c.target_enu_m[2]
        self.pending = [c for c in self.pending if c.apply_tick > k]
        t0 = self.t
        for j in range(1, self.n_steps + 1):
            for u in self.uavs:
                self.z[u] += 0.2 * (self.target[u] - self.z[u])
                self.history.append(u, t0 + j * gl.STEP_NS, self._row(u))
        self.t = t0 + self.n_steps * gl.STEP_NS
        self.history.release_before(self.t - 1_000_000_000)
        return _Res(k, t0, self.t, self.n_steps, (), _Tm())

    def close(self):
        return {"pose_sha256": "x"}


class _Inst:
    launcher = "local"
    stopped = False

    def stop(self):
        self.stopped = True


def _probe(be, alt_from_ns=0):
    return gl.ProbeTier(be, _Inst(), t_launch=0.0, t_socket=0.5, alt_from_ns=alt_from_ns, sampler=None,
                        sdf="<gravity>0 0 -9.80665</gravity>")


def test_probe_checks_grid_tick_boundary_and_altitude() -> None:
    from awr.phys.adapter import TierCommand

    tier = _probe(FakeBackend(["uav-01", "uav-02"]), alt_from_ns=20 * gl.TICK_NS)
    tier.open()
    tier.submit([TierCommand("uav-01", "goto", 5, target_enu_m=(0.0, 0.0, 31.0)), TierCommand("uav-02", "goto", 5, target_enu_m=(0.0, 0.0, 30.0))])
    for k in range(40):
        tier.advance(k)
    rep = tier.report(4.0)
    assert rep["ticks"] == 40 and not rep["bad_steps"] and rep["tick_diff_max_ns"] == 0
    assert rep["pose_history_hz"] == {"declared": 250, "step_hz": 250, "measured": {"uav-01": 250.0, "uav-02": 250.0}}
    assert rep["samples"] == {"uav-01": 1000, "uav-02": 1000}
    assert rep["alt_err"]["uav-01"]["n"] == 20 * 25 and rep["alt_err_max_m"] < 1e-6             # 窗口在收敛之后
    assert rep["rtf_phys"] == pytest.approx(4.0 / (40 * (1_000 + 2_000_000 + 3_000 + 4_000) / 1e9), rel=1e-3)
    assert rep["gravity"] == [0.0, 0.0, -G] and rep["vh_max_mps"]["uav-01"] == 9.0
    assert rep["startup"]["launch_to_socket_s"] == 0.5
    # 窗口提前到指令生效前：未收敛的样本计入误差
    tier2 = _probe(FakeBackend(["uav-01"]), alt_from_ns=0)
    tier2.submit([TierCommand("uav-01", "goto", 0, target_enu_m=(0.0, 0.0, 33.0))])
    tier2.advance(0)
    assert tier2.report(0.1)["alt_err_max_m"] == pytest.approx(3.0 * 0.8, abs=1e-9)
    assert tier2.close()["resources"] == {} and tier2.inst.stopped


def test_probe_flags_short_ticks() -> None:
    tier = _probe(FakeBackend(["uav-01"], n_steps=24))
    tier.advance(0)
    rep = tier.report(0.1)
    assert rep["bad_steps"] == [{"tick": 0, "uav": "uav-01", "n_steps": 24, "n_samples": 24}]
    assert rep["tick_diff_max_ns"] == gl.STEP_NS


class _PreBackend(FakeBackend):
    """FakeBackend + run_preroll 所需的 status()（地面静止，全部 GROUNDED）。"""

    def status(self):
        from awr.phys.adapter import VehicleStatus

        return dict.fromkeys(self.uavs, VehicleStatus.GROUNDED)


def test_probe_counts_preroll_ticks() -> None:
    """探针在 Env 预滚视图之下：推进 tick 数 = n_ticks + 预滚 tick，样本率与 RTF 按实际推进时长折算，定高窗口按预滚平移（S21 遗留 3）。"""
    from awr.agentsdk.env.preroll import plan_preroll, run_preroll
    from awr.phys.adapter import TierCommand

    plan = plan_preroll([("uav-01", False)], 2.0, 100)
    tier = _probe(_PreBackend(["uav-01"]), alt_from_ns=10 * gl.TICK_NS)
    tier.pre_ticks = plan.n_ticks
    tier.open()
    ev = run_preroll(tier, plan)
    ev.submit([TierCommand("uav-01", "goto", 0, target_enu_m=(0.0, 0.0, 31.0))])
    n = 40
    for k in range(n):
        ev.advance(k)
    rep = tier.report(n * gl.TICK_NS / 1e9)
    assert rep["ticks"] == n + 20 and rep["preroll_ticks"] == 20 and rep["sim_advanced_s"] == pytest.approx(6.0)
    assert rep["pose_history_hz"]["measured"] == {"uav-01": 250.0}                    # 旧口径：1500 / 4.0 s = 375
    assert rep["rtf_phys"] == pytest.approx(6.0 / (60 * 2_008_000 / 1e9), rel=1e-6)
    assert rep["alt_err"]["uav-01"]["n"] == (n - 10) * 25                              # 事件时间 1.0 s 起（档内 3.0 s）
    doc = _run_doc(ticks=n + 20, n_ticks=n, preroll_ticks=20)
    assert gl.summarize([doc])["checks"]["step_done_25"]["result"]
    assert not gl.summarize([_run_doc(ticks=n + 20, n_ticks=n)])["checks"]["step_done_25"]["result"]   # 旧记录无预滚计数


def test_probe_hello_once_and_startup_breakdown() -> None:
    be = FakeBackend(["uav-01"])
    calls = []
    be.open = lambda: calls.append(1) or {"t0_gz_ns": 0}
    inst = _Inst()
    inst.launch = {"cmd_s": 0.2, "socket_s": 0.9, "log_check_s": 0.03}
    tier = gl.ProbeTier(be, inst, t_launch=time.monotonic() - 1.0, t_socket=time.monotonic() - 0.1, alt_from_ns=0, sampler=None, sdf="")
    tier.hello()
    tier.open()
    assert calls == [1]                                                              # make_tier 已发送 Hello，reset 时不重发
    st = tier.startup
    assert st["launch_cmd_s"] == 0.2 and st["socket_seen_s"] == 0.9 and st["log_check_s"] == 0.03
    assert 0.9 <= st["launch_to_socket_s"] <= 0.95 and st["host_gap_s"] >= 0.09 and st["startup_s"] >= st["launch_to_socket_s"]


def test_startup_statistics_and_trace_parsing() -> None:
    samples = [{"index": i, "startup_s": 2.6, "hello_s": 1.6, "launch_to_socket_s": 1.0} for i in range(19)]
    s = gl.summarize_startup([*samples, {"index": 19, "startup_s": 5.2, "hello_s": 1.6, "launch_to_socket_s": 3.6}])
    assert s["n"] == 20 and s["over_threshold"] == [19] and s["worst_pass"] is False and s["p95_pass"] is True
    assert s["stats"]["startup_s"]["max"] == 5.2 and s["stats"]["startup_s"]["median"] == 2.6
    assert s["stats"]["startup_s"]["p95"] == pytest.approx(2.6 + 0.05 * 2.6, abs=1e-3)
    assert s["statistic"] == "p95" and s["samples_sufficient"] is False and s["pass"] is False          # 20 < 40 次：样本不足
    assert s["worst"] == {"source": "startup-19", "startup_s": 5.2, "segments": {"launch_to_socket_s": 3.6, "hello_s": 1.6},
                          "utc": None, "loadavg": None}
    ok = gl.summarize_startup(samples)
    assert ok["worst_pass"] and ok["p95_pass"] and not ok["over_threshold"]
    assert gl.summarize_startup([])["worst_pass"] is None and gl.summarize_startup([])["pass"] is None
    # ADR-216 (2)：≥ 40 次冷启动按 p95 判定；单个 6.2 s 长尾（最差实例，必报）不使条款不通过
    k04 = [{"index": i, "startup_s": 2.6} for i in range(41)] + [{"index": 41, "startup_s": 6.2, "launch_cmd_s": 4.1}]
    s40 = gl.summarize_startup(k04)
    assert s40["samples_sufficient"] and s40["p95_pass"] and s40["pass"] is True and s40["worst_pass"] is False
    assert s40["worst"]["source"] == "startup-41" and s40["worst"]["segments"] == {"launch_cmd_s": 4.1}
    slow = gl.summarize_startup([{"index": i, "startup_s": 5.5 if i % 10 == 0 else 2.6} for i in range(40)])
    assert slow["p95_pass"] is False and slow["pass"] is False

    t = gl.parse_rfc3339("2026-10-20T05:02:52.123456789Z")
    assert t == pytest.approx(1792472572.123456, abs=1e-6)
    assert gl.parse_rfc3339("2026-10-20T13:02:52+08:00") == pytest.approx(1792472572.0)
    assert gl.parse_rfc3339("0001-01-01T00:00:00Z") is None and gl.parse_rfc3339("bad") is None
    lines = ["2026-10-20T05:02:52.500000000Z [Msg] Gazebo Sim Server v8.15.0",
             "2026-10-20T05:02:52.900000000Z [Msg] [Drone4DClock] 3 vehicles, step 4000000 ns",
             "no timestamp", "2026-10-20T05:02:54.100000000Z [Msg] Found no publishers on /stats"]
    lo = gl.log_offsets(lines, 1792472572.0)
    assert lo == {"lines": 3, "first_s": 0.5, "last_s": 2.1, "clock_bound_s": 0.9}
    assert gl.startup_trace(_Inst()) == {}                                           # local 实例：无容器时刻


def test_summarize_startup_samples_and_cpu_pool() -> None:
    """实例启动条款按 p95 判定（ADR-216 (2)），并入 startup 子命令样本并报告最差实例；冷启动 < 40 次时样本不足、不计入；
    只有 physics 且未分配 GPU → cpu_pool，计为正式。"""
    runs = [_run_doc(idx=i, startup=s, gpu_assigned=None, gpu_snapshots=[{"available": False}]) for i, s in enumerate((2.6, 2.7, 2.8))]
    s = gl.summarize(runs)
    assert s["gpu_label"] == "cpu_pool" and _res(s)["rtf_phys"] == "pass"
    assert _res(s)["startup_s"] == "pass_insufficient_samples_not_counted" and not s["seal_ready"]
    probe = [{"index": i, "startup_s": 2.6, "hello_s": 1.5} for i in range(39)] + [{"index": 39, "startup_s": 6.2, "launch_cmd_s": 4.0}]
    s2 = gl.summarize(runs, startup=probe)
    a = next(x for x in s2["acceptance"] if x["clause"] == "startup_s")
    assert a["result"] == "pass" and a["measured"] == a["p95"] and a["p95"] < 5.0 and a["n"] == {"runs": 3, "probe": 40}
    assert a["statistic"].startswith("p95") and a["worst_pass"] is False and a["samples_sufficient"]
    assert a["worst"]["source"] == "startup-39" and a["worst"]["startup_s"] == 6.2 and a["worst"]["segments"] == {"launch_cmd_s": 4.0}
    assert s2["perf"]["startup_s_max"] == 6.2 and s2["perf"]["startup_s"] == 2.6
    assert s2["perf"]["startup_segments"]["hello_s"]["median"] == 1.5
    slow = gl.summarize(runs, startup=[{"index": i, "startup_s": 5.6 if i < 4 else 2.6} for i in range(40)])
    assert _res(slow)["startup_s"] == "fail" and not slow["passed"] and not slow["seal_ready"]
    excl = [*[_run_doc(idx=i) for i in range(3)], _run_doc("m25")]
    full = gl.summarize(excl, startup=probe)
    assert full["gpu_label"] == "exclusive" and full["passed"] and full["seal_ready"]
    few = gl.summarize(excl)                                                                      # 无冷启动样本：通过但不计入封板
    assert few["passed"] and not few["seal_ready"] and _res(few)["startup_s"] == "pass_insufficient_samples_not_counted"
    runs[1]["startup"]["startup_s"] = 7.0                                                         # Run 实例为最差实例
    w = next(x for x in gl.summarize(runs, startup=probe)["acceptance"] if x["clause"] == "startup_s")["worst"]
    assert w["source"] == "physics-1" and w["startup_s"] == 7.0
    old = gl.summarize([_run_doc(idx=i, snaps_other=None) for i in range(3)])                    # 旧记录（无 gpu_assigned）
    assert old["gpu_label"] == "shared" and _res(old)["startup_s"] == "pass_shared_not_counted"
    mixed = gl.summarize([*runs, _run_doc("m25", gpu_assigned=2)])
    assert mixed["gpu_label"] != "cpu_pool"


def test_cpu_pool_state(tmp_path: Path) -> None:
    root = tmp_path / "cg"
    (root / "d3.slice" / "d3-eval.slice").mkdir(parents=True)
    (root / "d3.slice" / "cgroup.procs").write_text("")
    (root / "d3.slice" / "d3-eval.slice" / "cgroup.procs").write_text("11\n12\n")
    st = gl.cpu_pool_state(str(root), "d3-K04", docker=str(tmp_path / "no-docker"))
    assert st == {"d3_slice_procs": 2}
    assert gl.cpu_pool_state(None, "x", docker=str(tmp_path / "no-docker")) == {}


def test_cli_startup_args() -> None:
    a = gl._args(["startup", "--events-root", "ev", "--out", "o"])
    assert a.cmd == "startup" and a.n == 40 and a.gz_verbose == 1 and a.gz_cpus == 2.0 and a.host_cgroup_root is None
    r = gl._args(["run", "--events-root", "ev", "--out", "o", "--host-cgroup-root", "/h"])
    assert r.repeats == 3 and r.host_cgroup_root == "/h" and r.kinds == "physics,m25"


# ---------------------------------------------------------------------------------------------------------------- 资源与 GPU
def test_resource_sampling() -> None:
    s = gl.parse_cgroup("usage_usec 2500000\nuser_usec 1\n", "1073741824\n", "anon 536870912\nfile 1048576\nshmem 0\n")
    assert s == {"cpu_s": 2.5, "mem_current_b": 2**30, "mem_anon_b": 2**29, "mem_file_b": 2**20}
    samples = [(float(i), {"cpu_s": 1.5 * i, "mem_anon_b": 2**28 * (1 + i % 2)}) for i in range(11)]
    out = gl.summarize_samples(samples)
    assert out["cpu_cores"] == {"p50": 1.5, "p95": 1.5, "max": 1.5, "mean": 1.5} and out["mem_anon_gb"]["max"] == 0.5
    me = gl.read_proc_group(os.getpgid(0))
    assert me["cpu_s"] > 0 and me["mem_anon_b"] > 0
    calls = iter([{"cpu_s": 0.0}, {"cpu_s": 1.0}])
    rs = gl.ResourceSampler(lambda: next(calls), period_s=60.0)
    rs.sample()
    assert rs.stop()["n"] == 2 and rs.errors == 0
    bad = gl.ResourceSampler(lambda: 1 / 0)
    bad.sample()
    assert bad.errors == 1


def test_gpu_snapshot_parsing_and_label() -> None:
    gpus = "0, GPU-aaa\n1, GPU-bbb\n2, GPU-ccc\n"
    apps = "GPU-ccc, 2026153, 31826 MiB\nGPU-aaa, 12, 10 MiB\n"
    s = gl.parse_compute_apps(gpus, apps, 2)
    assert s["gpu_uuid"] == "GPU-ccc" and s["n_other"] == 1 and s["compute_apps"][0]["pid"] == "2026153"
    one = gl.parse_compute_apps("0, GPU-ccc\n", "", 2)                                         # 容器内只见分配的卡
    assert one["gpu_uuid"] == "GPU-ccc" and one["n_other"] == 0
    assert gl.gpu_label([{"available": True, **one}, {"available": True, **one}]) == "exclusive"
    assert gl.gpu_label([{"available": True, **one}, {"available": True, **s}]) == "shared"
    assert gl.gpu_label([{"available": False}]) == "shared" and gl.gpu_label([]) == "shared"


# ---------------------------------------------------------------------------------------------------------------- 汇总判定
def _run_doc(kind="physics", idx=0, pose="p" * 64, rtf=4.0, alt=0.4, startup=2.0, snaps_other=0, **kw) -> dict:
    snap = {"available": True, "n_other": snaps_other}
    d = {"kind": kind, "index": idx, "seed": 1, "pose_sha256": pose, "control_sha256": "c", "ticks": 3600, "n_ticks": 3600, "bad_steps": [],
         "pose_history_hz": {"declared": 250, "measured": {"uav-01": 250.0}}, "gravity": [0.0, 0.0, -G], "rtf_phys": rtf, "gz_rtf_mean": 5.0,
         "rtf_env": 2.0, "startup": {"startup_s": startup}, "resources": {"cpu_cores": {"p95": 1.2, "max": 1.6}, "mem_anon_gb": {"max": 0.4}},
         "alt_err_max_m": alt, "tick_diff_max_ns": 0, "gpu": {"ok": True}, "vh_max_mps": {"uav-01": 12.1}, "contacts": ["uav-01:LIFTOFF"],
         "gpu_snapshots": [snap, snap], "obs_stream_sha256": {"uav-01": "o"}}
    if kind == "m25":
        d["frames"] = {"frames": {"cam0": 10800}, "max_abs_diff_ns": {"cam0": 0, "lidar0": 0}}
    d.update(kw)
    return d


def _res(summary) -> dict[str, str]:
    return {a["clause"]: a["result"] for a in summary["acceptance"]}


PROBES = [{"index": i, "startup_s": 2.5} for i in range(40)]


def test_summarize_exclusive_pass_and_medians() -> None:
    runs = [_run_doc(idx=i, rtf=r, startup=s) for i, (r, s) in enumerate(((3.5, 2.0), (4.5, 4.0), (4.0, 3.0)))] + [_run_doc("m25", rtf=1.0)]
    s = gl.summarize(runs, startup=PROBES)
    assert s["gpu_label"] == "exclusive" and s["passed"] and s["adr"] == "ADR-216"
    assert s["perf"]["rtf_phys"] == 4.0 and s["perf"]["repeats"] == 3 and s["perf"]["startup_s_max"] == 4.0   # rtf 只取 physics 的中位数
    assert set(_res(s).values()) == {"pass"} and s["same_seed"]["bitwise"] and s["m25_equal"]["bitwise"] and s["seal_ready"]
    rtf = next(a for a in s["acceptance"] if a["clause"] == "rtf_phys")
    assert "provisional" not in rtf and rtf["min_single"] == 3.5 and rtf["runs"] == [3.5, 4.5, 4.0]          # ADR-216 (1)：冻结，单次最低另报
    # 中位 ≥ 3.0 而单次低于 3.0（S21 第 1 组 2.92）：按中位判定通过
    low1 = gl.summarize([_run_doc(idx=i, rtf=r) for i, r in enumerate((2.92, 3.12, 3.65))] + [_run_doc("m25")], startup=PROBES)
    assert _res(low1)["rtf_phys"] == "pass" and low1["perf"]["rtf_phys_min"] == 2.92


def test_summarize_failures_and_shared_card() -> None:
    base = [_run_doc(idx=i) for i in range(3)]
    shared = gl.summarize([*base[:2], _run_doc(idx=2, snaps_other=1), _run_doc("m25")])
    r = _res(shared)
    assert shared["gpu_label"] == "shared" and shared["passed"]                                  # 共享卡：性能条款不计入，不阻塞
    assert r["rtf_phys"] == r["startup_s"] == "pass_shared_not_counted" and r["alt_err_m"] == "pass"
    slow = gl.summarize([*[_run_doc(idx=i, rtf=2.5, snaps_other=1) for i in range(3)], _run_doc("m25")])
    assert _res(slow)["rtf_phys"] == "fail_shared_not_counted" and slow["passed"] and not slow["seal_ready"]
    bad = gl.summarize([base[0], _run_doc(idx=1, pose="q" * 64), _run_doc("m25", pose="z" * 64, alt=2.1)])
    r = _res(bad)
    assert r["same_seed_bitwise"] == r["m25_load_bitwise"] == r["alt_err_m"] == "fail" and not bad["passed"]
    frames = gl.summarize([*base, _run_doc("m25", frames={"frames": {"cam0": 1}, "max_abs_diff_ns": {"cam0": 4}})])
    assert _res(frames)["frame_tick_diff_ns"] == "fail"
    only_phys = gl.summarize(base)
    assert _res(only_phys)["m25_load_bitwise"] == _res(only_phys)["frame_tick_diff_ns"] == "not_measured" and not only_phys["passed"]
    crash = gl.summarize([*base, _run_doc("m25", contacts=["uav-02:COLLISION_WORLD"], gpu={"ok": False})])
    assert _res(crash)["no_crash"] == _res(crash)["no_gpu"] == "fail"


def test_summary_files_roundtrip(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    for i in range(3):
        (tmp_path / "runs" / f"physics-{i}.json").write_text(json.dumps(_run_doc(idx=i)))
    (tmp_path / "runs" / "m25-0.json").write_text(json.dumps(_run_doc("m25")))
    assert gl.main(["summarize", "--out", str(tmp_path)]) == gl.EXIT_OK
    doc = json.loads((tmp_path / "gz_lockstep.json").read_text())
    assert doc["schema"] == gl.SCHEMA and len(doc["runs"]) == 4 and doc["summary"]["passed"]
    lk = gl.acquire_lock(tmp_path / "x.lock", 0.0)
    assert lk is not None and gl.acquire_lock(tmp_path / "x.lock", 0.0) is None                # 已被持有：等锁超时
    lk.close()
    assert gl.wait_load(None, 0.0)["ok"] and not gl.wait_load(-1.0, 0.0)["ok"]


# ---------------------------------------------------------------------------------------------------------------- dmax：gz 锁步
def _has_gz() -> bool:
    return shutil.which("gz") is not None and any(
        (Path(p) / "libDrone4DGz.so").exists() for p in os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "").split(":") if p)


WORLDS = Path(os.environ.get("AWR_GZ_LOCKSTEP_WORLDS", ROOT / "worlds"))
EVENTS = Path(os.environ.get("AWR_GZ_LOCKSTEP_EVENTS", ROOT / "d3data" / "events"))
SPEC = ROOT / "configs" / "bench.d3" / "crop-handoff-001.l2-3uav.run_spec.json"


@pytest.mark.skipif(not _has_gz(), reason="needs gz-sim with libDrone4DGz.so (d3-gz image)")
@pytest.mark.skipif(not (WORLDS / "oryahovo-crop1" / "geometry" / "mesh" / "raycast" / "index.json").exists()
                    or not (EVENTS / "crop-handoff-001").is_dir(), reason="needs worlds/oryahovo-crop1 and the crop-handoff-001 event")
def test_three_uav_lockstep_env(tmp_path: Path) -> None:
    """3 架 p600_bench 映射的 x500 由 BenchEnv 驱动（L1，无 M25 负载），同种子两次：逐位一致与 D3-AC-34 的非性能条款。"""
    pytest.importorskip("zmq")
    pytest.importorskip("open3d")
    from awr.agentsdk.env.core import _open_world
    from awr.world.export.gz import export_world

    dur = float(os.environ.get("AWR_GZ_LOCKSTEP_S", "360"))
    plan = gl.Maneuver()
    if dur < 360:                                                                              # 冒烟：低空、短航段
        plan = gl.Maneuver(alt_agl_m=(10.0, 16.0, 22.0), legs_from_s=22.0, leg_s=8.0, n_legs=max(1, int((dur - 30.0) // 8.0)), half_side_m=40.0)
    cfg = gl.RunCfg(SPEC, WORLDS, EVENTS, tmp_path, duration_s=dur, gz={"launcher": "local"}, plan=plan)
    world = _open_world(WORLDS / "oryahovo-crop1")
    exp = export_world(WORLDS / "oryahovo-crop1", out_root=tmp_path / "export")
    runs = [gl.run_once(cfg, "physics", i, export=exp, world=world) for i in range(2)]
    s = gl.summarize(runs)
    r = _res(s)
    print(json.dumps({"perf": s["perf"], "acceptance": s["acceptance"], "vh_max_mps": s["vh_max_mps"],
                      "alt_err": runs[0]["alt_err"], "startup": [x["startup"] for x in runs], "resources": runs[0]["resources"]}, indent=1))
    for clause in ("step_done_25_steps_per_tick", "pose_history_hz_250", "same_seed_bitwise", "alt_err_m", "step_done_tick_diff_ns",
                   "cpu_cores", "mem_gb", "no_gpu", "gravity_9_80665", "no_crash"):
        assert r[clause] in ("pass", "pass_shared_not_counted"), (clause, r[clause], s)
    assert runs[0]["n_ticks"] == round(dur * 10) and runs[0]["iterations"] == 25 * (runs[0]["n_ticks"] + runs[0]["preroll_ticks"])
    assert runs[0]["ticks"] == runs[0]["n_ticks"] + runs[0]["preroll_ticks"] and runs[0]["preroll_ticks"] == 20     # Env 预滚 2 s
    assert runs[0]["vehicle"]["mass_kg"] == 3.935
    if dur >= 300:
        assert s["vh_max_mps"] >= 8.0                                                          # 航段确实以 ≥ 8 m/s（p600_std 巡航）机动
    assert all(a["timeouts"] == 0 for a in runs[0]["agents"].values())


REPORT = os.environ.get("AWR_GZ_LOCKSTEP_REPORT")


@pytest.mark.skipif(not REPORT, reason="set AWR_GZ_LOCKSTEP_REPORT to a gz_lockstep.json produced on dmax")
def test_report_acceptance() -> None:
    """完整报告（physics × 3 + m25）：除共享卡上不计入的性能条款外全部通过，且 3 机 360 s、M25 帧逐 tick 对齐。"""
    doc = json.loads(Path(REPORT).read_text(encoding="utf-8"))
    s = doc["summary"]
    r = _res(s)
    assert all(v == "pass" or (v.endswith("_shared_not_counted") and v.startswith("pass")) for v in r.values()), r
    assert s["perf"]["repeats"] >= 3 and s["m25_equal"]["runs"] >= 1 and sum(s["frames"].values()) > 0
    assert doc["config"]["duration_s"] == 360.0 and doc["config"]["vehicle"]["name"] == "p600_bench"
