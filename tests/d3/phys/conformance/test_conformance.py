"""两档一致性用例集（M24-FR-081、FR-063；AWR-05 §5.3.5 一致性用例集行；D3-AC-35）：本机 CPU 部分。

- 用例集覆盖 §11 D3-AC-35 列出的机动（起降、悬停、速度阶跃、航点、环绕、接触与坠毁、风扰），每个用例登记位置 RMS 或到达时刻等
  指标的容差；
- Fast 档逐用例运行：接口层判定（指令回执、`VehicleStatus` 序列、接触事件种类、PoseHistory 网格、指令生效步）与用例期望一致；
- RefTracker（Realistic 档参考跟踪器）与 Fast 档同一参考生成器：驱动理想一阶速度环对象时，起降与悬停的轨迹与 Fast 档在厘米级
  一致（参考生成、模式切换与触地爬行的语义相同，档间差异只来自动力学）；
- GzSimBackend + RefTracker 对替身 gz：起飞怠速 1 s 内电机失能，随后逐物理步下发设定（apply_step 0..24）；
- 运动学注入文件布局与注入世界（无重力、`<inject>`）；PoseHistory 逐样本比较的判据（位置 1e-6 m、速度逐位）；
- 两份 PoseHistory 送入同一 M25：同一输入时 LiDAR 与 IMU 输出逐字节相同，位姿差 0.5 mm 时 LiDAR 逐点位姿差在 1 mm 内、
  2 mm 时超限；同一 PoseHistory 两次送入 M25 全路径（相机 CPU 替身、LiDAR、IMU）摘要逐字节相同；
- `tier_compare.json`（drone4d.phys.tier_compare.v1）字段齐全：每用例的接口判定、各档指标、档间差、容差与是否在容差内。

两档实测（d3-gz 容器内）见 tests/d3/gz/test_tier_compare.py。
运行：`PYTHONPATH=python .venv-d3/bin/python -m pytest -q tests/d3/phys/conformance -p no:warnings`。
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np
import pytest

from awr.phys.adapter import TierCommand
from awr.phys.pose_history import STATE_N
from awr.phys.tier_compare import (
    CASES,
    IMU_TOL,
    LIDAR_TOL_M,
    REF_CASE,
    SCHEMA,
    STEP_NS,
    TICK_NS,
    BoxWorld,
    Case,
    CaseRun,
    build_report,
    compare_case,
    inject_sdf,
    m25_compare,
    m25_digest,
    pose_history_diff,
    run_fast,
    write_inject,
)
from awr.phys.tracker import VEL_P_ACC, RefTracker

AC35_MANEUVERS = {"takeoff_land", "hover", "velocity_step", "waypoints", "orbit", "contact_crash", "wind_hover"}
GZ_REST_Z = 0.224                                  # x500_d4d 在 gz 中的静止 base_link 高度（替身对象的地面）


@pytest.fixture(scope="module")
def fast_runs() -> dict[str, CaseRun]:
    want = ("hover", "velocity_step", "yaw_step", "contact_crash", "wind_hover", "takeoff_land")
    return {c.id: run_fast(c) for c in CASES if c.id in want}


@pytest.fixture(scope="module")
def ref_run() -> CaseRun:
    return run_fast(Case(REF_CASE.id, REF_CASE.title, REF_CASE.spawn, 120, {k: v for k, v in REF_CASE.script.items() if k < 120},
                         grounded=REF_CASE.grounded))


def _cases() -> dict[str, Case]:
    return {c.id: c for c in CASES}


# ---------------------------------------------------------------------------------------------------------------- 用例集
def test_case_set_covers_ac35_maneuvers_with_tolerances() -> None:
    cs = _cases()
    assert set(cs) >= AC35_MANEUVERS
    for c in cs.values():
        assert c.tol, c.id
        assert all(v > 0 for v in c.tol.values())
        assert c.n_ticks * TICK_NS // STEP_NS >= 250
    for cid in ("takeoff_land", "hover", "velocity_step", "waypoints", "orbit", "wind_hover"):
        assert "pos_rms_m" in cs[cid].tol, cid
    assert "arrival_s_per_100m" in cs["waypoints"].tol and "t_reach_s" in cs["takeoff_land"].tol
    assert cs["wind_hover"].wind is not None and cs["wind_hover"].wind[1] % TICK_NS == 0
    assert cs["contact_crash"].boxes and {"COLLISION_WORLD", "IMPACT"} <= {k for v in cs["contact_crash"].expect_contacts.values() for k in v}
    # M24 §6.10.6 初始容差
    assert cs["velocity_step"].tol["rise_s"] == 0.3 and cs["velocity_step"].tol["overshoot_pct"] == 10.0
    assert cs["wind_hover"].tol == {"tilt_deg": 0.5, "pos_rms_m": 0.5}
    assert cs["waypoints"].tol == {"pos_rms_m": 1.5, "arrival_s_per_100m": 1.0}


def test_fast_tier_interface_matches_case_expectations(fast_runs: dict[str, CaseRun]) -> None:
    cs = _cases()
    for cid, run in fast_runs.items():
        r = compare_case(cs[cid], run, run)
        assert r["interface"]["pass"], (cid, r["failures"])
        assert r["interface"]["pose_history_grid"]["samples"] == cs[cid].n_ticks * TICK_NS // STEP_NS + 1
        assert all(r["within"].values()), (cid, r["diff"])
    kinds = {c["uav"]: c["kind"] for c in fast_runs["contact_crash"].contacts}
    assert kinds == {"wall": "COLLISION_WORLD", "drop": "IMPACT"}
    m = compare_case(cs["takeoff_land"], fast_runs["takeoff_land"], fast_runs["takeoff_land"])["metrics"]["fast"]
    assert m["t_touchdown_s"] is not None and abs(m["z_rest_m"] - 0.227) < 0.01


# ---------------------------------------------------------------------------------------------------------------- RefTracker
def _toy_plant(case: Case) -> np.ndarray:
    """RefTracker 驱动理想速度环对象（a = K_p·(v_cmd − v)，与 compose "px4" 档的 Lee 增益相同；无阻力、地面 z = 0.224）。"""
    u = case.uavs[0]
    (x, y, z), yaw = case.spawn[u]
    grounded = u in case.grounded
    p = np.array([x, y, GZ_REST_Z if grounded else z])
    v = np.zeros(3)
    psi, wz = yaw, 0.0
    tr = RefTracker()
    tr.tick_s = 0.1

    def row() -> np.ndarray:
        return np.r_[p, 0.0, 0.0, math.sin(psi / 2), math.cos(psi / 2), v, 0.0, 0.0, wz, 0.0, 0.0, 9.80665]

    tr.bind([u], {u: row()}, [u] if grounded else [])
    landed, ic, still = grounded, grounded, 0
    P = [p.copy()]
    for k in range(case.n_ticks):
        tr.observe(k * TICK_NS, {u: (row(), ic, landed, False)})
        for c in case.script.get(k, ()):
            tr.set(c, p)
            landed = landed and c.kind not in ("takeoff",)
        tr.observe(k * TICK_NS, {u: (row(), ic, landed, False)})
        for sp in tr.plan(k, k * TICK_NS, 25)[u]:
            if sp.enabled:
                a = np.clip(VEL_P_ACC * (np.array(sp.v_enu) - v), -9.8, 9.8)
                wz += (sp.yaw_rate - wz) * 0.08
            else:
                a, wz = np.array([0.0, 0.0, -9.80665]), 0.0
            v = v + a * 0.004
            p = p + v * 0.004
            psi += wz * 0.004
            ic = p[2] <= GZ_REST_Z
            if ic:
                p[2] = GZ_REST_Z
                v = np.zeros(3) if v[2] < 0 else v
            still = still + 1 if ic and not landed and k > 30 and float(np.linalg.norm(v)) < 0.25 else 0
            landed = landed or still >= 125
            P.append(p.copy())
    return np.array(P)


def test_ref_tracker_shares_fast_reference_generator(fast_runs: dict[str, CaseRun]) -> None:
    cs = _cases()
    for cid, tol in (("takeoff_land", 0.05), ("hover", 1e-6)):
        a = fast_runs[cid].state["u1"][:, 0:3]
        b = _toy_plant(cs[cid])
        assert a.shape == b.shape
        rms = float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
        assert rms < tol, (cid, rms)
    # 起降：起飞怠速 1 s、3 s 斜坡、LAND 剖面与触地爬行 → 触地时刻与 Fast 相差 < 0.1 s
    a = fast_runs["takeoff_land"].state["u1"]
    b = _toy_plant(cs["takeoff_land"])
    t = np.arange(len(b)) * STEP_NS / 1e9
    land_t = cs["takeoff_land"].params["land_tick"] * 0.1
    ta = t[np.flatnonzero((t > land_t) & (a[:, 2] <= 0.25))[0]]
    tb = t[np.flatnonzero((t > land_t) & (b[:, 2] <= 0.25))[0]]
    assert abs(ta - tb) < 0.1


def test_ref_tracker_rejects_bad_orbit_and_plans_per_step() -> None:
    tr = RefTracker()
    row = np.r_[0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 1.0, np.zeros(9)]
    tr.bind(["u"], {"u": row}, [])
    from awr.phys import PhysError
    from awr.phys.adapter import OrbitSpec

    with pytest.raises(PhysError) as e:
        tr.set(TierCommand("u", "orbit", 0, orbit=OrbitSpec((0.0, 0.0, 10.0), 0.5)), row[0:3])
    assert e.value.code == 630
    tr.observe(0, {"u": (row, False, False, False)})
    tr.set(TierCommand("u", "velocity", 0, vel_enu_mps=(3.0, 0.0, 0.0)), row[0:3])
    sps = tr.plan(0, 0, 25)["u"]
    assert len(sps) == 25 and all(s.enabled for s in sps)
    assert sps[0].v_enu[0] == pytest.approx(3.0 * (1.0 + 0.4 / 1.8 * 0.1), abs=1e-9)   # 速度模式：设定即指令 + 一个 tick 的积分
    assert tr.kind("u") == "velocity"


class _StaticClock:
    """REP 端替身 Drone4DClock：机体静止于 z = 0.224（地面），记录每个 Control 的 (apply_step, enabled, vz)。"""

    def __init__(self, endpoint: str) -> None:
        import zmq

        from awr.phys.gz import protocol as P

        self.P = P
        self.sock = zmq.Context.instance().socket(zmq.REP)
        self.sock.bind(endpoint)
        self.t = 0
        self.controls: list[list[tuple[int, bool, float]]] = []
        threading.Thread(target=self.run, daemon=True).start()

    def run(self) -> None:
        P = self.P
        row = np.zeros(STATE_N)
        row[2], row[6], row[15] = GZ_REST_Z, 1.0, 9.80665
        while True:
            body, m = P.decode(self.sock.recv())
            if body == "hello":
                self.sock.send(P.encode("hello_ack", P.msg("HelloAck", n_uav=1, t_ns=0, initial_state=row.tobytes())))
            elif body == "control":
                self.controls.append([(s.apply_step, s.enabled, s.vz) for s in m.setpoints])
                t = [self.t + (j + 1) * STEP_NS for j in range(m.grant_steps)]
                self.t += m.grant_steps * STEP_NS
                s = np.tile(row, (m.grant_steps, 1, 1))
                self.sock.send(P.encode("step_done", P.msg("StepDone", seq=m.seq, tick=m.tick, n_steps=m.grant_steps, n_uav=1, t_ns=t,
                                                           samples=s.tobytes())))
            else:
                self.sock.send(P.encode("bye", P.msg("Bye", iterations=self.t // STEP_NS)))
                self.sock.close(0)
                return


def test_backend_with_ref_tracker_spools_then_streams_step_setpoints(tmp_path: Path) -> None:
    pytest.importorskip("zmq")
    from awr.phys.gz.backend import GzSimBackend

    ep = f"ipc://{tmp_path / 'ctl.sock'}"
    fake = _StaticClock(ep)
    be = GzSimBackend(["u1"], ep, timeout_s=5.0, tracker=RefTracker(), retain_s=None, history_s=5.0)
    be.open()
    be.submit([TierCommand("u1", "takeoff", 1, alt_m=5.0)])
    for k in range(18):
        be.advance(k)
    be.close()
    c = fake.controls
    assert c[0] == [(0, False, 0.0)]                                # 地面出生：电机失能
    spool = [x for k in range(1, 11) for x in c[k]]                  # 起飞指令后 1 s 怠速：不下发使能设定
    assert all(not en for _j, en, _vz in spool)
    climb = c[12]
    assert len(climb) > 1 and {j for j, _e, _v in climb} <= set(range(25)) and all(en for _j, en, _v in climb)
    assert all(vz > 0.0 for _j, _e, vz in climb)


# ---------------------------------------------------------------------------------------------------------------- 注入
def test_inject_file_layout_and_world(tmp_path: Path, ref_run: CaseRun) -> None:
    from awr.phys.gz.compose import UavSpawn, compose_run

    f = write_inject(ref_run, tmp_path / "ref.f64")
    a = np.frombuffer(f.read_bytes(), "<f8").reshape(-1, 1, STATE_N)
    assert np.array_equal(a[:, 0], ref_run.state["u1"]) and len(a) == len(ref_run.t_ns)
    sdf = inject_sdf(compose_run([UavSpawn("u1", (0.0, 0.0, 1.0))], endpoint="ipc:///run/d4d/ctl.sock"), "/x/ref.f64")
    assert "<gravity>0 0 0</gravity>" in sdf and "<inject>/x/ref.f64</inject>" in sdf
    import xml.etree.ElementTree as ET

    ET.fromstring(sdf)
    with pytest.raises(ValueError):
        inject_sdf(compose_run([UavSpawn("u1", (0.0, 0.0, 1.0))], ground_plane=True), "/x/ref.f64")


def test_pose_history_diff_thresholds(ref_run: CaseRun) -> None:
    same = pose_history_diff(ref_run, ref_run)
    assert same["pass"] and same["t_bitwise"]
    st = {u: s.copy() for u, s in ref_run.state.items()}
    st["u1"][100:110, 0] += 5e-7
    ok = CaseRun(ref_run.case, "realistic", ref_run.t_ns, st, {}, [], [], {})
    assert pose_history_diff(ref_run, ok)["pass"]
    st2 = {u: s.copy() for u, s in st.items()}
    st2["u1"][200, 0] += 2e-6
    assert not pose_history_diff(ref_run, CaseRun(ref_run.case, "realistic", ref_run.t_ns, st2, {}, [], [], {}))["pass"]
    st3 = {u: s.copy() for u, s in ref_run.state.items()}
    st3["u1"][50, 8] = np.nextafter(st3["u1"][50, 8], np.inf)
    d = pose_history_diff(ref_run, CaseRun(ref_run.case, "realistic", ref_run.t_ns, st3, {}, [], [], {}))
    assert not d["pass"] and not d["uavs"]["u1"]["vel_bitwise"]
    t2 = ref_run.t_ns.copy()
    t2[-1] += 1
    assert not pose_history_diff(ref_run, CaseRun(ref_run.case, "realistic", t2, ref_run.state, {}, [], [], {}))["t_bitwise"]


# ---------------------------------------------------------------------------------------------------------------- M25
def _shift(run: CaseRun, dx: float, axis: int = 0) -> CaseRun:
    d = np.zeros(STATE_N)
    d[axis] = dx
    st = {u: s + d for u, s in run.state.items()}
    return CaseRun(run.case, "realistic", run.t_ns, st, run.status, run.contacts, run.dispatch, run.info)


def test_m25_sampling_consistency_lidar_and_imu(ref_run: CaseRun) -> None:
    world = BoxWorld()
    same = m25_compare(ref_run, ref_run, world=world, max_ticks=30)
    assert same["pass"] and same["lidar"]["points_equal_frames"] == same["lidar"]["frames"] == 30
    assert same["imu"]["bitwise_blocks"] == same["imu"]["blocks"] == 30 and same["lidar"]["points"] > 0
    near = m25_compare(ref_run, _shift(ref_run, 5e-4), world=world, max_ticks=30)
    assert near["lidar"]["pass"] and 4e-4 < near["lidar"]["origin_max_m"] <= LIDAR_TOL_M
    far = m25_compare(ref_run, _shift(ref_run, 2e-3), world=world, max_ticks=30)
    assert not far["lidar"]["pass"]
    assert near["imu"]["acc_max_mps2"] <= IMU_TOL["acc_mps2"]       # 平移不改 IMU（比力与角速度相同）


def test_m25_same_pose_history_is_bytewise_identical(ref_run: CaseRun) -> None:
    pytest.importorskip("PIL")
    a = m25_digest(ref_run, world=BoxWorld(), max_ticks=25)
    b = m25_digest(ref_run, world=BoxWorld(), max_ticks=25)
    assert a["sha256"] == b["sha256"] and a["frames"]["lidar0"] == 25 and a["frames"]["eo"] >= 25
    c = m25_digest(_shift(ref_run, 1e-3, axis=2), world=BoxWorld(), max_ticks=25)    # 竖直 1 mm：对地距离改变
    assert c["sha256"]["lidar0"] != a["sha256"]["lidar0"] and c["sha256"]["pose_history"] != a["sha256"]["pose_history"]


# ---------------------------------------------------------------------------------------------------------------- 报告
def test_report_fields_and_tolerance_flags(fast_runs: dict[str, CaseRun], tmp_path: Path) -> None:
    cs = _cases()
    rows = [compare_case(cs[cid], run, run) for cid, run in fast_runs.items()]
    bad = dict(rows[0])
    bad["pass"], bad["tol_pass"] = False, False
    doc = build_report(rows, sampling={"pass": True}, m25={"pass": True}, meta={"gz": {"plugin_build": "x"}})
    assert doc["schema"] == SCHEMA and doc["pass"] and doc["summary"]["cases"] == len(rows) == doc["summary"]["pass"]
    for k in ("vehicle", "clock_profile", "tick_ms", "pose_history_hz", "wind_mode", "turbulence_sigma", "drag_law",
              "collision_source", "m09_autonomy", "tolerance_status", "generated_utc", "sampling_consistency", "m25_bytewise"):
        assert k in doc, k
    assert doc["wind_mode"] == "field" and doc["turbulence_sigma"] == 0.0 and doc["pose_history_hz"] == 250
    for r in doc["cases"]:
        assert {"interface", "metrics", "diff", "tol", "within", "pos_rms_m", "pass"} <= set(r)
        assert set(r["within"]) == set(r["diff"]) and set(r["metrics"]) == {"fast", "gz"}
    assert not build_report([*rows, bad])["pass"]
    p = tmp_path / "tier_compare.json"
    p.write_text(json.dumps(doc, default=str), encoding="utf-8")
    assert json.loads(p.read_text(encoding="utf-8"))["schema"] == SCHEMA
