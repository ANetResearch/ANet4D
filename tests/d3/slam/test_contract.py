"""SLAM 导出合同测试（M25 §6.11.4 C1–C11、N1、N2；AWR-05 §5.4.6；D3-AC-13；M25-AC-026、027、029、030；CAT-AC-026）。

夹具为物理档口径的 PoseHistory（逐物理步 250 Hz 的位置、速度、姿态、机体系角速度与比力，解析微分平坦航迹）：空中出生，
预滚窗口 [−2 s, 0) 悬停保持，0–4 s 平滑加速到 6 m/s，其后匀速航段叠加缓慢横向蛇行与偏航。机型 `p600_mid360_inv`（倒装
MID-360，内置 IMU 的挂载与杆臂取 rig.json）。bag 时基 = 事件时刻 + t_pre（ROS1 时间不能为负，预滚窗口从 bag 时刻 0 开始）。

1. LiDAR（C1–C3）：md5、每帧 20,000 槽位、无回波零点占位、`offset_time = i · 5000`、`timebase` 为帧首且对齐 100 ms 网格；
   负向用例 N1：删除无回波点（或删除后重排 offset_time）的 bag 被拒绝（642）。
2. IMU（C4–C8、N2）：200 Hz、单位 g、`orientation_covariance[0] = −1`；悬停保持窗口与 `gravity_align_en`；与同算法 float64
   参考实现（`imu_reference.py`）之差 ≤ 1e-6（M25 输出、bag 读数、经 raw.mcap f32 存档的数据集导出路径三处），抗混叠理论误差
   另行报告；clean 档 5 s 纯惯导回代 < 5 cm；|ω| < 5 rad/s、|a| < 30 m/s² 断言（违例 642 并标注时间段）；单位写成 m/s² 的
   bag 被拒绝（642）。
3. SLAM 相机（C10）：rig 为全局快门、无云台、固定内参与曝光；飞行中逐帧内参、外参（挂载与云台真值）、曝光恒定且与 rig 一致，
   读出时间为 0（采集作业以 `slam_cam` 配置运行，GPU 以桩替代）。
4. 噪声参数（C11）：IMU 合成所用参数与参数表（AWR-20 §15.6.5，`imu_noise_tier`）、rig.json 逐项一致；样本统计（白噪声 sigma、
   上电偏置 sigma、偏置随机游走步长、harsh 振动幅值与频率）与参数一致；预滚窗口负网格序号的偏置随机游走连续且与分批无关。
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("yaml")

from awr.sensorsvc.errors import SensorSvcError
from awr.sensorsvc.imu import (
    ACC_SAT,
    DEFAULT_BPF_HZ,
    DYN_ACC_MAX_MPS2,
    DYN_OMEGA_MAX_RPS,
    GYRO_SAT,
    TIERS,
    ImuModel,
    assert_dynamics,
    check_dynamics,
    kinematics_from_poses,
    load_imu_model,
    synth_imu,
)
from awr.sensorsvc.slam.bag import (
    CUSTOM_POINT,
    IMU_TOPIC,
    LIDAR_TOPIC,
    LIVOX_CUSTOM_MSG_MD5,
    LIVOX_TYPES,
    Bag1Reader,
    Bag1Writer,
    SlamBagWriter,
    check_custom_msg,
    offset_time_ns,
    pack_custom_msg,
    pack_imu,
    unpack_custom_msg,
    unpack_imu,
    verify_slam_bag,
)
from awr.sensorsvc.slam.rig import check_init_window, check_slam_rig, require_init_window, rig_for_vehicle
from awr.sim.catalog.loader import Catalog, imu_noise_tier

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("d3_slam_imu_reference", Path(__file__).with_name("imu_reference.py"))
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
G, reference_imu, strapdown = _ref.G, _ref.reference_imu, _ref.strapdown
VID = "p600_mid360_inv"
NS = 1_000_000_000
PHYS_HZ = 250
T_PRE_NS = 2 * NS                     # 预滚窗口 [−2 s, 0)；bag 时刻 = 事件时刻 + T_PRE_NS
T_BAG_END_NS = 8 * NS                 # bag 覆盖事件时刻 [−2 s, 8 s]
SLOTS = 20_000
LIDAR_PERIOD_NS = 100_000_000
REF_TOL = 1e-6                        # C6：与 float64 参考实现之差（m/s²、rad/s）
INS_TOL_M = 0.05                      # C7：5 s 纯惯导回代位置误差
INS_WINDOWS_S = (-2.0, 0.0, 1.5, 3.0) # 5 s 回代窗口起点（事件时刻）：悬停 → 加速、加速段、加速 → 匀速、匀速蛇行

# 解析航迹参数：巡航速度、加速时长、横向蛇行幅值与角频率、偏航速率与其起始角频率
V_CRUISE, T_ACC, A_WEAVE, OM_WEAVE, YAW_RATE, OM_YAW, AGL = 6.0, 4.0, 1.0, 0.6, 0.15, 0.5, 30.0


# ---------------------------------------------------------------------------------------------------------------- 航迹夹具
def _flat(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """事件时刻 t（s）的位置、速度、加速度（ENU）与航向；t ≤ 0 悬停，加速段为五次平滑阶跃，加速度与急动度连续起步。"""
    if t <= 0:
        x = vx = ax = 0.0
    elif t < T_ACC:
        u = t / T_ACC
        x = V_CRUISE * T_ACC * (u**6 - 3 * u**5 + 2.5 * u**4)
        vx = V_CRUISE * (6 * u**5 - 15 * u**4 + 10 * u**3)
        ax = V_CRUISE / T_ACC * (30 * u**4 - 60 * u**3 + 30 * u**2)
    else:
        x, vx, ax = V_CRUISE * T_ACC * 0.5 + V_CRUISE * (t - T_ACC), V_CRUISE, 0.0
    tau = max(t - T_ACC, 0.0)
    y = A_WEAVE * (OM_WEAVE * tau - math.sin(OM_WEAVE * tau))
    vy = A_WEAVE * OM_WEAVE * (1 - math.cos(OM_WEAVE * tau))
    ay = A_WEAVE * OM_WEAVE**2 * math.sin(OM_WEAVE * tau)
    psi = YAW_RATE * (tau - math.sin(OM_YAW * tau) / OM_YAW)
    return np.array([x, y, AGL]), np.array([vx, vy, 0.0]), np.array([ax, ay, 0.0]), psi


def _R_wb(t: float) -> np.ndarray:
    """微分平坦：机体 z 轴沿比力方向，航向由 psi 给出（FLU 机体、ENU 世界）。"""
    _, _, a, psi = _flat(t)
    fw = a + np.array([0.0, 0.0, G])
    zb = fw / np.linalg.norm(fw)
    yb = np.cross(zb, [math.cos(psi), math.sin(psi), 0.0])
    yb /= np.linalg.norm(yb)
    return np.stack([np.cross(yb, zb), yb, zb], 1)


def _vee_log(R: np.ndarray) -> np.ndarray:
    th = math.acos(min(1.0, max(-1.0, (np.trace(R) - 1) / 2)))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (0.5 if th < 1e-9 else th / (2 * math.sin(th)))


def _omega_b(t: float, h: float = 1e-5) -> np.ndarray:
    return _vee_log(_R_wb(t - h).T @ _R_wb(t + h)) / (2 * h)


@cache
def _history(t0_s: float = -2.0, t1_s: float = 8.1) -> dict[str, np.ndarray]:
    """物理档 PoseHistory（逐物理步）：t_ns、p、v、R、omega_b、f_b。"""
    n = round((t1_s - t0_s) * PHYS_HZ) + 1
    t = round(t0_s * NS) + np.arange(n, dtype=np.int64) * (NS // PHYS_HZ)
    cols: dict[str, list] = {"p": [], "v": [], "R": [], "omega_b": [], "f_b": []}
    for tn in t:
        p, v, a, _ = _flat(tn / NS)
        R = _R_wb(tn / NS)
        cols["p"].append(p)
        cols["v"].append(v)
        cols["R"].append(R)
        cols["omega_b"].append(_omega_b(tn / NS))
        cols["f_b"].append(R.T @ (a + np.array([0.0, 0.0, G])))
    return {"t_ns": t, **{k: np.array(v) for k, v in cols.items()}}


def _imu_truth_state(rig: dict, t_ns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IMU 原点在事件时刻 t_ns 的世界姿态 R_wi、速度与位置（解析真值，含杆臂）。"""
    R_bi, r = _mount(rig)
    ts = t_ns / NS
    p, v, _, _ = _flat(ts)
    R = _R_wb(ts)
    return R @ R_bi, v + R @ np.cross(_omega_b(ts), r), p + R @ r


def _mount(rig: dict) -> tuple[np.ndarray, np.ndarray]:
    T = rig["imu"]["T_base_imu"]
    return np.asarray(T["R"], np.float64), np.asarray(T["t_m"], np.float64)


def _synth(rig: dict, tier: str = "clean", model: ImuModel | None = None, *, seed: int = 7, t_from_ns: int = -T_PRE_NS,
           t_to_ns: int = T_BAG_END_NS + 5_000_000, **kw):
    H = _history()
    R_bi, r = _mount(rig)
    return synth_imu(H["t_ns"], H["omega_b"], H["f_b"], R_bi=R_bi, r_imu_b=r, model=model or load_imu_model(), t_from_ns=t_from_ns,
                     t_to_ns=t_to_ns, tier=tier, seed=seed, uav_no=1, sensor_no=0, **kw)


@pytest.fixture(scope="module")
def cat():
    return Catalog.load(ROOT / "vehicles", d3=True)


@pytest.fixture(scope="module")
def rig() -> dict:
    """空中出生（悬停保持窗口，gravity_align_en 开启），clean 档。"""
    r = rig_for_vehicle(VID, vehicles_dir=ROOT / "vehicles", uav="uav-01", start="pose", noise_tier="clean")
    check_slam_rig(r)
    return r


# ---------------------------------------------------------------------------------------------------------------- bag 夹具
def _lidar_points(seq: int, frac: float = 0.6) -> np.ndarray:
    """一帧有回波槽位（槽位升序，offset_time 在网格上）；其余槽位由导出端补零点。"""
    rng = np.random.default_rng(1000 + seq)
    idx = np.sort(rng.choice(SLOTS, int(SLOTS * frac), replace=False))
    pts = np.zeros(len(idx), CUSTOM_POINT)
    pts["offset_time"] = offset_time_ns(SLOTS)[idx]
    xyz = rng.normal(size=(len(idx), 3))
    xyz *= rng.uniform(1.0, 60.0, len(idx))[:, None] / np.linalg.norm(xyz, axis=1)[:, None]
    pts["x"], pts["y"], pts["z"] = xyz.T.astype(np.float32)
    pts["reflectivity"] = rng.integers(1, 256, len(idx))
    pts["tag"] = 0x10
    pts["line"] = idx % 4
    return pts


@cache
def _jpeg() -> bytes:
    Image = pytest.importorskip("PIL.Image")

    b = io.BytesIO()
    Image.fromarray(np.full((16, 20, 3), 90, np.uint8)).save(b, format="JPEG", quality=90)
    return b.getvalue()


def _write_bag(path: Path, samples, *, acc_unit_g: bool = True, frames: int | None = None) -> Path:
    """LiDAR 10 Hz（帧首对齐 100 ms 网格）、IMU 200 Hz、相机 10 Hz（曝光中点 = 帧首 + 50 ms），bag 时刻 = 事件时刻 + t_pre。"""
    n_frames = frames if frames is not None else (T_PRE_NS + T_BAG_END_NS) // LIDAR_PERIOD_NS
    ev = []
    for k in range(n_frames):
        ev.append((k * LIDAR_PERIOD_NS, 0, "lidar", k))
        ev.append((k * LIDAR_PERIOD_NS + LIDAR_PERIOD_NS // 2, 2, "image", k))
    t_hi = n_frames * LIDAR_PERIOD_NS
    for i, t in enumerate(samples.t_ns):
        tb = int(t) + T_PRE_NS
        if 0 <= tb <= t_hi:
            ev.append((tb, 1, "imu", i))
    ev.sort(key=lambda e: (e[0], e[1]))
    with SlamBagWriter(path, slots=SLOTS) as w:
        for tb, _, kind, x in ev:
            if kind == "lidar":
                w.lidar_frame(tb, _lidar_points(x))
            elif kind == "image":
                w.image(tb, _jpeg())
            elif acc_unit_g:
                w.imu(tb, samples.gyro_rps[x], samples.acc_mps2[x])
            else:                                                 # N2：线加速度误写为 m/s²
                w.bag.write(w._c_imu, tb, pack_imu(w._next(IMU_TOPIC), tb, samples.gyro_rps[x], samples.acc_mps2[x]))
    return path


def _bag_imu(path: Path) -> dict[str, np.ndarray]:
    """读回 `/livox/imu`：事件时刻、角速度、线加速度（bag 原值，单位 g）与 orientation_covariance[0]。"""
    t, w, a, oc = [], [], [], []
    with Bag1Reader(path) as rd:
        for m in rd.messages([IMU_TOPIC]):
            d = unpack_imu(m.data)
            t.append(m.t_ns - T_PRE_NS)
            w.append(d["angular_velocity"])
            a.append(d["linear_acceleration"])
            oc.append(d["orientation_covariance"][0])
    return {"t_ns": np.array(t, np.int64), "gyro": np.array(w), "acc_g": np.array(a), "ocov0": np.array(oc)}


def imu_contract(path: Path, rig: dict) -> dict[str, np.ndarray]:
    """bag IMU 的合同检查（C4、C5、C8）：200 Hz 等间隔、`orientation_covariance[0] = −1`、单位 g（悬停或静止窗口比力均值模长
    1 ± 0.01 g，与 rig.json 的出生方式一致）、读数 |ω| < 5 rad/s 与 |a| < 30 m/s²。违反时抛 642。"""
    d = _bag_imu(path)
    if len(d["t_ns"]) < 2 or (np.diff(d["t_ns"]) != NS // int(rig["imu"]["rate_hz"])).any():
        raise SensorSvcError(642, "imu samples are not on the 200 Hz grid")
    if (d["ocov0"] != -1.0).any():
        raise SensorSvcError(642, "orientation_covariance[0] must be -1 (orientation unknown)")
    win = check_init_window(d["t_ns"], d["gyro"], d["acc_g"], mode=rig["init"]["mode"], t_event0_ns=0, acc_unit="g")
    require_init_window(win, rig)
    assert_dynamics(d["t_ns"], d["gyro"], d["acc_g"] * G)
    return d


@pytest.fixture(scope="module")
def clean_bag(tmp_path_factory, rig) -> Path:
    p = _write_bag(tmp_path_factory.mktemp("bag") / "clean.bag", _synth(rig, "clean"))
    yield p
    p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------------------------- C1–C3、N1
def test_lidar_custom_msg_contract(clean_bag: Path) -> None:
    """C1–C3：md5 与 livox_ros_driver 一致；每帧 20,000 槽位、零点占位、offset_time = i·5000 单调、timebase 为帧首且对齐 100 ms 网格；
    相机时刻落在 LiDAR 覆盖内（C10 时间部分）。"""
    rep = verify_slam_bag(clean_bag, slots=SLOTS)
    assert rep.md5[LIDAR_TOPIC] == LIVOX_CUSTOM_MSG_MD5 and rep.types[LIDAR_TOPIC] in LIVOX_TYPES
    assert rep.lidar_frames == 100 and rep.images == 100 and rep.imu_samples == 2001
    assert rep.returns_mean == pytest.approx(0.6 * SLOTS)
    grid = offset_time_ns(SLOTS)
    assert (grid == np.arange(SLOTS) * 5000).all() and (np.diff(grid.astype(np.int64)) > 0).all()
    with Bag1Reader(clean_bag) as rd:
        for k, m in enumerate(rd.messages([LIDAR_TOPIC])):
            msg = unpack_custom_msg(m.data)
            pts = msg["points"]
            assert msg["point_num"] == len(pts) == SLOTS and (pts["offset_time"] == grid).all()
            assert msg["timebase"] == msg["header"]["stamp_ns"] == m.t_ns == k * LIDAR_PERIOD_NS          # 帧首、100 ms 网格
            zero = (pts["x"] == 0) & (pts["y"] == 0) & (pts["z"] == 0)
            assert zero.sum() == SLOTS - int(0.6 * SLOTS)
            assert (pts["reflectivity"][zero] == 0).all() and (pts["tag"][zero] == 0).all()
            assert (pts["line"] == np.arange(SLOTS) % 4).all()
            assert check_custom_msg(msg, SLOTS) == int(0.6 * SLOTS)


def _bag_dropping_no_returns(path: Path, *, renumber: bool) -> Path:
    """错误导出：每帧只写有回波点（`renumber` 时再把 offset_time 按点序重排为 ⌊i·10⁸/N'⌋，看上去仍然"单调、等间隔"）。"""
    with Bag1Writer(path) as bw:
        c = bw.add_connection(LIDAR_TOPIC, LIVOX_TYPES[0])
        ci = bw.add_connection(IMU_TOPIC, "sensor_msgs/Imu")
        for k in range(3):
            pts = _lidar_points(k).copy()
            if renumber:
                pts["offset_time"] = offset_time_ns(len(pts))
            t = k * LIDAR_PERIOD_NS
            bw.write(c, t, pack_custom_msg(k, t, pts))
            for i in range(20):
                ti = t + i * 5_000_000
                bw.write(ci, ti, pack_imu(k * 20 + i, ti, (0.0, 0.0, 0.0), (0.0, 0.0, -1.0)))
    return path


@pytest.mark.parametrize("renumber", [False, True])
def test_bag_without_no_return_points_is_rejected(tmp_path: Path, renumber: bool) -> None:
    """N1：删除无回波点的 bag 被合同测试拒绝（642）；逐帧检查 `check_custom_msg` 同样拒绝。"""
    p = _bag_dropping_no_returns(tmp_path / "dropped.bag", renumber=renumber)
    with pytest.raises(SensorSvcError) as e:
        verify_slam_bag(p, slots=SLOTS)
    assert e.value.code == 642 and "zero placeholders" in e.value.detail
    with Bag1Reader(p) as rd:
        msg = unpack_custom_msg(next(iter(rd.messages([LIDAR_TOPIC]))).data)
    assert msg["point_num"] == len(msg["points"]) == int(0.6 * SLOTS)
    with pytest.raises(SensorSvcError) as e:
        check_custom_msg(msg, SLOTS)
    assert e.value.code == 642


# ---------------------------------------------------------------------------------------------------------------- C4、C5、N2
def test_imu_rate_unit_and_hover_window(clean_bag: Path, rig: dict) -> None:
    """C4、C5：200 Hz、单位 g、orientation_covariance[0] = −1；事件零点之前 2 s 悬停保持窗口，空中出生开启 gravity_align_en。"""
    assert rig["imu"]["acc_unit"] == "g" and rig["imu"]["rate_hz"] == 200 and rig["imu"]["gravity_align_en"] is True
    assert rig["init"] == {"start": "pose", "mode": "hover_hold", "t_pre_s": 2.0, "min_s": 0.5}
    d = imu_contract(clean_bag, rig)
    assert d["t_ns"][0] == -T_PRE_NS and (np.diff(d["t_ns"]) == 5_000_000).all()
    win = check_init_window(d["t_ns"], d["gyro"], d["acc_g"], mode="hover_hold", t_event0_ns=0, acc_unit="g")
    assert win.ok and win.window_s == pytest.approx(2.0) and win.still_s == pytest.approx(2.0)
    assert win.mean_f_g == pytest.approx(1.0, abs=1e-9)                                 # 悬停：比力模长 1 g（单位 g）
    # 倒装：内置 IMU 的 z 轴朝下，悬停读数 z = −1 g
    assert np.allclose(d["acc_g"][d["t_ns"] < 0], [0.0, 0.0, -1.0], atol=1e-12)


def test_imu_in_mps2_is_rejected(tmp_path: Path, rig: dict) -> None:
    """N2：线加速度写成 m/s² 的 bag 被拒绝（642，悬停窗口比力均值约 9.8 "g"）。"""
    p = _write_bag(tmp_path / "mps2.bag", _synth(rig, "clean", t_to_ns=NS), acc_unit_g=False, frames=30)
    verify_slam_bag(p, slots=SLOTS)                                                      # 结构检查本身不识别单位
    with pytest.raises(SensorSvcError) as e:
        imu_contract(p, rig)
    assert e.value.code == 642 and "mean specific force" in e.value.detail


# ---------------------------------------------------------------------------------------------------------------- C6
@pytest.mark.parametrize("source", ["pose_history", "script_poses"])
def test_imu_matches_float64_reference(clean_bag: Path, rig: dict, source: str) -> None:
    """C6：clean 档 M25 输出（单位换算前）、bag 读数（× g）与经 raw.mcap f32 存档的数据集导出读数，与同算法 float64 参考实现之差
    ≤ 1e-6（m/s²、rad/s）；参考实现与未滤波插值之差（抗混叠理论误差）另行报告。`script_poses` 为采集作业路径（位姿差分运动学）。"""
    H = _history()
    R_bi, r = _mount(rig)
    t = H["t_ns"]
    if source == "pose_history":
        w, f = H["omega_b"], H["f_b"]
    else:
        q = np.array([_quat_xyzw(R) for R in H["R"]])
        w, f = kinematics_from_poses(t, H["p"], q)
    kw = {"R_bi": R_bi, "r_imu_b": r, "t_from_ns": -T_PRE_NS, "t_to_ns": T_BAG_END_NS + 5_000_000}
    s = synth_imu(t, w, f, model=load_imu_model(), tier="clean", **kw)
    tj, g_ref, a_ref = reference_imu(t, w, f, **kw)
    assert np.array_equal(tj, s.t_ns) and len(tj) == 2001
    err = {"gyro_rps": float(np.abs(s.gyro_rps - g_ref).max()), "acc_mps2": float(np.abs(s.acc_mps2 - a_ref).max())}
    assert max(err.values()) <= REF_TOL
    f32 = {"gyro_rps": float(np.abs(s.gyro_rps.astype("<f4") - g_ref).max()),          # capture._imu_block 的 f32 存档
           "acc_mps2": float(np.abs(s.acc_mps2.astype("<f4") - a_ref).max())}
    assert max(f32.values()) <= REF_TOL
    _, g_un, a_un = reference_imu(t, w, f, unfiltered=True, **kw)
    aa = {"gyro_rps": float(np.abs(g_ref - g_un).max()), "acc_mps2": float(np.abs(a_ref - a_un).max())}
    assert aa["gyro_rps"] < 0.05 and aa["acc_mps2"] < 0.5                               # 报告值（量级检查，非判据）
    if source == "pose_history":
        d = _bag_imu(clean_bag)
        assert np.array_equal(d["t_ns"], tj)
        bag = {"gyro_rps": float(np.abs(d["gyro"] - g_ref).max()), "acc_mps2": float(np.abs(d["acc_g"] * G - a_ref).max())}
        assert max(bag.values()) <= REF_TOL
        err["bag"] = bag
    print(json.dumps({"C6": source, "m25_vs_ref": err, "f32_vs_ref": f32, "aa_theoretical": aa}))


def _quat_xyzw(R: np.ndarray) -> np.ndarray:
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([x, y, z, w])


# ---------------------------------------------------------------------------------------------------------------- C7
def _ins_error(rig: dict, t_ns: np.ndarray, gyro: np.ndarray, acc: np.ndarray, start_s: float, seconds: float = 5.0) -> float:
    i0 = int(np.searchsorted(t_ns, round(start_s * NS)))
    i1 = i0 + round(seconds * 200)
    assert t_ns[i0] == round(start_s * NS) and i1 < len(t_ns)
    R0, v0, p0 = _imu_truth_state(rig, int(t_ns[i0]))
    ps, _, _ = strapdown(t_ns[i0:i1 + 1], gyro[i0:i1 + 1], acc[i0:i1 + 1], R0=R0, v0=v0, p0=p0)
    return float(np.linalg.norm(ps[-1] - _imu_truth_state(rig, int(t_ns[i1]))[2]))


def test_clean_strapdown_5s_below_5cm(clean_bag: Path, rig: dict) -> None:
    """C7：clean 档（零偏置、零噪声）bag 读数 5 s 纯惯导回代位置误差 < 5 cm（悬停窗口、加速段与匀速航段各窗口）。"""
    d = imu_contract(clean_bag, rig)
    errs = {s: _ins_error(rig, d["t_ns"], d["gyro"], d["acc_g"] * G, s) for s in INS_WINDOWS_S}
    print(json.dumps({"C7_ins_error_m": errs}))
    assert max(errs.values()) < INS_TOL_M
    # 灵敏度：realistic 档（上电偏置 0.196 m/s²、0.5 °/s）同一回代不满足 5 cm，说明判据能区分零偏置与有偏置读数
    s = _synth(rig, "realistic", seed=3)
    assert _ins_error(rig, s.t_ns, s.gyro_rps, s.acc_mps2, 3.0) > INS_TOL_M


# ---------------------------------------------------------------------------------------------------------------- C8
def test_dynamics_assertion(clean_bag: Path, rig: dict) -> None:
    """C8：PoseHistory 与 bag 读数 |ω| < 5 rad/s、|a| < 30 m/s²；起飞速度阶跃与姿态跳变的轨迹在 SLAM 导出前被拒绝（642，标注时间段）。"""
    H = _history()
    rep = assert_dynamics(H["t_ns"], H["omega_b"], H["f_b"])
    assert rep.ok and rep.max_omega_rps < DYN_OMEGA_MAX_RPS and rep.max_acc_mps2 < DYN_ACC_MAX_MPS2
    br = verify_slam_bag(clean_bag, slots=SLOTS)
    assert br.gyro_max_rps < DYN_OMEGA_MAX_RPS and br.acc_max_g * G < DYN_ACC_MAX_MPS2
    # 违例轨迹：t = 1 s 竖直速度 0 → 3 m/s 阶跃（二阶导不连续），t = 2 s 横滚一步跳变 47°（r1 §4 的起飞横滚跳变）
    t = np.arange(3 * PHYS_HZ + 1, dtype=np.int64) * (NS // PHYS_HZ)
    ts = t / NS
    pos = np.stack([np.zeros_like(ts), np.zeros_like(ts), np.where(ts > 1.0, 3.0 * (ts - 1.0), 0.0)], 1)
    roll = np.where(ts >= 2.0, math.radians(47.0), 0.0)
    q = np.stack([np.sin(roll / 2), np.zeros_like(ts), np.zeros_like(ts), np.cos(roll / 2)], 1)
    w, f = kinematics_from_poses(t, pos, q)
    bad = check_dynamics(t, w, f)
    assert not bad.ok and len(bad.segments) == 2
    (s1, s2) = bad.segments
    assert s1["t0_ns"] <= NS <= s1["t1_ns"] + 8_000_000 and s1["max_acc_mps2"] >= DYN_ACC_MAX_MPS2
    assert s2["t0_ns"] <= 2 * NS <= s2["t1_ns"] + 8_000_000 and s2["max_omega_rps"] >= DYN_OMEGA_MAX_RPS
    with pytest.raises(SensorSvcError) as e:
        assert_dynamics(t, w, f)
    assert e.value.code == 642 and e.value.extra["segments"] == bad.segments


# ---------------------------------------------------------------------------------------------------------------- C10
SLAM_CAM_SPEC = {"name": "slam_cam", "profile": "rgb720-gs", "size": [640, 512], "hfov_deg": 80.0, "fps": "10", "exposure_s": 0.008,
                 "readout_s": 0.0, "mount_xyz_m": [0.12, 0.0, -0.08], "mount_rpy_deg": [0.0, 45.0, 0.0], "gimbal": {"az_deg": 0.0, "el_deg": 0.0},
                 "outputs": ["rgb"]}


def slam_cam_violations(records: list[dict], rig: dict, *, px_tol: float = 0.01) -> list[str]:
    """C10 飞行中部分：逐帧 frame.v1 记录（agent_view + ground_truth）的内参、畸变、分辨率、挂载、云台（真值与读数）与曝光在运行中
    恒定且与 rig.json 一致，读出时间为 0（全局快门，无卷帘字段）。返回违例列表。"""
    c = rig["camera"]
    bad: list[str] = []
    first = records[0]["agent_view"], records[0]["ground_truth"]
    want_K = [c["fx"], 0.0, c["cx"], 0.0, c["fy"], c["cy"], 0.0, 0.0, 1.0]
    if not np.allclose(first[0]["K"], want_K, atol=px_tol, rtol=0.0) or (first[0]["width"], first[0]["height"]) != (c["width"], c["height"]):
        bad.append("intrinsics differ from rig.json")
    if first[0]["mount_nominal"] != {"xyz_m": c["T_base_cam"]["xyz_m"], "rpy_deg": c["T_base_cam"]["rpy_deg"]}:
        bad.append("mount differs from rig.json")
    if first[0]["exposure_ns"] != c["exposure_ns"]:
        bad.append("exposure differs from rig.json")
    for rec in records:
        av, gt = rec["agent_view"], rec["ground_truth"]
        seq = av["seq"]
        if av["readout_ns"] != 0 or "rs_nodes" in gt["timing"]:
            bad.append(f"frame {seq}: rolling-shutter readout")
        bad += [f"frame {seq}: {key} changed in flight"
                for key in ("K", "dist", "width", "height", "pixel_convention", "mount_nominal", "exposure_ns", "gimbal_reading")
                if av[key] != first[0][key]]
        if gt["gimbal_true"] != first[1]["gimbal_true"]:
            bad.append(f"frame {seq}: camera-to-body rotation changed in flight")
    return bad


def _capture_slam_cam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tier: str) -> list[dict]:
    """以 `slam_cam` 配置运行采集作业（渲染以桩替代），读回真值文件中逐帧 frame.v1 记录。"""
    pytest.importorskip("PIL")
    pytest.importorskip("mcap_protobuf")
    import awr.sensorsvc.camera.raster as raster
    import awr.sensorsvc.camera.scene as scene
    from awr.bench.recorder.read import TruthStore
    from awr.bench.recorder.truth import unpack_camera_gt
    from awr.bench.runs_store import truth_paths
    from awr.sensorsvc import capture as cap

    monkeypatch.setattr(raster, "RasterCamera", _StubCamera)
    monkeypatch.setattr(scene, "load_appearance", lambda d: None)
    w = tmp_path / "worlds" / "w1"
    w.mkdir(parents=True)
    (w / "world.json").write_text(json.dumps({"contentVersion": "abc123def456"}))
    spec = {"run_id": f"slamcam-{tier}", "event_id": "capture", "world_id": "w1", "duration_s": 1.0, "noise_profile": tier, "seed": 5,
            "fleet": [{"uav": "uav-01", "path": {"kind": "orbit", "center": [0, 0], "radius_m": 60, "agl_m": 30, "period_s": 60},
                       "cameras": [SLAM_CAM_SPEC]}]}
    m = cap.run_capture(spec, tmp_path, log=lambda s: None)
    assert m["frames"] == 10
    ts = TruthStore(*truth_paths(tmp_path / f"dataset/capture/runs/slamcam-{tier}"))
    try:
        return [unpack_camera_gt(ts.get("uav-01", "slam_cam", k)[1])[0] for k in range(1, 11)]
    finally:
        ts.close()


class _StubCamera:
    """GPU 渲染的替身（同 tests/d3/sensorsvc/test_capture.py）：常量线性图经效果链参考实现得到交付 RGB。"""

    def __init__(self, scene) -> None:
        self.info = {"GL_RENDERER": "stub", "GL_VERSION": "4.5.0 stub 1.0"}

    def set_actors(self, actors) -> None:
        pass

    def set_sun(self, az_el) -> None:
        pass

    def render(self, cm, R, pos, *, linear: bool = False, fx=None, rs_nodes=None, actors=None):
        from awr.sensorsvc.camera.effects import apply_fx_reference
        from awr.sensorsvc.camera.raster import RenderOut

        h, w = cm.height, cm.width
        lin = np.full((h, w, 3), 0.1, np.float32)
        depth = np.full((h, w), float(pos[2]), np.float32)
        rgb, d_obs = apply_fx_reference(lin, depth, cm, fx)
        z = np.zeros((h, w), np.uint16)
        return RenderOut(rgb, depth, z, np.zeros((h, w), np.uint8), 1.0, 1.0, 1, depth_obs=d_obs,
                         meter_sub=lin[::8, ::8] if fx.meter else None, instance_unocc=None)

    def release(self) -> None:
        pass


def test_slam_cam_global_shutter_constant_intrinsics_extrinsics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rig: dict) -> None:
    """C10：rig 中 slam_cam 为全局快门、无云台、固定曝光；clean 档飞行中逐帧内参、外参与曝光恒定且与 rig 一致，读出为 0；
    篡改内参、挂载、云台或改为卷帘的记录序列被检出。"""
    c = rig["camera"]
    assert (c["shutter"], c["readout_s"], c["gimbal"], c["exposure_mode"]) == ("global", 0.0, False, "fixed")
    recs = _capture_slam_cam(tmp_path, monkeypatch, "clean")
    assert slam_cam_violations(recs, rig) == []
    assert {r["agent_view"]["readout_ns"] for r in recs} == {0}
    for mutate, what in ((lambda av, gt: av["K"].__setitem__(0, av["K"][0] * 1.001), "K changed"),
                         (lambda av, gt: av["mount_nominal"].__setitem__("rpy_deg", [0.0, 46.0, 0.0]), "mount_nominal changed"),
                         (lambda av, gt: gt.__setitem__("gimbal_true", {**gt["gimbal_true"], "pitch_deg": 1.0}), "rotation changed"),
                         (lambda av, gt: av.__setitem__("readout_ns", 20_000_000), "rolling-shutter")):
        bad = json.loads(json.dumps(recs))
        mutate(bad[5]["agent_view"], bad[5]["ground_truth"])
        assert any(what in v for v in slam_cam_violations(bad, rig)), what


@pytest.mark.xfail(reason="capture adds gimbal encoder noise to every camera; AWR-20 §15.6.5 limits it to cam0 (S19 follow-up)", strict=False)
def test_slam_cam_gimbal_reading_constant_in_realistic_tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rig: dict) -> None:
    """C10（realistic 档）：slam_cam 无云台，读数不应带编码器噪声；采集作业目前对所有相机加噪声，待修正后转为通过。"""
    recs = _capture_slam_cam(tmp_path, monkeypatch, "realistic")
    assert slam_cam_violations(recs, rig) == []


# ---------------------------------------------------------------------------------------------------------------- C11
TABLE_15_6_5 = {  # AWR-20 §15.6.5 MID-360 内置 IMU 行：(acc sigma g、gyro sigma rad/s、acc rw g/√s、gyro rw rad/s/√s、上电 m/s²、上电 °/s、振动 m/s²)
    "clean": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "realistic": (1e-3, 7e-4, 1e-5, 1e-5, 0.196, 0.5, 0.0),
    "harsh": (1e-3, 7e-4, 1e-5, 1e-5, 0.196, 0.5, 1.5),
}
_TABLE_KEYS = ("acc_sigma_sample_g", "gyro_sigma_sample_rps", "acc_bias_rw_g_rts", "gyro_bias_rw_rps_rts", "turn_on_acc_mps2", "turn_on_gyro_dps",
               "vib_amp_mps2")


def _same(a: dict, b: dict) -> list[str]:
    out = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        same = (x == y) if not isinstance(x, float) or not isinstance(y, float) else math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-15)
        if not same:
            out.append(f"{k}: {x!r} != {y!r}")
    return out


def test_noise_parameters_match_the_table(cat, rig: dict) -> None:
    """C11：IMU 合成使用的型号参数、参数表（`imu_noise_tier`，AWR-20 §15.6.5）与 rig.json 噪声段逐项一致；量程与振动频率一致。"""
    imu_doc = cat.overlay.sensor_model(rig["imu"]["model"])[0]
    assert imu_doc["noise_table_version"] == "d3-noise-1" and imu_doc["model_id"] == "imu_icm40609_livox"
    bpf = float(cat.overlay.variant(VID).doc["acoustic"]["bpf_hz"])
    assert bpf == DEFAULT_BPF_HZ == 150.0
    m = load_imu_model()
    assert m == ImuModel() and m.model_id == "imu_icm40609_livox" and m.noise_table_version == imu_doc["noise_table_version"]
    assert m.rate_hz == 200 and m.aa_cutoff_hz == 80.0 and m.vib_hz == bpf
    assert math.isclose(ACC_SAT, imu_doc["range"]["acc_g"] * G) and math.isclose(GYRO_SAT, math.radians(imu_doc["range"]["gyro_dps"]))
    for tier in TIERS:
        table = imu_noise_tier(imu_doc, tier, bpf_hz=bpf)
        assert _same(m.table(tier), table) == [], tier
        assert tuple(table[k] for k in _TABLE_KEYS) == pytest.approx(TABLE_15_6_5[tier], rel=1e-12, abs=0.0), tier
        r = rig_for_vehicle(VID, vehicles_dir=ROOT / "vehicles", noise_tier=tier)
        assert r["noise"]["noise_table_version"] == m.noise_table_version
        assert _same(m.table(tier), r["noise"]["imu"]) == [], tier
        if tier != "clean":
            assert ImuModel.from_noise(r["noise"]["imu"]) == m
    with pytest.raises(ValueError):
        ImuModel.from_noise(imu_noise_tier(imu_doc, "clean"))


def _hover(seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(round(seconds * PHYS_HZ) + 1, dtype=np.int64) * (NS // PHYS_HZ)
    return t, np.zeros((len(t), 3)), np.tile([0.0, 0.0, G], (len(t), 1))


def test_noise_statistics_follow_the_parameters() -> None:
    """C11（样本统计）：realistic 白噪声 sigma、上电偏置 sigma、偏置随机游走步长与参数一致；harsh = realistic + 加速度三轴振动
    （幅值 1.5 m/s²、频率 = 机型 bpf，陀螺不变）；clean 等于真值。"""
    m = load_imu_model()
    t, w, f = _hover(60.0)
    kw = {"R_bi": np.eye(3), "r_imu_b": np.zeros(3), "t_from_ns": 0, "t_to_ns": 60 * NS, "seed": 11, "uav_no": 2}
    c = synth_imu(t, w, f, model=m, tier="clean", **kw)
    assert np.array_equal(c.acc_mps2, c.acc_true) and np.array_equal(c.gyro_rps, c.gyro_true)
    a = synth_imu(t, w, f, model=m, tier="realistic", **kw)
    for name, x, sigma in (("gyro", a.gyro_rps, m.gyro_sigma), ("acc", a.acc_mps2, m.acc_sigma)):
        got = np.std(np.diff(x, axis=0), axis=0) / math.sqrt(2)                       # 差分去掉慢变偏置
        assert got == pytest.approx([sigma] * 3, rel=0.04), name
    # 偏置随机游走：只保留随机游走项，相邻样本增量 sigma = rw·sqrt(Δt)，b(0) = 0
    rw_only = replace(m, gyro_sigma=0.0, acc_sigma=0.0, gyro_turn_on=0.0, acc_turn_on=0.0)
    b = synth_imu(t, w, f, model=rw_only, tier="realistic", **kw)
    bias = np.hstack([b.gyro_rps - b.gyro_true, b.acc_mps2 - b.acc_true])
    assert np.abs(bias[0]).max() == 0.0
    step = np.std(np.diff(bias, axis=0), axis=0)
    assert step == pytest.approx([m.gyro_rw * math.sqrt(0.005)] * 3 + [m.acc_rw * math.sqrt(0.005)] * 3, rel=0.04)
    # 上电偏置：每运行抽一次（400 个种子 × 3 轴），j = 0 样本 = 真值 + 上电偏置 + 白噪声（白噪声占方差 < 0.7%）
    tt, ww, ff = _hover(0.02)
    on = np.array([np.hstack([s.gyro_rps[0] - s.gyro_true[0], s.acc_mps2[0] - s.acc_true[0]])
                   for s in (synth_imu(tt, ww, ff, model=m, tier="realistic", **{**kw, "t_to_ns": 10_000_000, "seed": k}) for k in range(400))])
    assert np.std(on[:, :3]) == pytest.approx(m.gyro_turn_on, rel=0.08)
    assert np.std(on[:, 3:]) == pytest.approx(m.acc_turn_on, rel=0.08)
    # harsh：陀螺与 realistic 逐位相同，加速度差为 A·sin(2π f t + φ_axis)（按输出采样时刻叠加，200 Hz 采样含混叠）
    h = synth_imu(t, w, f, model=m, tier="harsh", **kw)
    assert np.array_equal(h.gyro_rps, a.gyro_rps)
    vib = h.acc_mps2 - a.acc_mps2
    ts = h.t_ns / NS
    B = np.stack([np.sin(2 * math.pi * m.vib_hz * ts), np.cos(2 * math.pi * m.vib_hz * ts)], 1)
    for k in range(3):
        coef, *_ = np.linalg.lstsq(B, vib[:, k], rcond=None)
        assert math.hypot(*coef) == pytest.approx(m.vib_amp_mps2, abs=1e-9)
        assert np.abs(B @ coef - vib[:, k]).max() < 1e-9
    assert (h.flags == 0).all() and (a.flags == 0).all()


def test_preroll_bias_walk_negative_grid_indices() -> None:
    """预滚窗口（负网格序号，S14 后续 3）：偏置随机游走以 j = 0 为原点向两侧延伸，不回绕；跨零点与分批调用逐位一致。"""
    m = replace(load_imu_model(), gyro_sigma=0.0, acc_sigma=0.0, gyro_turn_on=0.0, acc_turn_on=0.0)
    t = np.arange(-2 * PHYS_HZ, 2 * PHYS_HZ + 1, dtype=np.int64) * (NS // PHYS_HZ)
    w, f = np.zeros((len(t), 3)), np.tile([0.0, 0.0, G], (len(t), 1))
    kw = {"R_bi": np.eye(3), "r_imu_b": np.zeros(3), "model": m, "tier": "realistic", "seed": 4, "uav_no": 1}
    full = synth_imu(t, w, f, t_from_ns=-2 * NS, t_to_ns=2 * NS, **kw)
    assert full.t_ns[0] == -2 * NS and len(full.t_ns) == 800
    neg = synth_imu(t, w, f, t_from_ns=-2 * NS, t_to_ns=0, **kw)
    pos = synth_imu(t, w, f, t_from_ns=0, t_to_ns=2 * NS, **kw)
    mid = synth_imu(t, w, f, t_from_ns=-NS // 2, t_to_ns=NS // 2, **kw)
    assert np.array_equal(np.vstack([neg.acc_mps2, pos.acc_mps2]), full.acc_mps2)
    assert np.array_equal(mid.gyro_rps, full.gyro_rps[300:500])
    bias = full.acc_mps2 - full.acc_true
    i0 = int(np.flatnonzero(full.t_ns == 0)[0])
    sd = m.acc_rw * math.sqrt(0.005)
    assert np.abs(bias[i0]).max() == 0.0                                                 # b(0) = 0
    d = np.diff(bias, axis=0)
    assert np.abs(d).max() < 6 * sd                                                      # 两侧均为连续随机游走（无回绕跳变）
    assert np.std(d[:i0]) == pytest.approx(sd, rel=0.1) and np.std(d[i0:]) == pytest.approx(sd, rel=0.1)
