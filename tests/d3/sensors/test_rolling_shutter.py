"""卷帘快门几何验收（D3-AC-60 卷帘条款；docs/05 §5.4.3 卷帘快门、ADR-179；M25 §15.4）。

阈值逐条对照 docs/05 §11.2 D3-AC-60：`rgb1080-rs`（1920 × 1080，读出 20 ms）在 8 m/s 平移叠加 90°/s 偏航的构造轨迹上，
逐顶点方法渲染的已知标记顶点像素位置相对 Warp 逐行射线参照的误差 p99 ≤ 0.5 px，行带方法（16 带）误差 ≤ 1.5 px；深度、实例、
语义与 RGB 的边缘对齐误差 ≤ 0.5 px；`CameraObs.t_mid_ns` 为中间行曝光中点、`readout_ns = 20,000,000`；静止机体时卷帘与全局快门
输出逐字节一致。

构造场景：400 m × 400 m 纹理地面（4 m 网格）上方悬浮的标记盒（1.6 m × 1.6 m × 0.4 m，`object` 类演员代理网格，底面离地 0.8 m，
朝向随机），按 6 m 网格铺满 t_mid 视场。三条构造轨迹：斜视 45° 前飞、斜视 45° 侧飞（反向偏航）、正下视对角飞行，均为 8 m/s 平移
叠加 ±90°/s 偏航（相机位姿含云台，轨迹为解析函数）。标记位置以标记掩膜质心度量（标记顶点位置的面积加权平均，像素中心采样
在两种渲染中相同）；只统计三种渲染中都完整位于画面内且面积 ≥ 120 px 的标记。p99 按全部轨迹的标记合并计算，并逐轨迹另行断言。

GPU 用例需要 moderngl、EGL 与 warp-lang（d3-sensorsvc 镜像，dmax 单卡绑定）；本机只运行 CPU 用例（Warp 用 CPU 设备）。
环境变量 `AWR_RS_REPORT=<path>` 时把逐轨迹度量写成 JSON（验收记录引用）。
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "world"))
from glbfix import png

from awr.sensorsvc.camera import CameraModel
from awr.sensorsvc.camera import rowref as rr
from awr.sensorsvc.camera.model import CAMERA_PROFILES
from awr.sensorsvc.camera.proj import gimbal_R, project_points

PROFILE = CAMERA_PROFILES["rgb1080-rs"]
T_RO = PROFILE["readout_s"]
HFOV = 84.0
SPEED, YAW_RATE = 8.0, 90.0                     # m/s，°/s
P99_VERTEX, MAX_BANDS, EDGE = 0.5, 1.5, 0.5     # px（D3-AC-60）
MARK_MM = [1600, 1600, 400]
MARK_Z, MARK_STEP, MIN_AREA = 0.8, 6.0, 120
SEM_OBJECT = 37
GROUND_RGB = (90, 120, 90)

# 构造轨迹：高度、云台俯仰、t_mid 偏航、平移方向（相对 t_mid 偏航）、偏航角速度
CASES = {
    "oblique-forward": {"alt": 30.0, "pitch": -45.0, "yaw0": 20.0, "heading": 0.0, "rate": YAW_RATE},
    "oblique-lateral": {"alt": 30.0, "pitch": -45.0, "yaw0": -35.0, "heading": 90.0, "rate": -YAW_RATE},
    "nadir-diagonal": {"alt": 40.0, "pitch": -90.0, "yaw0": 0.0, "heading": 45.0, "rate": YAW_RATE},
}


def cam_model(w: int = 1920, h: int = 1080, readout_s: float = T_RO) -> CameraModel:
    return CameraModel.from_hfov(w, h, HFOV, readout_s=readout_s, exposure_s=PROFILE["exposure_s"])


def trajectory(c: dict, *, speed: float = SPEED):
    """相对 t_mid 的时刻 t（s）→ (R_ws, p_w)：匀速平移叠加匀角速度偏航（相机位姿含云台俯仰）。"""
    hd = math.radians(c["yaw0"] + c["heading"])
    v = speed * np.array([math.cos(hd), math.sin(hd), 0.0])
    p0 = np.array([0.0, 0.0, c["alt"]])
    return lambda t: (gimbal_R(c["yaw0"] + c["rate"] * t, c["pitch"]), p0 + v * t)


def ground_scene(half: float = 200.0, step: float = 4.0):
    from awr.sensorsvc.camera.scene import AppearanceScene, Batch

    xs = np.arange(-half, half + 1e-9, step)
    xx, yy = np.meshgrid(xs, xs)
    V = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], 1)
    n = len(xs)
    a = (np.arange(n - 1)[:, None] * n + np.arange(n - 1)[None, :]).ravel()
    F = np.concatenate([np.stack([a, a + 1, a + n + 1], 1), np.stack([a, a + n + 1, a + n], 1)])
    UV = ((V[:, :2] + half) / (2 * half)).astype(np.float32)
    b = Batch((0, 0), np.zeros(3), 0, V.astype(np.float32), UV, F.astype(np.uint32), np.arange(len(F), dtype=np.uint32))
    return AppearanceScene("rs-fixture", [b], [("image/png", png(8, 8, GROUND_RGB))], len(F), V.min(0), V.max(0))


def markers(cm: CameraModel, pose_at, seed: int = 60):
    """t_mid 视场内（四周留 60 px，覆盖读出窗口内的位移）的 6 m 网格标记：返回 (演员表, ActorFrame)。"""
    from awr.sensorsvc.camera.raster import ActorFrame

    g = np.arange(-150.0, 150.0 + 1e-9, MARK_STEP)
    gx, gy = np.meshgrid(g, g)
    X = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, MARK_Z + 0.2)], 1)
    uv, z = project_points(cm, *pose_at(0.0), X)
    ok = (z > 2.0) & (uv[:, 0] > 60) & (uv[:, 0] < cm.width - 60) & (uv[:, 1] > 60) & (uv[:, 1] < cm.height - 60)
    P = X[ok]
    P[:, 2] = MARK_Z
    yaw = np.random.default_rng(seed).uniform(0.0, math.pi / 2, len(P))
    actors = [{"idx": i, "id": f"mk{i:04d}", "class": "object", "size_mm": MARK_MM, "appearance": {"tint_rgb": [230, 40, 40]}}
              for i in range(len(P))]
    return actors, ActorFrame(np.arange(len(P)), P, yaw)


def _q(a: np.ndarray, q: float) -> float:
    return float(np.quantile(a, q)) if len(a) else float("nan")


# ---------------------------------------------------------------- CPU：时间戳、参照与度量本身


def test_profile_readout_and_mid_row_timestamp() -> None:
    """`rgb1080-rs`：readout_ns = 20,000,000；t_mid 为中间行曝光中点（中间两行关于 t_mid 对称），首末行跨读出窗口。"""
    from awr.sensorsvc.timing import row_mid_ns, t_mid_ns

    W, H = PROFILE["size"]
    ro, e = round(PROFILE["readout_s"] * 1e9), round(PROFILE["exposure_s"] * 1e9)
    assert ro == 20_000_000 and (W, H) == (1920, 1080)
    t_end = 1_000_000_000
    tm = t_mid_ns(t_end, e, ro)
    assert tm == t_end - (e + ro) // 2
    assert row_mid_ns(tm, ro, H, (H - 1) / 2) == tm
    assert row_mid_ns(tm, ro, H, H // 2 - 1) - tm == -(row_mid_ns(tm, ro, H, H // 2) - tm)
    assert abs(row_mid_ns(tm, ro, H, H - 1) - row_mid_ns(tm, ro, H, 0) - ro * (H - 1) / H) <= 1
    t = rr.row_times_s(cam_model(), T_RO)
    assert t[0] == pytest.approx(-T_RO / 2 + T_RO / (2 * H)) and t[-1] == pytest.approx(T_RO / 2 - T_RO / (2 * H))
    assert np.allclose(np.round(t * 1e9) + tm, [row_mid_ns(tm, ro, H, i) for i in range(H)], atol=1)


def test_camera_obs_timing_fields_for_rgb1080_rs() -> None:
    """frame.v1：`agent_view.t_mid_ns`、`readout_ns = 20,000,000`，真值 timing 记录 9 个节点与读出窗口。"""
    pytest.importorskip("zstandard")
    pytest.importorskip("PIL")
    from awr.sensorsvc.camera.encode import CpuJpegEncoder
    from awr.sensorsvc.camera.raster import RenderOut
    from awr.sensorsvc.frame import FrameContext, camera_frame
    from awr.sensorsvc.poses import Mount
    from awr.sensorsvc.timing import t_mid_ns

    cm = cam_model()
    H, W = cm.height, cm.width
    out = RenderOut(np.zeros((H, W, 3), np.uint8), np.full((H, W), 50.0, np.float32), np.zeros((H, W), np.uint16),
                    np.zeros((H, W), np.uint8), 1.0, 1.0, 1)
    ro, e = round(cm.readout_s * 1e9), round(cm.exposure_s * 1e9)
    tm = t_mid_ns(300_000_000, e, ro)
    rec, _ = camera_frame(FrameContext("run-1", "sess-1", "v", "gpu", "drv"), uav="u1", sensor_id="cam0", seq=3, t_mid_ns=tm,
                          exposure_ns=e, readout_ns=ro, cm=cm, mount=Mount(), gimbal_reading=(0.0, -0.5, 0.0), gimbal_true=(0.0, -0.5, 0.0),
                          pos_enu_m=np.array([0.0, 0.0, 30.0]), R_ws=gimbal_R(0.0, -30.0), out=out, outputs={"rgb"},
                          rgb=b"\xff\xd8\xff\xd9", encoder=CpuJpegEncoder())
    av, tg = rec["agent_view"], rec["ground_truth"]["timing"]
    assert av["readout_ns"] == 20_000_000 and av["t_mid_ns"] == tm == 300_000_000 - (1_000_000 + 20_000_000) // 2
    assert tg["rs_nodes"] == 9 and tg["readout_window_ns"] == [tm - 10_000_000, tm + 10_000_000] and tg["t_truth_ns"] == [tm]


def test_project_rs_fixed_point_and_global_limit() -> None:
    """逐点卷帘投影满足不动点方程；读出为 0 或静止时与全局快门投影一致；偏航使上下两行看到的同一竖线错开。"""
    cm = cam_model()
    c = CASES["oblique-forward"]
    pose = trajectory(c)
    X = np.array([[30.0, 12.0, 0.0], [45.0, -8.0, 1.0], [25.0, 0.0, 0.5]])
    uv, z = rr.project_rs(cm, pose, T_RO, X)
    H = cm.height
    for (u, v), x in zip(uv, X, strict=True):
        uv1, _ = project_points(cm, *pose((v - H / 2) * T_RO / H), x[None])
        assert np.allclose(uv1[0], (u, v), atol=1e-6)
    uv0, _ = rr.project_rs(cm, pose, 0.0, X)
    assert np.allclose(uv0, project_points(cm, *pose(0.0), X)[0], atol=1e-9)
    assert np.allclose(rr.project_rs(cm, lambda t: pose(0.0), T_RO, X)[0], project_points(cm, *pose(0.0), X)[0], atol=1e-9)
    assert np.abs(uv - uv0).max() > 3.0 and np.all(z > 0)


def test_label_metrics_on_synthetic_masks() -> None:
    """质心与边缘偏移度量：整像素平移 1 列 → 质心差 1 px、边缘偏移 = 2·高 / 内边界像素数；同图为 0；触边标记被标出。"""
    a = np.zeros((40, 60), np.uint16)
    a[10:20, 10:30] = 1
    a[25:35, 40:50] = 2
    a[0:5, 0:5] = 3
    b = np.roll(a, 1, axis=1)
    ids = np.array([1, 2])
    assert np.allclose(rr.centroid_errors(a, b, ids), 1.0)
    st = rr.label_stats(a, 4)
    assert st["area"][1] == 200 and st["cx"][1] == pytest.approx(20.0) and st["cy"][1] == pytest.approx(15.0)
    assert st["boundary"][1] == 2 * 20 + 2 * 8 and st["border"][3] and not st["border"][1]
    assert np.allclose(rr.edge_offsets(a, b, ids), [2 * 10 / 56, 2 * 10 / 36])
    assert np.all(rr.edge_offsets(a, a, ids) == 0) and np.all(rr.centroid_errors(a, a, ids) == 0)
    z = rr.dilate_labels(a, 2)
    assert (z[8:22, 8:32] == 1).all() and z[30, 30] == 0


def test_warp_reference_rows_follow_exact_row_poses() -> None:
    """Warp 逐行射线（CPU 设备即可）：地面平面深度与逐行精确位姿的解析值一致，且与只取 t_mid 位姿的解析值明显不同。

    水平平移与绕竖直轴偏航不改变到水平面的平面深度，因此这里叠加 30 m/s 下降，使逐行深度随行时刻变化。"""
    pytest.importorskip("warp")
    cm = CameraModel.from_hfov(192, 108, HFOV, readout_s=T_RO)
    base = trajectory(CASES["oblique-forward"], speed=40.0)

    def pose(t: float):
        R, p = base(t)
        return R, p + np.array([0.0, 0.0, -30.0 * t])
    sc = ground_scene()
    ref = rr.RowRayReference.from_scene(sc, origin=pose(0.0)[1])
    out = ref.render(cm, pose, T_RO)
    want = rr.plane_depth(cm, pose, T_RO)
    hit = out.face >= 0
    assert hit.mean() > 0.99 and np.array_equal(hit, want > 0)
    assert np.abs(out.depth_m[hit] - want[hit]).max() <= 1e-4 * want[hit].max()
    assert (out.instance == 0).all() and (out.semantic[hit] == 0).all() and (out.semantic[~hit] == rr.SEM_SKY).all()
    gs = rr.plane_depth(cm, pose, 0.0)
    assert np.abs(gs[hit] - want[hit]).max() > 0.5                                   # 读出窗口内下降 0.6 m
    tris, inst, sem = rr.actor_triangles([{"class": "object", "size_mm": MARK_MM}], np.array([0]), np.array([[30.0, 5.0, 0.8]]),
                                         np.array([0.3]))
    assert tris.shape == (16, 3, 3) and set(inst) == {1} and set(sem) == {SEM_OBJECT}
    assert tris[..., 2].min() == pytest.approx(0.8) and tris[..., 2].max() == pytest.approx(1.2)


def test_rs_static_detects_static_readout_window() -> None:
    """静止判定：PoseHistory 恒定样本插值得到的 9 个节点判为静止；8 m/s 或 90°/s 时不判为静止。"""
    from awr.sensorsvc.camera.raster import rs_static

    pose = trajectory(CASES["oblique-lateral"])
    assert not rs_static(rr.rs_nodes(pose, T_RO))
    assert not rs_static(rr.rs_nodes(lambda t: (gimbal_R(10.0 + 90.0 * t, -45.0), np.zeros(3)), T_RO))
    assert not rs_static(rr.rs_nodes(lambda t: (gimbal_R(10.0, -45.0), np.array([8.0 * t, 0.0, 30.0])), T_RO))
    assert rs_static(rr.rs_nodes(lambda t: pose(0.0), T_RO))
    assert rs_static(_static_history_nodes()[0])


def _static_history_nodes():
    """悬停机体：PoseHistory 250 Hz 恒定样本、云台恒定，按 capture 的方式取读出窗口内 9 个节点。"""
    from awr.sensorsvc.poses import Mount, PoseHistory, sensor_pose

    Hs = PoseHistory("u1")
    yaw = math.radians(37.0)
    q = np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
    for k in range(51):
        Hs.append(k * 4_000_000, [12.5, -7.25, 31.0], q)
        Hs.append_gimbal(0, k * 4_000_000, 0.0, math.radians(-50.0))
    mount = Mount((0.1, 0.0, -0.08), (0.0, 0.0, 0.0))
    tm, ro = 100_000_000, 20_000_000
    nodes = []
    for k in range(9):
        p, R, _ = sensor_pose(Hs, mount, 0, tm - ro // 2 + (k * ro) // 8)
        nodes.append((R, p))
    p, R, _ = sensor_pose(Hs, mount, 0, tm)
    return nodes, (R, p)


# ---------------------------------------------------------------- GPU：光栅化对逐行射线参照


@pytest.fixture(scope="module")
def rig():
    moderngl = pytest.importorskip("moderngl")
    pytest.importorskip("PIL")
    pytest.importorskip("warp")
    try:
        ctx = moderngl.create_standalone_context(backend="egl", require=450)
    except Exception as e:                                   # 无 EGL 设备
        pytest.skip(f"EGL unavailable: {e}")
    from awr.sensorsvc.camera.raster import RasterCamera

    sc = ground_scene()
    cam = RasterCamera(sc, ctx=ctx)
    yield cam, sc
    cam.release()


def _labels_from_channels(r, zone: np.ndarray, plane: np.ndarray) -> dict[str, np.ndarray]:
    """光栅化各通道 → 标记图：实例直接取实例号；语义、RGB、深度按各自判据得到二值掩膜，再归属到参照标记的膨胀区。"""
    rgb = r.rgb8.astype(np.int16)
    return {"instance": r.instance.astype(np.int64),
            "semantic": np.where(r.semantic == SEM_OBJECT, zone, 0),
            "rgb": np.where(rgb[..., 0] > rgb[..., 1] + 40, zone, 0),
            "depth": np.where((r.depth_m > 0) & (r.depth_m < plane - 0.3), zone, 0)}


def _run_case(cam, sc, name: str) -> dict:
    c = CASES[name]
    cm = cam_model()
    pose = trajectory(c)
    actors, af = markers(cm, pose)
    cam.set_actors(actors)
    vx = cam.render(cm, *pose(0.0), rs_nodes=rr.rs_nodes(pose, T_RO), actors=af)
    bd = rr.render_bands(cam, cm, pose, T_RO, bands=rr.RS_BANDS_REF, actors=af)
    gs = cam.render(cm, *pose(0.0), actors=af)
    ref = rr.RowRayReference.from_scene(sc, actors, af, origin=pose(0.0)[1]).render(cm, pose, T_RO)
    n = len(actors) + 1
    L = {"ref": ref.instance.astype(np.int64), "vertex": vx.instance.astype(np.int64), "bands": bd.instance.astype(np.int64),
         "global": gs.instance.astype(np.int64)}
    st = {k: rr.label_stats(v, n) for k, v in L.items()}
    keep = np.ones(n, bool)
    keep[0] = False
    for s in st.values():
        keep &= (s["area"] >= MIN_AREA) & ~s["border"]
    ids = np.flatnonzero(keep)
    plane = rr.plane_depth(cm, pose, T_RO, z0=0.0)
    ground = (ref.instance == 0) & (ref.face >= 0)
    zone = rr.dilate_labels(ref.instance.astype(np.int64), 4)
    ch = _labels_from_channels(vx, zone, plane)
    e = {k: rr.centroid_errors(L["ref"], L[k], ids) for k in ("vertex", "bands", "global")}
    edge_ref = {k: rr.edge_offsets(L["ref"], v, ids) for k, v in ch.items()}
    edge_inter = {k: rr.edge_offsets(ch["instance"], v, ids) for k, v in ch.items() if k != "instance"}
    return {"case": name, "markers": len(actors), "measured": len(ids), "device": ref.device, "trace_ms": round(ref.trace_ms, 2),
            "err": e, "edge_ref": edge_ref, "edge_inter": edge_inter,
            "ref_plane_max_rel": float(np.abs(ref.depth_m[ground] - plane[ground]).max() / plane[ground].max()),
            "area_median": float(np.median(st["ref"]["area"][ids]))}


@pytest.fixture(scope="module")
def results(rig):
    cam, sc = rig
    res = {k: _run_case(cam, sc, k) for k in CASES}
    path = os.environ.get("AWR_RS_REPORT")
    if path:
        def summ(a: np.ndarray) -> dict:
            return {"p50": round(_q(a, 0.5), 4), "p99": round(_q(a, 0.99), 4), "max": round(float(a.max()), 4)}

        rep = {"ac": "D3-AC-60", "camera": {"size": PROFILE["size"], "hfov_deg": HFOV, "readout_s": T_RO}, "speed_mps": SPEED,
               "yaw_rate_dps": YAW_RATE, "bands": rr.RS_BANDS_REF, "cases": {}}
        for k, r in res.items():
            rep["cases"][k] = {"markers": r["markers"], "measured": r["measured"], "device": r["device"], "trace_ms": r["trace_ms"],
                               "area_median_px": r["area_median"], "ref_plane_max_rel": r["ref_plane_max_rel"],
                               "centroid_px": {m: summ(v) for m, v in r["err"].items()},
                               "edge_vs_ref_px": {m: summ(v) for m, v in r["edge_ref"].items()},
                               "edge_inter_channel_px": {m: summ(v) for m, v in r["edge_inter"].items()}}
        allv = np.concatenate([r["err"]["vertex"] for r in res.values()])
        allb = np.concatenate([r["err"]["bands"] for r in res.values()])
        rep["pooled"] = {"n": len(allv), "vertex": summ(allv), "bands": summ(allb)}
        Path(path).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    return res


def test_reference_agrees_with_rasterizer_without_motion(rig) -> None:
    """参照自检：读出为 0 时 Warp 射线与全局快门光栅化的实例图几乎逐像素相同（像素中心与内参约定一致），标记质心差 ≤ 0.05 px。"""
    cam, sc = rig
    c = CASES["oblique-forward"]
    cm = cam_model(readout_s=0.0)
    pose = trajectory(c)
    actors, af = markers(cm, pose)
    cam.set_actors(actors)
    gs = cam.render(cm, *pose(0.0), actors=af)
    ref = rr.RowRayReference.from_scene(sc, actors, af, origin=pose(0.0)[1]).render(cm, pose, 0.0)
    assert (ref.instance != gs.instance).mean() < 1e-4
    assert np.array_equal(ref.semantic == SEM_OBJECT, gs.semantic == SEM_OBJECT) or (ref.semantic != gs.semantic).mean() < 1e-4
    ids = np.unique(ref.instance[ref.instance > 0])
    assert len(ids) > 100 and rr.centroid_errors(ref.instance, gs.instance, ids).max() <= 0.05
    m = (ref.instance > 0) & (gs.instance == ref.instance)
    assert (np.abs(ref.depth_m[m] - gs.depth_m[m]) <= 1e-4 * gs.depth_m[m]).all()     # 平面深度同一定义（float32 求交，相对 1e-4）


def test_reference_row_timing_matches_plane_analytic(results) -> None:
    for r in results.values():
        assert r["ref_plane_max_rel"] <= 1e-4, r["case"]


def test_vertex_method_marker_error_p99(results) -> None:
    """逐顶点方法：标记位置误差 p99 ≤ 0.5 px（合并与逐轨迹）；全局快门同一度量 p99 > 3 px（构造轨迹确有可测的卷帘形变）。"""
    pooled = np.concatenate([r["err"]["vertex"] for r in results.values()])
    assert len(pooled) >= 300
    assert _q(pooled, 0.99) <= P99_VERTEX, _q(pooled, 0.99)
    for r in results.values():
        assert r["measured"] >= 40, r["case"]
        assert _q(r["err"]["vertex"], 0.99) <= P99_VERTEX, (r["case"], _q(r["err"]["vertex"], 0.99))
        assert _q(r["err"]["global"], 0.99) > 3.0, (r["case"], _q(r["err"]["global"], 0.99))


def test_band_method_16_bands_error(results) -> None:
    """行带方法（16 带）：标记位置误差 ≤ 1.5 px（逐标记最大值），且明显大于逐顶点方法（行带量化可测）。"""
    for r in results.values():
        e = r["err"]["bands"]
        assert float(e.max()) <= MAX_BANDS, (r["case"], float(e.max()))
        assert _q(e, 0.99) > _q(r["err"]["vertex"], 0.99), r["case"]


def test_depth_instance_semantic_rgb_edges_aligned(results) -> None:
    """边缘对齐：同一遍输出的深度、语义、RGB 掩膜与实例掩膜逐标记平均边缘偏移 ≤ 0.5 px；各通道相对 Warp 参照实例边缘 p99 ≤ 0.5 px。"""
    for r in results.values():
        for k, v in r["edge_inter"].items():
            assert float(v.max()) <= EDGE, (r["case"], k, float(v.max()))
        for k, v in r["edge_ref"].items():
            assert _q(v, 0.99) <= EDGE, (r["case"], k, _q(v, 0.99))


def test_static_body_rolling_equals_global_bytewise(rig) -> None:
    """静止机体：读出窗口内 9 个节点由悬停 PoseHistory 插值得到时，卷帘与全局快门的 RGB、深度、实例、语义、无遮挡实例与
    L2-depth 观测逐字节一致（含效果链噪声）。"""
    from awr.sensorsvc.camera.effects import DepthSensorConfig, FxParams

    cam, _sc = rig
    nodes, (R, p) = _static_history_nodes()
    cm = cam_model()
    actors, af = markers(cm, lambda t: (R, p))
    cam.set_actors(actors)
    fx = FxParams(1.1, 1.1, 0.3, 1.5, 0.4, 0x5EED_0060_0000_0001, DepthSensorConfig(a=1.0), 0x5EED_0060_0000_0002, True)
    for f in (None, fx):
        gs = cam.render(cm, R, p, actors=af, fx=f)
        rs = cam.render(cm, R, p, actors=af, fx=f, rs_nodes=nodes)
        for k in ("rgb8", "depth_m", "instance", "semantic", "instance_unocc", "depth_obs", "meter_sub"):
            a, b = getattr(gs, k), getattr(rs, k)
            assert (a is None and b is None) or (a.tobytes() == b.tobytes()), k
        assert int((gs.instance > 0).sum()) > 10_000
