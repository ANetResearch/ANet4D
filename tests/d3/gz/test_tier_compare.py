"""Fast 与 Realistic 两档对照（D3-AC-35；M24-AC-016、M24-AC-028、M25-AC-040）：dmax 实测部分。

1. `test_two_tier_conformance_and_report`（d3-gz 容器内，launcher = local）：
   - 一致性用例集（awr.phys.tier_compare.CASES：起降、悬停、速度阶跃、速度限幅、偏航阶跃、50 m 方形航线、环绕、撞墙与空中失能
     坠地、8 m/s 恒定风悬停）以同一驱动代码分别在 FastTier 与 GzSimBackend（RefTracker、compose "px4" 档、field 风、
     sigma_turb = 0）上运行，接口层全部一致，位置 RMS、到达时刻差与其余指标差在用例容差内；
   - 两档采样一致性（§6.10.7）：Fast 参考轨迹（250 Hz，含起飞、机动、触地）经 `Drone4DClock` 运动学注入在 gz 中复现，两侧
     PoseHistory 时刻逐位相同、位置差 ≤ 1e-6 m、姿态差 ≤ 1e-9 rad、速度、角速度与比力逐位相同；两份 PoseHistory 送入同一 M25：
     LiDAR 逐点位姿差 ≤ 1 mm、IMU 输出差在重采样容差内；
   - 同一 PoseHistory 两次送入 M25 全路径（相机 CPU 替身、LiDAR、IMU）输出逐字节相同；
   - 汇总 `tier_compare.json`（drone4d.phys.tier_compare.v1），`AWR_D3_TIER_OUT=<dir>` 时写入该目录（否则 tmp）。
2. `test_m25_gpu_camera_bytewise`（d3-sensorsvc 容器、GPU、已构建世界 `AWR_D3_TIER_WORLD=<world_dir>`）：同一 PoseHistory
   两次送入 M25 全路径（RasterCamera GPU 渲染 + JPEG、网格 LiDAR、IMU），逐传感器输出 sha256 相同。

运行（dmax，命令模板见 P09 记录）：d3-gz:p09（或 dev）容器内 `python3 -m pytest -q tests/d3/gz/test_tier_compare.py`。
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from awr.phys.tier_compare import (
    CASES,
    REF_CASE,
    SCHEMA,
    BoxWorld,
    CaseRun,
    build_report,
    compare_case,
    m25_compare,
    m25_digest,
    pose_history_diff,
    run_fast,
    run_gz,
    write_inject,
)

HAS_GZ = shutil.which("gz") is not None and any(
    (Path(p) / "libDrone4DGz.so").exists() for p in os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "").split(":") if p)
needs_gz = pytest.mark.skipif(not HAS_GZ, reason="needs gz-sim with libDrone4DGz.so (d3-gz image)")


def _out(tmp_path: Path) -> Path:
    d = Path(os.environ.get("AWR_D3_TIER_OUT") or tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    return d


@needs_gz
def test_two_tier_conformance_and_report(tmp_path: Path) -> None:
    out = _out(tmp_path)
    rows = []
    for c in CASES:
        fa = run_fast(c)
        re_ = run_gz(c, tmp_path / "gz" / c.id, launcher="local")
        fa.save(out)
        re_.save(out)
        r = compare_case(c, fa, re_)
        rows.append(r)
        print(f"[tier_compare] {c.id}: interface {r['interface']['pass']} tol {r['tol_pass']} diff {r['diff']}", flush=True)
    ref = run_fast(REF_CASE)
    ref.save(out)
    got = run_gz(REF_CASE, tmp_path / "gz" / "inject", launcher="local", inject=write_inject(ref, out / "inject_ref.f64"))
    got.save(out)
    assert got.info["plugin_build"].endswith("+inject")
    ph = pose_history_diff(ref, got)
    m25 = m25_compare(ref, got, world=BoxWorld())
    sampling = {"pose_history": ph, "m25": m25, "pass": bool(ph["pass"] and m25["pass"])}
    runs = [m25_digest(ref, world=BoxWorld()) for _ in range(2)]
    bytewise = {"runs": runs, "world": "BoxWorld", "pass": runs[0]["sha256"] == runs[1]["sha256"]}
    meta = {"kernel": rows and CaseRun.load(out, CASES[0].id, "fast").info.get("kernel"),
            "gz": {k: CaseRun.load(out, CASES[0].id, "gz").info.get(k) for k in ("gz_sim_version", "plugin_build")},
            "gz_tracker": "RefTracker (M08 refgen shared with Fast; compose velocity profile px4)"}
    doc = build_report(rows, sampling=sampling, m25=bytewise, meta=meta)
    (out / "tier_compare.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    fails = {r["id"]: (r["failures"], {k: v for k, v in r["diff"].items() if not r["within"][k]}) for r in rows if not r["pass"]}
    assert not fails, fails
    assert ph["t_bitwise"] and ph["pass"], ph
    assert m25["lidar"]["pass"] and m25["imu"]["pass"], m25
    assert bytewise["pass"]
    assert doc["schema"] == SCHEMA and doc["pass"] and doc["summary"]["pass"] == len(CASES)


def _gl_ok() -> bool:
    """EGL 上下文由硬件 GPU 提供（排除 llvmpipe 等软件渲染器）。"""
    try:
        import moderngl

        ctx = moderngl.create_standalone_context(backend="egl")
        name = str(ctx.info.get("GL_RENDERER", ""))
        ctx.release()
        return not any(k in name.lower() for k in ("llvmpipe", "softpipe", "swrast"))
    except Exception:
        return False


def test_m25_gpu_camera_bytewise(tmp_path: Path) -> None:
    wd = os.environ.get("AWR_D3_TIER_WORLD")
    if not wd or not (Path(wd) / "world.json").is_file():
        pytest.skip("[data] set AWR_D3_TIER_WORLD to a built world dir (worlds/<id>)")
    if not _gl_ok():
        pytest.skip("[gpu] hardware EGL context unavailable")
    from awr.sensorsvc.camera.encode import GpuEncoderUnavailable
    src = Path(os.environ.get("AWR_D3_TIER_OUT") or tmp_path)
    ref = CaseRun.load(src, REF_CASE.id, "fast") if (src / f"{REF_CASE.id}.fast.json").is_file() else run_fast(REF_CASE)
    dz = float(os.environ.get("AWR_D3_TIER_DZ", "60"))
    ticks = int(os.environ.get("AWR_D3_TIER_TICKS", "60"))
    cam = os.environ.get("AWR_D3_TIER_CAMERA", "rgb1080-rs")
    try:
        runs = [m25_digest(ref, world_dir=Path(wd), camera_profile=cam, dz_m=dz, max_ticks=ticks) for _ in range(2)]
    except GpuEncoderUnavailable as e:
        pytest.skip(f"[gpu] {cam} needs the GPU JPEG encoder: {e}")
    assert runs[0]["renderer"] == "RasterCamera" and runs[0]["frames"]["eo"] > 0 and runs[0]["frames"]["lidar0"] == ticks
    doc = {"runs": runs, "world": wd, "dz_m": dz, "pass": runs[0]["sha256"] == runs[1]["sha256"]}
    (_out(tmp_path) / "m25_gpu.json").write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
    assert doc["pass"], [r["sha256"] for r in runs]
