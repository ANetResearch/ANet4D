"""交付帧的公开标定与像素到公开地形的投影；只用相机读数、公开导航解与静态地形，不查演员。

`camera_metadata(frame, navigation, image_width)`：L2 相机帧记录 → 交付给智能体的 `sensor_metadata[uav]`（内参按交付尺寸缩放、
畸变、名义安装位姿、云台读数、采集时与交付时的导航解、图像与导航的时间差）。`project_pixel(metadata, pixel, terrain)`：像素 →
反畸变 → 机体航向 · 云台偏航 · 云台俯仰 → 以采集时导航解（匀速对齐到图像中间时刻）为原点的射线与 `terrain` 的第一个交点
（2 m 步进再二分 16 次）。结果是估计值（`approximate = True`），适合人的脚点与车辆接地点，不会把头部或屋顶像素纠正成真值位置。
"""

from __future__ import annotations

import copy
import math

import numpy as np

__all__ = ["camera_metadata", "project_pixel"]

_FRAME_FIELDS = ("sensor_id", "seq", "t_mid_ns", "exposure_ns", "readout_ns", "width", "height", "K", "dist", "mount_nominal", "gimbal_reading")
_NAV_FIELDS = ("t_ns", "pos_enu_m", "vel_enu_mps", "q_wb_xyzw", "omega_b_rps", "pos_std_m", "fix")


def camera_metadata(frame: dict, navigation: dict, image_width: int) -> dict:
    result = {k: copy.deepcopy(frame[k]) for k in _FRAME_FIELDS if k in frame}
    width, height = int(frame["width"]), int(frame["height"])
    scaled = min(width, int(image_width))
    scale = scaled / width
    result["native_dimensions"] = [width, height]
    result["width"] = scaled
    result["height"] = int(height * scale)
    if "K" in result:
        result["K"] = [v * scale if i < 6 else v for i, v in enumerate(result["K"])]
    result["navigation_at_delivery"] = {k: copy.deepcopy(navigation[k]) for k in _NAV_FIELDS if k in navigation}
    result["navigation_time_offset_s"] = (navigation["t_ns"] - frame["t_mid_ns"]) * 1e-9
    if "capture_navigation" in frame:
        result["navigation_at_capture"] = {k: copy.deepcopy(frame["capture_navigation"][k]) for k in _NAV_FIELDS if k in frame["capture_navigation"]}
    return result


def _rz(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def project_pixel(metadata: dict, pixel, terrain, max_range: float = 2000.0) -> dict:
    uv = np.asarray(pixel, float)
    if uv.shape != (2,) or not np.isfinite(uv).all() or not (0 <= uv[0] < metadata["width"] and 0 <= uv[1] < metadata["height"]):
        raise ValueError("Pixel must lie inside the delivered image")
    k = np.asarray(metadata["K"], float).reshape(3, 3)
    xd, yd = (uv - [k[0, 2], k[1, 2]]) / [k[0, 0], k[1, 1]]
    x, y = xd, yd
    dist = metadata.get("dist", [0.0] * 5)
    if len(dist) != 5:
        raise ValueError("Unsupported distortion model")
    k1, k2, p1, p2, k3 = dist
    for _ in range(20):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        if not np.isfinite(rad) or abs(rad) < 1e-8:
            raise ValueError("Unstable lens inversion")
        x, y = (xd - 2 * p1 * x * y - p2 * (r2 + 2 * x * x)) / rad, (yd - p1 * (r2 + 2 * y * y) - 2 * p2 * x * y) / rad
    nav = metadata.get("navigation_at_capture", metadata["navigation_at_delivery"])
    qx, qy, qz, qw = nav["q_wb_xyzw"]
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    g = metadata["gimbal_reading"]
    a, e = math.radians(g["yaw_deg"]), -math.radians(g["pitch_deg"])
    ry = np.array([[math.cos(e), 0, math.sin(e)], [0, 1, 0], [-math.sin(e), 0, math.cos(e)]])
    direction = _rz(yaw) @ _rz(a) @ ry @ np.array([1.0, -x, -y])
    direction /= np.linalg.norm(direction)
    if direction[2] >= -0.01:
        raise ValueError("Pixel ray does not point down toward ground")
    # Explicit constant-velocity time alignment, using only public nav data.
    offset = (nav["t_ns"] - metadata["t_mid_ns"]) * 1e-9 if "navigation_at_capture" in metadata else metadata["navigation_time_offset_s"]
    origin = np.asarray(nav["pos_enu_m"]) - np.asarray(nav["vel_enu_mps"]) * offset
    origin = origin + _rz(yaw) @ np.asarray(metadata["mount_nominal"]["t_bs_m"])

    def above(distance):
        p = origin + direction * distance
        return p[2] - terrain(float(p[0]), float(p[1]))

    if above(0) <= 0:
        raise ValueError("Camera estimate is below the terrain")
    lo = 0.0
    for hi in np.arange(2.0, max_range + 2.0, 2.0):
        if above(hi) <= 0:
            break
        lo = hi
    else:
        raise ValueError("No terrain intersection within range")
    for _ in range(16):
        mid = (lo + hi) / 2
        if above(mid) > 0:
            lo = mid
        else:
            hi = mid
    position = origin + direction * (lo + hi) / 2
    return dict(position_enu_m=position.tolist(), frame_seq=metadata["seq"],
                source="delivered camera calibration + public navigation + static DTM",
                approximate=True, navigation_position_std_m=nav.get("pos_std_m"),
                limitation="Ground-contact pixels only; DTM and nominal calibration errors remain. No actor identity or dynamic depth is queried.")
