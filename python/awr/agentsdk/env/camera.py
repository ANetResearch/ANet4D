"""L1 标称相机与云台（M26 §6.6.3；P0a）：针孔 1920 × 1080、HFOV 60°、无畸变，挂载于机头下方三轴云台。

相机覆盖（ADR-217 决策 (1)，D3-AC-18）：run_spec `fleet[].sensors[]` 中 `sensor_id = eo` 的 `overrides.hfov_deg` 改变水平视场（分辨率、
云台与挂载不变），只用于参考行（r6 行名）与不排名运行（`EnvSpec.from_run_spec` 校验）；覆盖后的型号记为 `<标称型号>+hfov<度>`，
运行清单 `camera` 字段记录实际相机，评测器的 AO oracle 按该相机计算（M27 `evaluator.run_camera`）。

云台角相对机体航向：`yaw_deg` 逆时针为正（与 ENU 偏航同口径），`pitch_deg` 向上为正（−90 为正下视）；指令角按
`rate_max_dps` 每控制步限速趋近。光学帧取 forward / right / down 三个 ENU 单位向量。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from awr.sim.sensors import servo as SV

__all__ = ["CAMERA_OVERRIDE_KEYS", "HFOV_RANGE_DEG", "CameraModel", "Gimbal", "cam_axes", "project"]

CAMERA_OVERRIDE_KEYS = frozenset({"hfov_deg"})      # L1 标称相机可覆盖的键（ADR-217）
HFOV_RANGE_DEG = (10.0, 120.0)                      # 覆盖视场的取值范围（与 run_spec 契约一致）


@dataclass(frozen=True)
class CameraModel:
    sensor_id: str = "eo"
    width: int = 1920
    height: int = 1080
    hfov_deg: float = 60.0
    rate_hz: float = 10.0
    pitch_range_deg: tuple[float, float] = (-90.0, 30.0)
    yaw_range_deg: tuple[float, float] = (-150.0, 150.0)
    rate_max_dps: float = 90.0
    mount_t_bs_m: tuple[float, float, float] = (0.12, 0.0, -0.08)
    model_ref: str = "d4d_eo1080@1"
    # 伺服档（awr.sim.sensors.servo）：l1 限速逼近（缺省，与以前相同）；l2 二阶伺服（度/s²、秒）
    servo_tier: str = "l1"
    acc_max_dps2: float = SV.L2_DEFAULTS["acc_max_deg_s2"]
    tau_s: float = SV.L2_DEFAULTS["tau_s"]

    @property
    def fx(self) -> float:
        return (self.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    @property
    def K(self) -> list[float]:
        f = self.fx
        return [f, 0.0, self.width / 2, 0.0, f, self.height / 2, 0.0, 0.0, 1.0]

    def with_overrides(self, ov: dict | None) -> CameraModel:
        """按相机覆盖（`{"hfov_deg": 度}`）派生：视场改为覆盖值，型号记为 `<标称型号>+hfov<度>`；空覆盖返回自身。"""
        if not ov:
            return self
        bad = set(ov) - CAMERA_OVERRIDE_KEYS
        if bad:
            raise ValueError(f"camera overrides {sorted(bad)} are not supported (only {sorted(CAMERA_OVERRIDE_KEYS)})")
        h = float(ov["hfov_deg"])
        if not HFOV_RANGE_DEG[0] <= h <= HFOV_RANGE_DEG[1]:
            raise ValueError(f"camera override hfov_deg {h} is outside {list(HFOV_RANGE_DEG)}")
        if h == self.hfov_deg:
            return self
        return replace(self, hfov_deg=h, model_ref=f"{self.model_ref.split('+', 1)[0]}+hfov{h:g}")

    def manifest(self) -> dict:
        """运行清单 `camera`：实际 L1 相机的型号、分辨率、视场与焦距（像素）。"""
        return {"model_ref": self.model_ref, "width": self.width, "height": self.height, "hfov_deg": self.hfov_deg, "fx": self.fx}

    def config(self) -> dict:
        """agent_config.v1 `sensors[]` 条目。"""
        return {"sensor_id": self.sensor_id, "kind": "camera", "model_ref": self.model_ref, "rate_hz": self.rate_hz,
                "mount_nominal": {"t_bs_m": list(self.mount_t_bs_m), "q_bs_xyzw": [0.0, 0.0, 0.0, 1.0]},
                "width": self.width, "height": self.height, "K": self.K, "dist": [0.0] * 5,
                "gimbal": {"pitch_range_deg": list(self.pitch_range_deg), "yaw_range_deg": list(self.yaw_range_deg),
                           "rate_max_dps": self.rate_max_dps}, "controllable": True}


@dataclass
class Gimbal:
    pitch_deg: float = -30.0
    yaw_deg: float = 0.0
    cmd_pitch_deg: float = -30.0
    cmd_yaw_deg: float = 0.0
    cam: CameraModel = field(default_factory=CameraModel)
    omega_pitch_dps: float = 0.0  # l2 伺服档的角速度状态
    omega_yaw_dps: float = 0.0

    def command(self, pitch_deg: float, yaw_deg: float) -> bool:
        """→ 是否被限幅。"""
        lo, hi = self.cam.pitch_range_deg
        p = min(max(pitch_deg, lo), hi)
        ylo, yhi = self.cam.yaw_range_deg
        y = (yaw_deg + 180.0) % 360.0 - 180.0
        yc = min(max(y, ylo), yhi)
        self.cmd_pitch_deg, self.cmd_yaw_deg = p, yc
        return p != pitch_deg or yc != y

    def slew(self, dt_s: float) -> None:
        c = self.cam
        if c.servo_tier == "l2":
            self.pitch_deg, self.omega_pitch_dps = SV.step_l2(self.pitch_deg, self.omega_pitch_dps, self.cmd_pitch_deg,
                                                              c.rate_max_dps, c.acc_max_dps2, c.tau_s, dt_s)
            self.yaw_deg, self.omega_yaw_dps = SV.step_l2(self.yaw_deg, self.omega_yaw_dps, self.cmd_yaw_deg,
                                                          c.rate_max_dps, c.acc_max_dps2, c.tau_s, dt_s)
            return
        self.pitch_deg = SV.step_l1(self.pitch_deg, self.cmd_pitch_deg, c.rate_max_dps, dt_s)
        self.yaw_deg = SV.step_l1(self.yaw_deg, self.cmd_yaw_deg, c.rate_max_dps, dt_s)


def cam_axes(yaw_body_rad: float, pitch_deg: float, yaw_rel_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ (forward, right, down)，ENU 单位向量。"""
    yw = yaw_body_rad + math.radians(yaw_rel_deg)
    pt = math.radians(pitch_deg)
    f = np.array([math.cos(pt) * math.cos(yw), math.cos(pt) * math.sin(yw), math.sin(pt)])
    r = np.array([math.sin(yw), -math.cos(yw), 0.0])
    return f, r, np.cross(f, r)


def project(cam: CameraModel, o: np.ndarray, axes, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pts (n, 3) → (u, v, z)；z 为沿光轴深度。"""
    f, r, d = axes
    rel = pts - o
    z = rel @ f
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.width / 2 + cam.fx * (rel @ r) / z
        v = cam.height / 2 + cam.fx * (rel @ d) / z
    return u, v, z
