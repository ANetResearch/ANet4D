"""相机效果链与 ISP 的参考实现（M25 §6.5；AWR-20 §15.6.5 噪声档；docs/05 §5.4.3 效果链；D3-AC-11 晴天档与最坏档）。

顺序（docs/05 §5.4.3、M25 §6.5.1；不含运动模糊，`blur_subframes = 1`）：在无畸变渲染域依次施加接地阴影（接触阴影贴花、
2048² 投影阴影图）→ 雨天湿润 → 雾与雨的消光（Koschmieder，沿视线斜距）→ 雨线；随后对全部输出施加同一畸变映射（`proj.build_remap`，
RGB 线性域双线性，深度、实例、语义与质量位最近邻）；再施加曝光增益（固定或自动曝光）→ 渐晕 → Poisson–Gaussian 噪声 → sRGB 编码
→ 8 位量化。L2-depth 深度传感器模型（§6.5.6）与云台编码器噪声（§15.6.5）同在本模块。全部计算为 numpy，随机数只取决于
（run_seed、uav_no、帧序号、像素、通道），按 `awr.sim.sensors.cbrng` 逐级派生（M25 §6.17.2，流 22，通道表 `CHANNELS_V2`）；
GPU 着色器版本（`raster`）以本模块为参照做一致性测试。投影阴影图需要从太阳方向光栅化演员，只有 GPU 实现，以几何判据验收。

天气（`Weather`）与 LiDAR 共用同一环境时间线：能见度 V（m）给出雾的消光系数 3.912 / V，降雨强度 R（mm/h）给出可见光消光
1.076·R^0.67 km⁻¹（r3）、雨线密度与湿润；两者相加为 `FxParams.fog_sigma_m`。

本实现设定（M25 未规定处）：
- 逐像素正态：k = sm(K_frame ^ (pix·m + c))，b = sm(k ^ C3)；u1 = ((k >> 40) + 0.5)·2⁻²⁴，u2 = (b >> 40)·2⁻²⁴（float32 可精确表示，
  便于 GLSL 复现），n = sqrt(−2 ln u1)·cos(2π u2)。pix 为交付图像行主序下标（第 0 行在上），RGB 的 m = 3、深度的 m = 1。
- 自动曝光测光：1/8 降采样（取每 8 × 8 块左上像素），中心 50% 区域（各边 25% 内缩）权重 2，log2 Y 在 [−12, 0] 等分 64 档，
  m = 2^(加权平均档中心)。测光对象为施加基础增益、未施加曝光误差的图像，使误差保持为显式的 ±ae_error_amp 调制而不被反馈抵消。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from awr.sensorsvc.camera.model import CameraModel, IspConfig, NoiseConfig
from awr.sim.sensors.cbrng import C3, key, sm, uniform

__all__ = ["CHANNELS_V2", "CONTACT_K", "FOG_LIN", "FX_VERSION", "NOISE_TABLE_VERSION", "PROFILES", "SHADOW_MAP_K", "STREAM_M25", "AeState",
           "DepthSensorConfig", "FxParams", "NoiseProfile", "Weather", "apply_fx_reference", "apply_rgb", "contact_casters", "depth_sensor",
           "finish_frame", "frame_params", "gimbal_reading", "meter", "meter_sub", "pixel_normals", "profile", "rain_density", "rain_streak",
           "shade_reference", "visible_sigma_per_m", "with_profile"]

FX_VERSION = "fx-full@2"
NOISE_TABLE_VERSION = "d3-noise-1-draft"
STREAM_M25 = 22
# M25 §6.17.2 通道表（流 22）：基址 + sensor_no；区间互不重叠
CHANNELS_V2: dict[str, tuple[int, int]] = {
    "cam_noise": (0x1000, 0x10FF), "cam_ae": (0x1100, 0x11FF), "cam_rain": (0x1200, 0x12FF), "cam_gimbal": (0x1300, 0x13FF),
    "depth_noise": (0x1400, 0x14FF), "lidar": (0x2000, 0x20FF), "imu": (0x3000, 0x37FF), "gnss": (0x3800, 0x3FFF), "nav": (0x4000, 0x40FF),
}
_U = np.uint64
_INV24 = np.float32(1.0 / 16777216.0)
AE_BINS = 64
AE_LOG2_RANGE = (-12.0, 0.0)
# 接地阴影与天气（docs/05 §5.4.3；r3 exp6、§3.5）
CONTACT_K = 0.5                       # 接触阴影贴花中心的衰减
SHADOW_MAP_K = 0.45                   # 投影阴影图完全遮挡时的衰减（环境光保留 55%）
CONTACT_SUN_EL_MIN = 0.2              # 太阳仰角正弦下限：阴影长度不超过演员高度的 5 倍
CONTACT_DZ_M = 0.6                    # 贴花只作用于演员脚底所在高度 ±0.6 m 内的表面
WET_K, WET_DESAT = 0.82, 0.15         # 雨天湿润：反照率 × 0.82，向亮度去饱和 15%
RAIN_DENSITY_PER_MMH, RAIN_DENSITY_MAX = 0.0025, 0.1
RAIN_CELL_PX = 3.0                    # 雨滴网格（像素）；每格至多一滴，深度 0.5–30 m
RAIN_SAMPLES = 16                     # 雨线沿屏幕位移方向的采样数
RAIN_FALL_MPS = 6.5                   # 雨滴末速度
RAIN_REF_M = 2.0                      # 雨线长度的参照距离
RAIN_STREAK_LIN, RAIN_ALPHA_K, RAIN_ALPHA_MAX = 0.6, 0.03, 0.5


def _srgb_lin(rgb) -> np.ndarray:
    c = np.asarray(rgb, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


FOG_LIN = _srgb_lin((190, 196, 202))   # 雾色（与 raster 远景雾色相同）


@dataclass(frozen=True)
class Weather:
    """可见光天气（与 LiDAR 同一环境时间线）：能见度（m，None 为无雾）与降雨强度（mm/h）。"""
    visibility_m: float | None = None
    rain_mmh: float = 0.0


def visible_sigma_per_m(w: Weather | None) -> float:
    """可见光消光系数（m⁻¹）：雾 3.912 / V 加雨 1.076·R^0.67 km⁻¹。"""
    if w is None:
        return 0.0
    s = 3.912 / float(w.visibility_m) if w.visibility_m else 0.0
    if w.rain_mmh > 0:
        s += 1.076 * float(w.rain_mmh) ** 0.67 / 1000.0
    return s


def rain_density(rain_mmh: float) -> float:
    return min(RAIN_DENSITY_PER_MMH * max(float(rain_mmh), 0.0), RAIN_DENSITY_MAX)


@dataclass(frozen=True)
class DepthSensorConfig:
    a: float = 0.0                        # sigma(d) = a·(0.005·d + 0.0005·d²)
    d_min_m: float = 0.3
    d_max_m: float = 200.0


@dataclass(frozen=True)
class NoiseProfile:
    name: str
    isp: IspConfig
    noise: NoiseConfig
    depth: DepthSensorConfig
    gimbal_sigma_deg: float


PROFILES: dict[str, NoiseProfile] = {
    "clean": NoiseProfile("clean", IspConfig(ae="fixed", ev=0.0), NoiseConfig(0.0, 0.0), DepthSensorConfig(0.0, 0.3, 200.0), 0.0),
    "realistic": NoiseProfile("realistic", IspConfig(ae="auto", ae_error_amp=0.2), NoiseConfig(2.0, 0.2), DepthSensorConfig(1.0, 0.3, 60.0), 0.05),
    "harsh": NoiseProfile("harsh", IspConfig(ae="auto", ae_error_amp=0.4), NoiseConfig(4.0, 0.4), DepthSensorConfig(2.0, 0.3, 40.0), 0.2),
}


def profile(name: str) -> NoiseProfile:
    if name not in PROFILES:
        raise ValueError(f"unknown noise profile {name!r} (clean, realistic, harsh)")
    return PROFILES[name]


def with_profile(cm: CameraModel, p: NoiseProfile) -> CameraModel:
    """把噪声档写入相机模型（保留调用方的 JPEG 质量与渐晕系数）。"""
    return replace(cm, isp=replace(p.isp, jpeg_quality=cm.isp.jpeg_quality, vignette_k=cm.isp.vignette_k), noise=p.noise)


@dataclass
class AeState:
    """每 (uav, sensor) 一份，按 seq 顺序推进（M25 §6.5.3）。"""
    gain_prev: float = 1.0
    meter_prev: float | None = None
    frames: int = 0
    history: list[float] = field(default_factory=list)


def _normals(K: np.uint64, idx: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        k = sm(_U(K) ^ idx)
        b = sm(k ^ C3)
    u1 = ((k >> _U(40)).astype(np.float32) + np.float32(0.5)) * _INV24
    u2 = (b >> _U(40)).astype(np.float32) * _INV24
    return np.sqrt(np.float32(-2.0) * np.log(u1)) * np.cos(np.float32(2.0 * math.pi) * u2)


def pixel_normals(seed: int, uav_no: int, frame_index: int, ch: int, n: int, m: int) -> np.ndarray:
    """(n, m) float32 标准正态；元素 (pix, c) 的计数器为 pix·m + c。"""
    K = key(seed, STREAM_M25, uav_no, frame_index, ch)
    return _normals(K, np.arange(n * m, dtype=np.uint64)).reshape(n, m)


def meter(lin: np.ndarray) -> float:
    """测光：1/8 降采样的中心加权 64 档 log2 直方图的加权平均（整数计数，确定）。"""
    return meter_sub(lin[::8, ::8])


def meter_sub(sub: np.ndarray) -> float:
    """`meter` 的直方图部分；`sub` 为已按 1/8 降采样的线性图像（GPU 路径读回的小图）。"""
    s = np.asarray(sub, np.float32)
    Y = 0.2126 * s[..., 0] + 0.7152 * s[..., 1] + 0.0722 * s[..., 2]
    lo, hi = AE_LOG2_RANGE
    b = np.clip(((np.log2(np.maximum(Y, 2.0 ** lo)) - lo) / (hi - lo) * AE_BINS).astype(np.int64), 0, AE_BINS - 1)
    h, w = Y.shape
    wgt = np.ones((h, w), np.int64)
    wgt[h // 4:h - h // 4, w // 4:w - w // 4] = 2
    hist = np.bincount(b.ravel(), weights=wgt.ravel(), minlength=AE_BINS).astype(np.int64)
    centre = lo + (np.arange(AE_BINS) + 0.5) * (hi - lo) / AE_BINS
    return float(2.0 ** (float((hist * centre).sum()) / max(int(hist.sum()), 1)))


def exposure_gain(state: AeState, isp: IspConfig, t_mid_s: float, *, seed: int, uav_no: int, sensor_no: int) -> tuple[float, float]:
    """返回 (基础增益 g, 含误差的增益 g·err)。"""
    if isp.ae == "fixed":
        g = 2.0 ** isp.ev
    else:
        m = state.meter_prev if state.meter_prev is not None else isp.ae_target
        g_prev = state.gain_prev if state.frames > 0 else 1.0
        g = float(np.clip(g_prev * (isp.ae_target / max(m, 1e-4)) ** isp.ae_alpha, *isp.ae_gain_range))
    phi = float(uniform(seed, STREAM_M25, uav_no, 0, CHANNELS_V2["cam_ae"][0] + sensor_no)) * 2.0 * math.pi
    err = 1.0 + isp.ae_error_amp * math.sin(2.0 * math.pi * t_mid_s / isp.ae_error_period_s + phi)
    return g, g * err


def _cos4(cm: CameraModel) -> np.ndarray:
    x = (np.arange(cm.width, dtype=np.float32) + np.float32(0.5) - np.float32(cm.cx)) / np.float32(cm.fx)
    y = (np.arange(cm.height, dtype=np.float32) + np.float32(0.5) - np.float32(cm.cy)) / np.float32(cm.fy)
    r2 = x[None, :] ** 2 + y[:, None] ** 2
    return (np.float32(1.0) / (np.float32(1.0) + r2)) ** 2          # cos⁴θ = 1 / (1 + r²)²


def _srgb_q(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, np.float32(0.0), np.float32(1.0))
    y = np.where(x <= np.float32(0.0031308), x * np.float32(12.92),
                 np.float32(1.055) * np.power(x, np.float32(1.0 / 2.4)) - np.float32(0.055))
    return np.floor(y * np.float32(255.0) + np.float32(0.5)).astype(np.uint8)


@dataclass(frozen=True)
class FxParams:
    """一帧效果链的全部参数（CPU 计算，GPU 着色器与参考实现共用）。"""
    gain: float                           # 含曝光误差的增益
    gain_base: float                      # 测光用的基础增益
    vignette_k: float
    read_sigma_dn: float
    shot_k_dn: float
    key_noise: int                        # 像素噪声帧键（uint64）
    depth: DepthSensorConfig | None       # None：不输出观测深度
    key_depth: int
    meter: bool                           # 自动曝光：本帧需要测光
    fog_sigma_m: float = 0.0              # 雾与雨的可见光消光系数（m⁻¹，`visible_sigma_per_m`）
    rain_mmh: float = 0.0                 # > 0：雨线与湿润
    key_rain: int = 0                     # 雨滴帧键（uint64，通道 cam_rain）
    contact_shadow: bool = False          # 接触阴影贴花（`CameraModel.shadow` 缺省 contact）
    shadow_map: bool = False              # 2048² 投影阴影图（`shadow = map`）

    @property
    def shade(self) -> bool:
        """是否需要渲染域的着色遍（阴影、湿润、消光或雨线）。"""
        return self.contact_shadow or self.shadow_map or self.fog_sigma_m > 0 or self.rain_mmh > 0


def frame_params(cm: CameraModel, state: AeState, *, seed: int, uav_no: int, sensor_no: int, frame_index: int, t_mid_s: float,
                 depth: DepthSensorConfig | None = None, weather: Weather | None = None, shadow: bool = True) -> FxParams:
    """计算一帧的效果链参数并推进自动曝光状态（测光结果由 `finish_frame` 写回）。`weather` 为本帧天气（None 为晴、无雾）；
    `shadow = False` 关闭接地阴影（缺省按 `cm.shadow`：contact 为接触阴影，map 为投影阴影图加接触阴影）。"""
    g, ge = exposure_gain(state, cm.isp, t_mid_s, seed=seed, uav_no=uav_no, sensor_no=sensor_no)
    state.gain_prev = g
    state.frames += 1
    state.history.append(ge)
    kn = int(key(seed, STREAM_M25, uav_no, frame_index, CHANNELS_V2["cam_noise"][0] + sensor_no))
    kd = int(key(seed, STREAM_M25, uav_no, frame_index, CHANNELS_V2["depth_noise"][0] + sensor_no))
    rain = float(weather.rain_mmh) if weather is not None else 0.0
    kr = int(key(seed, STREAM_M25, uav_no, frame_index, CHANNELS_V2["cam_rain"][0] + sensor_no)) if rain > 0 else 0
    return FxParams(ge, g, cm.isp.vignette_k, cm.noise.read_sigma_dn, cm.noise.shot_k_dn, kn, depth, kd, cm.isp.ae == "auto",
                    fog_sigma_m=visible_sigma_per_m(weather), rain_mmh=rain, key_rain=kr, contact_shadow=bool(shadow),
                    shadow_map=bool(shadow) and cm.shadow == "map")


def finish_frame(state: AeState, p: FxParams, sub: np.ndarray | None) -> None:
    """自动曝光：以本帧 1/8 降采样线性图像（未乘增益）乘基础增益测光，供下一帧使用。"""
    if p.meter and sub is not None:
        state.meter_prev = meter_sub(np.asarray(sub, np.float32) * np.float32(p.gain_base))


def _rgb_fx(lin: np.ndarray, cm: CameraModel, p: FxParams) -> np.ndarray:
    img = np.asarray(lin, np.float32) * np.float32(p.gain)
    if p.vignette_k:
        img = img * (np.float32(1.0) - np.float32(p.vignette_k) * (np.float32(1.0) - _cos4(cm)))[..., None]
    if p.read_sigma_dn > 0 or p.shot_k_dn > 0:
        h, w = img.shape[:2]
        x = np.float32(255.0) * np.clip(img, np.float32(0.0), np.float32(1.0))
        sigma = np.sqrt(np.float32(p.shot_k_dn) * x + np.float32(p.read_sigma_dn) ** 2)
        n = _normals(_U(p.key_noise), np.arange(h * w * 3, dtype=np.uint64)).reshape(h, w, 3)
        img = (x + sigma * n) / np.float32(255.0)
    return _srgb_q(img)


def _depth_fx(depth_m: np.ndarray, cfg: DepthSensorConfig, key_depth: int) -> np.ndarray:
    d = np.asarray(depth_m, np.float32)
    ok = (d >= np.float32(cfg.d_min_m)) & (d <= np.float32(cfg.d_max_m))
    out = np.where(ok, d, np.float32(0.0)).astype(np.float32)
    if cfg.a > 0:
        n = _normals(_U(key_depth), np.arange(d.size, dtype=np.uint64)).reshape(d.shape)
        sig = np.float32(cfg.a) * (np.float32(0.005) * d + np.float32(0.0005) * d * d)
        out = np.where(ok, np.maximum(out + sig * n, np.float32(cfg.d_min_m)), np.float32(0.0)).astype(np.float32)
    return out


def apply_fx_reference(lin: np.ndarray, depth_m: np.ndarray, cm: CameraModel, p: FxParams) -> tuple[np.ndarray, np.ndarray | None]:
    """参考实现：线性渲染与平面深度 → (交付 sRGB8, 观测深度或 None)；GPU 编码遍与之逐值一致（容差见一致性测试）。"""
    return _rgb_fx(lin, cm, p), (None if p.depth is None else _depth_fx(depth_m, p.depth, p.key_depth))


def apply_rgb(lin: np.ndarray, cm: CameraModel, state: AeState, *, seed: int, uav_no: int, sensor_no: int, frame_index: int,
              t_mid_s: float) -> np.ndarray:
    """线性渲染 (H, W, 3) → 交付 sRGB8；推进 `state`（自动曝光）。"""
    lin = np.asarray(lin, np.float32)
    p = frame_params(cm, state, seed=seed, uav_no=uav_no, sensor_no=sensor_no, frame_index=frame_index, t_mid_s=t_mid_s)
    finish_frame(state, p, lin[::8, ::8])
    return _rgb_fx(lin, cm, p)


def depth_sensor(depth_m: np.ndarray, cfg: DepthSensorConfig, *, seed: int, uav_no: int, sensor_no: int, frame_index: int) -> np.ndarray:
    """L2-depth（M25 §6.5.6）：越界、无几何为 0；其余加 sigma(d)·n，加噪后不低于 d_min（不产生负深度或 0 以外的越界值）。"""
    return _depth_fx(depth_m, cfg, int(key(seed, STREAM_M25, uav_no, frame_index, CHANNELS_V2["depth_noise"][0] + sensor_no)))


def gimbal_reading(g_true: tuple[float, float, float], sigma_deg: float, *, seed: int, uav_no: int, sensor_no: int,
                   frame_index: int) -> tuple[float, float, float]:
    """云台编码器读数（弧度）：每帧每轴白噪声 sigma（度，换算成弧度后相加；以前把度直接加在弧度上）。"""
    if sigma_deg <= 0:
        return tuple(float(v) for v in g_true)  # type: ignore[return-value]
    n = pixel_normals(seed, uav_no, frame_index, CHANNELS_V2["cam_gimbal"][0] + sensor_no, 3, 1).ravel()
    s = math.radians(sigma_deg)
    return tuple(float(v) + s * float(e) for v, e in zip(g_true, n, strict=True))  # type: ignore[return-value]


# ---------------------------------------------------------------------------------------------------- 渲染域着色（阴影、天气）
def contact_casters(pos: np.ndarray, dims: np.ndarray, sun: np.ndarray) -> np.ndarray:
    """接触阴影贴花参数（每演员 8 个 float32：中心 x、y，脚底 z，0；阴影方向单位向量 ux、uy，1/a，1/b）。

    演员脚底位置 `pos` (n, 3)、尺寸 `dims` (n, 3)（l、w、h，米）、指向太阳的单位向量 `sun`：阴影沿 −sun 的水平投影方向延伸，
    每米高度的地面长度 d = |sun_xy| / max(sun_z, 0.2)；贴花为椭圆，短半轴 b = max(l, w)/2 + 0.1，长半轴 a = b + 0.5·h·d，
    中心自脚底沿阴影方向偏移 0.5·h·d。"""
    pos = np.asarray(pos, np.float64).reshape(-1, 3)
    dims = np.asarray(dims, np.float64).reshape(-1, 3)
    s = np.asarray(sun, np.float64)
    dxy = -s[:2] / max(float(s[2]), CONTACT_SUN_EL_MIN)
    dl = float(np.hypot(*dxy))
    u = dxy / dl if dl > 1e-6 else np.array([1.0, 0.0])
    b = 0.5 * np.maximum(dims[:, 0], dims[:, 1]) + 0.1
    a = b + 0.5 * dims[:, 2] * dl
    out = np.zeros((len(pos), 8), np.float32)
    out[:, 0] = pos[:, 0] + u[0] * 0.5 * dims[:, 2] * dl
    out[:, 1] = pos[:, 1] + u[1] * 0.5 * dims[:, 2] * dl
    out[:, 2] = pos[:, 2]
    out[:, 4], out[:, 5] = u[0], u[1]
    out[:, 6], out[:, 7] = 1.0 / a, 1.0 / b
    return out


def rain_streak(cm: CameraModel, R_ws: np.ndarray) -> tuple[float, float]:
    """雨线的屏幕位移（像素，float32 可表示）：世界竖直向下方向在光学帧中的像平面分量方向，长度为参照距离 2 m 处的雨滴在
    曝光时间（不短于 2 ms）内下落 6.5 m/s 的投影，钳位到 [2, 40] px。"""
    from awr.sensorsvc.camera.proj import R_FLU_OPT

    Rwo = np.asarray(R_ws, np.float64) @ R_FLU_OPT
    d = Rwo.T @ np.array([0.0, 0.0, -1.0])
    v = np.array([d[0] * cm.fx, d[1] * cm.fy])
    n = float(np.hypot(*v))
    v = v / n if n > 1e-9 else np.array([0.0, 1.0])
    L = float(np.clip(cm.fy * RAIN_FALL_MPS * max(cm.exposure_s, 0.002) / RAIN_REF_M, 2.0, 40.0))
    return float(np.float32(v[0] * L)), float(np.float32(v[1] * L))


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _pcg(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.uint32)
    with np.errstate(over="ignore"):
        s = v * np.uint32(747796405) + np.uint32(2891336453)
        w = ((s >> ((s >> np.uint32(28)) + np.uint32(4))) ^ s) * np.uint32(277803737)
    return (w >> np.uint32(22)) ^ w


def _h01(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    return _pcg(x ^ _pcg(y ^ _pcg(np.uint32(k & 0xFFFFFFFF)))).astype(np.float32) * np.float32(1.0 / 4294967296.0)


def shade_reference(lin: np.ndarray, depth_m: np.ndarray, instance: np.ndarray, cm: CameraModel, R_ws: np.ndarray, p_w: np.ndarray,
                    p: FxParams, casters: np.ndarray | None = None) -> np.ndarray:
    """渲染域着色的参考实现（不含投影阴影图）：接触阴影（静态像素，`casters` 为 `contact_casters`）→ 湿润 → 消光（雾色混合，
    透过率 exp(−sigma·斜距)）→ 雨线（全部像素，雨滴深度小于场景斜距时可见）。`cm` 为渲染相机（含裕量、无畸变），返回 float32 (H, W, 3)。"""
    from awr.sensorsvc.camera.proj import R_FLU_OPT

    H, W = depth_m.shape
    c = np.asarray(lin, np.float64)[..., :3].copy()
    z = np.asarray(depth_m, np.float64)
    geo = z > 0
    xs = (np.arange(W) + 0.5 - cm.cx) / cm.fx
    ys = (np.arange(H) + 0.5 - cm.cy) / cm.fy
    rx, ry = np.meshgrid(xs, ys)
    rlen = np.sqrt(rx * rx + ry * ry + 1.0)
    Rwo = np.asarray(R_ws, np.float64) @ R_FLU_OPT
    X = (np.stack([rx * z, ry * z, z], -1) @ Rwo.T) + np.asarray(p_w, np.float64)
    if p.contact_shadow and casters is not None and len(casters):
        a = np.zeros((H, W))
        for cc in np.asarray(casters, np.float64):
            dx, dy = X[..., 0] - cc[0], X[..., 1] - cc[1]
            al = (dx * cc[4] + dy * cc[5]) * cc[6]
            ac = (-dx * cc[5] + dy * cc[4]) * cc[7]
            e = al * al + ac * ac
            att = (1.0 - _smoothstep(0.5, 1.0, e)) * (1.0 - _smoothstep(0.0, CONTACT_DZ_M, np.abs(X[..., 2] - cc[2])))
            a = np.maximum(a, att)
        m = geo & (np.asarray(instance) == 0)
        c[m] *= (1.0 - CONTACT_K * a[m])[:, None]
    if p.rain_mmh > 0:
        wet = c[geo] * WET_K
        g = wet @ np.array([0.2126, 0.7152, 0.0722])
        c[geo] = wet + (g[:, None] - wet) * WET_DESAT
    if p.fog_sigma_m > 0:
        T = np.exp(-p.fog_sigma_m * z * rlen)
        c[geo] = FOG_LIN + (c[geo] - FOG_LIN) * T[geo][:, None]
    if p.rain_mmh > 0:
        dx, dy = rain_streak(cm, R_ws)
        dens = np.float32(rain_density(p.rain_mmh))
        zz = np.where(geo, z * rlen, 1e4).astype(np.float32)
        qx, qy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        acc = np.zeros((H, W), np.float32)
        k0, k1 = p.key_rain & 0xFFFFFFFF, (p.key_rain >> 32) & 0xFFFFFFFF
        for i in range(RAIN_SAMPLES):
            t = np.float32(i) / np.float32(RAIN_SAMPLES - 1)
            sx = qx - np.float32(dx) * t
            sy = qy - np.float32(dy) * t
            cx = (np.maximum(sx, np.float32(0.0)) / np.float32(RAIN_CELL_PX)).astype(np.uint32)
            cy = (np.maximum(sy, np.float32(0.0)) / np.float32(RAIN_CELL_PX)).astype(np.uint32)
            r = _h01(cx, cy, k0)
            d = np.float32(0.5) + np.float32(29.5) * _h01(cx, cy, k1)
            hit = (r < dens) & (d < zz)
            acc += np.where(hit, np.float32(1.0) / (np.float32(1.0) + np.float32(0.15) * d), np.float32(0.0))
        al = np.clip(acc * np.float32(RAIN_ALPHA_K), 0.0, RAIN_ALPHA_MAX).astype(np.float64)
        c = c + (RAIN_STREAK_LIN - c) * al[..., None]
    return c.astype(np.float32)
