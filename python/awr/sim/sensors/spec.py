"""SensorSpec 装配（M13-FR-001、FR-002；M13 §6.4.1；AWR-16 §11.6；ADR-043；D2：M13-FR-201、FR-212、FR-215，AWR-20 §7.4、§7.5、§8.2）。

`vehicles/<model>/sensors/*.yaml` 经 contracts 的 schema 校验后装成冻结的 `SensorSpec`；另做 schema 表达不了的语义检查（云台限位
倒置、主点越界、RTK 固定解 sigma 一致）。任一失败抛 `SensorSpecError`（`110 PARAM_OUT_OF_RANGE`，detail = SENSOR_SPEC_INVALID），
机型被拒绝。

D1 类型：`vehicle/{camera,thermal,gnss,imu,lidar_livox}.schema.json`。D2 类型（schema 由 M19 起草，`sensor/*.schema.json`）：
`eo_zoom`（CAMERA）、`lwir`（THERMAL）、`radar_mmw`（RADAR）、`acoustic_array`（ACOUSTIC）、`nir_illuminator`（ILLUMINATOR）。
D2 挂载文件以 `spec_ref: <model_id>@<semver>` 引用型号库 `vehicles/sensor_models/`（M21，ADR-115），有效规格为
`deep_merge(型号, 挂载)`，校验规则 SN-1 至 SN-6、SN-8（AWR-20 §8.2；SN-7 只告警，SN-9 由 M21 目录检查）。

传感器组（rig）：机型 `params.yaml` 的 `sensors:` 映射给出槽位名与文件；字符串形式为 D1 槽位（剧本可选，沙盒不挂载），对象形式
`{file, default, removable}` 为 D2 槽位（AWR-20 §7.4）。`sensor_no`：映射中出现对象形式时按映射顺序编号（AWR-20 §7.4 第 2 条，
P600 v2：camera 0、thermal 1、mid360 2、gnss 3、imu 4、eo 5、nir_ill 6）；纯 D1 映射沿用"有视场的在前"（D1 映射本就如此排列，
编号不变）。roster `sensors[]` 只列有视场的传感器。

有效传感器组（`Rig.subset`）：同一机型按挂载掩码（按 sensor_no 的位）取子集，供沙盒实例与剧本子集使用（M13-FR-202）；云台列
`col_specs` 在子集上计算，因此 P600 v2 的沙盒实例（eo、mid360、gnss、imu、可选 nir_ill）以 eo 占云台列 0，D1 剧本以 camera 占列 0。

只依赖 numpy、yaml、jsonschema；无副作用（缓存按文件路径与修改时间）。
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass, field, replace
from functools import cache, cached_property
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .enums import KIND_BIT, SensorKind
from .frames import mount_R

__all__ = [
    "CODE_SPEC_INVALID",
    "D2_SCHEMAS",
    "DetectorSpec",
    "GimbalSpec",
    "IllumSpec",
    "Intrinsics",
    "LensSpec",
    "OutputSpec",
    "Rig",
    "SensorSpec",
    "SensorSpecError",
    "SlotMeta",
    "ZoomSpec",
    "load_rig",
    "load_sensor_file",
    "model_dirs",
    "resolve_doc",
    "rig_entries",
    "rig_for_model",
    "vehicles_dir",
]

CODE_SPEC_INVALID = 110
ROOT = Path(__file__).resolve().parents[4]
N_COLS = 2  # 云台列（与 block.N_GIMBAL 相同）

SCHEMA_KIND = {
    "awr.sensor.camera.v1": (SensorKind.CAMERA, "vehicle/camera.schema.json"),
    "awr.sensor.thermal.v1": (SensorKind.THERMAL, "vehicle/thermal.schema.json"),
    "awr.sensor.gnss.v1": (SensorKind.GNSS, "vehicle/gnss.schema.json"),
    "awr.sensor.imu.v1": (SensorKind.IMU, "vehicle/imu.schema.json"),
    "awr.sensor.lidar_livox.v1": (SensorKind.LIDAR, "vehicle/lidar_livox.schema.json"),
    "awr.sensor.eo_zoom.v1": (SensorKind.CAMERA, "sensor/eo_zoom.schema.json"),
    "awr.sensor.lwir.v1": (SensorKind.THERMAL, "sensor/lwir.schema.json"),
    "awr.sensor.radar_mmw.v1": (SensorKind.RADAR, "sensor/radar_mmw.schema.json"),
    "awr.sensor.acoustic_array.v1": (SensorKind.ACOUSTIC, "sensor/acoustic_array.schema.json"),
    "awr.sensor.nir_illuminator.v1": (SensorKind.ILLUMINATOR, "sensor/nir_illuminator.schema.json"),
}
D2_SCHEMAS = frozenset({"awr.sensor.eo_zoom.v1", "awr.sensor.lwir.v1", "awr.sensor.radar_mmw.v1",
                        "awr.sensor.acoustic_array.v1", "awr.sensor.nir_illuminator.v1"})
# 型号文件未给 `overridable[]` 时的缺省可覆盖字段（AWR-20 §7.5 类型专有字段表最后一列）
DEFAULT_OVERRIDABLE = {
    "awr.sensor.eo_zoom.v1": ("mount", "gimbal", "mass_kg", "power_w", "analysis.default"),
    "awr.sensor.nir_illuminator.v1": ("mount", "on_gimbal_of"),
    "awr.sensor.lwir.v1": ("mount", "gimbal", "lens_mm", "gimbal_mass_kg"),
    "awr.sensor.radar_mmw.v1": ("mount",),
    "awr.sensor.acoustic_array.v1": ("mount", "nr_db", "r_ego_m"),
    "awr.sensor.lidar_livox.v1": ("mount", "output"),
}
# 挂载文件总是允许的字段（AWR-20 §7.5 挂载文件表）
MOUNT_KEYS = frozenset({"schema", "sensor_id", "spec_ref", "mount", "on_gimbal_of", "lens_mm", "conf", "src"})
FOV_KINDS = (SensorKind.CAMERA, SensorKind.THERMAL, SensorKind.LIDAR, SensorKind.RADAR)
DEFAULT_RANGE_M = {SensorKind.CAMERA: 300.0, SensorKind.THERMAL: 400.0, SensorKind.LIDAR: 70.0}
D2_RANGE_M = 1000.0  # eo_zoom、lwir 的视锥绘制长度（本文设定；判据距离由 M19 计算）
# 缺省云台（M13 §6.5.2：俯仰 [−90°, +30°]、方位 [−150°, +150°]、90°/s，conf D）
GIMBAL_DEFAULTS = {"rate_max_deg_s": 90.0, "default_mode": "fixed", "default_yaw_deg": 0.0, "default_pitch_deg": -15.0}
SERVO_TIERS = ("l1", "l2")
SERVO_L2_DEFAULTS = {"acc_max_deg_s2": 400.0, "tau_s": 0.08}
SPEC_FOV_TOL_DEG = 0.2  # SN-5
LAMBDA_VIS_UM = 0.55
LAMBDA_NIR_UM = 0.85


class SensorSpecError(ValueError):
    """传感器文件无效（110 PARAM_OUT_OF_RANGE，detail = SENSOR_SPEC_INVALID；M13 §7.5）。"""

    code = CODE_SPEC_INVALID
    detail = "SENSOR_SPEC_INVALID"

    def __init__(self, where: str, problems: list[str]) -> None:
        self.where = where
        self.problems = list(problems)
        super().__init__(f"{where}: " + "; ".join(problems))

    def to_json(self) -> dict:
        return {"code": self.code, "detail": self.detail, "file": self.where, "problems": self.problems}


@dataclass(frozen=True)
class Intrinsics:
    w: int
    h: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, ...] = ()


@dataclass(frozen=True)
class GimbalSpec:
    az_min_rad: float
    az_max_rad: float
    el_min_rad: float
    el_max_rad: float
    rate_max_rad_s: float
    default_mode: str
    default_az_rad: float
    default_el_rad: float
    # 伺服档（awr.sim.sensors.servo）：l1 限速逼近（缺省，与以前逐位相同）；l2 二阶伺服（加速度上限、速度环时间常数）
    servo_tier: str = "l1"
    acc_max_rad_s2: float = math.inf
    tau_s: float = 0.0


@dataclass(frozen=True)
class DetectorSpec:
    capability: str
    p0: float
    r_fp_m: float
    t_look_s: float = 1.0
    pos_sigma_m: float = 0.0
    conf_model: str = "scenario_or_default"


@dataclass(frozen=True)
class OutputSpec:
    """eo_zoom 输出档（AWR-20 §6.2）：输出分辨率与靶面裁切（像素）。"""

    name: str
    w: int
    h: int
    crop_w: int
    crop_h: int
    fps: float


@dataclass(frozen=True)
class IllumSpec:
    """850 nm 照明器（AWR-04 §6.1；AWR-20 §6.3）：`R_nir = R_ref·(θ_ref/θ)·√(P/P_ref)·√(rho/rho_ref)·tau(R_nir)`。

    `follow_hfov = True` 为 GX40 机内补光（发散角随变焦取 HFOV）；否则发散角固定为 `divergence_rad`。"""

    wavelength_nm: float
    p_w: float
    follow_hfov: bool
    divergence_rad: float
    r_ref_m: float
    p_ref_w: float
    theta_ref_rad: float
    rho_ref: float
    r_spec_max_m: float = float("nan")
    builtin: bool = False

    def theta(self, hfov_rad: float = float("nan")) -> float:
        return float(hfov_rad) if self.follow_hfov else self.divergence_rad

    def range_m(self, theta_rad: float, rho: float | None = None, sigma_per_m: float = 0.0) -> float:
        """有效距离（m）；`sigma_per_m` 为 850 nm 消光系数（/m），按压缩映射迭代至 |Δx| < 0.1 m。"""
        th = float(theta_rad)
        if not th > 0.0:
            return float("nan")
        r = self.rho_ref if rho is None else float(rho)
        r0 = self.r_ref_m * (self.theta_ref_rad / th) * math.sqrt(self.p_w / self.p_ref_w) * math.sqrt(max(r, 0.0) / self.rho_ref)
        if not sigma_per_m > 0.0:
            return r0
        x = r0
        for _ in range(64):
            nx = r0 * math.exp(-sigma_per_m * x)
            if abs(nx - x) < 0.1:
                return nx
            x = nx
        return x


@dataclass(frozen=True)
class ZoomSpec:
    """光学变焦（AWR-20 §6.2；M13-FR-204）：靶面、焦距范围、F 数律、全程变焦时间、输出档与分析分辨率。"""

    f_min_mm: float
    f_max_mm: float
    fnum_min: float
    fnum_max: float
    law: str
    zoom_time_s: float
    pixel_um: float
    width_px: int
    height_px: int
    outputs: tuple[OutputSpec, ...]
    analysis_default: str
    k4_requires: str | None
    exposure: dict = field(default_factory=dict)
    jitter_urad: dict = field(default_factory=dict)
    spec_fov: dict = field(default_factory=dict)

    @property
    def sensor_w_mm(self) -> float:
        return self.width_px * self.pixel_um * 1e-3

    @property
    def sensor_h_mm(self) -> float:
        return self.height_px * self.pixel_um * 1e-3

    @property
    def log_rate(self) -> float:
        """|d ln f / dt| 上限（/s）：ln(f_max/f_min) / t_zoom_full。"""
        t = self.zoom_time_s
        return math.log(self.f_max_mm / self.f_min_mm) / t if t > 0 else float("inf")

    def output(self, name: str | None = None) -> OutputSpec:
        n = name or self.analysis_default
        for o in self.outputs:
            if o.name == n:
                return o
        return self.outputs[0]

    def crop_mm(self, name: str | None = None) -> tuple[float, float]:
        o = self.output(name)
        return o.crop_w * self.pixel_um * 1e-3, o.crop_h * self.pixel_um * 1e-3

    def p_eff_um(self, name: str | None = None) -> float:
        o = self.output(name)
        return self.pixel_um * o.crop_w / o.w

    def hfov(self, f_mm: float, name: str | None = None) -> float:
        return 2.0 * math.atan(self.crop_mm(name)[0] / (2.0 * f_mm))

    def vfov(self, f_mm: float, name: str | None = None) -> float:
        return 2.0 * math.atan(self.crop_mm(name)[1] / (2.0 * f_mm))

    def dfov(self, f_mm: float, name: str | None = None) -> float:
        w, h = self.crop_mm(name)
        return 2.0 * math.atan(math.hypot(w, h) / (2.0 * f_mm))

    def fnum(self, f_mm: float) -> float:
        a, b = self.fnum_min, self.fnum_max
        if self.f_max_mm <= self.f_min_mm:
            return a
        if self.law == "linear_ln_f":
            u = math.log(f_mm / self.f_min_mm) / math.log(self.f_max_mm / self.f_min_mm)
        else:
            u = (f_mm - self.f_min_mm) / (self.f_max_mm - self.f_min_mm)
        return a + (b - a) * min(max(u, 0.0), 1.0)

    def p_prime_um(self, f_mm: float, name: str | None = None, lambda_um: float = LAMBDA_VIS_UM) -> float:
        """有效像元 `p' = max(p_eff, λ·F#)`（AWR-04 §6.1 衍射项）。"""
        return max(self.p_eff_um(name), lambda_um * self.fnum(f_mm))

    def f_for_hfov(self, hfov_rad: float, name: str | None = None) -> float:
        return self.crop_mm(name)[0] / (2.0 * math.tan(0.5 * hfov_rad))

    def intrinsics(self, f_mm: float, name: str | None = None) -> Intrinsics:
        o = self.output(name)
        pw = self.pixel_um * o.crop_w / o.w
        ph = self.pixel_um * o.crop_h / o.h
        return Intrinsics(o.w, o.h, f_mm * 1e3 / pw, f_mm * 1e3 / ph, o.w / 2.0, o.h / 2.0, ())


@dataclass(frozen=True)
class LensSpec:
    f_mm: float
    f_number: float
    mass_kg: float
    power_w: float


@dataclass(frozen=True)
class SlotMeta:
    """`sensors` 映射中的槽位属性（AWR-20 §7.4）：字符串形式为 D1 槽位（legacy）。"""

    legacy: bool = True
    default: bool = True
    removable: bool = True


@dataclass(frozen=True)
class SensorSpec:
    name: str
    kind: SensorKind
    sensor_no: int
    mount_R: np.ndarray
    mount_t: np.ndarray
    intr: Intrinsics | None
    hfov_rad: float
    vfov_rad: float
    range_m: float
    gimbal: GimbalSpec | None
    detector: DetectorSpec | None
    noise: dict = field(default_factory=dict)
    rate_hz: float = float("nan")
    conf: str = "D"
    src: str = ""
    file: str = ""
    doc: dict = field(default_factory=dict)
    # D2（M13-FR-201）
    slot: SlotMeta = SlotMeta()
    schema_id: str = ""
    model_id: str = ""
    spec_ref: str = ""
    mass_kg: float = float("nan")
    power_w: float = float("nan")
    on_gimbal_of: str | None = None
    zoom: ZoomSpec | None = None
    illum: IllumSpec | None = None
    lenses: tuple[LensSpec, ...] = ()
    lens_mm: float = float("nan")
    env_limits: tuple[float, float] | None = None

    @property
    def bit(self) -> int:
        return KIND_BIT[self.kind]

    @property
    def has_fov(self) -> bool:
        return self.kind in FOV_KINDS

    @property
    def d2(self) -> bool:
        return self.schema_id in D2_SCHEMAS

    @property
    def wire_kind(self) -> str:
        """`awr.uav.payload.v1` 的 `sensors[].kind`（eo、lwir、radar、acoustic、lidar、nir_illuminator）。"""
        return {"awr.sensor.eo_zoom.v1": "eo", "awr.sensor.lwir.v1": "lwir", "awr.sensor.radar_mmw.v1": "radar",
                "awr.sensor.acoustic_array.v1": "acoustic", "awr.sensor.nir_illuminator.v1": "nir_illuminator",
                "awr.sensor.lidar_livox.v1": "lidar", "awr.sensor.thermal.v1": "lwir"}.get(self.schema_id, "eo")


@dataclass(frozen=True)
class Rig:
    """一个机型的全部传感器（按 sensor_no 排序）；`subset()` 取挂载子集。"""

    model: str
    specs: tuple[SensorSpec, ...]
    base: Rig | None = None
    mask: int = 0

    def by_name(self, name: str) -> SensorSpec | None:
        for s in self.specs:
            if s.name == name:
                return s
        return None

    def by_no(self, sensor_no: int) -> SensorSpec | None:
        for s in self.specs:
            if s.sensor_no == sensor_no:
                return s
        return None

    def by_kind(self, kind: SensorKind) -> SensorSpec | None:
        for s in self.specs:
            if s.kind == kind:
                return s
        return None

    @property
    def has_bits(self) -> int:
        b = 0
        for s in self.specs:
            b |= s.bit
        if (imu := self.by_kind(SensorKind.IMU)) is not None and isinstance(imu.doc.get("baro"), dict):
            b |= KIND_BIT[SensorKind.BARO]
        return b

    def fov_specs(self) -> tuple[SensorSpec, ...]:
        return tuple(s for s in self.specs if s.has_fov)

    @property
    def d2(self) -> bool:
        """映射中有对象形式的槽位（schema 1.1 的 D2 机型）。"""
        root = self.base or self
        return any(not s.slot.legacy for s in root.specs)

    @property
    def all_mask(self) -> int:
        m = 0
        for s in self.specs:
            m |= 1 << s.sensor_no
        return m

    @cached_property
    def col_specs(self) -> tuple[SensorSpec | None, ...]:
        """云台列的所有者：列 0 取首个 CAMERA 类、列 1 取首个 THERMAL 类（跟随他人云台的不占列）；纯 D1 组与 by_kind 相同。"""
        return tuple(next((s for s in self.specs if s.kind == kind and s.on_gimbal_of is None), None)
                     for kind in (SensorKind.CAMERA, SensorKind.THERMAL))

    def col_of(self, name: str) -> int | None:
        """传感器所在云台列：列所有者为本列，`on_gimbal_of` 跟随者取被跟随者的列；无云台为 None。"""
        s = self.by_name(name)
        if s is None:
            return None
        if s.on_gimbal_of is not None:
            s = self.by_name(s.on_gimbal_of)
            if s is None:
                return None
        for k, c in enumerate(self.col_specs):
            if c is not None and c.name == s.name and c.gimbal is not None:
                return k
        return None

    def followers(self, k: int) -> tuple[SensorSpec, ...]:
        """跟随云台列 k 的传感器（照明器、吊舱内第二传感器）。"""
        c = self.col_specs[k] if 0 <= k < len(self.col_specs) else None
        if c is None:
            return ()
        return tuple(s for s in self.specs if s.on_gimbal_of == c.name)

    def mask_of(self, names: list[str] | tuple[str, ...] | set[str]) -> int:
        m = 0
        for s in self.specs:
            if s.name in names:
                m |= 1 << s.sensor_no
        return m

    def default_mask(self, *, sandbox: bool) -> int:
        """缺省挂载（AWR-20 §7.4）：沙盒实例只装对象形式且 `default = true` 的槽位；剧本另含全部 D1 槽位。"""
        m = 0
        for s in self.specs:
            if (s.slot.default and not s.slot.legacy) or (s.slot.legacy and not sandbox):
                m |= 1 << s.sensor_no
        return m

    def subset(self, mask: int, lens: tuple[float, ...] = ()) -> Rig:
        """按 sensor_no 位掩码取子集（同一 SensorSpec 对象，编号不变）；`lens` 为 (列, 焦距) 的展开替换（LWIR 换镜头）。"""
        root = self.base or self
        key = (id(root), int(mask), tuple(lens))
        r = _SUBSETS.get(key)
        if r is not None and r.base is root:
            return r
        specs = [s for s in root.specs if (int(mask) >> s.sensor_no) & 1]
        if lens:
            sel = dict(zip(lens[0::2], lens[1::2], strict=True))
            specs = [with_lens(s, sel[float(s.sensor_no)]) if float(s.sensor_no) in sel else s for s in specs]
        r = Rig(root.model, tuple(specs), root, int(mask))
        _SUBSETS[key] = r
        return r


_SUBSETS: dict[tuple, Rig] = {}


def vehicles_dir() -> Path:
    return Path(os.environ.get("AWR_VEHICLES_DIR") or (ROOT / "vehicles"))


def model_dirs(extra: list[Path] | tuple[Path, ...] | None = None) -> list[Path]:
    """型号库搜索路径：显式给出者在前，其后 `AWR_SENSOR_MODELS_DIR`（以 os.pathsep 分隔），最后 `vehicles/sensor_models/`。"""
    out: list[Path] = [Path(p) for p in (extra or ())]
    env = os.environ.get("AWR_SENSOR_MODELS_DIR")
    if env:
        out += [Path(p) for p in env.split(os.pathsep) if p]
    out.append(vehicles_dir() / "sensor_models")
    return out


# ---------------------------------------------------------------- schema 校验
def _registry():
    from awr.sim.schemas import contracts_registry  # 与机型、剧本校验共用一次扫描（ADR-061）

    return contracts_registry()


@cache
def _validator(rel: str):
    from jsonschema import Draft202012Validator

    from awr.contracts._paths import contracts_root

    schema = json.loads((contracts_root() / rel).read_text(encoding="utf-8"))
    return Draft202012Validator(schema, registry=_registry())


def schema_problems(doc: dict) -> list[str]:
    sid = doc.get("schema") if isinstance(doc, dict) else None
    if sid not in SCHEMA_KIND:
        return [f"schema: unknown sensor schema {sid!r}"]
    v = _validator(SCHEMA_KIND[sid][1])
    return [f"schema {'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message[:160]}" for e in v.iter_errors(doc)]


def semantic_problems(doc: dict) -> list[str]:
    """schema 之外的检查（M13-AC-001；SN-5、SN-6）：云台限位倒置、主点越界、速率非正、RTK 固定解 sigma 一致、焦距范围、
    由靶面与焦距复算的视场与 `spec_fov` 误差。"""
    out: list[str] = []
    g = doc.get("gimbal")
    if isinstance(g, dict):
        if "pitch_min_deg" in g and "pitch_max_deg" in g and not float(g["pitch_min_deg"]) < float(g["pitch_max_deg"]):
            out.append("gimbal: pitch_min_deg must be < pitch_max_deg (GIMBAL_LIMITS_INVERTED)")
        if "yaw_min_deg" in g and "yaw_max_deg" in g and not float(g["yaw_min_deg"]) < float(g["yaw_max_deg"]):
            out.append("gimbal: yaw_min_deg must be < yaw_max_deg (GIMBAL_LIMITS_INVERTED)")
        for k in ("default_pitch_deg", "default_yaw_deg"):
            lo, hi = ("pitch_min_deg", "pitch_max_deg") if "pitch" in k else ("yaw_min_deg", "yaw_max_deg")
            if k in g and lo in g and hi in g and not float(g[lo]) <= float(g[k]) <= float(g[hi]):
                out.append(f"gimbal: {k} outside [{lo}, {hi}]")
        if "rate_max_deg_s" in g and not float(g["rate_max_deg_s"]) > 0.0:
            out.append("gimbal: rate_max_deg_s must be > 0")
    if "cx" in doc and "width" in doc and not 0.0 <= float(doc["cx"]) <= float(doc["width"]):
        out.append("intrinsics: cx outside [0, width]")
    if "cy" in doc and "height" in doc and not 0.0 <= float(doc["cy"]) <= float(doc["height"]):
        out.append("intrinsics: cy outside [0, height]")
    if doc.get("schema") == "awr.sensor.gnss.v1":
        rtk = doc.get("rtk") or {}
        fx = rtk.get("fixed") or {}
        out.extend(f"rtk.fixed.{k} must equal rtk.{k}" for k in ("sigma_h_m", "sigma_v_m")
                   if k in fx and k in rtk and float(fx[k]) != float(rtk[k]))
        if rtk.get("enabled") and ("sigma_h_m" not in rtk or "sigma_v_m" not in rtk):
            out.append("rtk: sigma_h_m and sigma_v_m are required when enabled")
    if doc.get("schema") in D2_SCHEMAS:  # M13-FR-201：每个 D2 有效规格带置信度与来源
        out.extend(f"{k} is required (M13-FR-201)" for k in ("conf", "src") if not doc.get(k))
    if doc.get("schema") == "awr.sensor.eo_zoom.v1":
        out += _eo_problems(doc)
    if doc.get("schema") == "awr.sensor.lwir.v1":
        lenses = [float(x.get("f_mm", 0.0)) for x in (doc.get("lenses") or [])]
        lm = doc.get("lens_mm")
        if lm is not None and lenses and not any(abs(float(lm) - f) < 1e-9 for f in lenses):
            out.append(f"lens_mm {lm} not in lenses[] {lenses}")
    if doc.get("schema") == "awr.sensor.nir_illuminator.v1":
        ref = doc.get("ref") or {}
        miss = [k for k in ("r_m", "p_w", "theta_deg", "rho") if k not in ref]
        if miss or "p_w" not in doc or "divergence_deg" not in doc:
            out.append(f"illuminator: ref fields missing {miss or ['p_w/divergence_deg']} (SN-8)")
    return out


def _eo_problems(doc: dict) -> list[str]:
    out: list[str] = []
    op = doc.get("optics") or {}
    sn = doc.get("sensor") or {}
    try:
        f0, f1 = float(op["f_min_mm"]), float(op["f_max_mm"])
    except (KeyError, TypeError, ValueError):
        return ["optics: f_min_mm and f_max_mm are required"]
    if not 0.0 < f0 <= f1:
        out.append("optics: f_min_mm must be in (0, f_max_mm] (FOCAL_RANGE_INVERTED)")
        return out
    if not sn:
        return [*out, "sensor: width_px, height_px and pixel_um are required"]
    names = [str(o.get("name")) for o in (doc.get("outputs") or [])]
    ad = (doc.get("analysis") or {}).get("default")
    if ad is not None and names and ad not in names:
        out.append(f"analysis.default {ad!r} not in outputs[] {names}")
    sf = doc.get("spec_fov") or {}
    if sf:
        w = int(sn["width_px"]) * float(sn["pixel_um"]) * 1e-3
        h = int(sn["height_px"]) * float(sn["pixel_um"]) * 1e-3
        calc = {"hfov_deg": (w, w), "vfov_deg": (h, h), "dfov_deg": (math.hypot(w, h), math.hypot(w, h))}
        for k, (a, _b) in calc.items():
            v = sf.get(k)
            if not v:
                continue
            got = (math.degrees(2 * math.atan(a / (2 * f0))), math.degrees(2 * math.atan(a / (2 * f1))))
            out.extend(f"spec_fov.{k}[{i}] {v[i]} vs computed {got[i]:.3f} (> {SPEC_FOV_TOL_DEG} deg, SN-5)"
                       for i in (0, 1) if abs(got[i] - float(v[i])) > SPEC_FOV_TOL_DEG)
    return out


# ---------------------------------------------------------------- spec_ref（AWR-20 §7.5；SN-1、SN-2、SN-4）
def _yaml_load(text: str):
    """`yaml.safe_load` 的等价物（有 libyaml 时用 CSafeLoader；sim-core 启动路径，FX-SIM1）。"""
    return yaml.load(text, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


_MODEL_CACHE: dict[tuple[str, int], dict] = {}


def _read_model(p: Path) -> dict:
    st = p.stat()
    key = (str(p), st.st_mtime_ns)
    d = _MODEL_CACHE.get(key)
    if d is None:
        try:
            d = _yaml_load(p.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as e:
            raise SensorSpecError(str(p), [f"read: {e} (SN-1)"]) from e
        if not isinstance(d, dict):
            raise SensorSpecError(str(p), ["model file is not a mapping (SN-1)"])
        probs = schema_problems(d)
        if not probs and not (d.get("model_id") and d.get("model_version")):
            probs = ["model file needs model_id and model_version (SN-1)"]
        if probs:
            raise SensorSpecError(str(p), [f"{x} (SN-1)" for x in probs])
        _MODEL_CACHE[key] = d
    return d


def find_model(spec_ref: str, dirs: list[Path] | None = None) -> tuple[dict, Path]:
    """`<model_id>@<semver>` -> (型号文档, 路径)；查 `<id>@<ver>.yaml` 再查 `<id>.yaml`（版本须一致）；找不到为 SN-2。"""
    if "@" not in spec_ref:
        raise SensorSpecError(spec_ref, [f"spec_ref {spec_ref!r} is not <model_id>@<semver> (SN-2)"])
    mid, ver = spec_ref.split("@", 1)
    seen: list[str] = []
    for d in model_dirs(dirs):
        for p in (d / f"{mid}@{ver}.yaml", d / f"{mid}.yaml"):
            if not p.is_file():
                continue
            doc = _read_model(p)
            seen.append(f"{p}: {doc.get('model_id')}@{doc.get('model_version')}")
            if str(doc.get("model_id")) == mid and str(doc.get("model_version")) == ver:
                return doc, p
    hint = "; found " + ", ".join(seen) if seen else ""
    raise SensorSpecError(spec_ref, [f"spec_ref {spec_ref} not found in the sensor model library{hint} (SN-2)"])


def _get_path(d: Any, path: str) -> Any:
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return _MISSING
        cur = cur[k]
    return cur


_MISSING = object()


def deep_merge(a: dict, b: dict) -> dict:
    """对象逐键递归，数组与标量整体替换（与 D1 `variants` 同一规则）。"""
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _sn4(model: dict, mount: dict) -> list[str]:
    allowed = tuple(model.get("overridable") or DEFAULT_OVERRIDABLE.get(str(model.get("schema")), ("mount",)))
    out: list[str] = []

    def covered(p: str) -> bool:
        return any(p == a or p.startswith(a + ".") for a in allowed)

    def walk(m: dict, path: str) -> None:
        for k, v in m.items():
            p = f"{path}.{k}" if path else str(k)
            if not path and k in MOUNT_KEYS:
                continue
            if covered(p):
                continue
            if isinstance(v, dict) and any(a.startswith(p + ".") for a in allowed):
                walk(v, p)
                continue
            if _get_path(model, p) != v:
                out.append(f"{p}: mount file overrides a field outside overridable[] {list(allowed)} (SN-4)")

    walk(mount, "")
    return out


def resolve_doc(doc: dict, where: str = "", dirs: list[Path] | None = None) -> dict:
    """挂载文件 -> 有效规格（无 `spec_ref` 时原样返回）。"""
    ref = doc.get("spec_ref") if isinstance(doc, dict) else None
    if not ref:
        return doc
    mount_probs = schema_problems(doc)
    if mount_probs:
        raise SensorSpecError(where or str(ref), mount_probs)
    model, mp = find_model(str(ref), dirs)
    if model.get("schema") != doc.get("schema"):
        raise SensorSpecError(where or str(ref), [f"schema {doc.get('schema')} differs from the model {model.get('schema')} at {mp} (SN-2)"])
    probs = _sn4(model, doc)
    if probs:
        raise SensorSpecError(where or str(ref), probs)
    eff = deep_merge(model, doc)
    eff.pop("overridable", None)
    if "mount" not in eff:
        raise SensorSpecError(where or str(ref), ["mount is required in the effective spec (SN-3)"])
    return eff


# ---------------------------------------------------------------- 装配
def _fov(w: float, h: float, fx: float, fy: float, cx: float, cy: float) -> tuple[float, float]:
    """非对称主点时取两侧之和（M13 §6.4.1）。"""
    return (math.atan(cx / fx) + math.atan((w - cx) / fx), math.atan(cy / fy) + math.atan((h - cy) / fy))


def _mount(doc: dict) -> tuple[np.ndarray, np.ndarray]:
    m = doc.get("mount") or {}
    t = np.asarray(m.get("xyz_m") or m.get("t") or [0.0, 0.0, 0.0], np.float64)
    if "q_xyzw" in m:
        from .frames import quat_to_R

        q = np.asarray(m["q_xyzw"], np.float64)
        R = quat_to_R((q / np.linalg.norm(q))[None])[0]
    else:
        R = mount_R(m.get("rpy_deg") or [0.0, 0.0, 0.0])
    R.flags.writeable = False
    t.flags.writeable = False
    return R, t


def _gimbal(doc: dict) -> GimbalSpec | None:
    g = doc.get("gimbal")
    if not isinstance(g, dict):
        return None
    d = {**GIMBAL_DEFAULTS, **g}
    if "default_pitch_deg" not in g:  # 缺省俯仰钳到限位内（eo_zoom 型号文件不给 default_*）
        d["default_pitch_deg"] = min(max(float(d["default_pitch_deg"]), float(d["pitch_min_deg"])), float(d["pitch_max_deg"]))
    r = math.radians
    servo = d.get("servo") or {}
    tier = str(servo.get("tier", "l1"))
    if tier not in SERVO_TIERS:
        raise SensorSpecError("gimbal.servo", [f"tier must be one of {SERVO_TIERS}, got {tier!r}"])
    acc = float(servo.get("acc_max_deg_s2", SERVO_L2_DEFAULTS["acc_max_deg_s2"]))
    tau = float(servo.get("tau_s", SERVO_L2_DEFAULTS["tau_s"]))
    if tier == "l2" and not (acc > 0 and tau > 0):
        raise SensorSpecError("gimbal.servo", ["l2 needs acc_max_deg_s2 > 0 and tau_s > 0"])
    return GimbalSpec(r(float(d["yaw_min_deg"])), r(float(d["yaw_max_deg"])), r(float(d["pitch_min_deg"])),
                      r(float(d["pitch_max_deg"])), r(float(d["rate_max_deg_s"])), str(d["default_mode"]),
                      r(float(d["default_yaw_deg"])), r(float(d["default_pitch_deg"])),
                      servo_tier=tier, acc_max_rad_s2=r(acc) if tier == "l2" else math.inf, tau_s=tau if tier == "l2" else 0.0)


def _detector(doc: dict) -> DetectorSpec | None:
    d = doc.get("detector")
    if not isinstance(d, dict) or "capability" not in d:
        return None
    return DetectorSpec(str(d["capability"]), float(d["p0"]), float(d["r_fp_m"]), float(d.get("t_look_s", 1.0)),
                        float(d.get("pos_sigma_m", 0.0)))


def _val(x: Any) -> float:
    if isinstance(x, dict):
        x = x.get("value")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _zoom(doc: dict) -> ZoomSpec:
    sn, op = doc["sensor"], doc["optics"]
    fn = op.get("f_number") or {}
    w, h = int(sn["width_px"]), int(sn["height_px"])
    outs = tuple(OutputSpec(str(o["name"]), int(o["width_px"]), int(o["height_px"]), int(o.get("crop_w_px") or w),
                            int(o.get("crop_h_px") or round(int(o.get("crop_w_px") or w) * h / w)), float(o.get("fps") or 30.0))
                 for o in (doc.get("outputs") or [{"name": "native", "width_px": w, "height_px": h, "crop_w_px": w}]))
    an = doc.get("analysis") or {}
    names = [o.name for o in outs]
    ad = str(an.get("default") or ("1080p" if "1080p" in names else names[0]))
    return ZoomSpec(float(op["f_min_mm"]), float(op["f_max_mm"]), float(fn.get("at_min", 2.0)),
                    float(fn.get("at_max", fn.get("at_min", 2.0))), str(fn.get("law", "linear_f")),
                    float(op.get("zoom_time_s", 4.0)), float(sn["pixel_um"]), w, h, outs, ad, an.get("k4_requires"),
                    dict(doc.get("exposure") or {}), dict(doc.get("jitter_urad") or {}), dict(doc.get("spec_fov") or {}))


def _illum_builtin(doc: dict) -> IllumSpec | None:
    il = (doc.get("night") or {}).get("illuminator")
    if not isinstance(il, dict):
        return None
    th_ref = math.radians(float(il.get("theta_ref_deg", 6.6)))
    follow = str(il.get("divergence", "follow_hfov")) == "follow_hfov"
    return IllumSpec(float(il.get("wavelength_nm", 850.0)), float(il["p_w"]), follow,
                     math.radians(float(il.get("divergence_deg", 0.0) or 0.0)) if not follow else float("nan"),
                     float(il["r_ref_m"]), float(il.get("p_ref_w", il["p_w"])), th_ref, float(il.get("rho_ref", 0.3)),
                     float(il.get("r_spec_max_m", float("nan"))), True)


def _illum_ext(doc: dict) -> IllumSpec:
    ref = doc.get("ref") or {}
    return IllumSpec(float(doc.get("wavelength_nm", 850.0)), float(doc["p_w"]), False, math.radians(float(doc["divergence_deg"])),
                     float(ref["r_m"]), float(ref["p_w"]), math.radians(float(ref["theta_deg"])), float(ref["rho"]), float("nan"), False)


def _lenses(doc: dict) -> tuple[LensSpec, ...]:
    return tuple(LensSpec(float(x["f_mm"]), float(x.get("f_number", 1.0)), float(x.get("mass_kg", float("nan"))),
                          float(x.get("power_w", float("nan")))) for x in (doc.get("lenses") or []))


def _lwir_intr(doc: dict, f_mm: float) -> Intrinsics:
    det = doc.get("detector") or {}
    w, h, p = int(det.get("width_px", 640)), int(det.get("height_px", 512)), float(det.get("pixel_um", 12.0))
    f_px = f_mm * 1e3 / p
    return Intrinsics(w, h, f_px, f_px, w / 2.0, h / 2.0, ())


def with_lens(s: SensorSpec, f_mm: float) -> SensorSpec:
    """LWIR 换镜头（实例覆盖 `lens_mm`，M13-FR-202）：内参、视场、质量与功率按镜头表。"""
    if s.schema_id != "awr.sensor.lwir.v1" or not s.lenses:
        return s
    ln = next((x for x in s.lenses if abs(x.f_mm - f_mm) < 1e-9), None)
    if ln is None:
        raise SensorSpecError(s.file or s.name, [f"lens_mm {f_mm} not in lenses[] {[x.f_mm for x in s.lenses]}"])
    it = _lwir_intr(s.doc, f_mm)
    hf, vf = _fov(it.w, it.h, it.fx, it.fy, it.cx, it.cy)
    gm = float((s.doc.get("gimbal_mass_kg") or 0.0) if s.gimbal is not None else 0.0)
    mass = ln.mass_kg + gm if math.isfinite(ln.mass_kg) else s.mass_kg
    return replace(s, intr=it, hfov_rad=hf, vfov_rad=vf, lens_mm=float(f_mm), mass_kg=mass,
                   power_w=ln.power_w if math.isfinite(ln.power_w) else s.power_w)


def build_spec(name: str, doc: dict, sensor_no: int, file: str = "", *, slot: SlotMeta | None = None,
               model_dirs_: list[Path] | None = None) -> SensorSpec:
    where = file or name
    eff = resolve_doc(doc, where, model_dirs_) if isinstance(doc, dict) and doc.get("spec_ref") else doc
    problems = schema_problems(eff)
    if not problems:
        problems = semantic_problems(eff)
    if problems:
        raise SensorSpecError(where, problems)
    sid = str(eff["schema"])
    kind = SCHEMA_KIND[sid][0]
    R, t = _mount(eff)
    intr = None
    hf = vf = float("nan")
    zoom = illum = None
    lenses: tuple[LensSpec, ...] = ()
    lens_mm = float("nan")
    rng: Any = eff.get("range_m")
    if sid == "awr.sensor.eo_zoom.v1":
        zoom = _zoom(eff)
        intr = zoom.intrinsics(zoom.f_min_mm)
        hf, vf = zoom.hfov(zoom.f_min_mm), zoom.vfov(zoom.f_min_mm)
        illum = _illum_builtin(eff)
        rng = D2_RANGE_M
    elif sid == "awr.sensor.lwir.v1":
        lenses = _lenses(eff)
        lm = eff.get("lens_mm", lenses[0].f_mm if lenses else None)
        if lm is None:
            raise SensorSpecError(where, ["lwir: lens_mm is required in the mount file"])
        lens_mm = float(lm)
        intr = _lwir_intr(eff, lens_mm)
        hf, vf = _fov(intr.w, intr.h, intr.fx, intr.fy, intr.cx, intr.cy)
        rng = D2_RANGE_M
    elif sid == "awr.sensor.radar_mmw.v1":
        fov = eff.get("fov") or {}
        hf = 2.0 * math.radians(float(fov.get("az_deg", 60.0)))  # az_deg、el_deg 为半角（±）
        vf = 2.0 * math.radians(float(fov.get("el_deg", 15.0)))
        rng = eff.get("r_max_m", eff.get("r_ref_m"))
    elif sid == "awr.sensor.nir_illuminator.v1":
        illum = _illum_ext(eff)
    elif kind in (SensorKind.CAMERA, SensorKind.THERMAL):
        k = eff.get("distortion", {}).get("k") or []
        intr = Intrinsics(int(eff["width"]), int(eff["height"]), float(eff["fx"]), float(eff["fy"]), float(eff["cx"]),
                          float(eff["cy"]), tuple(float(x) for x in k))
        hf, vf = _fov(intr.w, intr.h, intr.fx, intr.fy, intr.cx, intr.cy)
    elif kind == SensorKind.LIDAR:
        fov = eff["fov"]
        hf = math.radians(float(fov["h_deg"]))
        vf = math.radians(float(fov["v_max_deg"]) - float(fov["v_min_deg"]))
    if kind == SensorKind.LIDAR:
        rng = (eff.get("range") or {}).get("hard_max_m", rng)
    range_m = float(rng) if rng is not None else DEFAULT_RANGE_M.get(kind, float("nan"))
    rate = eff.get("rate_hz", eff.get("frame_rate_hz"))
    if rate is None and sid == "awr.sensor.lwir.v1":
        rate = (eff.get("detector") or {}).get("frame_hz")
    noise: dict[str, Any] = {}
    if kind == SensorKind.GNSS:
        noise = {k: eff.get(k) for k in ("gauss_markov", "dgps", "rtk", "timing", "sats", "hdop", "fix_type", "warm_start")}
        R = np.eye(3)
        R.flags.writeable = False
        t = np.asarray((eff.get("antenna") or {}).get("xyz_m") or [0.0, 0.0, 0.0], np.float64)
        t.flags.writeable = False
    elif kind == SensorKind.IMU:
        noise = {k: eff.get(k) for k in ("gyro", "accel", "baro")}
    elif kind == SensorKind.LIDAR:
        noise = dict(eff.get("noise") or {})
    elif sid == "awr.sensor.thermal.v1":
        noise = {"netd_k": float(eff["netd_k"])}
    elif sid == "awr.sensor.lwir.v1":
        noise = {"netd_k": float((eff.get("detector") or {}).get("netd_mk", 50.0)) * 1e-3}
    el = (eff.get("env_limits") or {}).get("temp_c")
    on_g = eff.get("on_gimbal_of")
    spec = SensorSpec(name=name, kind=kind, sensor_no=int(sensor_no), mount_R=R, mount_t=t, intr=intr, hfov_rad=hf,
                      vfov_rad=vf, range_m=range_m, gimbal=None if on_g else _gimbal(eff), detector=_detector(eff), noise=noise,
                      rate_hz=float(rate) if rate is not None else float("nan"), conf=str(eff.get("conf", "D")),
                      src=str(eff.get("src", "")), file=file, doc=eff, slot=slot or SlotMeta(), schema_id=sid,
                      model_id=str(eff.get("model_id") or ""), spec_ref=str(doc.get("spec_ref") or ""),
                      mass_kg=_val(eff.get("mass_kg")), power_w=_val(eff.get("power_w")),
                      on_gimbal_of=str(on_g) if on_g else None, zoom=zoom, illum=illum, lenses=lenses, lens_mm=lens_mm,
                      env_limits=(float(el[0]), float(el[1])) if el else None)
    if sid == "awr.sensor.lwir.v1" and lenses:
        spec = with_lens(spec, lens_mm)
    return spec


def load_sensor_file(path: Path | str, name: str | None = None, sensor_no: int = 0, *, slot: SlotMeta | None = None,
                     model_dirs_: list[Path] | None = None) -> SensorSpec:
    p = Path(path)
    try:
        doc = _yaml_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise SensorSpecError(str(p), [f"read: {e}"]) from e
    if not isinstance(doc, dict):
        raise SensorSpecError(str(p), ["document is not a mapping"])
    return build_spec(name or p.stem, doc, sensor_no, str(p), slot=slot, model_dirs_=model_dirs_)


def _order(entries: list[tuple[str, dict, str, SlotMeta]]) -> list[tuple[str, dict, str, SlotMeta]]:
    if any(not e[3].legacy for e in entries):  # D2 映射：按映射顺序（AWR-20 §7.4 第 2 条）
        return list(entries)
    fov = [e for e in entries if SCHEMA_KIND.get(e[1].get("schema"), (None,))[0] in FOV_KINDS]
    rest = [e for e in entries if e not in fov]
    return fov + rest


def _slot_entry(name: str, v: Any) -> tuple[str, SlotMeta]:
    if isinstance(v, dict):
        return str(v.get("file")), SlotMeta(False, bool(v.get("default", False)), bool(v.get("removable", True)))
    return str(v), SlotMeta(True, True, True)


def rig_problems(specs: tuple[SensorSpec, ...]) -> list[str]:
    """组级检查：SN-8（`on_gimbal_of` 指向存在、带云台的 eo_zoom 槽位）；对象形式槽位名合法。"""
    out: list[str] = []
    by = {s.name: s for s in specs}
    for s in specs:
        if s.on_gimbal_of is None:
            continue
        t = by.get(s.on_gimbal_of)
        if t is None or t.schema_id != "awr.sensor.eo_zoom.v1" or t.gimbal is None:
            out.append(f"{s.name}: on_gimbal_of {s.on_gimbal_of!r} is not an eo_zoom slot with a gimbal (SN-8)")
    return out


def load_rig(model_dir: Path | str, sensors_map: dict[str, Any] | None = None, *, model_dirs_: list[Path] | None = None) -> Rig:
    """读取一个机型目录下的传感器组；`sensors_map` 缺省取 `params.yaml` 的 `sensors:`（无该键时为空组）。"""
    d = Path(model_dir)
    if sensors_map is None:
        try:
            params = _yaml_load((d / "params.yaml").read_text(encoding="utf-8")) or {}
        except OSError:
            params = {}
        sensors_map = dict(params.get("sensors") or {})
    raw: list[tuple[str, dict, str, SlotMeta]] = []
    for name, v in sensors_map.items():
        rel, meta = _slot_entry(str(name), v)
        p = d / rel
        try:
            doc = _yaml_load(p.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as e:
            raise SensorSpecError(str(p), [f"read: {e}"]) from e
        if not isinstance(doc, dict):
            raise SensorSpecError(str(p), ["document is not a mapping"])
        raw.append((str(name), doc, str(p), meta))
    dirs = [*(model_dirs_ or []), d.parent / "sensor_models"]  # 与机型目录同级的型号库（vehicles/sensor_models/）
    specs = tuple(build_spec(n, doc, i, f, slot=m, model_dirs_=dirs) for i, (n, doc, f, m) in enumerate(_order(raw)))
    probs = rig_problems(specs)
    if probs:
        raise SensorSpecError(str(d), probs)
    return Rig(d.name, specs)


_RIGS: dict[tuple[str, float], Rig] = {}


def rig_for_model(model: str, base: Path | None = None) -> Rig:
    """按机型目录名（roster `model`，例如 p600、x500）取传感器组；按 params.yaml 修改时间缓存。"""
    d = (base or vehicles_dir()) / model
    try:
        mt = (d / "params.yaml").stat().st_mtime
    except OSError:
        return Rig(model, ())
    key = (str(d), mt)
    r = _RIGS.get(key)
    if r is None:
        r = _RIGS[key] = load_rig(d)
    return r


def rig_entries(rig: Rig, names: list[str] | None = None) -> list[dict]:
    """roster `sensors[]`（M13-FR-002）：只列有视场的传感器；`names` 为剧本 `vehicles[].sensors` 子集（缺省全部）。"""
    kind_name = {SensorKind.CAMERA: "camera", SensorKind.LIDAR: "lidar", SensorKind.THERMAL: "thermal",
                 SensorKind.RADAR: "radar"}
    out = []
    for s in rig.specs:
        if not s.has_fov or (names is not None and s.name not in names):
            continue
        out.append({"sensor_no": s.sensor_no, "name": s.name, "kind": kind_name[s.kind]})
    return out
