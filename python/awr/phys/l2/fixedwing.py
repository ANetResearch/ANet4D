"""Fixed-wing domain model of the unified L2 physics engine (``awr.phys.l2.fixedwing``): aeroplanes on the shared rigid body.

gz-sim flies fixed wings through a lift-drag plugin configured per link, not through an aircraft model; the domain here
integrates the small-UAV flight dynamics literature instead: the coefficient-table 6-DoF model of Beard & McLain (*Small
Unmanned Aircraft*, ch. 4; the same form the PX4 ``advanced_lift_drag`` plugin descends from) with the successive-loop-closure
autopilot of its ch. 6, plus what a sandbox needs (launch and recovery, a course / altitude / airspeed reference generator,
world collision):

- **aerodynamics**: lift ``C_L(alpha)`` as the linear ``C_L0 + C_L_alpha·alpha`` blended into the flat-plate ``2·sign(alpha)·sin²alpha·cos alpha`` past
  the stall angle (sigmoid of sharpness ``STALL_M`` around ``alpha0 = (C_Lmax − C_L0)/C_L_alpha``, so the profile's ``cl_max`` is where
  the wing stalls), drag ``C_D0 + C_L²/(π·e·AR)`` (the profile's ``cd0`` and ``oswald_e``), the pitch-rate, elevator,
  sideslip, roll-rate, yaw-rate, aileron and rudder derivatives of the table; forces in the stability axes turned into the
  body (``f_x = −D cos alpha + L sin alpha``, ``f_z = −D sin alpha − L cos alpha``), moments ``l, m, n`` scaled by ``b`` and ``c``;
- **propulsion**: thrust along the body x axis, ``T = δt · min(T_static, η_prop·P_max / V_a)`` (static-thrust and power
  limited), throttle through a first-order motor lag; the propeller's own torque and gyroscopic terms are dropped;
- **actuation**: elevator, aileron and rudder deflections through a first-order servo lag, saturated at the package limits;
- **controller** (shared with the oracle, so the comparison isolates dynamics and integration): course → coordinated-turn
  roll command (``φ_c = atan(V_a·ψ̇_c / g)`` from the same course law as the fixed-wing fast tier, stall-protected
  ``φ_max(V_a)``, slewed at ``roll_rate_max``) → roll PD on the ailerons; climb rate → pitch command (flight-path angle +
  angle of attack + PI on the climb-rate error) → pitch PD on the elevator; airspeed → throttle (drag feed-forward + PI,
  anti-windup), the airspeed reference ramped at the package's ``accel_max / decel_max``; sideslip → rudder. The roll
  and pitch gains come from the aircraft's own control derivatives at cruise (the saturation design of Beard & McLain
  §6.1.1), nothing is tuned per aircraft here;
- **phases**: ``GROUND`` (at rest on the launcher), ``LAUNCH`` (catapult: the rail, elevated ``RAIL_ELEV_RAD``, carries
  the body at constant acceleration to ``v_min + LAUNCH_MARGIN`` over ``launch.t_s``, attitude along the rail), ``FLY``, ``CHUTE`` (parachute recovery:
  motor off, controls neutral, a drag of terminal velocity ``recovery.descent_mps`` relative to the air, the body hangs
  level and drifts with the wind), ``OFF`` (kill: motor off, controls neutral, the airframe is left to the air);
- **references**: the tier's command set. ``WingTracker`` turns commands into airspeed, course (or course rate), course-rate
  feed-forward and climb-rate references with the vector-field guidance of the fixed-wing fast tier
  (``awr.sim.fixedwing.kinematics``: straight-line and circle fields): goto / follow_path fly the legs and loiter at the
  end, orbit is a loiter circle of at least the loiter radius, hover is a loiter about the point, velocity is an airspeed
  along a course (a yaw rate without a course is a turn at that rate), takeoff is the catapult launch and a climb to the
  takeoff altitude followed by a loiter, land is the parachute, kill is motor and controls off. A fixed wing never stops in
  the air: the loiter is its hover.

Parameters come from the vehicle profile (``airframe``, ``fw_dynamics``, ``inertia_kgm2``; ``awr.vehicle.v1`` schema 1.2).
Coordinates: world ENU, body FLU, quaternion xyzw WORLD←FLU (``awr.phys.l2.rigid``); the aerodynamic equations are
evaluated in the FRD / NED convention of the literature and mapped (``y → −y``, ``z → −z``) at the boundary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from awr.phys.l2 import rigid as RB
from awr.sim.fixedwing import kinematics as KN

__all__ = ["ALT_TOL_M", "CHUTE", "COS_MIN", "DAMP_Z", "E_PHI_MAX", "E_THETA_MAX", "FLY", "GROUND", "K_BETA", "K_HDOT_I", "K_HDOT_P",
           "K_V", "K_V_I", "LAUNCH", "LAUNCH_MARGIN_MPS", "OFF", "RAIL_ELEV_RAD", "RHO0", "STALL_M", "THETA_MAX", "V_SP_MIN", "WingFleet", "WingParams",
           "WingRef", "WingTracker", "euler_flu", "q_from_euler_flu"]

RHO0 = 1.225
G = RB.G
STALL_M = 100.0                  # sharpness of the blend between the linear lift and the flat plate (B&M use 50 with a far-off alpha0)
STALL_BLEND_OFFSET = 3.0         # the blend centre sits 3/M past the linear stall angle, so C_L still reaches ≈ 0.95·C_Lmax there
COS_MIN = KN.COS_MIN             # cos(χ − ψ) floor of the course-rate → heading-rate conversion (the fast tier's value)
E_PHI_MAX = math.radians(15.0)   # roll error that commands full aileron (saturation design, B&M §6.1.1.1)
E_THETA_MAX = math.radians(10.0) # pitch error that commands full elevator (B&M §6.1.2.1)
DAMP_Z = 0.707                   # closed-loop damping of the roll and pitch loops
THETA_MAX = math.radians(30.0)   # pitch command limit
K_HDOT_P = 0.05                  # pitch command per m/s of climb-rate error, rad/(m/s)
K_HDOT_I = 0.015                 # rad/(m/s)/s
H_INT_MAX = math.radians(10.0)
K_THETA_I = 0.5                  # elevator integrator on the pitch error, 1/s (the trim elevator of a banked turn, no retuning per aircraft)
DE_INT_MAX = math.radians(15.0)
K_V = 0.15                       # throttle per m/s of airspeed error
K_V_I = 0.05                     # per m/s per s
V_INT_MAX = 0.2
K_BETA = 1.5                     # rudder per rad of sideslip
STALL_MARGIN = 1.15              # the bank limit keeps the load factor below (V_a / (STALL_MARGIN·V_s))²
ALPHA_MAX_FRAC = 0.85            # angle-of-attack protection: pitch commands never ask for more than this fraction of the stall angle
V_PROTECT = 1.1                  # below V_PROTECT·V_s the climb command gives way to speed (nose down)
V_SP_MIN = 0.1                   # a horizontal velocity command below this is a hover (loiter)
LAUNCH_MARGIN_MPS = 1.5          # catapult release speed above v_min
RAIL_ELEV_RAD = math.radians(12.0)   # catapult rail elevation: the body leaves the rail climbing, attitude along the rail
ALT_TOL_M = 3.0                  # takeoff: the climb is complete within this of the target altitude
CHUTE_LEVEL_TAU_S = 1.0          # under the parachute the body hangs level with this time constant
CHUTE_RATE_DAMP = 2.0            # rate damping under the parachute, 1/s
GROUND, LAUNCH, FLY, CHUTE, OFF = 0, 1, 2, 3, 4
MODE_NAMES = ("GROUND", "LAUNCH", "FLY", "CHUTE", "OFF")
_HALF_PI = math.pi / 2.0


# ------------------------------------------------------------------------------------------------------------ attitude helpers
def euler_flu(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(roll, pitch, yaw) of WORLD←FLU quaternions (n, 4): roll positive right wing down, pitch positive nose up, yaw ENU
    (counter-clockwise from east). ZYX sequence yaw → −pitch about body y → roll about body x."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + w * z)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return np.arctan2(r21, r22), np.arcsin(np.clip(r20, -1.0, 1.0)), np.arctan2(r10, r00)


def q_from_euler_flu(roll, pitch, yaw) -> np.ndarray:
    """Inverse of ``euler_flu`` (arrays of equal length → (n, 4) xyzw)."""
    roll, pitch, yaw = (np.asarray(a, np.float64).reshape(-1) for a in (roll, pitch, yaw))
    z = np.zeros_like(yaw)
    qz = np.stack([z, z, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], axis=1)
    qy = np.stack([z, np.sin(-pitch / 2.0), z, np.cos(-pitch / 2.0)], axis=1)
    qx = np.stack([np.sin(roll / 2.0), z, z, np.cos(roll / 2.0)], axis=1)
    return RB.q_mul(RB.q_mul(qz, qy), qx)


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ------------------------------------------------------------------------------------------------------------ parameters
@dataclass(frozen=True)
class WingParams:
    """Physical, aerodynamic, actuation and envelope parameters of one fixed wing, all from its vehicle profile."""
    profile_id: str
    mass_kg: float
    inertia: tuple[float, float, float]
    S: float                               # wing area, m²
    b: float                               # span, m
    c: float                               # mean chord, m
    collision_radius_m: float
    cd0: float
    oswald_e: float
    cl_max: float
    cl0: float
    cl_alpha: float
    cl_q: float
    cl_de: float
    cd_q: float
    cd_de: float
    cm0: float
    cm_alpha: float
    cm_q: float
    cm_de: float
    cy0: float
    cy_beta: float
    cy_p: float
    cy_r: float
    cy_da: float
    cy_dr: float
    cr0: float                             # rolling-moment coefficients (C_l in the literature; "cr" to keep them apart from lift)
    cr_beta: float
    cr_p: float
    cr_r: float
    cr_da: float
    cr_dr: float
    cn0: float
    cn_beta: float
    cn_p: float
    cn_r: float
    cn_da: float
    cn_dr: float
    de_max: float                          # rad
    da_max: float
    dr_max: float
    tau_servo: float
    t_static: float                        # N
    p_max_w: float
    eta_prop: float
    tau_motor: float
    v_min: float
    v_loiter: float
    v_cruise: float
    v_max: float
    phi_max: float                         # rad
    roll_rate_max: float                   # rad/s
    accel_max: float
    decel_max: float
    climb_max: float
    sink_max: float
    r_loiter_default: float
    launch_type: str
    launch_t_s: float
    launch_climb_mps: float
    launch_alt_agl_m: float
    recovery_type: str
    chute_descent_mps: float

    @property
    def aspect_ratio(self) -> float:
        return self.b * self.b / self.S

    @property
    def alpha_stall(self) -> float:
        """Where the linear lift reaches ``cl_max``."""
        return (self.cl_max - self.cl0) / self.cl_alpha

    @property
    def alpha0(self) -> float:
        """Centre of the lift blend into the flat plate: ``STALL_BLEND_OFFSET / STALL_M`` past the linear stall angle."""
        return self.alpha_stall + STALL_BLEND_OFFSET / STALL_M

    @property
    def alpha_max(self) -> float:
        """Angle of attack the autopilot never asks past."""
        return ALPHA_MAX_FRAC * self.alpha_stall

    @property
    def v_stall(self) -> float:
        return math.sqrt(2.0 * self.mass_kg * G / (RHO0 * self.S * self.cl_max))

    def phi_max_at(self, va: float) -> float:
        """Bank available at airspeed ``va``: ``min(φ_max, acos((STALL_MARGIN·V_s/V_a)²))`` (the fast tier's stall protection
        with a margin: the load factor of the turn stays below ``(V_a / 1.15 V_s)²``)."""
        r = STALL_MARGIN * self.v_stall / max(float(va), 1e-6)
        lim = math.acos(min(1.0, r * r)) if r < 1.0 else 0.0
        return min(self.phi_max, lim)

    def r_loiter_min(self, va: float) -> float:
        return 1.2 * float(va) ** 2 / (G * math.tan(max(self.phi_max_at(va), 1e-6)))

    def loiter_radius(self, cmd_radius_m: float | None = None) -> float:
        """``max(R_min(v_loiter), r_loiter_default, commanded)`` (``FwAero.loiter_radius`` in still air)."""
        r = max(self.r_loiter_min(self.v_loiter), self.r_loiter_default)
        return r if cmd_radius_m is None else max(r, float(cmd_radius_m))

    def t_max(self, va) -> np.ndarray:
        """Full-throttle thrust at airspeed ``va`` (array ok): static-thrust and power limited."""
        return np.minimum(self.t_static, self.eta_prop * self.p_max_w / np.maximum(np.asarray(va, np.float64), 0.5))

    @classmethod
    def from_profile(cls, prof) -> WingParams:
        from awr.phys import PhysError
        from awr.sim.catalog.derive import fw_aero

        doc = prof.doc if isinstance(prof.doc, dict) else {}
        fw = fw_aero(doc)
        dyn = doc.get("fw_dynamics")
        J = _val(doc.get("inertia_kgm2"))
        if fw is None or not isinstance(dyn, dict) or not isinstance(J, dict):
            raise PhysError(630, f"vehicle {prof.profile_id!r}: the L2 fixed-wing model needs airframe, fw_dynamics and inertia_kgm2 in the profile")
        lift, drag, pitch = dyn["lift"], dyn.get("drag") or {}, dyn["pitch"]
        side, roll, yaw = dyn["side"], dyn["roll"], dyn["yaw"]
        ctl, prop = dyn["controls"], dyn["propulsion"]
        g = lambda d, k, default=0.0: float(_val(d.get(k, default)))  # noqa: E731
        la, rc = fw.launch or {}, fw.recovery or {}
        return cls(profile_id=prof.profile_id, mass_kg=float(prof.mass_kg), inertia=(float(J["xx"]), float(J["yy"]), float(J["zz"])),
                   S=fw.wing_area_m2, b=fw.span_m, c=g(dyn, "chord_m", fw.wing_area_m2 / fw.span_m), collision_radius_m=float(prof.collision_radius_m),
                   cd0=fw.cd0, oswald_e=fw.oswald_e, cl_max=fw.cl_max,
                   cl0=g(lift, "cl0"), cl_alpha=g(lift, "cl_alpha"), cl_q=g(lift, "cl_q"), cl_de=g(lift, "cl_de"),
                   cd_q=g(drag, "cd_q"), cd_de=g(drag, "cd_de"),
                   cm0=g(pitch, "cm0"), cm_alpha=g(pitch, "cm_alpha"), cm_q=g(pitch, "cm_q"), cm_de=g(pitch, "cm_de"),
                   cy0=g(side, "cy0"), cy_beta=g(side, "cy_beta"), cy_p=g(side, "cy_p"), cy_r=g(side, "cy_r"), cy_da=g(side, "cy_da"), cy_dr=g(side, "cy_dr"),
                   cr0=g(roll, "cl0"), cr_beta=g(roll, "cl_beta"), cr_p=g(roll, "cl_p"), cr_r=g(roll, "cl_r"), cr_da=g(roll, "cl_da"), cr_dr=g(roll, "cl_dr"),
                   cn0=g(yaw, "cn0"), cn_beta=g(yaw, "cn_beta"), cn_p=g(yaw, "cn_p"), cn_r=g(yaw, "cn_r"), cn_da=g(yaw, "cn_da"), cn_dr=g(yaw, "cn_dr"),
                   de_max=math.radians(g(ctl, "de_max_deg", 25.0)), da_max=math.radians(g(ctl, "da_max_deg", 25.0)),
                   dr_max=math.radians(g(ctl, "dr_max_deg", 25.0)), tau_servo=g(ctl, "tau_servo_s", 0.05),
                   t_static=g(prop, "t_static_n"), p_max_w=fw.p_max_w, eta_prop=fw.eta_prop, tau_motor=g(prop, "tau_motor_s", 0.2),
                   v_min=fw.v_min_mps, v_loiter=fw.v_loiter_mps, v_cruise=fw.v_cruise_mps, v_max=fw.v_max_mps,
                   phi_max=math.radians(fw.phi_max_deg), roll_rate_max=math.radians(fw.roll_rate_max_deg_s),
                   accel_max=fw.accel_max_mps2, decel_max=fw.decel_max_mps2, climb_max=fw.climb_max_mps, sink_max=fw.sink_max_mps,
                   r_loiter_default=fw.r_loiter_default_m,
                   launch_type=str(la.get("type", "catapult")), launch_t_s=float(la.get("t_s", 1.0)),
                   launch_climb_mps=float(la.get("climb_mps", fw.climb_max_mps)), launch_alt_agl_m=float(la.get("alt_agl_m", 60.0)),
                   recovery_type=str(rc.get("type", "parachute")), chute_descent_mps=float(rc.get("descent_mps", 5.0)))

    # ---- trim (level flight at airspeed V, still air): angle of attack, elevator, throttle
    def trim(self, va: float) -> tuple[float, float, float]:
        qS = 0.5 * RHO0 * va * va * self.S
        cl_req = self.mass_kg * G / qS
        alpha = (cl_req - self.cl0) / self.cl_alpha
        de = -(self.cm0 + self.cm_alpha * alpha) / self.cm_de
        cd = self.cd0 + (self.cl0 + self.cl_alpha * alpha) ** 2 / (math.pi * self.oswald_e * self.aspect_ratio) + self.cd_de * de
        thrust = qS * cd * math.cos(alpha)                                              # along the body x axis (lift has no x component at trim)
        dt = float(np.clip(thrust / float(self.t_max(va)), 0.0, 1.0))
        return alpha, de, dt


def _val(d):
    return d.get("value") if isinstance(d, dict) and "value" in d else d


# ------------------------------------------------------------------------------------------------------------ references
@dataclass
class WingRef:
    """Per-tick references of one fixed wing: airspeed, course (NaN: hold / rate), course-rate feed-forward, climb rate,
    course rate (used when ``chi`` is NaN), the phase the tracker asks for and whether actuation is enabled."""
    va: float
    chi: float
    chi_ff: float
    hdot: float
    chidot: float
    mode: int
    enabled: bool = True


class WingFleet:
    """Vectorised coefficient-table dynamics and the autopilot loops for ``n`` fixed wings on the shared rigid body."""

    def __init__(self, params: list[WingParams]) -> None:
        self.params = list(params)
        n = self.n = len(params)
        f = lambda a: np.array([float(getattr(p, a)) for p in params])  # noqa: E731
        self.m = f("mass_kg")
        self.J = np.array([p.inertia for p in params])
        self.S, self.b, self.c = f("S"), f("b"), f("c")
        self.AR = self.b * self.b / self.S
        self.collision_r = f("collision_radius_m")
        self.cd0, self.e, self.cl_max = f("cd0"), f("oswald_e"), f("cl_max")
        for k in ("cl0", "cl_alpha", "cl_q", "cl_de", "cd_q", "cd_de", "cm0", "cm_alpha", "cm_q", "cm_de", "cy0", "cy_beta", "cy_p", "cy_r",
                  "cy_da", "cy_dr", "cr0", "cr_beta", "cr_p", "cr_r", "cr_da", "cr_dr", "cn0", "cn_beta", "cn_p", "cn_r", "cn_da", "cn_dr",
                  "de_max", "da_max", "dr_max", "tau_servo", "t_static", "p_max_w", "eta_prop", "tau_motor", "v_min", "v_loiter", "v_cruise",
                  "v_max", "phi_max", "roll_rate_max", "accel_max", "decel_max", "climb_max", "sink_max", "launch_t_s", "launch_climb_mps",
                  "chute_descent_mps"):
            setattr(self, k, f(k))
        self.alpha0 = np.array([p.alpha0 for p in params])
        self.alpha_max = np.array([p.alpha_max for p in params])
        self.v_stall = np.array([p.v_stall for p in params])
        self.sgn_de = np.sign(self.cm_de)
        self.sgn_da = np.sign(self.cr_da)
        self.sgn_dr = np.sign(self.cn_dr)
        # roll and pitch loop gains from the control derivatives at cruise (saturation design, B&M §6.1.1.1 / §6.1.2.1)
        qS = 0.5 * RHO0 * self.v_cruise ** 2 * self.S
        a_phi1 = -qS * self.b * self.cr_p * self.b / (2.0 * self.v_cruise) / self.J[:, 0]
        a_phi2 = np.abs(qS * self.b * self.cr_da / self.J[:, 0])
        self.k_p_phi = self.da_max / E_PHI_MAX
        wn_phi = np.sqrt(a_phi2 * self.k_p_phi)
        self.k_d_phi = np.maximum((2.0 * DAMP_Z * wn_phi - a_phi1) / a_phi2, 0.0)
        a_th1 = -qS * self.c * self.cm_q * self.c / (2.0 * self.v_cruise) / self.J[:, 1]
        a_th2 = -qS * self.c * self.cm_alpha / self.J[:, 1]
        a_th3 = np.abs(qS * self.c * self.cm_de / self.J[:, 1])
        self.k_p_theta = self.de_max / E_THETA_MAX
        wn_th = np.sqrt(np.maximum(a_th2 + a_th3 * self.k_p_theta, 1e-9))
        self.k_d_theta = np.maximum((2.0 * DAMP_Z * wn_th - a_th1) / a_th3, 0.0)
        # actuator and controller state
        self.delta = np.zeros((n, 3))                      # elevator, aileron, rudder (rad), actual
        self.delta_cmd = np.zeros((n, 3))
        self.thr = np.zeros(n)                             # throttle, actual
        self.thr_cmd = np.zeros(n)
        self.v_ref = np.zeros(n)                           # ramped airspeed reference
        self.v_int = np.zeros(n)
        self.h_int = np.zeros(n)
        self.de_int = np.zeros(n)                          # elevator integrator, rad (in the "nose up" direction of sgn_de)
        self.phi_c = np.zeros(n)                           # slewed roll command
        self.mode = np.zeros(n, np.int64)
        self.t_mode = np.zeros(n)                          # time in the current mode, s
        self.rail_dir = np.zeros((n, 3))                   # catapult rail direction (unit, ENU, elevated by RAIL_ELEV_RAD)
        self.rail_yaw = np.zeros(n)
        self.va = np.zeros(n)
        self.alpha = np.zeros(n)
        self.beta = np.zeros(n)
        self.thrust = np.zeros(n)
        self.aero_f = np.zeros((n, 3))                     # last aerodynamic force, body FLU
        self.aero_tau = np.zeros((n, 3))

    # ------------------------------------------------------------------------------------------------------ helpers
    def set_mode(self, i: int, mode: int, heading: float | None = None) -> None:
        if int(self.mode[i]) != int(mode):
            self.mode[i] = mode
            self.t_mode[i] = 0.0
            if mode == LAUNCH and heading is not None:
                ce, se = math.cos(RAIL_ELEV_RAD), math.sin(RAIL_ELEV_RAD)
                self.rail_dir[i] = (ce * math.cos(heading), ce * math.sin(heading), se)
                self.rail_yaw[i] = heading
            if mode in (CHUTE, OFF, GROUND):
                self.thr_cmd[i] = self.thr[i] = 0.0
                self.delta_cmd[i] = 0.0
                self.v_int[i] = self.h_int[i] = self.de_int[i] = 0.0
                self.v_ref[i] = 0.0

    def set_trim(self, idx: np.ndarray, S: RB.RigidState) -> None:
        """Airborne spawn: level trimmed cruise along the heading (velocity, pitch, elevator, throttle)."""
        for i in idx:
            p = self.params[i]
            alpha, de, dt = p.trim(p.v_cruise)
            yaw = float(RB.yaw_of(S.q[i:i + 1])[0])
            S.v[i] = (p.v_cruise * math.cos(yaw), p.v_cruise * math.sin(yaw), 0.0)
            S.q[i] = q_from_euler_flu([0.0], [alpha], [yaw])[0]
            S.w[i] = 0.0
            self.delta[i] = self.delta_cmd[i] = (de, 0.0, 0.0)
            self.de_int[i] = de * self.sgn_de[i]
            self.thr[i] = self.thr_cmd[i] = dt
            self.v_ref[i] = p.v_cruise
            self.mode[i] = FLY

    def air_state(self, S: RB.RigidState, wind: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Body-relative air velocity (FLU), airspeed, angle of attack and sideslip (FRD convention: alpha positive nose up
        relative to the flow, β positive flow from the right)."""
        vb = RB.q_rot_inv(S.q, S.v - wind)
        va = np.linalg.norm(vb, axis=1)
        alpha = np.where(va > 1e-6, np.arctan2(-vb[:, 2], vb[:, 0]), 0.0)
        beta = np.where(va > 1e-6, np.arcsin(np.clip(-vb[:, 1] / np.maximum(va, 1e-9), -1.0, 1.0)), 0.0)
        return vb, va, alpha, beta

    # ------------------------------------------------------------------------------------------------------ control
    def control(self, S: RB.RigidState, refs: list[WingRef], wind: np.ndarray, dt: float) -> None:
        """References → surface and throttle commands (``delta_cmd``, ``thr_cmd``), all vehicles at once; the modes the
        references carry are applied first. Shared with the Beard–McLain oracle."""
        n = self.n
        for i, r in enumerate(refs):
            if r.mode != self.mode[i]:
                self.set_mode(i, r.mode, RB.yaw_of(S.q[i:i + 1])[0] if r.mode == LAUNCH else None)
        self.t_mode += dt
        va_c = np.array([r.va for r in refs])
        chi_d = np.array([r.chi for r in refs])
        chi_ff = np.array([r.chi_ff for r in refs])
        hdot_c = np.array([r.hdot for r in refs])
        chidot_c = np.array([r.chidot for r in refs])
        en = np.array([r.enabled for r in refs]) & (self.mode != CHUTE) & (self.mode != OFF) & (self.mode != GROUND)
        _vb, va, alpha, beta = self.air_state(S, wind)
        self.va, self.alpha, self.beta = va, alpha, beta
        roll, pitch, yaw = euler_flu(S.q)
        p, q_frd, _r_frd = S.w[:, 0], -S.w[:, 1], -S.w[:, 2]
        vg = np.hypot(S.v[:, 0], S.v[:, 1])
        chi = np.where(vg > 1.0, np.arctan2(S.v[:, 1], S.v[:, 0]), yaw)
        va_s = np.maximum(va, 0.1)
        # --- lateral: course → heading rate → coordinated-turn bank → aileron; sideslip → rudder
        rate_only = np.isnan(chi_d)
        chidot = np.where(rate_only, chidot_c, chi_ff + KN.K_CHI * _wrap(np.where(rate_only, 0.0, chi_d) - chi))
        cosd = np.maximum(np.cos(chi - yaw), COS_MIN)
        psidot_c = chidot * np.maximum(vg, 0.1) / (va_s * cosd)
        phi_c = -np.arctan(va_s * psidot_c / G)                                              # ENU heading rate is counter-clockwise: a left turn needs a left bank (φ < 0)
        pm = np.array([self.params[i].phi_max_at(va[i]) for i in range(n)])
        phi_c = np.clip(phi_c, -pm, pm)
        slew = self.roll_rate_max * dt
        self.phi_c = np.where(en, np.clip(phi_c, self.phi_c - slew, self.phi_c + slew), roll)
        da = self.sgn_da * (self.k_p_phi * (self.phi_c - roll) - self.k_d_phi * p)
        dr = self.sgn_dr * K_BETA * beta
        # --- longitudinal: climb rate → pitch → elevator (speed has priority near the stall, angle of attack is capped)
        margin = np.clip((va - V_PROTECT * self.v_stall) / np.maximum(self.v_min - V_PROTECT * self.v_stall, 0.1), 0.0, 1.0)
        hdot_c = np.clip(hdot_c, -self.sink_max, self.climb_max)
        hdot_c = np.where(hdot_c > 0.0, hdot_c * margin, hdot_c)
        hdot_c = np.where(margin <= 0.0, np.minimum(hdot_c, -0.5 * self.sink_max), hdot_c)
        hdot = S.v[:, 2]
        gamma_c = np.arcsin(np.clip(hdot_c / va_s, -0.5, 0.5))
        n_load = 1.0 / np.cos(np.clip(self.phi_c, -1.0, 1.0))
        qS_now = 0.5 * RHO0 * va_s ** 2 * self.S
        alpha_max = self.alpha_max
        alpha_req = np.clip((self.m * G * n_load * np.cos(gamma_c) / qS_now - self.cl0) / self.cl_alpha, -alpha_max, alpha_max)
        e_h = hdot_c - hdot
        theta_c = gamma_c + alpha_req + K_HDOT_P * e_h + self.h_int
        theta_c = np.minimum(theta_c, (pitch - alpha) + alpha_max)                        # never ask for more than alpha_max
        theta_c = np.clip(theta_c, -THETA_MAX, THETA_MAX)
        # pitch PD with the coordinated turn's body pitch rate removed from the damping term, plus an elevator integrator
        # (the elevator a banked turn needs against the pitch damping C_mq is not something a P loop can hold)
        q_turn = G / va_s * np.tan(roll) * np.sin(roll) * np.cos(pitch)
        e_th = theta_c - pitch
        de_n = self.k_p_theta * e_th - self.k_d_theta * (q_frd - q_turn) + self.de_int                # "nose up" units
        de = self.sgn_de * de_n
        de_sat = np.abs(de_n) >= self.de_max
        self.de_int = np.where(en, np.where(de_sat & (np.sign(e_th) == np.sign(de_n)), self.de_int,
                                            np.clip(self.de_int + K_THETA_I * e_th * dt, -DE_INT_MAX, DE_INT_MAX)), 0.0)
        self.h_int = np.where(en, np.where(de_sat & (np.sign(e_h) == np.sign(de_n)), self.h_int,
                                           np.clip(self.h_int + K_HDOT_I * e_h * dt, -H_INT_MAX, H_INT_MAX)), 0.0)
        # --- airspeed: ramped reference → throttle (drag feed-forward + PI); the reference never drops below the stall margin of the turn
        v_sp = np.clip(np.maximum(va_c, STALL_MARGIN * self.v_stall * np.sqrt(n_load)), self.v_min, self.v_max)
        self.v_ref = np.where(en & (self.v_ref <= 0.0), va, self.v_ref)                  # a fresh reference starts at the airspeed
        dv = np.clip((v_sp - self.v_ref) / 2.0, -self.decel_max, self.accel_max) * dt        # the fast tier's law: τ_V = 2 s inside the clamps
        self.v_ref = np.where(en, self.v_ref + dv, 0.0)
        launch = self.mode == LAUNCH
        self.v_ref = np.where(launch, va, self.v_ref)                                      # on the rail the reference is the rail speed
        qS = 0.5 * RHO0 * self.v_ref ** 2 * self.S
        cl_req = self.m * G * n_load * np.cos(gamma_c) / np.maximum(qS, 1e-6)
        gamma_ff = np.where(margin < 1.0, np.maximum(gamma_c, 0.0), gamma_c)                # a nose-down for speed recovery is not a descent to throttle back for
        t_req = qS * (self.cd0 + cl_req ** 2 / (math.pi * self.e * self.AR)) + self.m * G * np.sin(gamma_ff)
        t_max = np.minimum(self.t_static, self.eta_prop * self.p_max_w / np.maximum(self.v_ref, 0.5))
        e_v = self.v_ref - va
        thr = t_req / np.maximum(t_max, 1e-6) + K_V * e_v + self.v_int + (1.0 - margin)      # slow: full power as well as nose down
        sat = ((thr >= 1.0) & (e_v > 0.0)) | ((thr <= 0.0) & (e_v < 0.0))
        self.v_int = np.where(en, np.where(sat, self.v_int, np.clip(self.v_int + K_V_I * e_v * dt, -V_INT_MAX, V_INT_MAX)), 0.0)
        thr = np.where(launch, 1.0, thr)                                                      # full power off the rail
        de = np.where(launch, self.sgn_de * (self.k_p_theta * (np.arcsin(np.clip(self.launch_climb_mps / va_s, -0.5, 0.5)) + alpha_req - pitch) + self.de_int), de)
        da = np.where(launch, 0.0, da)
        self.delta_cmd = np.where(en[:, None], np.clip(np.stack([de, da, dr], axis=1), -np.stack([self.de_max, self.da_max, self.dr_max], axis=1),
                                                       np.stack([self.de_max, self.da_max, self.dr_max], axis=1)), 0.0)
        self.thr_cmd = np.where(en, np.clip(thr, 0.0, 1.0), 0.0)

    # ------------------------------------------------------------------------------------------------------ dynamics
    def actuate(self, dt: float) -> None:
        """Servos and motor follow their commands through first-order lags."""
        self.delta += (self.delta_cmd - self.delta) * np.minimum(dt / self.tau_servo, 1.0)[:, None]
        self.thr += (self.thr_cmd - self.thr) * np.minimum(dt / self.tau_motor, 1.0)

    def aero(self, S: RB.RigidState, wind: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Aerodynamic + propulsive force (body FLU) and moment (body FLU) from the coefficient table at the current state."""
        _vb, va, alpha, beta = self.air_state(S, wind)
        self.va, self.alpha, self.beta = va, alpha, beta
        qS = 0.5 * rho * va * va * self.S
        vs = np.maximum(va, 1e-3)
        p, q, r = S.w[:, 0], -S.w[:, 1], -S.w[:, 2]
        qh, ph, rh = self.c / (2.0 * vs) * q, self.b / (2.0 * vs) * p, self.b / (2.0 * vs) * r
        de, da, dr = self.delta[:, 0], self.delta[:, 1], self.delta[:, 2]
        ex1 = np.exp(np.clip(-STALL_M * (alpha - self.alpha0), -60.0, 60.0))
        ex2 = np.exp(np.clip(STALL_M * (alpha + self.alpha0), -60.0, 60.0))
        sig = (1.0 + ex1 + ex2) / ((1.0 + ex1) * (1.0 + ex2))
        cl_lin = self.cl0 + self.cl_alpha * alpha
        cl = (1.0 - sig) * cl_lin + sig * 2.0 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha)
        cd = self.cd0 + cl_lin ** 2 / (math.pi * self.e * self.AR)
        lift = qS * (cl + self.cl_q * qh + self.cl_de * de)
        drag = qS * (cd + self.cd_q * qh + self.cd_de * de)
        m_pitch = qS * self.c * (self.cm0 + self.cm_alpha * alpha + self.cm_q * qh + self.cm_de * de)
        fy = qS * (self.cy0 + self.cy_beta * beta + self.cy_p * ph + self.cy_r * rh + self.cy_da * da + self.cy_dr * dr)
        l_roll = qS * self.b * (self.cr0 + self.cr_beta * beta + self.cr_p * ph + self.cr_r * rh + self.cr_da * da + self.cr_dr * dr)
        n_yaw = qS * self.b * (self.cn0 + self.cn_beta * beta + self.cn_p * ph + self.cn_r * rh + self.cn_da * da + self.cn_dr * dr)
        ca, sa = np.cos(alpha), np.sin(alpha)
        fx = -drag * ca + lift * sa
        fz_frd = -drag * sa - lift * ca
        self.thrust = self.thr * np.minimum(self.t_static, self.eta_prop * self.p_max_w / np.maximum(va, 0.5)) * (rho / RHO0)
        F = np.stack([fx + self.thrust, -fy, -fz_frd], axis=1)
        tau = np.stack([l_roll, -m_pitch, -n_yaw], axis=1)
        self.aero_f, self.aero_tau = F, tau
        return F, tau

    def on_rail(self, S: RB.RigidState, i: int, p_prev: np.ndarray, dt: float) -> None:
        """Catapult kinematics for vehicle ``i`` after the integration step: the rail carries the body at constant acceleration
        to ``v_min + LAUNCH_MARGIN_MPS`` over ``launch_t_s`` along ``rail_dir`` (elevated by ``RAIL_ELEV_RAD``), attitude held
        along the rail, no rates. Called while the mode is ``LAUNCH``; the forces of the step are discarded."""
        v_rel = float(self.v_min[i]) + LAUNCH_MARGIN_MPS
        a = v_rel / max(float(self.launch_t_s[i]), 1e-3)
        v = min(a * float(self.t_mode[i]), v_rel)
        S.v[i] = self.rail_dir[i] * v
        S.p[i] = p_prev + S.v[i] * dt
        S.q[i] = q_from_euler_flu(0.0, RAIL_ELEV_RAD, float(self.rail_yaw[i]))[0]
        S.w[i] = 0.0

    def forces(self, S: RB.RigidState, wind: np.ndarray, rho: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Advance the actuators one step and return (world force without gravity, body torque) for the current mode (on the
        rail the aero and thrust are returned too; ``on_rail`` overrides the state afterwards)."""
        self.actuate(dt)
        F_b, tau = self.aero(S, wind, rho)
        F = RB.q_rot(S.q, F_b)
        chute = self.mode == CHUTE
        if chute.any():
            vr = S.v - wind
            spd = np.linalg.norm(vr, axis=1)
            k = self.m * G / self.chute_descent_mps ** 2
            F_ch = -k[:, None] * spd[:, None] * vr
            tau_ch = -CHUTE_RATE_DAMP * S.J * S.w
            F = np.where(chute[:, None], F_ch, F)
            tau = np.where(chute[:, None], tau_ch, tau)
        ground = self.mode == GROUND
        if ground.any():
            F = np.where(ground[:, None], 0.0, F)
            tau = np.where(ground[:, None], 0.0, tau)
        return F, tau

    def level_chute(self, S: RB.RigidState, idx: np.ndarray, dt: float) -> None:
        """Under the parachute the body hangs level (heading kept)."""
        for i in idx:
            yaw = float(RB.yaw_of(S.q[i:i + 1])[0])
            lev = RB.q_from_yaw(yaw)[None]
            qi = S.q[i:i + 1]
            if float(np.sum(qi * lev)) < 0.0:
                lev = -lev
            a_ = min(dt / CHUTE_LEVEL_TAU_S, 1.0)
            qn = qi + (lev - qi) * a_
            S.q[i] = qn[0] / np.linalg.norm(qn[0])

    def stop(self, idx: np.ndarray) -> None:
        self.thr[idx] = self.thr_cmd[idx] = 0.0
        self.delta[idx] = self.delta_cmd[idx] = 0.0
        self.v_int[idx] = self.h_int[idx] = self.de_int[idx] = 0.0
        self.v_ref[idx] = 0.0
        self.mode[idx] = OFF


# ------------------------------------------------------------------------------------------------------------ tracker
_LOITER_ACC = 30.0               # leg acceptance radius floor, m (the fast tier's r_acc)


class WingTracker:
    """Per-tick references for fixed wings from the tier's commands, with the vector-field guidance of the fixed-wing fast
    tier (``awr.sim.fixedwing.kinematics``): legs (goto, follow_path) end in a loiter, orbit and hover are loiters, velocity
    is an airspeed along a course, takeoff is the catapult launch and climb, land the parachute, kill motor and controls off.
    Angles are ENU (counter-clockwise from east); the fast tier's compass-angle fields are converted at the call."""

    def __init__(self, params: Mapping[str, WingParams], tick_s: float = 0.1) -> None:
        self.params = dict(params)
        self.tick_s = float(tick_s)
        self.st: dict[str, dict] = {u: {"kind": "hover", "mode": GROUND} for u in self.params}
        self.stats = {"plans": 0, "legs_done": 0}

    def kind(self, uav: str) -> str:
        return str(self.st[uav]["kind"])

    def mode(self, uav: str) -> int:
        return int(self.st[uav]["mode"])

    # ---- commands
    def set(self, cmd, pos: np.ndarray, vel: np.ndarray, grounded: bool, ground_z: float) -> None:
        p = self.params[cmd.uav]
        st = self.st[cmd.uav]
        was = int(st["mode"])
        pos = np.asarray(pos, np.float64)
        k = cmd.kind
        if k == "kill":
            st.update(kind="kill", mode=OFF)
            return
        if k == "land":
            st.update(kind="land", mode=GROUND if was == GROUND else CHUTE)
            return
        if k == "takeoff":
            if was != GROUND:
                st.update(kind="takeoff")
                return
            alt = cmd.alt_m if cmd.alt_m is not None and float(cmd.alt_m) >= 10.0 else p.launch_alt_agl_m
            st.update(kind="takeoff", mode=LAUNCH, z_t=float(ground_z) + float(alt), launch_t=0.0)
            return
        if was == GROUND:
            st.update(kind=k)                                                   # a flight command on the ground: nothing moves until a takeoff
            return
        if k == "hover":
            st.update(kind="hover", mode=FLY, loiter=self._loiter(p, pos, None, True, None))
        elif k == "orbit":
            o = cmd.orbit
            st.update(kind="orbit", mode=FLY, loiter=self._loiter(p, np.asarray(o.center_enu_m, np.float64), o.radius_m, o.cw, o.turns,
                                                                   va=self._speed(p, cmd.speed_mps)))
        elif k == "goto":
            tgt = np.asarray(cmd.target_enu_m, np.float64)
            st.update(kind="goto", mode=FLY, legs=[pos.copy(), tgt], i=1, va=self._speed(p, cmd.speed_mps))
        elif k == "follow_path":
            pts = [pos.copy(), *[np.asarray(w, np.float64) for w in np.asarray(cmd.path_enu_m, np.float64).reshape(-1, 3)]]
            st.update(kind="follow_path", mode=FLY, legs=pts, i=1, va=self._speed(p, cmd.speed_mps))
        elif k == "velocity":
            v = np.asarray(cmd.vel_enu_mps or (0.0, 0.0, 0.0), np.float64)
            yr = float(cmd.yaw_rate_rps or 0.0)
            sp = float(np.hypot(v[0], v[1]))
            if sp < V_SP_MIN and abs(yr) < 1e-9:
                st.update(kind="velocity", mode=FLY, loiter=self._loiter(p, pos, None, True, None), vz=float(v[2]))
            else:
                st.update(kind="velocity", mode=FLY, loiter=None, va=self._speed(p, sp if sp >= V_SP_MIN else None),
                          chi=math.atan2(v[1], v[0]) if sp >= V_SP_MIN and abs(yr) < 1e-9 else math.nan, chidot=yr, vz=float(v[2]))

    @staticmethod
    def _speed(p: WingParams, v) -> float:
        return p.v_cruise if v is None else min(max(float(v), p.v_min), p.v_max)

    @staticmethod
    def _loiter(p: WingParams, c: np.ndarray, r_cmd, cw: bool, turns, va: float | None = None) -> dict:
        return {"c": np.asarray(c, np.float64).copy(), "r": p.loiter_radius(r_cmd), "lam": 1.0 if cw else -1.0,
                "turns": None if turns is None else float(turns), "acc": 0.0, "th": None, "va": p.v_loiter if va is None else float(va)}

    # ---- references
    def setpoint(self, uav: str, pos: np.ndarray, vel: np.ndarray, q: np.ndarray) -> WingRef:
        p = self.params[uav]
        st = self.st[uav]
        self.stats["plans"] += 1
        mode = int(st["mode"])
        x, y, z = (float(a) for a in pos[:3])
        vg = float(np.hypot(vel[0], vel[1]))
        if mode in (GROUND, CHUTE, OFF):
            return WingRef(p.v_cruise, math.nan, 0.0, 0.0, 0.0, mode, enabled=mode != OFF)
        if mode == LAUNCH:
            if float(st.get("launch_t", 0.0)) >= p.launch_t_s - 1e-9:                      # the rail has reached the release speed
                st["mode"] = FLY
                st["climb_to"] = st["z_t"]
                st["chi_hold"] = float(RB.yaw_of(np.asarray(q, np.float64).reshape(1, 4))[0])
            st["launch_t"] = float(st.get("launch_t", 0.0)) + self.tick_s
            return WingRef(p.v_cruise, math.nan, 0.0, p.launch_climb_mps, 0.0, int(st["mode"]))
        if st.get("climb_to") is not None:                                                   # takeoff climb: straight ahead, then loiter
            if z >= float(st["climb_to"]) - ALT_TOL_M:
                st["climb_to"] = None
                st["loiter"] = self._loiter(p, np.array([x, y, st["z_t"]]), None, True, None)
            else:
                return WingRef(p.v_cruise, float(st["chi_hold"]), 0.0, min(p.launch_climb_mps, p.climb_max), 0.0, FLY)
        kind = st["kind"]
        if kind == "velocity" and st.get("loiter") is None:
            return WingRef(float(st["va"]), float(st["chi"]), 0.0, float(st["vz"]), float(st["chidot"]), FLY)
        if kind in ("goto", "follow_path") and st.get("legs") is not None:
            legs, i = st["legs"], int(st["i"])
            a, b = legs[i - 1], legs[i]
            chi_c, _e, along = KN.line_field(x, y, float(a[0]), float(a[1]), float(b[0]), float(b[1]))
            dist = math.hypot(float(b[0]) - x, float(b[1]) - y)
            r_acc = max(_LOITER_ACC, 0.5 * p.loiter_radius(None))
            if dist < r_acc or along < 0.0:
                self.stats["legs_done"] += 1
                if i + 1 < len(legs):
                    st["i"] = i + 1
                else:
                    st["legs"] = None
                    st["loiter"] = self._loiter(p, b, None, True, None, va=float(st["va"]))
            if st.get("legs") is not None:
                return WingRef(float(st["va"]), _wrap(_HALF_PI - chi_c), 0.0, KN.K_HDOT * (float(b[2]) - z), 0.0, FLY)
        lo = st.get("loiter")
        if lo is None:                                                                      # nothing asked yet in the air: hold
            st["loiter"] = lo = self._loiter(p, np.array([x, y, z]), None, True, None)
        c = lo["c"]
        chi_c, ff, dev, th = KN.circle_field(x, y, float(c[0]), float(c[1]), float(lo["r"]), float(lo["lam"]), vg)
        if lo["turns"] is not None and abs(dev) < 0.5 * lo["r"]:
            if lo["th"] is not None:
                lo["acc"] += lo["lam"] * _wrap(th - lo["th"])
            lo["th"] = th
        vz = float(st.get("vz", 0.0)) if kind == "velocity" else KN.K_HDOT * (float(c[2]) - z)
        return WingRef(float(lo["va"]), _wrap(_HALF_PI - chi_c), -ff, vz, 0.0, FLY)

    def turns(self, uav: str) -> float:
        lo = self.st[uav].get("loiter")
        return 0.0 if lo is None else float(lo["acc"]) / (2.0 * math.pi)
