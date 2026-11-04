"""Multirotor model of the unified L2 physics engine (``awr.phys.l2.multirotor``): per-rotor dynamics on the shared rigid body.

What L1 (PX4-lite, ``awr.sim.fleet``) collapses into one thrust scalar with an ideal rate loop, this model resolves:

- **control allocation**: the attitude and rate loops ask for a collective thrust and three body torques; the mixer maps
  them onto ``n`` rotors through the pseudo-inverse of the frame's effectiveness matrix (rotor positions from
  ``geometry.arm_xy_m`` and the PX4 ``quad_x`` / ``hex_x`` layout, spin directions, moment constant ``motor.km_m``), with
  PX4-style desaturation (yaw authority is given up first, then the collective is shifted);
- **rotor lag**: each rotor speed follows its command through a first-order lag ``motor.tau_s``; thrust is ``k_f·Ω²``;
- **rigid body with inertia**: torques from the rotor geometry act on ``inertia_kgm2`` through Euler's equation
  (gyroscopic term included); attitude is a quaternion integrated from body rates;
- **rate loop**: PX4 rate PID (``MC_*RATE_P/I/D``) in PX4's normalised-torque units, converted through the same
  effectiveness normalisation PX4's control allocation applies (roll and pitch by the RMS of the mixer columns, yaw by
  the largest entry), with the gyroscopic feed-forward; attitude P (``MC_ROLL_P``, ``MC_PITCH_P``, ``MC_YAW_P``) with the
  rate limits ``MC_*RATE_MAX`` — the same attitude law as L1, which then assumes the rates are reached at once;
- **velocity loop**: the PX4 velocity PID of L1's ``pos_ctrl`` (``MPC_*_VEL_P/I/D_ACC``, feed-forward acceleration,
  anti-reset windup against the achieved horizontal acceleration, vertical saturation freeze) run at the physics rate
  inside the engine; the reference tracker (``awr.phys.tracker.RefTracker`` in ``inner_pid`` mode) supplies the same
  velocity, acceleration and heading references it supplies to the gz tier;
- **aerodynamics**: the same composite law as L1 (``½ρ·CdA·‖v_r‖·v_r`` plus rotor-induced drag ``ΣΩ·c_rd`` on the
  in-plane relative velocity) or the linear ``k_dv`` law, with the measured ``ΣΩ`` instead of L1's uniform-thrust estimate;
  the wind enters through the relative velocity only.

Parameters come from the vehicle profile (``awr.sim.fleet.profiles.Profile`` and its document); nothing is tuned per
vehicle here. Coordinates: world ENU, body FLU, quaternion xyzw WORLD←FLU (``awr.phys.l2.rigid``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from awr.phys.l2 import rigid as RB
from awr.sim.fleet import params_px4 as P

__all__ = ["ACC_DOWN_MAX", "ACC_UP_MAX", "K_ATT", "K_RATE_D_PX4", "K_RATE_I_PX4", "K_RATE_P_PX4", "K_VEL", "K_VEL_D", "K_VEL_I",
           "RATE_MAX", "THR_MAX", "THR_MIN", "TILT_MAX", "MultirotorFleet", "MultirotorParams", "rotor_layout"]

# PX4 defaults (awr.sim.fleet.params_px4) in the units this model uses
K_VEL = np.array([P.MPC_XY_VEL_P_ACC, P.MPC_XY_VEL_P_ACC, P.MPC_Z_VEL_P_ACC])      # acceleration per velocity error, 1/s
K_VEL_I = np.array([P.MPC_XY_VEL_I_ACC, P.MPC_XY_VEL_I_ACC, P.MPC_Z_VEL_I_ACC])    # 1/s²
K_VEL_D = np.array([P.MPC_XY_VEL_D_ACC, P.MPC_XY_VEL_D_ACC, P.MPC_Z_VEL_D_ACC])    # on the measured acceleration
ACC_UP_MAX = P.MPC_ACC_UP_MAX
ACC_DOWN_MAX = P.MPC_ACC_DOWN_MAX
TILT_MAX = P.TILTMAX_RAD
THR_MIN = P.MPC_THR_MIN
THR_MAX = P.MPC_THR_MAX
K_ATT = np.array([P.MC_ROLL_P, P.MC_PITCH_P, P.MC_YAW_P])                         # attitude P, 1/s
RATE_MAX = np.radians([P.MC_ROLLRATE_MAX, P.MC_PITCHRATE_MAX, P.MC_YAWRATE_MAX])
K_RATE_P_PX4 = np.array([0.15, 0.15, 0.2])                                        # MC_ROLLRATE_P, MC_PITCHRATE_P, MC_YAWRATE_P
K_RATE_I_PX4 = np.array([0.2, 0.2, 0.1])                                          # MC_ROLLRATE_I, MC_PITCHRATE_I, MC_YAWRATE_I
K_RATE_D_PX4 = np.array([0.003, 0.003, 0.0])                                      # MC_ROLLRATE_D, MC_PITCHRATE_D, MC_YAWRATE_D
RATE_I_MAX = np.array([0.3, 0.3, 0.3])                                            # MC_RR_INT_LIM, MC_PR_INT_LIM, MC_YR_INT_LIM
RHO0 = P.RHO0
E3 = RB.E3


def rotor_layout(frame_type: str, n: int, arm_xy_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotor positions (n, 2) in FLU and spin signs (+1 counter-clockwise seen from above) for the PX4 geometries.

    ``quad_x`` in PX4 motor order: 1 front-right CCW, 2 rear-left CCW, 3 front-left CW, 4 rear-right CW (the FRD layout
    mirrored into FLU). ``hex_x``: six arms at 30° + 60°·k with alternating spin. Any other frame with ``n`` rotors gets
    a regular star with alternating spin (the physics does not depend on the motor numbering)."""
    L = float(arm_xy_m)
    if frame_type == "quad_x" and n == 4:
        pos = np.array([[L, -L], [-L, L], [L, L], [-L, -L]])
        spin = np.array([1.0, 1.0, -1.0, -1.0])
        return pos, spin
    r = L * math.sqrt(2.0)                                                         # arm length from the XY component
    ang = np.radians(30.0) + np.arange(n) * 2.0 * math.pi / n
    pos = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
    spin = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return pos, spin


@dataclass(frozen=True)
class MultirotorParams:
    """Physical and control parameters of one multirotor, all taken from its vehicle profile."""
    profile_id: str
    mass_kg: float
    inertia: tuple[float, float, float]
    rotor_xy: np.ndarray                   # (n, 2) FLU
    spin: np.ndarray                       # (n,) +1 CCW
    k_f: float                             # N per (rad/s)²
    k_m: float                             # m (torque per thrust)
    omega_max: float
    t_max_rotor: float
    tau_motor: float
    aero: str                              # composite | linear
    cda_m2: float
    c_rd: float
    k_dv: float
    collision_radius_m: float
    vxy_max: float
    z_up_max: float
    z_dn_max: float
    yawrate_max: float

    @classmethod
    def from_profile(cls, prof, limits) -> MultirotorParams:
        doc = prof.doc if isinstance(prof.doc, dict) else {}
        mot = doc.get("motor") or {}
        km = mot.get("km_m")
        km = float(km.get("value") if isinstance(km, dict) else (km if km is not None else 0.016))
        pos, spin = rotor_layout(prof.frame_type, int(prof.n_rot), float(prof.arm_xy_m))
        return cls(prof.profile_id, float(prof.mass_kg), tuple(float(x) for x in prof.inertia_kgm2), pos, spin,
                   float(prof.t_max_rotor_n) / float(prof.omega_max_rad_s) ** 2, km, float(prof.omega_max_rad_s),
                   float(prof.t_max_rotor_n), float(prof.tau_motor_s), str(prof.aero), float(prof.cda_m2), float(prof.c_rd),
                   float(prof.k_dv), float(prof.collision_radius_m), float(limits.vxy_max_mps), float(limits.z_up_mps),
                   float(limits.z_dn_mps), math.radians(float(limits.yawrate_max_deg_s)))

    @property
    def n_rot(self) -> int:
        return len(self.spin)

    @property
    def t_max(self) -> float:
        return self.n_rot * self.t_max_rotor

    @property
    def hover_omega(self) -> float:
        return math.sqrt(self.mass_kg * RB.G / self.n_rot / self.k_f)

    def effectiveness(self) -> np.ndarray:
        """(4, n): [collective thrust; roll torque; pitch torque; yaw torque] per unit rotor thrust (FLU, z up)."""
        x, y = self.rotor_xy[:, 0], self.rotor_xy[:, 1]
        return np.stack([np.ones(self.n_rot), y, -x, -self.spin * self.k_m], axis=0)

    def torque_authority(self) -> np.ndarray:
        """Torque per unit of PX4's normalised torque command, per axis (N·m).

        PX4's control allocation (``ControlAllocationPseudoInverse::normalizeControlAllocationMatrix``) builds the mixer as
        the pseudo-inverse of the effectiveness matrix in actuator units (fraction of a rotor's maximum thrust) and scales
        the roll and pitch columns by the RMS of their entries over half the actuators (one shared scale) and the yaw
        column by its largest entry, so that a unit command drives the actuators by at most about one. The physical
        torque that unit command produces is what the rate gains are tuned against."""
        A = self.effectiveness() * self.t_max_rotor                                  # torque per unit actuator (fraction of max)
        mix = np.linalg.pinv(A)                                                      # (n, 4): actuator per unit control
        rp = math.sqrt(max(float(np.sum(mix[:, 1] ** 2)), float(np.sum(mix[:, 2] ** 2))) / (self.n_rot / 2.0))
        yaw = float(np.max(np.abs(mix[:, 3])))
        scale = np.array([rp, rp, yaw])
        out = np.empty(3)
        for k in range(3):
            col = mix[:, k + 1] / scale[k]
            out[k] = float((A @ col)[k + 1])
        return out


class MultirotorFleet:
    """``n`` multirotors stepped together (vectorised over vehicles; rotor arrays are padded to the widest frame).

    ``control`` turns per-step references into rotor speed commands; ``forces`` turns rotor speeds, attitude and relative
    wind into the world force and body torque for ``RigidState.integrate``. Both are deterministic functions of their
    inputs and the fleet's own rotor and integrator state."""

    def __init__(self, params: list[MultirotorParams]) -> None:
        self.params = list(params)
        n = len(params)
        R = max(p.n_rot for p in params)
        self.n, self.R = n, R
        self.m = np.array([p.mass_kg for p in params])
        self.J = np.array([p.inertia for p in params])
        self.rotor_xy = np.zeros((n, R, 2))
        self.spin = np.zeros((n, R))
        self.live = np.zeros((n, R), bool)
        self.k_f = np.array([p.k_f for p in params])
        self.k_m = np.array([p.k_m for p in params])
        self.omega_max = np.array([p.omega_max for p in params])
        self.t_max_rotor = np.array([p.t_max_rotor for p in params])
        self.tau = np.array([p.tau_motor for p in params])
        self.t_max = np.array([p.t_max for p in params])
        self.composite = np.array([p.aero != "linear" for p in params])
        self.cda = np.array([p.cda_m2 for p in params])
        self.c_rd = np.array([p.c_rd for p in params])
        self.k_dv = np.array([p.k_dv for p in params])
        self.collision_r = np.array([p.collision_radius_m for p in params])
        self.vxy_max = np.array([p.vxy_max for p in params])
        self.yawrate_max = np.array([p.yawrate_max for p in params])
        self.A = np.zeros((n, 4, R))
        self.A_pinv = np.zeros((n, R, 4))
        self.auth = np.zeros((n, 3))
        for i, p in enumerate(params):
            r = p.n_rot
            self.rotor_xy[i, :r] = p.rotor_xy
            self.spin[i, :r] = p.spin
            self.live[i, :r] = True
            A = p.effectiveness()
            self.A[i, :, :r] = A
            self.A_pinv[i, :r] = np.linalg.pinv(A)
            self.auth[i] = p.torque_authority()                                    # N·m per unit normalised torque
        self.k_rate = K_RATE_P_PX4 * self.auth                                       # N·m per rad/s
        self.k_rate_i = K_RATE_I_PX4 * self.auth                                     # N·m per rad
        self.k_rate_d = K_RATE_D_PX4 * self.auth                                     # N·m per rad/s²
        self.i_max = RATE_I_MAX * self.auth                                          # N·m
        self.omega_r = np.zeros((n, R))                                              # rotor speeds, rad/s
        self.rate_int = np.zeros((n, 3))                                             # N·m
        self.vel_int = np.zeros((n, 3))                                              # m/s²
        self.w_prev = np.zeros((n, 3))
        self.T_i = np.zeros((n, R))
        self.tilt_cmd = np.zeros(n)

    # ------------------------------------------------------------------------------------------------------ control
    def control(self, S: RB.RigidState, v_sp: np.ndarray, a_ff: np.ndarray, yaw_sp: np.ndarray, yaw_rate_sp: np.ndarray,
                a_meas: np.ndarray, enabled: np.ndarray, dt: float) -> np.ndarray:
        """References → rotor speed commands (n, R). ``yaw_sp`` NaN means rate-only yaw. Disabled vehicles command zero
        and hold their integrators at zero."""
        n = self.n
        en = enabled[:, None]
        # velocity PID (L1 pos_ctrl, PX4 PositionControl): acceleration setpoint
        e = v_sp - S.v
        vi = self.vel_int.copy()
        vi[:, 2] = np.clip(vi[:, 2], -RB.G, RB.G)
        acs = K_VEL * e + vi - K_VEL_D * a_meas + a_ff
        f = self.m[:, None] * (acs + RB.G * E3)                                      # desired thrust vector, world
        fz = np.maximum(f[:, 2], THR_MIN * self.t_max)                               # never ask for less than the minimum collective
        fxy = np.linalg.norm(f[:, :2], axis=1)
        cap = fz * math.tan(TILT_MAX)
        sc = np.where(fxy > cap, cap / np.maximum(fxy, 1e-12), 1.0)                  # PX4 limitTilt: shrink the horizontal part
        f = np.concatenate([f[:, :2] * sc[:, None], fz[:, None]], axis=1)
        Tn = np.linalg.norm(f, axis=1)
        T = np.clip(Tn, THR_MIN * self.t_max, THR_MAX * self.t_max)
        b3 = f / np.maximum(Tn[:, None], 1e-12)
        self.tilt_cmd = np.arccos(np.clip(b3[:, 2], -1.0, 1.0))
        # anti-reset windup: the horizontal error only integrates what the thrust vector could deliver
        ap = b3[:, :2] * T[:, None] / self.m[:, None]
        need = np.sum(acs[:, :2] ** 2, axis=1) > np.sum(ap ** 2, axis=1)
        arw = 2.0 / K_VEL[0]
        e[:, :2] = np.where(need[:, None], e[:, :2] - arw * (acs[:, :2] - ap), e[:, :2])
        sat = ((THR_MIN * self.t_max * (1 + 1e-9) >= T) & (e[:, 2] >= 0.0)) | ((THR_MAX * self.t_max * (1 - 1e-9) <= T) & (e[:, 2] <= 0.0))
        e[:, 2] = np.where(sat, 0.0, e[:, 2])
        self.vel_int = np.where(en, vi + K_VEL_I * e * dt, 0.0)
        # attitude loop (quaternion P, the L1 law): body z toward b3, heading yaw_sp (or the current heading when rate-only)
        yaw = RB.yaw_of(S.q)
        rate_only = np.isnan(yaw_sp)
        q_sp = RB.q_from_z_yaw(b3, np.where(rate_only, yaw, yaw_sp))
        qe = RB.q_mul(RB.q_inv(S.q), q_sp)
        sgn = np.where(qe[:, 3] < 0.0, -1.0, 1.0)
        w_sp = 2.0 * K_ATT * qe[:, 0:3] * sgn[:, None]
        w_sp[:, 2] = np.where(rate_only, yaw_rate_sp, w_sp[:, 2])
        lim = np.stack([np.full(n, RATE_MAX[0]), np.full(n, RATE_MAX[1]), np.minimum(RATE_MAX[2], self.yawrate_max)], axis=1)
        w_sp = np.clip(w_sp, -lim, lim)
        # rate loop (PX4 PID in normalised torque, converted by the frame's authority), gyroscopic feed-forward
        er = w_sp - S.w
        self.rate_int = np.where(en, np.clip(self.rate_int + self.k_rate_i * er * dt, -self.i_max, self.i_max), 0.0)
        wdot = (S.w - self.w_prev) / dt
        self.w_prev = S.w.copy()
        tau = self.k_rate * er + self.rate_int - self.k_rate_d * wdot + RB.cross(S.w, self.J * S.w)
        # allocation with desaturation: yaw first, then the collective
        u = np.concatenate([T[:, None], tau], axis=1)                                 # (n, 4)
        Ti = np.einsum("nrk,nk->nr", self.A_pinv, u)
        hi = self.t_max_rotor[:, None]
        over = np.where(self.live, np.maximum(Ti - hi, 0.0) + np.maximum(-Ti, 0.0), 0.0)
        bad = over.max(axis=1) > 1e-9
        if bad.any():
            dyaw = self.A_pinv[:, :, 3] * u[:, 3:4]                                     # yaw contribution per rotor
            base = Ti - dyaw
            with np.errstate(divide="ignore", invalid="ignore"):
                room_hi = np.where(dyaw > 1e-12, (hi - base) / dyaw, np.inf)
                room_lo = np.where(dyaw < -1e-12, -base / dyaw, np.inf)
            s = np.clip(np.minimum(np.where(self.live, room_hi, np.inf), np.where(self.live, room_lo, np.inf)).min(axis=1), 0.0, 1.0)
            Ti = np.where(bad[:, None], base + dyaw * s[:, None], Ti)
            mx = np.where(self.live, Ti, -np.inf).max(axis=1)
            mn = np.where(self.live, Ti, np.inf).min(axis=1)
            shift = np.where(mx > hi[:, 0], mx - hi[:, 0], 0.0) - np.where(mn < 0.0, -mn, 0.0)
            Ti = np.where(bad[:, None], Ti - shift[:, None], Ti)
        Ti = np.clip(Ti, 0.0, hi)
        om = np.sqrt(Ti / self.k_f[:, None])
        om = np.minimum(om, self.omega_max[:, None])
        return np.where(en & self.live, om, 0.0)

    # ------------------------------------------------------------------------------------------------------ dynamics
    def forces(self, S: RB.RigidState, omega_cmd: np.ndarray, wind: np.ndarray, rho: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Advance the rotors one step and return (world force without gravity, body torque)."""
        am = 1.0 - np.exp(-dt / self.tau)
        self.omega_r = self.omega_r + (omega_cmd - self.omega_r) * am[:, None]
        rr = (rho / RHO0)[:, None]
        Ti = self.k_f[:, None] * self.omega_r * self.omega_r * rr
        self.T_i = Ti
        Tsum = Ti.sum(axis=1)
        tau = np.einsum("nkr,nr->nk", self.A[:, 1:4, :], Ti)
        b3 = RB.q_rot(S.q, np.tile(E3, (self.n, 1)))
        F = b3 * Tsum[:, None]
        vr = S.v - wind
        spd = np.linalg.norm(vr, axis=1)
        dotb = np.sum(vr * b3, axis=1)
        perp = vr - dotb[:, None] * b3
        kq = 0.5 * rho * self.cda
        kr = self.omega_r.sum(axis=1) * self.c_rd
        comp = -kq[:, None] * spd[:, None] * vr - kr[:, None] * perp
        lin = -self.k_dv[:, None] * vr
        F = F + np.where(self.composite[:, None], comp, lin)
        return F, tau

    def set_hover(self, idx: np.ndarray) -> None:
        """Rotors at hover speed (air spawn)."""
        for i in idx:
            self.omega_r[i, :] = np.where(self.live[i], self.params[i].hover_omega, 0.0)

    def stop(self, idx: np.ndarray) -> None:
        self.omega_r[idx] = 0.0
        self.rate_int[idx] = 0.0
        self.vel_int[idx] = 0.0
