"""Beard–McLain 6-DoF oracle (``awr.phys.l2.bm_oracle``): the small-UAV flight dynamics equations in the form of the textbook
(*Small Unmanned Aircraft*, ch. 3–4: NED position, FRD body velocity, Euler angles, body rates), stepped with RK4, as the
baseline the fixed-wing domain of the unified L2 engine is compared against (``awr.phys.tier_compare --set wing --tier bm``).

State per aircraft ``(p_n, p_e, p_d, u, v, w, φ, θ, ψ, p, q, r, δe, δa, δr, δt)``; the kinematics are eq. 3.14–3.17 with a
diagonal inertia (``J_xz`` dropped, as the engine's rigid core has no product of inertia), the forces and moments are
eq. 4.18–4.20 with the same coefficient table the engine reads from the profile (lift blended into the flat plate past
``alpha0``, drag ``C_D0 + C_L²/(π e AR)``, rate and control derivatives), the thrust is the engine's power-limited propeller law,
and the servo and motor lags are states of the ODE. The controller (``WingFleet.control``: course → bank → aileron, climb
rate → pitch → elevator, airspeed → throttle, sideslip → rudder) is the engine's own, evaluated once per step on the oracle
state mapped into the engine's ENU / FLU convention, so the two tiers differ in how the dynamics are written and
integrated, not in what they are asked to do.

This is a comparison tool, not a tier for the sandbox: airborne only (no ground, no launch, no parachute, no collisions);
takeoff, land and kill are rejected with 630.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping, Sequence

import numpy as np

from awr.phys.adapter import (
    ClockProfile,
    ContactEvent,
    DispatchResult,
    TickResult,
    TickTiming,
    TierCommand,
    VehicleStatus,
)
from awr.phys.l2 import rigid as RB
from awr.phys.l2.fixedwing import FLY, RHO0, STALL_M, WingFleet, WingParams, WingTracker
from awr.phys.pose_history import G_W, STATE_N, PoseHistoryBuffer

__all__ = ["BMOracleTier", "bm_rhs", "oracle_params"]

COMMANDS = ("velocity", "goto", "follow_path", "orbit", "hover")
G = RB.G
_P = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])     # NED → ENU
_T = np.diag([1.0, -1.0, -1.0])                                          # FLU → FRD (and back)


def oracle_params(vehicles: Mapping[str, str], uavs: Sequence[str]) -> tuple[dict[str, str], list[WingParams]]:
    from awr.phys.fast import resolve_profile
    from awr.sim.fleet.profiles import ProfileTable

    T = ProfileTable()
    ids = {u: resolve_profile(vehicles.get(u, "awr_f1"), T) for u in uavs}
    return ids, [WingParams.from_profile(T.get(ids[u])) for u in uavs]


def _rot_body_to_ned(phi, theta, psi) -> np.ndarray:
    """R_b^v of eq. 3.14 (body FRD → vehicle NED), (n, 3, 3)."""
    cf, sf, ct, st, cp, sp = np.cos(phi), np.sin(phi), np.cos(theta), np.sin(theta), np.cos(psi), np.sin(psi)
    R = np.empty((len(phi), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp
    R[:, 1, 0], R[:, 1, 1], R[:, 1, 2] = ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp
    R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = -st, sf * ct, cf * ct
    return R


def bm_rhs(x: np.ndarray, delta_cmd: np.ndarray, thr_cmd: np.ndarray, F: WingFleet, wind_ned: np.ndarray) -> np.ndarray:
    """ẋ for n aircraft (eq. 3.14–3.17 and 4.18–4.20 of Beard & McLain with the engine's coefficient table and thrust law)."""
    vb = x[:, 3:6]
    phi, theta, psi = x[:, 6], x[:, 7], x[:, 8]
    p, q, r = x[:, 9], x[:, 10], x[:, 11]
    de, da, dr, dt_ = x[:, 12], x[:, 13], x[:, 14], x[:, 15]
    R = _rot_body_to_ned(phi, theta, psi)
    # air-relative velocity, airspeed, angle of attack, sideslip (eq. 2.7–2.8)
    wb = np.einsum("nji,nj->ni", R, wind_ned)                                       # R^T·wind
    vr = vb - wb
    ur, vr_, wr = vr[:, 0], vr[:, 1], vr[:, 2]
    va = np.linalg.norm(vr, axis=1)
    alpha = np.where(va > 1e-6, np.arctan2(wr, ur), 0.0)
    beta = np.where(va > 1e-6, np.arcsin(np.clip(vr_ / np.maximum(va, 1e-9), -1.0, 1.0)), 0.0)
    vs = np.maximum(va, 1e-3)
    qS = 0.5 * RHO0 * va * va * F.S
    ex1 = np.exp(np.clip(-STALL_M * (alpha - F.alpha0), -60.0, 60.0))
    ex2 = np.exp(np.clip(STALL_M * (alpha + F.alpha0), -60.0, 60.0))
    sig = (1.0 + ex1 + ex2) / ((1.0 + ex1) * (1.0 + ex2))
    cl_lin = F.cl0 + F.cl_alpha * alpha
    CL = (1.0 - sig) * cl_lin + sig * 2.0 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha)
    CD = F.cd0 + cl_lin ** 2 / (math.pi * F.e * F.AR)
    c2v, b2v = F.c / (2.0 * vs), F.b / (2.0 * vs)
    ca, sa = np.cos(alpha), np.sin(alpha)
    # eq. 4.18 (lift and drag resolved into body x, z) and the pitching moment; eq. 4.19–4.20 (lateral)
    CX = -CD * ca + CL * sa
    CXq = -F.cd_q * ca + F.cl_q * sa
    CXde = -F.cd_de * ca + F.cl_de * sa
    CZ = -CD * sa - CL * ca
    CZq = -F.cd_q * sa - F.cl_q * ca
    CZde = -F.cd_de * sa - F.cl_de * ca
    thrust = dt_ * np.minimum(F.t_static, F.eta_prop * F.p_max_w / np.maximum(va, 0.5))
    fx = qS * (CX + CXq * c2v * q + CXde * de) + thrust
    fy = qS * (F.cy0 + F.cy_beta * beta + F.cy_p * b2v * p + F.cy_r * b2v * r + F.cy_da * da + F.cy_dr * dr)
    fz = qS * (CZ + CZq * c2v * q + CZde * de)
    ll = qS * F.b * (F.cr0 + F.cr_beta * beta + F.cr_p * b2v * p + F.cr_r * b2v * r + F.cr_da * da + F.cr_dr * dr)
    m = qS * F.c * (F.cm0 + F.cm_alpha * alpha + F.cm_q * c2v * q + F.cm_de * de)
    nn = qS * F.b * (F.cn0 + F.cn_beta * beta + F.cn_p * b2v * p + F.cn_r * b2v * r + F.cn_da * da + F.cn_dr * dr)
    # gravity in the body frame (eq. 4.3)
    gx, gy, gz = -F.m * G * np.sin(theta), F.m * G * np.cos(theta) * np.sin(phi), F.m * G * np.cos(theta) * np.cos(phi)
    u, v, w = vb[:, 0], vb[:, 1], vb[:, 2]
    f = np.empty_like(x)
    f[:, 0:3] = np.einsum("nij,nj->ni", R, vb)                                      # eq. 3.14
    f[:, 3] = r * v - q * w + (fx + gx) / F.m                                        # eq. 3.15
    f[:, 4] = p * w - r * u + (fy + gy) / F.m
    f[:, 5] = q * u - p * v + (fz + gz) / F.m
    cf, sf, ct, tt = np.cos(phi), np.sin(phi), np.cos(theta), np.tan(theta)
    f[:, 6] = p + q * sf * tt + r * cf * tt                                          # eq. 3.16
    f[:, 7] = q * cf - r * sf
    f[:, 8] = (q * sf + r * cf) / np.maximum(ct, 1e-6)
    Jx, Jy, Jz = F.J[:, 0], F.J[:, 1], F.J[:, 2]
    f[:, 9] = (Jy - Jz) / Jx * q * r + ll / Jx                                       # eq. 3.17 with J_xz = 0
    f[:, 10] = (Jz - Jx) / Jy * p * r + m / Jy
    f[:, 11] = (Jx - Jy) / Jz * p * q + nn / Jz
    f[:, 12:15] = (delta_cmd - x[:, 12:15]) / F.tau_servo[:, None]
    f[:, 15] = (thr_cmd - dt_) / F.tau_motor
    return f


class _State:
    """The oracle state in the engine's convention (ENU position and velocity, WORLD←FLU quaternion, FLU rates) for the shared controller."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.p = np.zeros((n, 3))
        self.v = np.zeros((n, 3))
        self.q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
        self.w = np.zeros((n, 3))
        self.m = np.ones(n)
        self.J = np.ones((n, 3))

    def load(self, x: np.ndarray) -> None:
        R = _rot_body_to_ned(x[:, 6], x[:, 7], x[:, 8])
        self.p[:] = x[:, 0:3] @ _P.T
        self.v[:] = np.einsum("nij,nj->ni", R, x[:, 3:6]) @ _P.T
        self.q[:] = RB.q_from_R(np.einsum("ij,njk,kl->nil", _P, R, _T))
        self.w[:] = x[:, 9:12] @ _T


class BMOracleTier:
    """PhysicsTier-shaped driver of the Beard–McLain model for fixed wings (same commands, tracker and PoseHistory as ``L2Tier``)."""

    name = "bm"

    def __init__(self, uavs: Sequence[str], spawn: Mapping[str, tuple[tuple[float, float, float], float]], world=None, *,
                 grounded: Sequence[str] | None = None, vehicles: Mapping[str, str] | None = None, clock_profile: str = "d1_250",
                 tick_ms: int = 100, outputs: Sequence[str] = (), history_s: float = 2.0, retain_s: float | None = 1.0) -> None:
        from awr.phys.fast import clock_of

        self.uavs = tuple(uavs)
        self.n = len(self.uavs)
        self.cp: ClockProfile = clock_of(clock_profile)
        self.tick_ms = int(tick_ms)
        self.n_steps = self.tick_ms * self.cp.physics_hz // 1000
        self.step_ns = 1_000_000_000 // self.cp.physics_hz
        self.dt = self.step_ns * 1e-9
        self.history_s, self.retain_s = float(history_s), retain_s
        self.profile_ids, self.params = oracle_params(dict(vehicles or {}), self.uavs)
        self.fleet = WingFleet(self.params)                                          # the controller only; the plant is the ODE
        self.S = _State(self.n)
        self.S.m[:] = self.fleet.m
        self.S.J[:] = self.fleet.J
        self.x = np.zeros((self.n, 16))
        for i, u in enumerate(self.uavs):
            (px, py, pz), yaw = spawn[u]
            p = self.params[i]
            alpha, de, dt_ = p.trim(p.v_cruise)
            self.x[i, 0:3] = (py, px, -pz)
            self.x[i, 3:6] = (p.v_cruise * math.cos(alpha), 0.0, p.v_cruise * math.sin(alpha))
            self.x[i, 6:9] = (0.0, alpha, math.pi / 2.0 - yaw)
            self.x[i, 12:16] = (de, 0.0, 0.0, dt_)
            self.fleet.delta[i] = self.fleet.delta_cmd[i] = (de, 0.0, 0.0)
            self.fleet.thr[i] = self.fleet.thr_cmd[i] = dt_
            self.fleet.v_ref[i] = p.v_cruise
            self.fleet.mode[i] = FLY
        self.tracker = WingTracker({u: self.params[i] for i, u in enumerate(self.uavs)}, tick_s=self.tick_ms / 1000.0)
        for u in self.uavs:
            self.tracker.st[u]["mode"] = FLY
        self.wind = np.zeros((self.n, 3))
        self._pending: list[TierCommand] = []
        self.history = PoseHistoryBuffer(self.uavs, hz=self.cp.physics_hz, capacity_s=self.history_s)
        self._t = np.zeros(self.n_steps, np.int64)
        self._blk = np.zeros((self.n_steps, self.n, STATE_N))
        self.t_k_ns = 0
        self.digest = hashlib.sha256()
        self._ticks = 0
        self._physics_ns = 0
        self.S.load(self.x)
        self._prev_rows = self._rows(np.zeros((self.n, 3)))
        self.stats = {"steps": 0}

    def set_wind(self, wind_enu: np.ndarray) -> None:
        self.wind = np.array(wind_enu, np.float64).reshape(self.n, 3)

    def _rows(self, a_w: np.ndarray) -> np.ndarray:
        S = self.S
        out = np.empty((self.n, STATE_N))
        out[:, 0:3] = S.p
        out[:, 3:7] = S.q
        out[:, 7:10] = S.v
        out[:, 10:13] = S.w
        out[:, 13:16] = RB.q_rot_inv(S.q, a_w - G_W)
        return out

    def caps(self) -> dict:
        return {"tier": self.name, "physics_hz": self.cp.physics_hz, "steps_per_tick": self.n_steps, "engine": "awr.phys.l2.bm_oracle",
                "commands": list(COMMANDS), "domains": ["fixed_wing"], "collision": False}

    def open(self) -> dict:
        init = self._rows(np.zeros((self.n, 3)))
        self._prev_rows = init
        self.history.append_block(np.array([0], np.int64), init[None])
        self.digest.update(np.array([0], "<i8").tobytes() + init.astype("<f8").tobytes())
        return {"tier": self.name, "t0_gz_ns": 0, "engine": "awr.phys.l2.bm_oracle"}

    def submit(self, cmds: Sequence[TierCommand]) -> list[DispatchResult]:
        out = []
        for c in cmds:
            if c.uav not in self.uavs:
                out.append(DispatchResult(c.uav, False, 630, detail="unknown vehicle"))
            elif c.kind not in COMMANDS:
                out.append(DispatchResult(c.uav, False, 630, detail=f"{c.kind} is not an airborne fixed-wing command of the oracle"))
            elif c.kind == "follow_path" and (c.path_enu_m is None or not 1 <= len(c.path_enu_m) <= 256):
                out.append(DispatchResult(c.uav, False, 630, detail="follow_path needs 1..256 points"))
            elif c.kind == "orbit" and (c.orbit is None or not 1.0 <= c.orbit.radius_m <= 1000.0):
                out.append(DispatchResult(c.uav, False, 630, detail="orbit radius must be in [1, 1000] m"))
            else:
                self._pending.append(c)
                out.append(DispatchResult(c.uav, True))
        return out

    def advance(self, k: int) -> TickResult:
        w0 = time.perf_counter_ns()
        keep = []
        for c in sorted(self._pending, key=lambda c: c.apply_tick):
            if c.apply_tick > k:
                keep.append(c)
                continue
            i = self.uavs.index(c.uav)
            self.tracker.set(c, self.S.p[i], self.S.v[i], False, 0.0)
        self._pending = keep
        refs = [self.tracker.setpoint(u, self.S.p[i], self.S.v[i], self.S.q[i]) for i, u in enumerate(self.uavs)]
        dt = self.dt
        F = self.fleet
        wind_ned = self.wind @ _P.T                                                   # _P is its own inverse
        for j in range(self.n_steps):
            t_ns = self.t_k_ns + (j + 1) * self.step_ns
            self._t[j] = t_ns
            F.delta[:] = self.x[:, 12:15]
            F.thr[:] = self.x[:, 15]
            F.control(self.S, refs, self.wind, dt)
            dc, tc = F.delta_cmd.copy(), F.thr_cmd.copy()
            x = self.x
            k1 = bm_rhs(x, dc, tc, F, wind_ned)
            k2 = bm_rhs(x + 0.5 * dt * k1, dc, tc, F, wind_ned)
            k3 = bm_rhs(x + 0.5 * dt * k2, dc, tc, F, wind_ned)
            k4 = bm_rhs(x + dt * k3, dc, tc, F, wind_ned)
            xn = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            xn[:, 6] = (xn[:, 6] + math.pi) % (2.0 * math.pi) - math.pi
            xn[:, 8] = (xn[:, 8] + math.pi) % (2.0 * math.pi) - math.pi
            self.x = xn
            self.stats["steps"] += 1
            self.S.load(xn)
            rows = self._rows(np.zeros((self.n, 3)))
            a_w = (rows[:, 7:10] - self._prev_rows[:, 7:10]) / dt
            rows[:, 13:16] = RB.q_rot_inv(rows[:, 3:7], a_w - G_W)
            self._blk[j] = rows
            self._prev_rows = rows
        t_start = self.t_k_ns
        self.t_k_ns = int(self._t[-1])
        self.history.append_block(self._t, self._blk)
        self.digest.update(self._t.astype("<i8").tobytes() + self._blk.astype("<f8").tobytes())
        if self.retain_s is not None:
            self.history.release_before(self.t_k_ns - int(self.retain_s * 1e9))
        self._ticks += 1
        dt_ns = time.perf_counter_ns() - w0
        self._physics_ns += dt_ns
        contacts: tuple[ContactEvent, ...] = ()
        return TickResult(k, t_start, self.t_k_ns, self.n_steps, contacts, self.status(), TickTiming(physics=dt_ns))

    def status(self) -> Mapping[str, VehicleStatus]:
        return dict.fromkeys(self.uavs, VehicleStatus.AIRBORNE)

    def kind(self, uav: str) -> str:
        return self.tracker.kind(uav)

    def close(self) -> dict:
        return {"ticks": self._ticks, "t_end_ns": self.t_k_ns, "pose_sha256": self.digest.hexdigest(), "tier": self.name,
                "engine": "awr.phys.l2.bm_oracle", "kernel": "beard-mclain-6dof-rk4", "physics_hz": self.cp.physics_hz, "steps_per_tick": self.n_steps,
                "profiles": dict(self.profile_ids), "domains": dict.fromkeys(self.uavs, "fixed_wing"), "physics_wall_s": round(self._physics_ns * 1e-9, 3),
                **self.stats, "tracker": dict(self.tracker.stats)}
