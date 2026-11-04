"""Fossen 3-DoF oracle (``awr.phys.l2.fossen_oracle``): the published manoeuvring equations in matrix form, stepped with RK4 in
the body frame, as the baseline the surface vessel domain of the unified L2 engine is compared against
(``awr.phys.tier_compare --set water --tier fossen``).

The model is ``M·nu_dot + C_RB(nu)·nu + C_A(nu_r)·nu_r + D(nu_r)·nu_r = τ`` with ``η̇ = R(ψ)·nu`` (Fossen 2021, ch. 6–7; the Otter
of the MSS ``python_vehicle_simulator``: rigid-body and added mass matrices about the body origin with ``x_g``, the Munk
moment in yaw dropped, linear + quadratic damping, the strip-integral cross-flow drag, twin thrusters ``k·n|n|`` with a
first-order shaft lag inside the ODE). State per vessel ``(x, y, ψ, u, v, r, n_l, n_r)``; the controller (speed, heading,
allocation) is the engine's own ``UsvFleet.control`` evaluated once per step, so the two tiers differ in how the dynamics are
written and integrated, not in what they are asked to do. Added for the comparison only: the shared stop rule (a vessel told
to stop in still water and slower than ``V_STOP`` is at rest).

This is a comparison tool, not a tier for the sandbox: no world, no grounding, no collisions, no wind.
"""

from __future__ import annotations

import hashlib
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
from awr.phys.l2.ground import UsvTable
from awr.phys.l2.usv import N_STRIPS, R_STOP, V_SP_MIN, V_STOP, UsvFleet, WaterTracker
from awr.phys.pose_history import G_W, STATE_N, PoseHistoryBuffer

__all__ = ["FossenOracleTier", "fossen_rhs"]

COMMANDS = ("velocity", "goto", "follow_path", "orbit", "hover", "kill")


def fossen_rhs(x: np.ndarray, n_cmd: np.ndarray, F: UsvFleet, current: np.ndarray) -> np.ndarray:
    """ẍ for n vessels: x (n, 8) = (x, y, ψ, u, v, r, n_l, n_r), n_cmd (n, 2) shaft speed commands, current (n, 2) ENU."""
    n = x.shape[0]
    psi, nu, sh = x[:, 2], x[:, 3:6], x[:, 6:8]
    c, s = np.cos(psi), np.sin(psi)
    nu_c = np.stack([current[:, 0] * c + current[:, 1] * s, -current[:, 0] * s + current[:, 1] * c, np.zeros(n)], axis=1)
    nu_r = nu - nu_c
    u, v, r = nu[:, 0], nu[:, 1], nu[:, 2]
    ur, vr = nu_r[:, 0], nu_r[:, 1]
    # rigid-body and added-mass Coriolis matrices (Fossen, 3-DoF, origin ahead of the cg by x_g)
    C_RB = np.zeros((n, 3, 3))
    C_RB[:, 0, 2] = -F.m * (F.x_g * r + v)
    C_RB[:, 1, 2] = F.m * u
    C_RB[:, 2, 0] = F.m * (F.x_g * r + v)
    C_RB[:, 2, 1] = -F.m * u
    C_A = np.zeros((n, 3, 3))
    C_A[:, 0, 2] = -F.a_v * vr
    C_A[:, 1, 2] = F.a_u * ur
    # C_A[2, 0] and C_A[2, 1] (the Munk moment) are zero as in otter.py
    D = np.zeros((n, 3, 3))
    D[:, 0, 0] = F.d_u + F.d_uu * np.abs(ur)
    D[:, 1, 1] = F.d_v + F.d_vv * np.abs(vr)
    D[:, 2, 2] = F.d_r + F.d_rr * np.abs(r)
    # cross-flow drag by the strip integral (MSS crossFlowDrag, midpoint rule)
    xs = F.L[:, None] * ((np.arange(N_STRIPS) + 0.5) / N_STRIPS - 0.5)[None, :]
    vc = vr[:, None] + xs * r[:, None]
    ucf = np.abs(vc) * vc
    k = 0.5 * F.rho * F.T * F.cd * F.L / N_STRIPS
    tau_cf = np.stack([np.zeros(n), -k * ucf.sum(axis=1), -k * (xs * ucf).sum(axis=1)], axis=1)
    thrust = np.where(sh >= 0.0, F.k_pos[:, None], F.k_neg[:, None]) * sh * np.abs(sh)
    tau = np.stack([thrust[:, 0] + thrust[:, 1], np.zeros(n), F.thr_y * (thrust[:, 1] - thrust[:, 0])], axis=1)
    rhs = tau + tau_cf - np.einsum("nij,nj->ni", C_RB, nu) - np.einsum("nij,nj->ni", C_A + D, nu_r)
    nu_dot = np.einsum("nij,nj->ni", F.Minv, rhs)
    f = np.empty_like(x)
    f[:, 0] = u * c - v * s
    f[:, 1] = u * s + v * c
    f[:, 2] = r
    f[:, 3:6] = nu_dot
    f[:, 6:8] = (n_cmd - sh) / F.tau_n[:, None]
    return f


class FossenOracleTier:
    """PhysicsTier-shaped driver of the Fossen model for surface vessels (same commands, tracker and PoseHistory as ``L2Tier``)."""

    name = "fossen"

    def __init__(self, uavs: Sequence[str], spawn: Mapping[str, tuple[tuple[float, float, float], float]], world=None, *,
                 grounded: Sequence[str] | None = None, vehicles: Mapping[str, str] | None = None, clock_profile: str = "d1_250",
                 tick_ms: int = 100, outputs: Sequence[str] = (), water_z: float = 0.0, history_s: float = 2.0, retain_s: float | None = 1.0) -> None:
        from awr.phys.fast import clock_of

        self.uavs = tuple(uavs)
        self.n = len(self.uavs)
        self.cp: ClockProfile = clock_of(clock_profile)
        self.tick_ms = int(tick_ms)
        self.n_steps = self.tick_ms * self.cp.physics_hz // 1000
        self.step_ns = 1_000_000_000 // self.cp.physics_hz
        self.dt = self.step_ns * 1e-9
        self.water_z = float(water_z)
        self.history_s, self.retain_s = float(history_s), retain_s
        T = UsvTable()
        vehicles = dict(vehicles or {})
        self.profile_ids = {u: T.key(vehicles.get(u, "otter")) for u in self.uavs}
        self.params = [T.get(self.profile_ids[u]) for u in self.uavs]
        self.fleet = UsvFleet(self.params)                                       # the controller only; the plant is the ODE below
        self.x = np.zeros((self.n, 8))
        for i, u in enumerate(self.uavs):
            (px, py, _z), yaw = spawn[u]
            self.x[i, 0], self.x[i, 1], self.x[i, 2] = px, py, yaw
        self.current = np.zeros((self.n, 2))
        self.tracker = WaterTracker({u: self.params[i] for i, u in enumerate(self.uavs)}, tick_s=self.tick_ms / 1000.0)
        self._pending: list[TierCommand] = []
        self.history = PoseHistoryBuffer(self.uavs, hz=self.cp.physics_hz, capacity_s=self.history_s)
        self._t = np.zeros(self.n_steps, np.int64)
        self._blk = np.zeros((self.n_steps, self.n, STATE_N))
        self.t_k_ns = 0
        self.digest = hashlib.sha256()
        self._ticks = 0
        self._physics_ns = 0
        self._moving = np.zeros(self.n, bool)
        self._prev_rows = self._rows(np.zeros((self.n, 3)))
        self.stats = {"rest_steps": 0}

    def set_current(self, current_enu: np.ndarray) -> None:
        self.current = np.array(current_enu, np.float64).reshape(self.n, -1)[:, :2]

    def _rows(self, a_w: np.ndarray) -> np.ndarray:
        out = np.empty((self.n, STATE_N))
        x = self.x
        c, s = np.cos(x[:, 2]), np.sin(x[:, 2])
        out[:, 0], out[:, 1] = x[:, 0], x[:, 1]
        out[:, 2] = self.water_z + np.array([p.cg_above_water_m for p in self.params])
        q = RB.q_from_yaw(x[:, 2])
        out[:, 3:7] = q
        out[:, 7] = x[:, 3] * c - x[:, 4] * s
        out[:, 8] = x[:, 3] * s + x[:, 4] * c
        out[:, 9] = 0.0
        out[:, 10:12] = 0.0
        out[:, 12] = x[:, 5]
        out[:, 13:16] = RB.q_rot_inv(q, a_w - G_W)
        return out

    def caps(self) -> dict:
        return {"tier": self.name, "physics_hz": self.cp.physics_hz, "steps_per_tick": self.n_steps, "engine": "awr.phys.l2.fossen_oracle",
                "commands": list(COMMANDS), "domains": ["usv"], "collision": False}

    def open(self) -> dict:
        init = self._rows(np.zeros((self.n, 3)))
        self._prev_rows = init
        self.history.append_block(np.array([0], np.int64), init[None])
        self.digest.update(np.array([0], "<i8").tobytes() + init.astype("<f8").tobytes())
        return {"tier": self.name, "t0_gz_ns": 0, "engine": "awr.phys.l2.fossen_oracle"}

    def submit(self, cmds: Sequence[TierCommand]) -> list[DispatchResult]:
        out = []
        for c in cmds:
            if c.uav not in self.uavs:
                out.append(DispatchResult(c.uav, False, 630, detail="unknown vehicle"))
            elif c.kind not in COMMANDS:
                out.append(DispatchResult(c.uav, False, 630, detail=f"{c.kind} is not a surface vessel command"))
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
            self.tracker.set(c, self._prev_rows[i, 0:3])
        self._pending = keep
        speed = np.zeros(self.n)
        head = np.full(self.n, np.nan)
        en = np.ones(self.n, bool)
        for i, u in enumerate(self.uavs):
            r = self._prev_rows[i]
            speed[i], head[i], en[i] = self.tracker.setpoint(u, r[0:3], r[7:10], r[3:7], (float(self.current[i, 0]), float(self.current[i, 1])))
        dt = self.dt
        F = self.fleet
        still = np.hypot(self.current[:, 0], self.current[:, 1]) < 1e-9
        for j in range(self.n_steps):
            t_ns = self.t_k_ns + (j + 1) * self.step_ns
            self._t[j] = t_ns
            x = self.x
            u_r = x[:, 3] - (self.current[:, 0] * np.cos(x[:, 2]) + self.current[:, 1] * np.sin(x[:, 2]))
            n_cmd = F.control(u_r, x[:, 5], x[:, 2], speed, head, en, dt)
            k1 = fossen_rhs(x, n_cmd, F, self.current)
            k2 = fossen_rhs(x + 0.5 * dt * k1, n_cmd, F, self.current)
            k3 = fossen_rhs(x + 0.5 * dt * k2, n_cmd, F, self.current)
            k4 = fossen_rhs(x + dt * k3, n_cmd, F, self.current)
            xn = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            xn[:, 6:8] = np.clip(xn[:, 6:8], F.n_min[:, None], F.n_max[:, None])
            want_stop = (speed < V_SP_MIN) | ~en
            rest = (want_stop & still & (np.abs(x[:, 3]) <= V_STOP) & (np.abs(x[:, 4]) <= V_STOP) & (np.abs(x[:, 5]) <= R_STOP)
                    & (np.abs(xn[:, 3]) <= V_STOP + 1e-12) & (np.abs(xn[:, 5]) <= R_STOP + 1e-12))
            xn[rest, 3:8] = 0.0
            self.stats["rest_steps"] += int(rest.sum())
            F.n_shaft = xn[:, 6:8].copy()
            self.x = xn
            rows = self._rows(np.zeros((self.n, 3)))
            a_w = (rows[:, 7:10] - self._prev_rows[:, 7:10]) / dt
            rows[:, 13:16] = RB.q_rot_inv(rows[:, 3:7], a_w - G_W)
            self._blk[j] = rows
            self._prev_rows = rows
            self._moving = np.any(xn[:, 3:6] != 0.0, axis=1) | (speed > 0.0)
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
        return {u: VehicleStatus.MOVING if self._moving[i] else VehicleStatus.AFLOAT for i, u in enumerate(self.uavs)}

    def close(self) -> dict:
        return {"ticks": self._ticks, "t_end_ns": self.t_k_ns, "pose_sha256": self.digest.hexdigest(), "tier": self.name,
                "engine": "awr.phys.l2.fossen_oracle", "kernel": "fossen-3dof-rk4", "physics_hz": self.cp.physics_hz, "steps_per_tick": self.n_steps,
                "profiles": dict(self.profile_ids), "domains": dict.fromkeys(self.uavs, "usv"), "physics_wall_s": round(self._physics_ns * 1e-9, 3),
                **self.stats, "tracker": dict(self.tracker.stats)}
