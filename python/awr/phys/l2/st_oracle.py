"""CommonRoad single-track oracle (``awr.phys.l2.st_oracle``): the published ST equations stepped with RK4, as the baseline
the ground domain of the unified L2 engine is compared against (``awr.phys.tier_compare`` ``--tier st``).

The model is ``vehicle_dynamics_st`` from the CommonRoad vehicle models as implemented in f1tenth_gym: state
``(s_x, s_y, δ, v, ψ, ψ̇, β)``, inputs ``(steering rate, acceleration)``, linear tyres with load transfer on the acceleration
input, and the switch to the kinematic single-track below 0.5 m/s. Added for the comparison only: gravity along the slope
(the ST model is flat) and the shared stop rule (a vehicle told to stop and slower than ``V_STOP`` is at rest; the ODE alone
only decays asymptotically). The controller (speed loop, heading loop, steering limits) is the engine's own
``UgvFleet.control``, so the two tiers differ in the dynamics and the integration scheme, not in what they are asked to do.

This is a comparison tool, not a tier for the sandbox: no world, no collisions, flat attitude (yaw only) on slopes.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence

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
from awr.phys.l2.ground import GroundTable
from awr.phys.l2.ugv import V_ROLL, V_SP_MIN, V_STOP, GroundTracker, UgvFleet
from awr.phys.pose_history import G_W, STATE_N, PoseHistoryBuffer

__all__ = ["KS_SWITCH_MPS", "STOracleTier", "st_rhs"]

KS_SWITCH_MPS = 0.5
RHO = 1.225
COMMANDS = ("velocity", "goto", "follow_path", "orbit", "hover", "kill")
G = RB.G


def st_rhs(x: np.ndarray, u: np.ndarray, p: UgvFleet, slope: np.ndarray) -> np.ndarray:
    """ẍ of the CommonRoad ST model (f1tenth_gym form) for n vehicles: x (n, 7), u (n, 2) = (sv, a), slope = dz/ds along the heading."""
    d, v, psi, dpsi, beta = (x[:, k] for k in range(2, 7))
    sv, a_in = u[:, 0], u[:, 1]
    mu, m, Iz, lf, lr, h = p.mu, p.m, p.I, p.lf, p.lr, p.h
    # the engine's resistances and the slope on top of the published model (the load transfer stays on the input, as in ST)
    a = a_in - G * slope - p.c_rr * G * np.clip(v / V_ROLL, -1.0, 1.0) - 0.5 * RHO * p.cda * np.abs(v) * v / m
    Csf, Csr = p.csf, p.csr
    L = lf + lr
    f = np.empty_like(x)
    ks = np.abs(v) < KS_SWITCH_MPS
    vs = np.where(ks, np.sign(v) * KS_SWITCH_MPS + (v == 0.0) * KS_SWITCH_MPS, v)    # the ST terms are only evaluated where |v| ≥ 0.5
    # kinematic single-track (|v| < 0.5 m/s, as f1tenth_gym)
    f[:, 0] = np.where(ks, v * np.cos(psi), v * np.cos(psi + beta))
    f[:, 1] = np.where(ks, v * np.sin(psi), v * np.sin(psi + beta))
    f[:, 2] = sv
    f[:, 3] = a
    f[:, 4] = np.where(ks, v / L * np.tan(d), dpsi)
    gf, gr = G * lr - a_in * h, G * lf + a_in * h
    f5 = (-mu * m / (vs * Iz * L) * (lf ** 2 * Csf * gf + lr ** 2 * Csr * gr) * dpsi
          + mu * m / (Iz * L) * (lr * Csr * gr - lf * Csf * gf) * beta
          + mu * m / (Iz * L) * lf * Csf * gf * d)
    f6 = ((mu / (vs ** 2 * L) * (Csr * gr * lr - Csf * gf * lf) - 1.0) * dpsi
          - mu / (vs * L) * (Csr * gr + Csf * gf) * beta
          + mu / (vs * L) * Csf * gf * d)
    f[:, 5] = np.where(ks, a / L * np.tan(d) + v / (L * np.cos(d) ** 2) * sv, f5)
    f[:, 6] = np.where(ks, 0.0, f6)
    return f


class STOracleTier:
    """PhysicsTier-shaped driver of the ST model for ground vehicles (same commands, tracker and PoseHistory as ``L2Tier``)."""

    name = "st"

    def __init__(self, uavs: Sequence[str], spawn: Mapping[str, tuple[tuple[float, float, float], float]], world=None, *,
                 grounded: Sequence[str] | None = None, vehicles: Mapping[str, str] | None = None, clock_profile: str = "d1_250",
                 tick_ms: int = 100, outputs: Sequence[str] = (), ground: Callable[[np.ndarray], float] | None = None,
                 history_s: float = 2.0, retain_s: float | None = 1.0) -> None:
        from awr.phys.fast import clock_of

        self.uavs = tuple(uavs)
        self.n = len(self.uavs)
        self.cp: ClockProfile = clock_of(clock_profile)
        self.tick_ms = int(tick_ms)
        self.n_steps = self.tick_ms * self.cp.physics_hz // 1000
        self.step_ns = 1_000_000_000 // self.cp.physics_hz
        self.dt = self.step_ns * 1e-9
        self._ground = ground
        self.history_s, self.retain_s = float(history_s), retain_s
        T = GroundTable()
        vehicles = dict(vehicles or {})
        self.profile_ids = {u: T.key(vehicles.get(u, "f1tenth")) for u in self.uavs}
        self.params = [T.get(self.profile_ids[u]) for u in self.uavs]
        self.fleet = UgvFleet(self.params)                                       # the controller only; the plant is the ST ODE
        self.x = np.zeros((self.n, 7))
        for i, u in enumerate(self.uavs):
            (px, py, _z), yaw = spawn[u]
            self.x[i, 0], self.x[i, 1], self.x[i, 4] = px, py, yaw
        self.tracker = GroundTracker({u: self.params[i] for i, u in enumerate(self.uavs)}, tick_s=self.tick_ms / 1000.0)
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
        self.stats = {"ks_steps": 0}

    def ground_h(self, xy) -> float:
        return 0.0 if self._ground is None else float(self._ground(np.asarray(xy, np.float64).reshape(-1)[:2]))

    def _slope(self) -> np.ndarray:
        if self._ground is None:
            return np.zeros(self.n)
        out = np.empty(self.n)
        for i in range(self.n):
            c, s = math.cos(self.x[i, 4]), math.sin(self.x[i, 4])
            hl = self.params[i].wheelbase / 2.0
            out[i] = (self.ground_h((self.x[i, 0] + hl * c, self.x[i, 1] + hl * s)) - self.ground_h((self.x[i, 0] - hl * c, self.x[i, 1] - hl * s))) / (2 * hl)
        return out

    def _rows(self, a_w: np.ndarray) -> np.ndarray:
        out = np.empty((self.n, STATE_N))
        x = self.x
        z = np.array([self.ground_h(x[i, :2]) for i in range(self.n)]) + np.array([p.cg_h for p in self.params])
        out[:, 0], out[:, 1], out[:, 2] = x[:, 0], x[:, 1], z
        q = RB.q_from_yaw(x[:, 4])
        out[:, 3:7] = q
        out[:, 7] = x[:, 3] * np.cos(x[:, 4] + x[:, 6])
        out[:, 8] = x[:, 3] * np.sin(x[:, 4] + x[:, 6])
        out[:, 9] = 0.0
        out[:, 10:12] = 0.0
        out[:, 12] = x[:, 5]
        out[:, 13:16] = RB.q_rot_inv(q, a_w - G_W)
        return out

    def caps(self) -> dict:
        return {"tier": self.name, "physics_hz": self.cp.physics_hz, "steps_per_tick": self.n_steps, "engine": "awr.phys.l2.st_oracle",
                "commands": list(COMMANDS), "domains": ["ugv"], "collision": False}

    def open(self) -> dict:
        init = self._rows(np.zeros((self.n, 3)))
        self._prev_rows = init
        self.history.append_block(np.array([0], np.int64), init[None])
        self.digest.update(np.array([0], "<i8").tobytes() + init.astype("<f8").tobytes())
        return {"tier": self.name, "t0_gz_ns": 0, "engine": "awr.phys.l2.st_oracle"}

    def submit(self, cmds: Sequence[TierCommand]) -> list[DispatchResult]:
        out = []
        for c in cmds:
            if c.uav not in self.uavs:
                out.append(DispatchResult(c.uav, False, 630, detail="unknown vehicle"))
            elif c.kind not in COMMANDS:
                out.append(DispatchResult(c.uav, False, 630, detail=f"{c.kind} is not a ground vehicle command"))
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
            speed[i], head[i], en[i] = self.tracker.setpoint(u, r[0:3], r[7:10], r[3:7])
        dt = self.dt
        for j in range(self.n_steps):
            t_ns = self.t_k_ns + (j + 1) * self.step_ns
            self._t[j] = t_ns
            x = self.x
            a, sv = self.fleet.control(x[:, 3], x[:, 4], x[:, 2], speed, head, en, dt)
            u = np.stack([sv, a], axis=1)
            slope = self._slope()
            k1 = st_rhs(x, u, self.fleet, slope)
            k2 = st_rhs(x + 0.5 * dt * k1, u, self.fleet, slope)
            k3 = st_rhs(x + 0.5 * dt * k2, u, self.fleet, slope)
            k4 = st_rhs(x + dt * k3, u, self.fleet, slope)
            xn = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            xn[:, 2] = np.clip(xn[:, 2], -self.fleet.s_max, self.fleet.s_max)
            self.stats["ks_steps"] += int(np.sum(np.abs(x[:, 3]) < KS_SWITCH_MPS))
            rest = ((np.abs(speed) < V_SP_MIN) | ~en) & (np.abs(x[:, 3]) <= V_STOP) & (np.abs(xn[:, 3]) <= V_STOP + 1e-12)
            xn[rest, 3] = 0.0
            xn[rest, 5] = 0.0
            xn[rest, 6] = 0.0
            self.fleet.delta = xn[:, 2].copy()
            self.x = xn
            rows = self._rows(np.zeros((self.n, 3)))
            a_w = (rows[:, 7:10] - self._prev_rows[:, 7:10]) / dt
            rows[:, 13:16] = RB.q_rot_inv(rows[:, 3:7], a_w - G_W)
            self._blk[j] = rows
            self._prev_rows = rows
            self._moving = (np.abs(xn[:, 3]) > 0.0) | (speed > 0.0)
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
        return {u: VehicleStatus.MOVING if self._moving[i] else VehicleStatus.GROUNDED for i, u in enumerate(self.uavs)}

    def close(self) -> dict:
        return {"ticks": self._ticks, "t_end_ns": self.t_k_ns, "pose_sha256": self.digest.hexdigest(), "tier": self.name,
                "engine": "awr.phys.l2.st_oracle", "kernel": "commonroad-st-rk4", "physics_hz": self.cp.physics_hz, "steps_per_tick": self.n_steps,
                "profiles": dict(self.profile_ids), "domains": dict.fromkeys(self.uavs, "ugv"), "physics_wall_s": round(self._physics_ns * 1e-9, 3),
                **self.stats, "tracker": dict(self.tracker.stats)}
