"""``FwKinTier`` (``awr.phys.fwkin``): the fixed-wing fast tier (``fw`` of the device ladder, stage ``fw_kin`` in the sandbox,
``awr.sim.fixedwing.kinematics``) as a PhysicsTier, so that the unified L2 engine's fixed-wing domain can be compared with it
through the same interface the other tiers use (``awr.phys.tier_compare --set wing --tier fw``).

The plant is the coordinated-turn kinematics of the fast tier (airspeed first-order to its command within the acceleration
limits, roll first-order to the coordinated-turn bank within the roll-rate limit, ``ψ̇ = g·tan φ / V_a``, climb rate clamped,
ground velocity = air velocity + wind); the references come from the same ``WingTracker`` the L2 engine uses (course and
climb-rate vector fields), converted to the compass angles the kinematics expect. Airborne only: a fixed wing spawned in the
air is in level cruise along its heading; takeoff, land and kill are rejected (630), like the other baselines. PoseHistory
rows are built as the sandbox builds them (``quat_xyzw`` from heading, flight-path angle and bank, FLU rates from the heading
and roll rates, specific force from the velocity difference).
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
from awr.phys.l2.fixedwing import FLY, WingTracker
from awr.phys.pose_history import G_W, STATE_N, PoseHistoryBuffer
from awr.sim.fixedwing import kinematics as KN

__all__ = ["FwKinTier"]

COMMANDS = ("velocity", "goto", "follow_path", "orbit", "hover")
_HALF_PI = math.pi / 2.0


class FwKinTier:
    """PhysicsTier-shaped driver of the fixed-wing kinematic fast tier."""

    name = "fw"

    def __init__(self, uavs: Sequence[str], spawn: Mapping[str, tuple[tuple[float, float, float], float]], world=None, *,
                 grounded: Sequence[str] | None = None, vehicles: Mapping[str, str] | None = None, clock_profile: str = "d1_250",
                 tick_ms: int = 100, outputs: Sequence[str] = (), history_s: float = 2.0, retain_s: float | None = 1.0) -> None:
        from awr.phys.fast import clock_of, resolve_profile
        from awr.phys.l2.bm_oracle import oracle_params
        from awr.sim.catalog.derive import fw_aero
        from awr.sim.fleet.profiles import ProfileTable

        self.uavs = tuple(uavs)
        self.n = len(self.uavs)
        self.cp: ClockProfile = clock_of(clock_profile)
        self.tick_ms = int(tick_ms)
        self.n_steps = self.tick_ms * self.cp.physics_hz // 1000
        self.step_ns = 1_000_000_000 // self.cp.physics_hz
        self.dt = self.step_ns * 1e-9
        self.history_s, self.retain_s = float(history_s), retain_s
        vehicles = dict(vehicles or {})
        self.profile_ids, self.params = oracle_params(vehicles, self.uavs)
        T = ProfileTable()
        self.aero = [fw_aero(T.get(resolve_profile(vehicles.get(u, "awr_f1"), T)).doc) for u in self.uavs]
        self.kin: list[KN.FwKin] = []
        for i, u in enumerate(self.uavs):
            (px, py, pz), yaw = spawn[u]
            k = KN.FwKin(x=float(px), y=float(py), z=float(pz), va=self.params[i].v_cruise, psi=_HALF_PI - float(yaw))
            k.vge, k.vgn = k.va * math.sin(k.psi), k.va * math.cos(k.psi)
            self.kin.append(k)
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
        self._prev_rows = self._rows()
        self.stats = {"steps": 0}

    def set_wind(self, wind_enu: np.ndarray) -> None:
        self.wind = np.array(wind_enu, np.float64).reshape(self.n, 3)

    def _rows(self) -> np.ndarray:
        out = np.empty((self.n, STATE_N))
        a_w = np.empty((self.n, 3))
        for i, k in enumerate(self.kin):
            q = KN.quat_xyzw(k.psi, k.gamma, k.phi)
            out[i, 0:3] = (k.x, k.y, k.z)
            out[i, 3:7] = q
            out[i, 7:10] = (k.vge, k.vgn, k.vgu)
            out[i, 10:13] = KN.omega_flu(q, k.psi_dot, k.phi_dot)
            a_w[i] = (k.ae, k.an, k.au)
        out[:, 13:16] = RB.q_rot_inv(out[:, 3:7], a_w - G_W)
        return out

    def caps(self) -> dict:
        return {"tier": self.name, "physics_hz": self.cp.physics_hz, "steps_per_tick": self.n_steps, "engine": "awr.sim.fixedwing.kinematics",
                "commands": list(COMMANDS), "domains": ["fixed_wing"], "collision": False}

    def open(self) -> dict:
        init = self._rows()
        self._prev_rows = init
        self.history.append_block(np.array([0], np.int64), init[None])
        self.digest.update(np.array([0], "<i8").tobytes() + init.astype("<f8").tobytes())
        return {"tier": self.name, "t0_gz_ns": 0, "engine": "awr.sim.fixedwing.kinematics"}

    def submit(self, cmds: Sequence[TierCommand]) -> list[DispatchResult]:
        out = []
        for c in cmds:
            if c.uav not in self.uavs:
                out.append(DispatchResult(c.uav, False, 630, detail="unknown vehicle"))
            elif c.kind not in COMMANDS:
                out.append(DispatchResult(c.uav, False, 630, detail=f"{c.kind} is not an airborne fixed-wing command of the fast tier"))
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
        rows = self._prev_rows
        keep = []
        for c in sorted(self._pending, key=lambda c: c.apply_tick):
            if c.apply_tick > k:
                keep.append(c)
                continue
            i = self.uavs.index(c.uav)
            self.tracker.set(c, rows[i, 0:3], rows[i, 7:10], False, 0.0)
        self._pending = keep
        dt = self.dt
        refs = [self.tracker.setpoint(u, rows[i, 0:3], rows[i, 7:10], rows[i, 3:7]) for i, u in enumerate(self.uavs)]   # per tick, as the engine
        for j in range(self.n_steps):
            t_ns = self.t_k_ns + (j + 1) * self.step_ns
            self._t[j] = t_ns
            for i, _u in enumerate(self.uavs):
                r = refs[i]
                p, a = self.params[i], self.aero[i]
                chi_d = None if math.isnan(r.chi) else _HALF_PI - r.chi
                KN.step(self.kin[i], va_c=r.va, chi_d=chi_d, chi_ff=-r.chi_ff, hdot_c=r.hdot, wind=tuple(float(x) for x in self.wind[i]), dt=dt,
                        v_min=p.v_min, v_max=p.v_max, v_stall=p.v_stall, phi_max=p.phi_max, roll_rate_max=p.roll_rate_max,
                        tau_v=a.tau_v_s, tau_phi=a.tau_phi_s, acc_max=p.accel_max, dec_max=p.decel_max, climb_max=p.climb_max,
                        sink_max=p.sink_max, psi_dot_cmd=None if chi_d is not None else -r.chidot)
            self.stats["steps"] += 1
            self._blk[j] = self._prev_rows = self._rows()
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
                "engine": "awr.sim.fixedwing.kinematics", "kernel": "fw-kin", "physics_hz": self.cp.physics_hz, "steps_per_tick": self.n_steps,
                "profiles": dict(self.profile_ids), "domains": dict.fromkeys(self.uavs, "fixed_wing"), "physics_wall_s": round(self._physics_ns * 1e-9, 3),
                **self.stats, "tracker": dict(self.tracker.stats)}
