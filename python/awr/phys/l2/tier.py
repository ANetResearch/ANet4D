"""``L2Tier`` (``physics = "l2"``): the unified L2 physics engine as a PhysicsTier.

Same interface as ``KinematicTier``, ``FastTier`` and ``GzSimBackend`` (``open``, ``submit``, ``advance``, ``status``, ``close``,
``history``), same command mapping and reference generator as the other two dynamic tiers (``awr.phys.tracker.RefTracker``:
PX4-lite refgen stepped per physics step, position loop, velocity integral and feed-forward as velocity equivalents), same
landing, lift-off and crash rules (M24 §6.5.9) and the same PoseHistory grid (4 ms, body origin, ENU / WORLD←FLU / FLU,
specific force from the velocity difference). What differs from Fast is the plant: every vehicle is a rigid body with
inertia driven by a domain model (``awr.phys.l2.multirotor`` today; ground and surface vehicles and fixed wings join the
same body and the same tier as their models land), with rotor-level actuation, a rate loop and control allocation.

Contact:
- ground = ``ground(xy)`` (DSM nearest cell of ``world``, or a callable) + ``GEAR_Z_M``; touching the ground faster than
  ``IMPACT_VZ_MPS`` downward while airborne is an IMPACT crash; on the ground the horizontal velocity is stopped by friction
  (``MU_GROUND``), the body levels and the rates are damped; landing = contact and ‖v‖ < 0.25 m/s for 0.5 s (land) or
  1.0 s (other), never within ``TAKEOFF_GRACE_S`` of a flight command;
- world geometry: when ``world`` offers ``closest_points_batch`` the exact distance is refreshed whenever the lower bound
  (last distance minus displacement since) drops below ``collision_radius + MARGIN_M``; a point within the collision radius
  outside the 45° cone below the body is COLLISION_WORLD (the vehicle returns to its last position, keeps only its descent,
  motors off; landing on a roof is a ground contact through the DSM instead).

Determinism: all state is numpy float64 updated in a fixed order; ``close()`` returns ``pose_sha256`` over every sample.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from awr.phys import PhysError
from awr.phys.adapter import (
    IMPACT_VZ_MPS,
    LAND_STILL_S,
    STILL_S,
    STILL_V_MPS,
    TAKEOFF_GRACE_S,
    ClockProfile,
    ContactEvent,
    DispatchResult,
    StatusSignals,
    TickResult,
    TickTiming,
    TierCommand,
    VehicleStatus,
    derive_status,
)
from awr.phys.kinematic import GEAR_Z_M
from awr.phys.l2 import rigid as RB
from awr.phys.l2.multirotor import MultirotorFleet, MultirotorParams
from awr.phys.pose_history import G_W, STATE_N, PoseHistoryBuffer
from awr.phys.tracker import RefTracker

__all__ = ["COMMANDS", "DOMAINS", "MARGIN_M", "MU_GROUND", "L2Tier", "domain_of"]

COMMANDS = ("velocity", "goto", "follow_path", "orbit", "hover", "takeoff", "land", "kill")
MAX_PATH_POINTS = 256
MARGIN_M = 0.5
COS_NADIR = math.cos(math.radians(45.0))
MU_GROUND = 0.8                                   # ground friction on the gear, as a deceleration μ·g
SNAP_M = 0.3                                      # a DSM rise above this within one step is a side hit, not ground
LEVEL_TAU_S = 0.1                                 # on the ground the body levels with this time constant
_FLY = frozenset({"takeoff", "goto", "velocity", "follow_path", "orbit"})
DOMAINS = {0: "multirotor", 1: "fixed_wing", 2: "vtol"}   # FleetSim frame_kind → domain model name
IMPLEMENTED = ("multirotor",)


def domain_of(prof) -> str:
    return DOMAINS.get(int(prof.frame_kind), "multirotor")


@dataclass
class _V:
    uav: str
    idx: int
    landed: bool
    contact: bool
    crash: str = "none"
    still_ns: int = 0
    fly_t_ns: int = -10**18
    clear_m: float = 0.0                              # lower bound on the distance to world geometry (0 = unknown, query)


class L2Tier:
    """See the module docstring. ``spawn[u] = ((x, y, z), yaw)`` with z the body-origin height for air spawns."""

    name = "l2"

    def __init__(self, uavs: Sequence[str], spawn: Mapping[str, tuple[tuple[float, float, float], float]], world: Any = None, *,
                 grounded: Sequence[str] | None = None, vehicles: Mapping[str, str] | None = None,
                 clock_profile: str | Mapping | ClockProfile | None = "d1_250", tick_ms: int = 100, outputs: Sequence[str] = (),
                 ground: Callable[[np.ndarray], float] | None = None, history_s: float = 2.0, retain_s: float | None = 1.0) -> None:
        from awr.phys.fast import check_physics_hz, clock_of, resolve_profile
        from awr.sim.fleet.profiles import ProfileTable

        self.uavs = tuple(uavs)
        self.n = len(self.uavs)
        self.cp = clock_of(clock_profile)
        check_physics_hz(self.cp, outputs)
        self.tick_ms = int(tick_ms)
        if (self.tick_ms * self.cp.physics_hz) % 1000:
            raise PhysError(637, f"tick {self.tick_ms} ms is not a multiple of the physics step at {self.cp.physics_hz} Hz")
        self.n_steps = self.tick_ms * self.cp.physics_hz // 1000
        self.step_ns = 1_000_000_000 // self.cp.physics_hz
        self.dt = self.step_ns * 1e-9
        self.world = world
        self._ground = ground
        self.history_s, self.retain_s = float(history_s), retain_s
        T = ProfileTable()
        vehicles = dict(vehicles or {})
        self.profile_ids: dict[str, str] = {}
        params: list[MultirotorParams] = []
        for u in self.uavs:
            pid = resolve_profile(vehicles.get(u, "x500"), T)
            prof = T.get(pid)
            dom = domain_of(prof)
            if dom not in IMPLEMENTED:
                raise PhysError(630, f"{u}: vehicle {pid!r} is a {dom}; the L2 engine has no {dom} model yet (multirotor only)")
            self.profile_ids[u] = pid
            params.append(MultirotorParams.from_profile(prof, T.limits[T.limit_ids[T.limits_index(None, pid)]]))
        self.fleet = MultirotorFleet(params)
        self.S = RB.RigidState(self.n)
        self.S.m[:] = self.fleet.m
        self.S.J[:] = self.fleet.J
        g = set(self.uavs if grounded is None else grounded)
        self._v: list[_V] = []
        for i, u in enumerate(self.uavs):
            (x, y, z), yaw = spawn[u]
            p = np.array([x, y, z], np.float64)
            if u in g:
                p[2] = self.ground_h(p[:2]) + GEAR_Z_M
            self.S.p[i] = p
            self.S.q[i] = RB.q_from_yaw(float(yaw))
            self._v.append(_V(u, i, landed=u in g, contact=u in g))
        air = np.array([i for i, u in enumerate(self.uavs) if u not in g], np.int64)
        if air.size:
            self.fleet.set_hover(air)
        self.tracker = RefTracker(vehicles={u: self.profile_ids[u] for u in self.uavs}, ground_h=self.ground_h, step_ns=self.step_ns,
                                  inner_pid=True)
        self.tracker.tick_s = self.tick_ms / 1000.0
        self.wind = np.zeros((self.n, 3))
        self.rho = np.full(self.n, 1.225)
        self._pending: list[TierCommand] = []
        self.history = PoseHistoryBuffer(self.uavs, hz=self.cp.physics_hz, capacity_s=self.history_s)
        self._t = np.zeros(self.n_steps, np.int64)
        self._blk = np.zeros((self.n_steps, self.n, STATE_N))
        self.t_k_ns = 0
        self.digest = hashlib.sha256()
        self._ticks = 0
        self._physics_ns = 0
        self._v_prev = self.S.v.copy()
        self._a_w = np.zeros((self.n, 3))
        self.stats = {"collision_queries": 0, "crashes": 0, "ground_snaps": 0}

    # ------------------------------------------------------------------------------------------------------ world
    def ground_h(self, xy) -> float:
        xy = np.asarray(xy, np.float64).reshape(-1)[:2]
        if self._ground is not None:
            return float(self._ground(xy))
        if self.world is None:
            return 0.0
        g = self.world.dsm_grid()
        a = np.asarray(g.a)
        h, w = a.shape
        c = int(np.clip(math.floor((float(xy[0]) - g.x0_m) / g.cell_m), 0, w - 1))
        r = int(np.clip(math.floor((float(xy[1]) - g.y0_m) / g.cell_m), 0, h - 1))
        return float(a[r, c])

    def set_wind(self, wind_enu: np.ndarray) -> None:
        """Per-vehicle wind (n, 3) ENU for the coming steps (the environment writes it at the tick start)."""
        self.wind = np.array(wind_enu, np.float64).reshape(self.n, 3)

    def caps(self) -> dict:
        return {"tier": self.name, "physics_hz": self.cp.physics_hz, "pose_history_hz": self.cp.physics_hz, "steps_per_tick": self.n_steps,
                "clock_mode": "lockstep", "clock_profile": {"name": self.cp.name, "main_hz": self.cp.main_hz, "physics_hz": self.cp.physics_hz,
                                                            "l1_hz": self.cp.l1_hz},
                "ranked_allowed": False, "determinism": "1b", "requires_gpu": False, "max_vehicles": 1024,
                "frame_kinds": list(IMPLEMENTED), "commands": list(COMMANDS), "engine": "awr.phys.l2",
                "collision": bool(self.world is not None and hasattr(self.world, "closest_points_batch"))}

    # ------------------------------------------------------------------------------------------------------ samples
    def _rows(self, a_w: np.ndarray) -> np.ndarray:
        out = np.empty((self.n, STATE_N))
        out[:, 0:3] = self.S.p
        out[:, 3:7] = self.S.q
        out[:, 7:10] = self.S.v
        out[:, 10:13] = self.S.w
        out[:, 13:16] = RB.q_rot_inv(self.S.q, a_w - G_W)
        return out

    def open(self) -> dict:
        init = self._rows(np.zeros((self.n, 3)))
        self.history.append_block(np.array([0], np.int64), init[None])
        self.digest.update(np.array([0], "<i8").tobytes() + init.astype("<f8").tobytes())
        self.tracker.bind(list(self.uavs), {u: init[i] for i, u in enumerate(self.uavs)}, [v.uav for v in self._v if v.landed])
        self._observe()
        return {"tier": self.name, "t0_gz_ns": 0, "engine": "awr.phys.l2"}

    def _observe(self) -> None:
        rows = self._rows(np.zeros((self.n, 3)))
        self.tracker.observe(self.t_k_ns, {v.uav: (rows[v.idx], v.contact, v.landed, v.crash != "none") for v in self._v})

    # ------------------------------------------------------------------------------------------------------ commands
    def submit(self, cmds: Sequence[TierCommand]) -> list[DispatchResult]:
        out = []
        for c in cmds:
            if c.uav not in self.uavs:
                out.append(DispatchResult(c.uav, False, 630, detail="unknown vehicle"))
            elif c.kind not in COMMANDS:
                out.append(DispatchResult(c.uav, False, 630, detail=f"{c.kind} is not a tier command"))
            elif c.kind == "follow_path" and (c.path_enu_m is None or not 1 <= len(c.path_enu_m) <= MAX_PATH_POINTS):
                out.append(DispatchResult(c.uav, False, 630, detail=f"follow_path needs 1..{MAX_PATH_POINTS} points"))
            elif c.kind == "orbit" and (c.orbit is None or not 1.0 <= c.orbit.radius_m <= 1000.0):
                out.append(DispatchResult(c.uav, False, 630, detail="orbit radius must be in [1, 1000] m"))
            else:
                self._pending.append(c)
                out.append(DispatchResult(c.uav, True))
        return out

    # ------------------------------------------------------------------------------------------------------ tick
    def advance(self, k: int) -> TickResult:
        w0 = time.perf_counter_ns()
        keep = []
        for c in sorted(self._pending, key=lambda c: c.apply_tick):
            if c.apply_tick > k:
                keep.append(c)
                continue
            v = self._v[self.uavs.index(c.uav)]
            if v.crash != "none":
                continue
            self.tracker.set(c, self.S.p[v.idx])
            if c.kind in _FLY:
                v.landed = False
                v.fly_t_ns = self.t_k_ns
            v.still_ns = 0
        self._pending = keep
        self._observe()
        plan = self.tracker.plan(k, self.t_k_ns, self.n_steps)
        V = np.zeros((self.n_steps, self.n, 3))
        AF = np.zeros((self.n_steps, self.n, 3))
        YAW = np.full((self.n_steps, self.n), np.nan)
        YR = np.zeros((self.n_steps, self.n))
        EN = np.zeros((self.n_steps, self.n), bool)
        for i, u in enumerate(self.uavs):
            for j, sp in enumerate(plan[u]):
                V[j, i] = sp.v_enu
                AF[j, i] = sp.a_ff
                YAW[j, i] = sp.yaw
                YR[j, i] = sp.yaw_rate
                EN[j, i] = sp.enabled
        contacts: list[ContactEvent] = []
        for j in range(self.n_steps):
            t_ns = self.t_k_ns + (j + 1) * self.step_ns
            self._t[j] = t_ns
            a_w = self._step(V[j], AF[j], YAW[j], YR[j], EN[j], t_ns, contacts)
            self._blk[j] = self._rows(a_w)
        t_start = self.t_k_ns
        self.t_k_ns = int(self._t[-1])
        self.history.append_block(self._t, self._blk)
        self.digest.update(self._t.astype("<i8").tobytes() + self._blk.astype("<f8").tobytes())
        if self.retain_s is not None:
            self.history.release_before(self.t_k_ns - int(self.retain_s * 1e9))
        self._ticks += 1
        dt_ns = time.perf_counter_ns() - w0
        self._physics_ns += dt_ns
        return TickResult(k, t_start, self.t_k_ns, self.n_steps, tuple(contacts), self.status(), TickTiming(physics=dt_ns))

    def _step(self, v_sp: np.ndarray, a_ff: np.ndarray, yaw_sp: np.ndarray, yaw_rate: np.ndarray, enabled: np.ndarray, t_ns: int,
              contacts: list[ContactEvent]) -> np.ndarray:
        S, F, dt = self.S, self.fleet, self.dt
        crashed = np.array([v.crash != "none" for v in self._v])
        landed = np.array([v.landed for v in self._v])
        en = enabled & ~crashed & ~landed
        om_cmd = F.control(S, v_sp, a_ff, yaw_sp, yaw_rate, self._a_w, en, dt)
        Fw, tau = F.forces(S, om_cmd, self.wind, self.rho, dt)
        Fw = Fw + S.m[:, None] * G_W
        p_prev = S.p.copy()
        v0 = S.v.copy()
        S.integrate(Fw, tau, dt)
        # ground contact per vehicle
        for v in self._v:
            i = v.idx
            zg = self.ground_h(S.p[i, :2]) + GEAR_Z_M
            if S.p[i, 2] <= zg:
                vz = float(S.v[i, 2])
                if not v.contact and v.crash == "none" and zg - S.p[i, 2] > SNAP_M:
                    # the ground rose under the body by more than a step could descend: a wall or roof edge in the DSM, not a landing
                    v.crash = "world"
                    self.stats["crashes"] += 1
                    self.stats["ground_snaps"] += 1
                    at = S.p[i].copy()
                    S.p[i] = p_prev[i]
                    S.v[i, :2] = 0.0
                    S.v[i, 2] = min(vz, 0.0)
                    S.w[i] = 0.0
                    F.stop(np.array([i]))
                    contacts.append(ContactEvent(v.uav, t_ns, "COLLISION_WORLD", "dsm", tuple(float(x) for x in at), None,
                                                 float(np.linalg.norm(v0[i]))))
                    continue
                if not v.contact and v.crash == "none" and -vz > IMPACT_VZ_MPS:
                    v.crash = "impact"
                    self.stats["crashes"] += 1
                    contacts.append(ContactEvent(v.uav, t_ns, "IMPACT", "dsm", (float(S.p[i, 0]), float(S.p[i, 1]), zg), (0.0, 0.0, 1.0), -vz))
                S.p[i, 2] = zg
                S.v[i, 2] = max(vz, 0.0)
                h = math.hypot(S.v[i, 0], S.v[i, 1])
                if h > 0.0:
                    S.v[i, :2] *= max(h - MU_GROUND * RB.G * dt, 0.0) / h
                S.w[i, :2] = 0.0
                # the gear levels the body: roll and pitch relax toward zero, the heading is kept
                yaw = float(RB.yaw_of(S.q[i:i + 1])[0])
                lev = RB.q_from_yaw(yaw)[None]
                a_ = min(dt / LEVEL_TAU_S, 1.0)
                qi = S.q[i:i + 1]
                if float(np.sum(qi * lev)) < 0.0:
                    lev = -lev
                qn = qi + (lev - qi) * a_
                S.q[i] = qn[0] / np.linalg.norm(qn[0])
                v.contact = True
            elif v.contact and S.p[i, 2] > zg + 0.02:
                v.contact = False
                if not v.landed and v.crash == "none":
                    contacts.append(ContactEvent(v.uav, t_ns, "LIFTOFF", "dsm", (float(S.p[i, 0]), float(S.p[i, 1]), zg), (0.0, 0.0, 1.0),
                                                 float(np.linalg.norm(S.v[i]))))
            if v.crash != "none":
                if v.contact:
                    S.v[i] = 0.0
                    S.w[i] = 0.0
                continue
            if v.landed:
                S.v[i] = 0.0
                S.w[i] = 0.0
                continue
            # landing judgement
            if v.contact:
                still = float(np.linalg.norm(S.v[i])) < STILL_V_MPS
                v.still_ns = v.still_ns + self.step_ns if still else 0
                need = LAND_STILL_S if self.tracker.kind(v.uav) == "land" else STILL_S
                if v.still_ns >= int(need * 1e9) and t_ns - v.fly_t_ns >= int(TAKEOFF_GRACE_S * 1e9):
                    v.landed = True
                    S.v[i] = 0.0
                    S.w[i] = 0.0
                    F.stop(np.array([i]))
                    contacts.append(ContactEvent(v.uav, t_ns, "TOUCHDOWN", "dsm", (float(S.p[i, 0]), float(S.p[i, 1]), float(S.p[i, 2])),
                                                 (0.0, 0.0, 1.0), 0.0))
            else:
                v.still_ns = 0
        self._collide(p_prev, t_ns, contacts)
        self._v_prev = v0
        self._a_w = (S.v - v0) / dt
        return self._a_w

    def _collide(self, p_prev: np.ndarray, t_ns: int, contacts: list[ContactEvent]) -> None:
        w = self.world
        if w is None or not hasattr(w, "closest_points_batch"):
            return
        S = self.S
        moved = np.linalg.norm(S.p - p_prev, axis=1)
        ask = []
        for v in self._v:
            if v.crash != "none" or v.contact:
                continue
            v.clear_m -= float(moved[v.idx])
            if v.clear_m < self.fleet.collision_r[v.idx] + MARGIN_M:
                ask.append(v)
        if not ask:
            return
        idx = np.array([v.idx for v in ask], np.int64)
        self.stats["collision_queries"] += 1
        d, pts, prim = w.closest_points_batch(S.p[idx], float(self.fleet.collision_r.max() + MARGIN_M + 2.0))
        for n_, v in enumerate(ask):
            i = v.idx
            v.clear_m = float(d[n_])
            r = float(self.fleet.collision_r[i])
            if d[n_] >= r:
                continue
            dvec = np.asarray(pts[n_], np.float64) - S.p[i]
            dn = float(np.linalg.norm(dvec))
            below = dn > 1e-9 and (-dvec[2] / dn) >= COS_NADIR
            if below:
                continue                                                     # ground-like contact below the body: the DSM handles it
            v.crash = "world"
            self.stats["crashes"] += 1
            pid = int(prim[n_])
            sem = int(w.semantic_of(np.array([pid]))[0]) if hasattr(w, "semantic_of") else None
            at = S.p[i].copy()
            S.p[i] = p_prev[i]
            S.v[i, :2] = 0.0
            S.v[i, 2] = min(float(S.v[i, 2]), 0.0)
            S.w[i] = 0.0
            self.fleet.stop(np.array([i]))
            nrm = None if dn <= 1e-9 else tuple(float(x) for x in (-dvec / dn))
            contacts.append(ContactEvent(v.uav, t_ns, "COLLISION_WORLD", "mesh", tuple(float(x) for x in at), nrm,
                                         float(np.linalg.norm(self._v_prev[i])), prim_id=pid, semantic=sem))

    # ------------------------------------------------------------------------------------------------------ status
    def status(self) -> Mapping[str, VehicleStatus]:
        out = {}
        for v in self._v:
            kind = self.tracker.kind(v.uav)
            out[v.uav] = derive_status(StatusSignals(crash_cause=v.crash, landed=v.landed, in_air=not v.landed and v.crash == "none",
                                                     cmd_mode="LAND" if kind == "land" else kind.upper()))
        return out

    def close(self) -> dict:
        return {"ticks": self._ticks, "t_end_ns": self.t_k_ns, "pose_sha256": self.digest.hexdigest(), "tier": self.name,
                "engine": "awr.phys.l2", "kernel": "l2-numpy", "physics_hz": self.cp.physics_hz, "steps_per_tick": self.n_steps,
                "clock_profile": self.caps()["clock_profile"], "profiles": dict(self.profile_ids), "domains": {u: "multirotor" for u in self.uavs},
                "physics_wall_s": round(self._physics_ns * 1e-9, 3), **self.stats, "tracker": dict(self.tracker.stats)}
