"""``L2Tier`` (``physics = "l2"``): the unified L2 physics engine as a PhysicsTier.

Same interface as ``KinematicTier``, ``FastTier`` and ``GzSimBackend`` (``open``, ``submit``, ``advance``, ``status``, ``close``,
``history``), same command mapping and reference generators as the other tiers (``awr.phys.tracker``), same landing,
lift-off and crash rules (M24 §6.5.9) and the same PoseHistory grid (4 ms, body origin, ENU / WORLD←FLU / FLU, specific
force from the velocity difference). What differs from Fast is the plant: every vehicle is a rigid body with inertia
(``awr.phys.l2.rigid``) driven by the domain model of its device family, in one tier and one step loop:

- ``multirotor`` (``awr.phys.l2.multirotor``): rotor-level actuation, PX4 rate loop, control allocation, composite
  aerodynamics; references from ``RefTracker`` (PX4-lite refgen, ``inner_pid`` mode);
- ``ugv`` (``awr.phys.l2.ugv``): single-track dynamics with linear tyres, load transfer, steering and drive limits,
  terrain following; references from ``GroundTracker`` (the shared M24 tracker as speed and heading);
- ``usv`` (``awr.phys.l2.usv``): Fossen 3-DoF manoeuvring dynamics (added mass, Coriolis, linear + quadratic + cross-flow
  damping), twin thrusters with shaft lag, current and wind; references from ``WaterTracker`` (speed and heading, hover as
  station keeping);
- ``fixed_wing`` (``awr.phys.l2.fixedwing``): coefficient-table 6-DoF aerodynamics (Beard & McLain form), power-limited
  propeller thrust, servo and motor lags, successive-loop-closure autopilot; references from ``WingTracker`` (airspeed,
  course, climb rate with the fast tier's vector-field guidance; catapult launch, parachute recovery, loiter as hover).

A vehicle's family comes from its package: air vehicles through the FleetSim profile table (``frame_kind``: multirotor or
fixed wing), ground vehicles through ``vehicles/ground`` and surface vessels through ``vehicles/usv`` (``awr.phys.l2.ground``).
VTOL joins the same body and the same tier when its model lands; until then the tier rejects it with 630 instead of pretending.

Contact:
- air: ground = ``ground(xy)`` (DSM nearest cell of ``world``, or a callable) + ``GEAR_Z_M``; touching the ground faster than
  ``IMPACT_VZ_MPS`` downward while airborne is an IMPACT crash; on the ground the horizontal velocity is stopped by friction
  (``MU_GROUND``), the body levels and the rates are damped; landing = contact and ‖v‖ < 0.25 m/s for 0.5 s (land) or
  1.0 s (other), never within ``TAKEOFF_GRACE_S`` of a flight command;
- fixed wings: a flying aeroplane that meets the ground is an IMPACT crash whatever its speed (it has no gear to land on
  outside its recovery); under the parachute the touchdown at the chute's descent rate is the designed landing (TOUCHDOWN
  once still); on the catapult rail the body is held at gear height until release (LIFTOFF when it leaves the ground);
- ground vehicles ride on ``ground(xy)`` + cg height with the attitude on the terrain normal; a DSM rise above ``SNAP_M``
  within one step (a wall, a curb) is COLLISION_WORLD;
- surface vessels ride on the water level ``water_z`` (one level per tier) + cg height above water, yaw-only attitude;
  ``ground(xy) > water_z − draft`` is a grounding (COLLISION_WORLD, detector ``dsm``; without a world or ``ground`` the water
  is deep); the vessel stays where it struck;
- world geometry (both): when ``world`` offers ``closest_points_batch`` the exact distance is refreshed whenever the lower
  bound (last distance minus displacement since) drops below ``collision_radius + MARGIN_M``; a point within the collision
  radius outside the 45° cone below the body is COLLISION_WORLD (the vehicle returns to its last position, keeps only its
  descent, actuation off). Ground vehicles query from ``collision_radius`` above the cg so the surface they ride on does not
  mask a wall.

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

__all__ = ["COMMANDS", "DOMAINS", "GROUND_COMMANDS", "IMPLEMENTED", "MARGIN_M", "MU_GROUND", "WATER_COMMANDS", "WING_COMMANDS", "L2Tier", "domain_of"]

COMMANDS = ("velocity", "goto", "follow_path", "orbit", "hover", "takeoff", "land", "kill")
GROUND_COMMANDS = ("velocity", "goto", "follow_path", "orbit", "hover", "kill")
WATER_COMMANDS = GROUND_COMMANDS
WING_COMMANDS = COMMANDS                          # a fixed wing takes every tier command (takeoff = launch, land = parachute, hover = loiter)
MAX_PATH_POINTS = 256
MARGIN_M = 0.5
COS_NADIR = math.cos(math.radians(45.0))
MU_GROUND = 0.8                                   # ground friction on the gear, as a deceleration μ·g
SNAP_M = 0.3                                      # a DSM rise above this within one step is a side hit, not ground
LEVEL_TAU_S = 0.1                                 # on the ground the body levels with this time constant
_FLY = frozenset({"takeoff", "goto", "velocity", "follow_path", "orbit"})
DOMAINS = {0: "multirotor", 1: "fixed_wing", 2: "vtol"}   # FleetSim frame_kind → domain model name
IMPLEMENTED = ("multirotor", "ugv", "usv", "fixed_wing")
_AIR = frozenset({"multirotor", "fixed_wing"})      # domains that fly (ground contact, landing and crash rules of the air)


def domain_of(prof) -> str:
    return DOMAINS.get(int(prof.frame_kind), "multirotor")


@dataclass
class _V:
    uav: str
    gi: int                                           # row in the tier (PoseHistory order)
    dom: str                                          # domain model name
    li: int                                           # index inside the domain's RigidState
    landed: bool
    contact: bool
    crash: str = "none"
    still_ns: int = 0
    fly_t_ns: int = -10**18
    clear_m: float = 0.0                              # lower bound on the distance to world geometry (0 = unknown, query)


class _Air:
    """The multirotor group: its own RigidState, fleet model and PX4-lite reference tracker."""

    name = "multirotor"

    def __init__(self, tier: L2Tier, members: list[tuple[str, int]], params: list[MultirotorParams], profile_ids: dict[str, str]) -> None:
        self.uavs = [u for u, _ in members]
        self.gi = np.array([g for _, g in members], np.int64)
        self.fleet = MultirotorFleet(params)
        self.n = len(members)
        self.S = RB.RigidState(self.n)
        self.S.m[:] = self.fleet.m
        self.S.J[:] = self.fleet.J
        self.tracker = RefTracker(vehicles={u: profile_ids[u] for u in self.uavs}, ground_h=tier.ground_h, step_ns=tier.step_ns, inner_pid=True)
        self.tracker.tick_s = tier.tick_ms / 1000.0
        self.wind = np.zeros((self.n, 3))
        self.rho = np.full(self.n, 1.225)
        self._v_prev = np.zeros((self.n, 3))
        self._a_w = np.zeros((self.n, 3))
        self.collision_r = self.fleet.collision_r

    def stop(self, li: int) -> None:
        self.fleet.stop(np.array([li]))


class _Ground:
    """The ground vehicle group: its own RigidState, single-track fleet model and speed / heading tracker."""

    name = "ugv"

    def __init__(self, tier: L2Tier, members: list[tuple[str, int]], params: list) -> None:
        from awr.phys.l2.ugv import GroundTracker, UgvFleet

        self.uavs = [u for u, _ in members]
        self.gi = np.array([g for _, g in members], np.int64)
        self.fleet = UgvFleet(params)
        self.n = len(members)
        self.S = RB.RigidState(self.n)
        self.S.m[:] = self.fleet.m
        self.S.J[:, 0] = self.fleet.I
        self.S.J[:, 1] = self.fleet.I
        self.S.J[:, 2] = self.fleet.I
        self.tracker = GroundTracker({u: p for u, p in zip(self.uavs, params, strict=True)}, tick_s=tier.tick_ms / 1000.0)
        self.wind = np.zeros((self.n, 3))
        self.rho = np.full(self.n, 1.225)
        self._v_prev = np.zeros((self.n, 3))
        self._a_w = np.zeros((self.n, 3))
        self.collision_r = self.fleet.collision_r
        self.half_l = self.fleet.L / 2.0
        self.half_w = self.fleet.track / 2.0
        self.speed_sp = np.zeros(self.n)
        self.head_sp = np.full(self.n, np.nan)
        self.enabled = np.ones(self.n, bool)

    def stop(self, li: int) -> None:
        self.fleet.stop(np.array([li]))


class _Water:
    """The surface vessel group: its own RigidState, Fossen 3-DoF fleet model and speed / heading tracker."""

    name = "usv"

    def __init__(self, tier: L2Tier, members: list[tuple[str, int]], params: list) -> None:
        from awr.phys.l2.usv import UsvFleet, WaterTracker

        self.uavs = [u for u, _ in members]
        self.gi = np.array([g for _, g in members], np.int64)
        self.fleet = UsvFleet(params)
        self.n = len(members)
        self.S = RB.RigidState(self.n)
        self.S.m[:] = self.fleet.m
        self.S.J[:, 0] = self.fleet.I
        self.S.J[:, 1] = self.fleet.I
        self.S.J[:, 2] = self.fleet.I
        self.tracker = WaterTracker({u: p for u, p in zip(self.uavs, params, strict=True)}, tick_s=tier.tick_ms / 1000.0)
        self.wind = np.zeros((self.n, 3))
        self.current = np.zeros((self.n, 2))
        self.rho = np.full(self.n, 1.225)
        self._v_prev = np.zeros((self.n, 3))
        self._a_w = np.zeros((self.n, 3))
        self.collision_r = self.fleet.collision_r
        self.draft = self.fleet.T
        self.speed_sp = np.zeros(self.n)
        self.head_sp = np.full(self.n, np.nan)
        self.enabled = np.ones(self.n, bool)

    def stop(self, li: int) -> None:
        self.fleet.stop(np.array([li]))


class _Wing:
    """The fixed-wing group: its own RigidState, coefficient-table fleet model and course / altitude / airspeed tracker."""

    name = "fixed_wing"

    def __init__(self, tier: L2Tier, members: list[tuple[str, int]], params: list) -> None:
        from awr.phys.l2.fixedwing import GROUND, WingFleet, WingRef, WingTracker

        self.uavs = [u for u, _ in members]
        self.gi = np.array([g for _, g in members], np.int64)
        self.fleet = WingFleet(params)
        self.n = len(members)
        self.S = RB.RigidState(self.n)
        self.S.m[:] = self.fleet.m
        self.S.J[:] = self.fleet.J
        self.tracker = WingTracker({u: p for u, p in zip(self.uavs, params, strict=True)}, tick_s=tier.tick_ms / 1000.0)
        self.wind = np.zeros((self.n, 3))
        self.rho = np.full(self.n, 1.225)
        self._v_prev = np.zeros((self.n, 3))
        self._a_w = np.zeros((self.n, 3))
        self.collision_r = self.fleet.collision_r
        self.refs = [WingRef(p.v_cruise, math.nan, 0.0, 0.0, 0.0, GROUND) for p in params]

    def stop(self, li: int) -> None:
        self.fleet.stop(np.array([li]))


class _Kinds:
    def __init__(self, tier: L2Tier) -> None:
        self._t = tier

    def kind(self, uav: str) -> str:
        return self._t.kind(uav)

    @property
    def stats(self) -> dict:
        return {g.name: dict(g.tracker.stats) for g in self._t.groups}


class L2Tier:
    """See the module docstring. ``spawn[u] = ((x, y, z), yaw)`` with z the body-origin height for air spawns; ground
    vehicles are always placed on the terrain, surface vessels on the water level ``water_z``."""

    name = "l2"

    def __init__(self, uavs: Sequence[str], spawn: Mapping[str, tuple[tuple[float, float, float], float]], world: Any = None, *,
                 grounded: Sequence[str] | None = None, vehicles: Mapping[str, str] | None = None,
                 clock_profile: str | Mapping | ClockProfile | None = "d1_250", tick_ms: int = 100, outputs: Sequence[str] = (),
                 ground: Callable[[np.ndarray], float] | None = None, water_z: float = 0.0, history_s: float = 2.0,
                 retain_s: float | None = 1.0) -> None:
        from awr.phys.fast import check_physics_hz, clock_of, resolve_profile
        from awr.phys.l2.fixedwing import GROUND as WING_GROUND
        from awr.phys.l2.fixedwing import WingParams
        from awr.phys.l2.ground import GroundTable, UsvTable
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
        self.water_z = float(water_z)
        self.history_s, self.retain_s = float(history_s), retain_s
        vehicles = dict(vehicles or {})
        GT = GroundTable()
        UT = UsvTable()
        T = None
        self.profile_ids: dict[str, str] = {}
        self.domains: dict[str, str] = {}
        air: list[tuple[str, int]] = []
        air_params: list[MultirotorParams] = []
        gnd: list[tuple[str, int]] = []
        gnd_params = []
        sea: list[tuple[str, int]] = []
        sea_params = []
        wing: list[tuple[str, int]] = []
        wing_params: list[WingParams] = []
        for gi, u in enumerate(self.uavs):
            ref = vehicles.get(u, "x500")
            if GT.has(ref):
                pid = GT.key(ref)
                self.profile_ids[u], self.domains[u] = pid, "ugv"
                gnd.append((u, gi))
                gnd_params.append(GT.get(pid))
                continue
            if UT.has(ref):
                pid = UT.key(ref)
                self.profile_ids[u], self.domains[u] = pid, "usv"
                sea.append((u, gi))
                sea_params.append(UT.get(pid))
                continue
            if T is None:
                T = ProfileTable()
            pid = resolve_profile(ref, T)
            prof = T.get(pid)
            dom = domain_of(prof)
            if dom not in IMPLEMENTED:
                raise PhysError(630, f"{u}: vehicle {pid!r} is a {dom}; the L2 engine has no {dom} model yet ({', '.join(IMPLEMENTED)})")
            self.profile_ids[u], self.domains[u] = pid, dom
            if dom == "fixed_wing":
                wing.append((u, gi))
                wing_params.append(WingParams.from_profile(prof))
                continue
            air.append((u, gi))
            air_params.append(MultirotorParams.from_profile(prof, T.limits[T.limit_ids[T.limits_index(None, pid)]]))
        self.air = _Air(self, air, air_params, self.profile_ids) if air else None
        self.gnd = _Ground(self, gnd, gnd_params) if gnd else None
        self.sea = _Water(self, sea, sea_params) if sea else None
        self.wing = _Wing(self, wing, wing_params) if wing else None
        self.groups = [g for g in (self.air, self.gnd, self.sea, self.wing) if g is not None]
        self.collision_r = np.zeros(self.n)
        for g in self.groups:
            self.collision_r[g.gi] = g.collision_r
        g_set = set(self.uavs if grounded is None else grounded)
        self._v: list[_V] = []
        for gi, u in enumerate(self.uavs):
            (x, y, z), yaw = spawn[u]
            p = np.array([x, y, z], np.float64)
            if self.domains[u] == "ugv":
                A = self.gnd
                li = A.uavs.index(u)
                p[2] = self.ground_h(p[:2]) + A.fleet.h[li]
                A.S.p[li] = p
                A.S.q[li] = RB.q_from_yaw(float(yaw))
                self._v.append(_V(u, gi, "ugv", li, landed=False, contact=True))
            elif self.domains[u] == "usv":
                A = self.sea
                li = A.uavs.index(u)
                p[2] = self.water_z + A.fleet.h[li]
                A.S.p[li] = p
                A.S.q[li] = RB.q_from_yaw(float(yaw))
                self._v.append(_V(u, gi, "usv", li, landed=False, contact=True))
            elif self.domains[u] == "fixed_wing":
                A = self.wing
                li = A.uavs.index(u)
                if u in g_set:
                    p[2] = self.ground_h(p[:2]) + GEAR_Z_M
                A.S.p[li] = p
                A.S.q[li] = RB.q_from_yaw(float(yaw))
                if u in g_set:
                    A.fleet.mode[li] = WING_GROUND
                    A.tracker.st[u]["mode"] = WING_GROUND
                else:
                    A.fleet.set_trim(np.array([li]), A.S)                                   # airborne spawn: trimmed cruise along the heading
                    A.tracker.st[u]["mode"] = A.fleet.mode[li]
                self._v.append(_V(u, gi, "fixed_wing", li, landed=u in g_set, contact=u in g_set))
            else:
                A = self.air
                li = A.uavs.index(u)
                if u in g_set:
                    p[2] = self.ground_h(p[:2]) + GEAR_Z_M
                A.S.p[li] = p
                A.S.q[li] = RB.q_from_yaw(float(yaw))
                self._v.append(_V(u, gi, "multirotor", li, landed=u in g_set, contact=u in g_set))
        if self.air is not None:
            up = np.array([self.air.uavs.index(u) for u in self.air.uavs if u not in g_set], np.int64)
            if up.size:
                self.air.fleet.set_hover(up)
        self._pending: list[TierCommand] = []
        self.history = PoseHistoryBuffer(self.uavs, hz=self.cp.physics_hz, capacity_s=self.history_s)
        self._t = np.zeros(self.n_steps, np.int64)
        self._blk = np.zeros((self.n_steps, self.n, STATE_N))
        self.t_k_ns = 0
        self.digest = hashlib.sha256()
        self._ticks = 0
        self._physics_ns = 0
        self.stats = {"collision_queries": 0, "crashes": 0, "ground_snaps": 0, "groundings": 0}

    # ------------------------------------------------------------------------------------------------------ world
    @property
    def has_floor(self) -> bool:
        """Whether there is terrain under the water at all (a world DSM or a ``ground`` callable); without one the water is deep."""
        return self._ground is not None or self.world is not None

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
        w = np.array(wind_enu, np.float64).reshape(self.n, 3)
        for g in self.groups:
            g.wind = w[g.gi]

    def set_current(self, current_enu: np.ndarray) -> None:
        """Per-vehicle water current (n, 2 or 3) ENU for the coming steps; only surface vessels feel it."""
        if self.sea is None:
            return
        cur = np.array(current_enu, np.float64).reshape(self.n, -1)[:, :2]
        self.sea.current = cur[self.sea.gi]

    def caps(self) -> dict:
        return {"tier": self.name, "physics_hz": self.cp.physics_hz, "pose_history_hz": self.cp.physics_hz, "steps_per_tick": self.n_steps,
                "clock_mode": "lockstep", "clock_profile": {"name": self.cp.name, "main_hz": self.cp.main_hz, "physics_hz": self.cp.physics_hz,
                                                            "l1_hz": self.cp.l1_hz},
                "ranked_allowed": False, "determinism": "1b", "requires_gpu": False, "max_vehicles": 1024,
                "frame_kinds": list(IMPLEMENTED), "domains": list(IMPLEMENTED), "commands": list(COMMANDS), "ground_commands": list(GROUND_COMMANDS),
                "water_commands": list(WATER_COMMANDS), "wing_commands": list(WING_COMMANDS), "water_z_m": self.water_z,
                "engine": "awr.phys.l2", "collision": bool(self.world is not None and hasattr(self.world, "closest_points_batch"))}

    # ------------------------------------------------------------------------------------------------------ samples
    def _rows(self) -> np.ndarray:
        out = np.empty((self.n, STATE_N))
        for g in self.groups:
            S = g.S
            out[g.gi, 0:3] = S.p
            out[g.gi, 3:7] = S.q
            out[g.gi, 7:10] = S.v
            out[g.gi, 10:13] = S.w
            out[g.gi, 13:16] = RB.q_rot_inv(S.q, g._a_w - G_W)
        return out

    def open(self) -> dict:
        for g in self.groups:
            g._a_w[:] = 0.0
        init = self._rows()
        self.history.append_block(np.array([0], np.int64), init[None])
        self.digest.update(np.array([0], "<i8").tobytes() + init.astype("<f8").tobytes())
        if self.air is not None:
            self.air.tracker.bind(list(self.air.uavs), {u: init[self.uavs.index(u)] for u in self.air.uavs},
                                  [v.uav for v in self._v if v.dom == "multirotor" and v.landed])
            self._observe_air(init)
        return {"tier": self.name, "t0_gz_ns": 0, "engine": "awr.phys.l2"}

    def _observe_air(self, rows: np.ndarray) -> None:
        self.air.tracker.observe(self.t_k_ns, {v.uav: (rows[v.gi], v.contact, v.landed, v.crash != "none") for v in self._v if v.dom == "multirotor"})

    # ------------------------------------------------------------------------------------------------------ commands
    def submit(self, cmds: Sequence[TierCommand]) -> list[DispatchResult]:
        out = []
        for c in cmds:
            if c.uav not in self.uavs:
                out.append(DispatchResult(c.uav, False, 630, detail="unknown vehicle"))
            elif c.kind not in COMMANDS:
                out.append(DispatchResult(c.uav, False, 630, detail=f"{c.kind} is not a tier command"))
            elif self.domains[c.uav] == "ugv" and c.kind not in GROUND_COMMANDS:
                out.append(DispatchResult(c.uav, False, 630, detail=f"{c.kind} is not a ground vehicle command"))
            elif self.domains[c.uav] == "usv" and c.kind not in WATER_COMMANDS:
                out.append(DispatchResult(c.uav, False, 630, detail=f"{c.kind} is not a surface vessel command"))
            elif c.kind == "follow_path" and (c.path_enu_m is None or not 1 <= len(c.path_enu_m) <= MAX_PATH_POINTS):
                out.append(DispatchResult(c.uav, False, 630, detail=f"follow_path needs 1..{MAX_PATH_POINTS} points"))
            elif c.kind == "orbit" and (c.orbit is None or not 1.0 <= c.orbit.radius_m <= 1000.0):
                out.append(DispatchResult(c.uav, False, 630, detail="orbit radius must be in [1, 1000] m"))
            else:
                self._pending.append(c)
                out.append(DispatchResult(c.uav, True))
        return out

    def _group(self, v: _V):
        return self.air if v.dom == "multirotor" else self.gnd if v.dom == "ugv" else self.sea if v.dom == "usv" else self.wing

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
            g = self._group(v)
            if v.dom == "fixed_wing":
                g.tracker.set(c, g.S.p[v.li], g.S.v[v.li], v.landed, self.ground_h(g.S.p[v.li, :2]))
                if c.kind == "takeoff" and v.landed:                                        # the launch: the only way off the ground
                    v.landed = False
                    v.fly_t_ns = self.t_k_ns
            else:
                g.tracker.set(c, g.S.p[v.li])
            if v.dom == "multirotor" and c.kind in _FLY:
                v.landed = False
                v.fly_t_ns = self.t_k_ns
            v.still_ns = 0
        self._pending = keep
        rows = self._rows()
        A, Gd, W, Wg = self.air, self.gnd, self.sea, self.wing
        if A is not None:
            self._observe_air(rows)
            plan = A.tracker.plan(k, self.t_k_ns, self.n_steps)
            V = np.zeros((self.n_steps, A.n, 3))
            AF = np.zeros((self.n_steps, A.n, 3))
            YAW = np.full((self.n_steps, A.n), np.nan)
            YR = np.zeros((self.n_steps, A.n))
            EN = np.zeros((self.n_steps, A.n), bool)
            for i, u in enumerate(A.uavs):
                for j, sp in enumerate(plan[u]):
                    V[j, i] = sp.v_enu
                    AF[j, i] = sp.a_ff
                    YAW[j, i] = sp.yaw
                    YR[j, i] = sp.yaw_rate
                    EN[j, i] = sp.enabled
        if Gd is not None:
            for i, u in enumerate(Gd.uavs):
                r = rows[Gd.gi[i]]
                Gd.speed_sp[i], Gd.head_sp[i], Gd.enabled[i] = Gd.tracker.setpoint(u, r[0:3], r[7:10], r[3:7])
        if W is not None:
            for i, u in enumerate(W.uavs):
                r = rows[W.gi[i]]
                W.speed_sp[i], W.head_sp[i], W.enabled[i] = W.tracker.setpoint(u, r[0:3], r[7:10], r[3:7], (float(W.current[i, 0]), float(W.current[i, 1])))
        if Wg is not None:
            for i, u in enumerate(Wg.uavs):
                r = rows[Wg.gi[i]]
                Wg.refs[i] = Wg.tracker.setpoint(u, r[0:3], r[7:10], r[3:7])
        contacts: list[ContactEvent] = []
        for j in range(self.n_steps):
            t_ns = self.t_k_ns + (j + 1) * self.step_ns
            self._t[j] = t_ns
            p_prev = {}
            if A is not None:
                p_prev["multirotor"] = self._step_air(V[j], AF[j], YAW[j], YR[j], EN[j], t_ns, contacts)
            if Gd is not None:
                p_prev["ugv"] = self._step_ground(t_ns, contacts)
            if W is not None:
                p_prev["usv"] = self._step_water(t_ns, contacts)
            if Wg is not None:
                p_prev["fixed_wing"] = self._step_wing(t_ns, contacts)
            self._collide(p_prev, t_ns, contacts)
            for g in self.groups:
                g._a_w = (g.S.v - g._v_prev) / self.dt
            self._blk[j] = self._rows()
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

    # ------------------------------------------------------------------------------------------------------ multirotor step
    def _step_air(self, v_sp: np.ndarray, a_ff: np.ndarray, yaw_sp: np.ndarray, yaw_rate: np.ndarray, enabled: np.ndarray, t_ns: int,
                  contacts: list[ContactEvent]) -> np.ndarray:
        A = self.air
        S, F, dt = A.S, A.fleet, self.dt
        vs = [v for v in self._v if v.dom == "multirotor"]
        crashed = np.array([v.crash != "none" for v in vs])
        landed = np.array([v.landed for v in vs])
        en = enabled & ~crashed & ~landed
        om_cmd = F.control(S, v_sp, a_ff, yaw_sp, yaw_rate, A._a_w, en, dt)
        Fw, tau = F.forces(S, om_cmd, A.wind, A.rho, dt)
        Fw = Fw + S.m[:, None] * G_W
        p_prev = S.p.copy()
        v0 = S.v.copy()
        S.integrate(Fw, tau, dt)
        # ground contact per vehicle
        for v in vs:
            i = v.li
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
                need = LAND_STILL_S if A.tracker.kind(v.uav) == "land" else STILL_S
                if v.still_ns >= int(need * 1e9) and t_ns - v.fly_t_ns >= int(TAKEOFF_GRACE_S * 1e9):
                    v.landed = True
                    S.v[i] = 0.0
                    S.w[i] = 0.0
                    F.stop(np.array([i]))
                    contacts.append(ContactEvent(v.uav, t_ns, "TOUCHDOWN", "dsm", (float(S.p[i, 0]), float(S.p[i, 1]), float(S.p[i, 2])),
                                                 (0.0, 0.0, 1.0), 0.0))
            else:
                v.still_ns = 0
        A._v_prev = v0
        return p_prev

    # ------------------------------------------------------------------------------------------------------ ground step
    def _step_ground(self, t_ns: int, contacts: list[ContactEvent]) -> np.ndarray:
        from awr.phys.l2.ugv import terrain

        Gd = self.gnd
        S, F, dt = Gd.S, Gd.fleet, self.dt
        vs = [v for v in self._v if v.dom == "ugv"]
        crashed = np.array([v.crash != "none" for v in vs])
        en = Gd.enabled & ~crashed
        psi = RB.yaw_of(S.q)
        z0, grade = terrain(self.ground_h, S.p[:, :2], psi, Gd.half_l, Gd.half_w, SNAP_M)
        p_prev = S.p.copy()
        v0 = S.v.copy()
        F.step(S, np.where(crashed, 0.0, Gd.speed_sp), Gd.head_sp, en, Gd.wind, Gd.rho, grade, dt)
        # terrain following: z on the ground, attitude on the terrain normal, no roll / pitch rates
        yaw = RB.yaw_of(S.q)
        for v in vs:
            i = v.li
            if v.crash != "none":
                S.p[i] = p_prev[i]
                S.v[i] = 0.0
                S.w[i] = 0.0
                continue
            zg = self.ground_h(S.p[i, :2])
            if zg - z0[i] > SNAP_M:                                               # a wall or curb in the DSM: the car stops against it
                v.crash = "world"
                self.stats["crashes"] += 1
                self.stats["ground_snaps"] += 1
                at = S.p[i].copy()
                S.p[i] = p_prev[i]
                S.v[i] = 0.0
                S.w[i] = 0.0
                F.stop(np.array([i]))
                contacts.append(ContactEvent(v.uav, t_ns, "COLLISION_WORLD", "dsm", tuple(float(x) for x in at), None, float(np.linalg.norm(v0[i]))))
                continue
            z_new = zg + F.h[i]
            S.v[i, 2] = (z_new - p_prev[i, 2]) / dt
            S.p[i, 2] = z_new
            nrm = np.array([-grade[i, 0], -grade[i, 1], 1.0])
            nrm /= np.linalg.norm(nrm)
            S.q[i] = RB.q_from_z_yaw(nrm[None], yaw[i:i + 1])[0]
            S.w[i, :2] = 0.0
        Gd._v_prev = v0
        return p_prev

    # ------------------------------------------------------------------------------------------------------ water step
    def _step_water(self, t_ns: int, contacts: list[ContactEvent]) -> np.ndarray:
        W = self.sea
        S, F, dt = W.S, W.fleet, self.dt
        vs = [v for v in self._v if v.dom == "usv"]
        crashed = np.array([v.crash != "none" for v in vs])
        en = W.enabled & ~crashed
        p_prev = S.p.copy()
        v0 = S.v.copy()
        F.step(S, np.where(crashed, 0.0, W.speed_sp), W.head_sp, en, W.wind, W.current, dt)
        yaw = RB.yaw_of(S.q)
        for v in vs:
            i = v.li
            if v.crash != "none":
                S.p[i] = p_prev[i]
                S.v[i] = 0.0
                S.w[i] = 0.0
                continue
            if self.has_floor and self.ground_h(S.p[i, :2]) > self.water_z - W.draft[i]:   # the sea floor rises into the hull: grounded
                v.crash = "world"
                self.stats["crashes"] += 1
                self.stats["groundings"] += 1
                at = S.p[i].copy()
                S.p[i] = p_prev[i]
                S.v[i] = 0.0
                S.w[i] = 0.0
                F.stop(np.array([i]))
                contacts.append(ContactEvent(v.uav, t_ns, "COLLISION_WORLD", "dsm", tuple(float(x) for x in at), None, float(np.linalg.norm(v0[i]))))
                continue
            S.p[i, 2] = self.water_z + F.h[i]                                            # 3-DoF: no heave, roll or pitch
            S.v[i, 2] = 0.0
            S.q[i] = RB.q_from_yaw(float(yaw[i]))
            S.w[i, :2] = 0.0
        W._v_prev = v0
        return p_prev

    # ------------------------------------------------------------------------------------------------------ fixed-wing step
    def _step_wing(self, t_ns: int, contacts: list[ContactEvent]) -> np.ndarray:
        from awr.phys.l2.fixedwing import CHUTE, GROUND, LAUNCH

        Wg = self.wing
        S, F, dt = Wg.S, Wg.fleet, self.dt
        vs = [v for v in self._v if v.dom == "fixed_wing"]
        F.control(S, Wg.refs, Wg.wind, dt)
        Fw, tau = F.forces(S, Wg.wind, Wg.rho, dt)
        Fw = Fw + S.m[:, None] * G_W
        static = np.array([v.landed or v.crash != "none" for v in vs])
        Fw[static] = 0.0
        tau[static] = 0.0
        p_prev = S.p.copy()
        v0 = S.v.copy()
        S.integrate(Fw, tau, dt)
        for v in vs:
            i = v.li
            if v.crash != "none" or v.landed:
                S.p[i] = p_prev[i]
                S.v[i] = 0.0
                S.w[i] = 0.0
                continue
            mode = int(F.mode[i])
            zg = self.ground_h(S.p[i, :2]) + GEAR_Z_M
            if mode == LAUNCH:                                                          # on the rail: carried along it, attitude held
                F.on_rail(S, i, p_prev[i], dt)
                v.contact = True
                continue
            if mode == CHUTE:
                F.level_chute(S, np.array([i]), dt)
            if S.p[i, 2] <= zg:
                vz = float(S.v[i, 2])
                if not v.contact and mode != CHUTE:
                    # a flying aeroplane meeting the ground: there is no gear to land on outside the recovery
                    v.crash = "impact"
                    self.stats["crashes"] += 1
                    contacts.append(ContactEvent(v.uav, t_ns, "IMPACT", "dsm", (float(S.p[i, 0]), float(S.p[i, 1]), zg), (0.0, 0.0, 1.0),
                                                 float(np.linalg.norm(v0[i]))))
                    S.p[i, 2] = zg
                    S.v[i] = 0.0
                    S.w[i] = 0.0
                    F.stop(np.array([i]))
                    v.contact = True
                    continue
                S.p[i, 2] = zg
                S.v[i, 2] = max(vz, 0.0)
                h = math.hypot(S.v[i, 0], S.v[i, 1])
                if h > 0.0:
                    S.v[i, :2] *= max(h - MU_GROUND * RB.G * dt, 0.0) / h
                S.w[i, :2] = 0.0
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
                if v.crash == "none":
                    contacts.append(ContactEvent(v.uav, t_ns, "LIFTOFF", "dsm", (float(S.p[i, 0]), float(S.p[i, 1]), zg), (0.0, 0.0, 1.0),
                                                 float(np.linalg.norm(S.v[i]))))
            if v.contact:
                still = float(np.linalg.norm(S.v[i])) < STILL_V_MPS
                v.still_ns = v.still_ns + self.step_ns if still else 0
                need = LAND_STILL_S if Wg.tracker.kind(v.uav) == "land" else STILL_S
                if v.still_ns >= int(need * 1e9) and t_ns - v.fly_t_ns >= int(TAKEOFF_GRACE_S * 1e9):
                    v.landed = True
                    S.v[i] = 0.0
                    S.w[i] = 0.0
                    F.set_mode(i, GROUND)
                    Wg.tracker.st[v.uav]["mode"] = GROUND
                    contacts.append(ContactEvent(v.uav, t_ns, "TOUCHDOWN", "dsm", (float(S.p[i, 0]), float(S.p[i, 1]), float(S.p[i, 2])),
                                                 (0.0, 0.0, 1.0), 0.0))
            else:
                v.still_ns = 0
        Wg._v_prev = v0
        return p_prev

    # ------------------------------------------------------------------------------------------------------ collision
    def _collide(self, p_prev: Mapping[str, np.ndarray], t_ns: int, contacts: list[ContactEvent]) -> None:
        w = self.world
        if w is None or not hasattr(w, "closest_points_batch"):
            return
        ask: list[_V] = []
        q = []
        for v in self._v:
            if v.crash != "none":
                continue
            g = self._group(v)
            if v.dom in _AIR and v.contact:
                continue
            moved = float(np.linalg.norm(g.S.p[v.li] - p_prev[v.dom][v.li]))
            v.clear_m -= moved
            r = float(self.collision_r[v.gi])
            if v.clear_m < r + MARGIN_M:
                ask.append(v)
                pt = g.S.p[v.li].copy()
                if v.dom in ("ugv", "usv"):
                    pt[2] += r                                                   # query above the cg: the surface the vehicle rides on is not an obstacle
                q.append(pt)
        if not ask:
            return
        self.stats["collision_queries"] += 1
        d, pts, prim = w.closest_points_batch(np.array(q), float(self.collision_r.max() + MARGIN_M + 2.0))
        for n_, v in enumerate(ask):
            g = self._group(v)
            S, i = g.S, v.li
            v.clear_m = float(d[n_])
            r = float(self.collision_r[v.gi])
            if d[n_] >= r:
                continue
            dvec = np.asarray(pts[n_], np.float64) - q[n_]
            dn = float(np.linalg.norm(dvec))
            below = dn > 1e-9 and (-dvec[2] / dn) >= COS_NADIR
            if below:
                continue                                                     # ground-like contact below the body: the DSM handles it
            v.crash = "world"
            self.stats["crashes"] += 1
            pid = int(prim[n_])
            sem = int(w.semantic_of(np.array([pid]))[0]) if hasattr(w, "semantic_of") else None
            at = S.p[i].copy()
            S.p[i] = p_prev[v.dom][i]
            S.v[i, :2] = 0.0
            S.v[i, 2] = min(float(S.v[i, 2]), 0.0) if v.dom in _AIR else 0.0
            S.w[i] = 0.0
            g.stop(i)
            nrm = None if dn <= 1e-9 else tuple(float(x) for x in (-dvec / dn))
            contacts.append(ContactEvent(v.uav, t_ns, "COLLISION_WORLD", "mesh", tuple(float(x) for x in at), nrm,
                                         float(np.linalg.norm(g._v_prev[i])), prim_id=pid, semantic=sem))

    # ------------------------------------------------------------------------------------------------------ status
    def status(self) -> Mapping[str, VehicleStatus]:
        out = {}
        for v in self._v:
            g = self._group(v)
            if v.dom in ("ugv", "usv"):
                if v.crash != "none":
                    out[v.uav] = VehicleStatus.CRASHED
                else:
                    moving = bool(np.any(g.S.v[v.li, :2] != 0.0)) or g.speed_sp[v.li] > 0.0 or (v.dom == "usv" and bool(g.S.w[v.li, 2] != 0.0))
                    rest = VehicleStatus.GROUNDED if v.dom == "ugv" else VehicleStatus.AFLOAT
                    out[v.uav] = VehicleStatus.MOVING if moving else rest
                continue
            kind = g.tracker.kind(v.uav)
            out[v.uav] = derive_status(StatusSignals(crash_cause=v.crash, landed=v.landed, in_air=not v.landed and v.crash == "none",
                                                     cmd_mode="LAND" if kind == "land" else kind.upper()))
        return out

    @property
    def tracker(self):
        """What hosts ask a tier's tracker for (``kind(uav)``, ``stats``), dispatched to the vehicle's domain group."""
        return _Kinds(self)

    @property
    def fleet(self):
        """The multirotor fleet model (rotor speeds, integrators); ``gnd.fleet`` / ``sea.fleet`` / ``wing.fleet`` are the other domains' models."""
        return self.air.fleet if self.air is not None else None

    def kind(self, uav: str) -> str:
        v = self._v[self.uavs.index(uav)]
        return self._group(v).tracker.kind(uav)

    def close(self) -> dict:
        tr = {g.name: dict(g.tracker.stats) for g in self.groups}
        return {"ticks": self._ticks, "t_end_ns": self.t_k_ns, "pose_sha256": self.digest.hexdigest(), "tier": self.name,
                "engine": "awr.phys.l2", "kernel": "l2-numpy", "physics_hz": self.cp.physics_hz, "steps_per_tick": self.n_steps,
                "clock_profile": self.caps()["clock_profile"], "profiles": dict(self.profile_ids), "domains": dict(self.domains),
                "physics_wall_s": round(self._physics_ns * 1e-9, 3), **self.stats, "tracker": tr}
