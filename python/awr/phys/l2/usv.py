"""Surface vessel domain model of the unified L2 physics engine (``awr.phys.l2.usv``): boats on the shared rigid body.

gz-sim's surface vessel support is a set of plugins (buoyancy, a lift-drag hull, a thruster) tuned per model in VRX, not a
vessel model; the domain here integrates the manoeuvring literature instead: Fossen's 3-DoF horizontal-plane model
(surge, sway, yaw; *Handbook of Marine Craft Hydrodynamics and Motion Control*, 2nd ed., ch. 6–7) in the form the MSS
``python_vehicle_simulator`` uses for the Otter, plus what a sandbox needs (a speed / heading controller, thrust
allocation, station keeping, grounding, world collision, a path follower):

- **rigid body and added mass**: ``M = M_RB(m, x_g, I_z) + diag(X_u̇, Y_v̇, N_ṙ)``; the Coriolis / centripetal terms
  ``C_RB(nu)·nu`` on the body velocity and ``C_A(nu_r)·nu_r`` on the current-relative velocity (the Munk moment in yaw is dropped
  as in ``otter.py``: it is otherwise balanced by an extra nonlinear yaw damping term);
- **damping**: linear ``diag(X_u, Y_v, N_r)`` and quadratic ``diag(X_|u|u, Y_|v|v, N_|r|r)`` on ``nu_r``, plus the 2-D cross-flow
  drag integral along the hull (``½ρ·T·C_d·∫|v_r + x·r|(v_r + x·r)·{1, x} dx``, Hoerner C_d from the package) for vessels whose
  package asks for it (Otter: yes; WAM-V: the VRX quadratic sway damping instead);
- **actuation**: two fixed thrusters at ``(x_t, ±y_t)`` with ``T = k_pos·n|n|`` (``k_neg`` astern), first-order shaft lag
  ``τ_n``, revolution limits ``[n_min, n_max]``; surge force ``T_l + T_r``, yaw moment ``y_t·(T_r − T_l)``. The controller is a
  ramped speed reference (``a_max``) with feed-forward and a PI surge loop (anti-windup as the ground domain), a heading loop
  ``r_sp = K_HEADING·ψ_err`` within ``r_max`` and a P yaw-rate loop, both with damping compensation, allocated to the two
  shafts and saturated per shaft (yaw has priority when the pair saturates); the surge reference is scaled by ``cos ψ_err``
  (clipped at 0), so a boat pointing the wrong way turns before it runs;
- **environment**: wind drag ``½ρ_air·C_dA·|w_rel|·w_rel`` on the above-water body, a water current ``set_current`` entering
  every hydrodynamic term through ``nu_r`` (a vessel with its thrusters off drifts with the current);
- **surface**: the body rides on the tier's water level (``z = water_z + cg_above_water``), yaw-only attitude, no heave, roll or
  pitch (3-DoF); the sea floor / shore is the DSM: ``ground(xy) > water_z − draft`` is a grounding (COLLISION_WORLD);
- **references**: the tier's command set (velocity, goto, follow_path, orbit, hover, kill); ``WaterTracker`` turns the shared
  M24 tracker's velocity set-points into speed and heading, stops inside ``ARRIVE_M`` of a goto target / final waypoint /
  orbit end point (or when the point has been overshot and lies astern within a hull length: a boat cannot turn on the
  spot), and makes ``hover`` station keeping (ground speed ``K_HOLD``·offset back to the hold point once it has drifted
  ``HOLD_M``; the hold point is where the hull can stop, the command position plus the braking distance). takeoff and land
  are rejected (630); kill is thrusters off.

Parameters come from the vessel package (``awr.phys.l2.ground.UsvTable``, ``vehicles/usv``); nothing is tuned per vessel here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from awr.phys.l2 import rigid as RB
from awr.phys.l2.ground import UsvParams
from awr.phys.l2.ugv import ARRIVE_M, TWO_PI, V_SP_MIN, V_STOP
from awr.phys.tracker import Tracker, TrackerLimits

__all__ = ["ARRIVE_M", "HOLD_M", "K_HEADING", "K_HOLD", "N_STRIPS", "RHO_AIR", "R_STOP", "V_HOLD_MAX", "V_SP_MIN", "V_STOP", "UsvFleet", "WaterTracker", "crossflow"]

K_HEADING = 1.0                  # yaw-rate demand per heading error, 1/s (a boat turns slower than a car)
R_STOP = 0.02                    # a vessel told to stop, slower than V_STOP and yawing slower than this is at rest
HOLD_M = 0.3                     # station keeping: the vessel moves back when it has drifted this far from the hold point
K_HOLD = 0.3                     # station keeping: ground speed demand per metre of offset, 1/s (slow: an overshoot means a U-turn)
V_HOLD_MAX = 1.0                 # station keeping: ground speed cap of the approach, m/s
N_STRIPS = 20                    # hull strips of the cross-flow drag integral (MSS crossFlowDrag: dx = L / 20)
RHO_AIR = 1.225


def crossflow(v_r: np.ndarray, r: np.ndarray, L: np.ndarray, T: np.ndarray, cd: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2-D cross-flow drag (sway force, yaw moment) of ``n`` hulls from the strip integral (MSS ``crossFlowDrag``, midpoint rule)."""
    xs = (np.arange(N_STRIPS) + 0.5) / N_STRIPS - 0.5                     # strip centres, fraction of L
    x = L[:, None] * xs[None, :]
    dx = L / N_STRIPS
    vc = v_r[:, None] + x * r[:, None]
    ucf = np.abs(vc) * vc
    k = 0.5 * rho * T * cd * dx
    return -k * np.sum(ucf, axis=1), -k * np.sum(x * ucf, axis=1)


class UsvFleet:
    """Vectorised Fossen 3-DoF dynamics and the speed / heading / allocation loops for ``n`` surface vessels."""

    def __init__(self, params: list[UsvParams]) -> None:
        self.params = list(params)
        n = self.n = len(params)
        f = lambda a: np.array([float(getattr(p, a)) for p in params])  # noqa: E731
        self.m, self.I, self.x_g = f("mass_kg"), f("izz"), f("x_g")
        self.L, self.B, self.T = f("length_m"), f("beam_m"), f("draft_m")
        self.h = f("cg_above_water_m")
        self.collision_r = f("collision_radius_m")
        self.a_u, self.a_v, self.a_r = f("a_u"), f("a_v"), f("a_r")
        self.d_u, self.d_v, self.d_r = f("d_u"), f("d_v"), f("d_r")
        self.d_uu, self.d_vv, self.d_rr = f("d_uu"), f("d_vv"), f("d_rr")
        self.cd, self.rho, self.cda = f("cd_crossflow"), f("rho_water"), f("cda_wind")
        self.thr_x, self.thr_y = f("thr_x"), f("thr_y")
        self.k_pos, self.k_neg = f("k_pos"), f("k_neg")
        self.n_max, self.n_min, self.tau_n = f("n_max"), f("n_min"), f("tau_n")
        self.v_max, self.a_max, self.r_max = f("v_max"), f("a_max"), f("r_max")
        self.k_speed, self.k_yaw = f("k_speed"), f("k_yaw")
        self.T_fwd = self.k_pos * self.n_max * np.abs(self.n_max)         # per-shaft thrust limits
        self.T_rev = self.k_neg * self.n_min * np.abs(self.n_min)          # negative
        # mass matrix (body origin) and its inverse
        M = np.zeros((n, 3, 3))
        M[:, 0, 0] = self.m + self.a_u
        M[:, 1, 1] = self.m + self.a_v
        M[:, 1, 2] = M[:, 2, 1] = self.m * self.x_g
        M[:, 2, 2] = self.I + self.a_r
        self.M = M
        self.Minv = np.linalg.inv(M)
        self.m_surge = M[:, 0, 0]
        self.i_yaw = M[:, 2, 2]
        self.n_shaft = np.zeros((n, 2))                                   # shaft speeds (left, right), rad/s
        self.thrust = np.zeros((n, 2))                                    # thrusts of the last step, N
        self.v_int = np.zeros(n)                                          # surge loop integrator, m/s²
        self.v_ref = np.zeros(n)                                          # ramped speed reference, m/s
        self.tau = np.zeros((n, 3))                                       # actuation (X, Y, N) of the last step
        self.nu_dot = np.zeros((n, 3))
        self.at_rest = np.ones(n, bool)

    # ------------------------------------------------------------------------------------------------------ control
    def control(self, u: np.ndarray, r: np.ndarray, psi: np.ndarray, speed_sp: np.ndarray, heading_sp: np.ndarray, enabled: np.ndarray,
                dt: float) -> np.ndarray:
        """Speed and heading references → commanded shaft speeds (n, 2). ``u`` is the surge speed through the water
        (``u − u_current``) and ``speed_sp`` the reference for it (``WaterTracker`` subtracts the current from the ground
        velocity set-point). Shared with the Fossen oracle, so the two differ in dynamics and integration only. ``heading_sp``
        NaN keeps the current heading."""
        err = np.where(np.isnan(heading_sp), 0.0, (heading_sp - psi + math.pi) % TWO_PI - math.pi)
        # the surge reference is scaled by the heading alignment: a boat pointing the wrong way turns first, it does not run off
        v_sp = np.where(enabled, np.clip(speed_sp, 0.0, self.v_max) * np.clip(np.cos(err), 0.0, 1.0), 0.0)
        dv = np.clip(v_sp - self.v_ref, -self.a_max * dt, self.a_max * dt)
        self.v_ref = np.where(enabled, self.v_ref + dv, 0.0)
        a_ff = np.where(enabled, dv / dt, 0.0)
        e = self.v_ref - u
        a = a_ff + self.k_speed * e + self.v_int
        X = self.m_surge * a + self.d_u * u + self.d_uu * np.abs(u) * u                 # damping compensation on the measured speed
        X = np.where((u >= self.v_max) & (X > 0.0), 0.0, X)
        hi, lo = 2.0 * self.T_fwd, 2.0 * self.T_rev
        Xc = np.clip(X, lo, hi)
        sat = ((Xc >= hi - 1e-9) & (e > 0.0)) | ((Xc <= lo + 1e-9) & (e < 0.0))
        live = enabled & ((speed_sp >= V_SP_MIN) | (np.abs(u) > V_STOP))
        self.v_int = np.where(live, np.where(sat, self.v_int, np.clip(self.v_int + 0.5 * self.k_speed * e * dt, -self.a_max, self.a_max)), 0.0)
        self.v_ref = np.where(live, self.v_ref, 0.0)
        r_sp = np.clip(K_HEADING * err, -self.r_max, self.r_max)
        N = self.i_yaw * self.k_yaw * (r_sp - r) + self.d_r * r + self.d_rr * np.abs(r) * r
        # allocation to the two shafts; the pair saturates on yaw first (surge gives way)
        Nmax = self.thr_y * (self.T_fwd - self.T_rev)
        N = np.clip(N, -Nmax, Nmax)
        half = 0.5 * N / self.thr_y
        Xs = np.clip(0.5 * Xc, self.T_rev + np.abs(half), self.T_fwd - np.abs(half))   # surge per shaft within what yaw leaves
        T_l, T_r = Xs - half, Xs + half
        off = ~enabled | (~live & (np.abs(err) < 1e-9))
        T = np.where(off[:, None], 0.0, np.stack([T_l, T_r], axis=1))
        T = np.clip(T, self.T_rev[:, None], self.T_fwd[:, None])
        return np.sign(T) * np.sqrt(np.abs(T) / np.where(T >= 0.0, self.k_pos[:, None], self.k_neg[:, None]))

    # ------------------------------------------------------------------------------------------------------ dynamics
    def shaft_thrust(self, n: np.ndarray) -> np.ndarray:
        return np.where(n >= 0.0, self.k_pos[:, None], self.k_neg[:, None]) * n * np.abs(n)

    def hydro(self, nu: np.ndarray, nu_r: np.ndarray) -> np.ndarray:
        """``−C_RB(nu)nu − C_A(nu_r)nu_r − D(nu_r)nu_r`` + cross-flow drag, (n, 3), for the body velocity ``nu = (u, v, r)`` and the
        current-relative ``nu_r``."""
        u, v, r = nu[:, 0], nu[:, 1], nu[:, 2]
        ur, vr = nu_r[:, 0], nu_r[:, 1]
        m, xg = self.m, self.x_g
        c = np.empty_like(nu)
        c[:, 0] = -m * (xg * r + v) * r - self.a_v * vr * r
        c[:, 1] = m * u * r + self.a_u * ur * r
        c[:, 2] = m * (xg * r + v) * u - m * u * v                                      # Munk moment (a_v − a_u)·u_r·v_r dropped as in otter.py
        d = np.empty_like(nu)
        d[:, 0] = (self.d_u + self.d_uu * np.abs(ur)) * ur
        d[:, 1] = (self.d_v + self.d_vv * np.abs(vr)) * vr
        d[:, 2] = (self.d_r + self.d_rr * np.abs(r)) * r
        Yh, Nh = crossflow(vr, r, self.L, self.T, self.cd, self.rho)
        out = -c - d
        out[:, 1] += Yh
        out[:, 2] += Nh
        return out

    def step(self, S: RB.RigidState, speed_sp: np.ndarray, heading_sp: np.ndarray, enabled: np.ndarray, wind: np.ndarray, current: np.ndarray,
             dt: float) -> None:
        """One physics step of the horizontal dynamics on the shared body (the caller constrains z and the attitude to the
        water surface afterwards). ``wind`` (n, 3) and ``current`` (n, 2) are ENU."""
        psi = RB.yaw_of(S.q)
        c, s = np.cos(psi), np.sin(psi)
        u = S.v[:, 0] * c + S.v[:, 1] * s
        v = -S.v[:, 0] * s + S.v[:, 1] * c
        r = S.w[:, 2]
        nu = np.stack([u, v, r], axis=1)
        uc = current[:, 0] * c + current[:, 1] * s
        vc = -current[:, 0] * s + current[:, 1] * c
        nu_r = np.stack([u - uc, v - vc, r], axis=1)
        n_cmd = self.control(nu_r[:, 0], r, psi, speed_sp, heading_sp, enabled, dt)
        self.n_shaft = np.clip(self.n_shaft + (n_cmd - self.n_shaft) * np.minimum(dt / self.tau_n[:, None], 1.0), self.n_min[:, None], self.n_max[:, None])
        self.thrust = self.shaft_thrust(self.n_shaft)
        tau = np.zeros((self.n, 3))
        tau[:, 0] = self.thrust[:, 0] + self.thrust[:, 1]
        tau[:, 2] = self.thr_y * (self.thrust[:, 1] - self.thrust[:, 0])
        wrel = wind[:, :2] - S.v[:, :2]
        Fw = 0.5 * RHO_AIR * self.cda[:, None] * np.hypot(wrel[:, 0], wrel[:, 1])[:, None] * wrel
        tau[:, 0] += Fw[:, 0] * c + Fw[:, 1] * s
        tau[:, 1] += -Fw[:, 0] * s + Fw[:, 1] * c
        self.tau = tau
        rhs = tau + self.hydro(nu, nu_r)
        nu_dot = np.einsum("nij,nj->ni", self.Minv, rhs)
        # a vessel told to stop in still water comes to rest (the ODE alone only decays asymptotically)
        want_stop = (speed_sp < V_SP_MIN) | ~enabled
        still = np.hypot(current[:, 0], current[:, 1]) < 1e-9
        slow = (np.abs(u) <= V_STOP) & (np.abs(v) <= V_STOP) & (np.abs(r) <= R_STOP)
        slow_next = (np.abs(u + nu_dot[:, 0] * dt) <= V_STOP + 1e-12) & (np.abs(r + nu_dot[:, 2] * dt) <= R_STOP + 1e-12)
        rest = want_stop & still & slow & slow_next
        self.at_rest = rest
        self.nu_dot = nu_dot
        # body acceleration → world: d/dt(R nu) = R(nu_dot + r k × nu)
        ab_x = nu_dot[:, 0] - r * v
        ab_y = nu_dot[:, 1] + r * u
        F_world = np.stack([S.m * (ab_x * c - ab_y * s), S.m * (ab_x * s + ab_y * c), np.zeros(self.n)], axis=1)
        tau_b = np.stack([np.zeros(self.n), np.zeros(self.n), S.J[:, 2] * nu_dot[:, 2]], axis=1)
        if rest.any():
            F_world[rest] = -S.m[rest, None] * S.v[rest] / dt
            tau_b[rest] = -S.J[rest] * S.w[rest] / dt
        S.integrate(F_world, tau_b, dt)
        if rest.any():
            S.v[rest] = 0.0
            S.w[rest] = 0.0
            self.n_shaft[rest] = 0.0
            self.thrust[rest] = 0.0

    def stop(self, idx: np.ndarray) -> None:
        self.n_shaft[idx] = 0.0
        self.thrust[idx] = 0.0
        self.v_int[idx] = 0.0
        self.v_ref[idx] = 0.0
        self.tau[idx] = 0.0


class WaterTracker:
    """Per-tick speed and heading references for surface vessels from the shared M24 command mapping (``awr.phys.tracker.Tracker``),
    with the vessel's speed limit, a stop inside ``ARRIVE_M`` of a goto target / the final waypoint / the orbit end point, and
    ``hover`` as station keeping at the position the command was given."""

    def __init__(self, params: Mapping[str, UsvParams], tick_s: float = 0.1) -> None:
        self.params = dict(params)
        self.T: dict[str, Tracker] = {}
        self.hold: dict[str, np.ndarray | None] = {}
        self._vel: dict[str, np.ndarray] = {}
        for u, p in self.params.items():
            lim = TrackerLimits(v_xy_mps=p.v_max, v_z_up_mps=0.0, v_z_dn_mps=0.0, yaw_rate_rps=p.r_max, takeoff_vz_mps=0.0, land_vz_mps=0.0,
                                arrive_m=ARRIVE_M, acc_mps2=p.a_max, orbit_entry_m=2.0, path_switch_m=2.0)
            self.T[u] = Tracker(lim, tick_s=tick_s)
            self.hold[u] = None
        self.stats = {"plans": 0}

    def set(self, cmd, pos: np.ndarray) -> None:
        self.T[cmd.uav].set(cmd, pos)
        self.hold[cmd.uav] = None
        if cmd.kind == "hover":
            # hold where the hull can stop, not where it was told: a boat under way brakes over |v|²/(2 a_max), and
            # anchoring behind it would mean a U-turn back
            v = self._vel.get(cmd.uav, np.zeros(2))
            sp = float(np.hypot(v[0], v[1]))
            self.hold[cmd.uav] = np.array(pos[:2], np.float64) + (v * (sp / (2.0 * self.params[cmd.uav].a_max)) if sp > 0.0 else 0.0)

    def kind(self, uav: str) -> str:
        return self.T[uav].kind(uav)

    def setpoint(self, uav: str, pos: np.ndarray, vel: np.ndarray, q: np.ndarray, current: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float, bool]:
        """tick start state → (speed reference through the water m/s, heading reference rad or NaN, enabled). The ground
        velocity set-point of the command mapping minus the ``current`` is what the hull has to make through the water: a
        vessel told to hold a ground velocity (or a position) in a stream heads up into it."""
        T = self.T[uav]
        self._vel[uav] = np.asarray(vel, np.float64)[:2]
        sp = T.setpoint(uav, pos, vel, q)
        vx, vy = float(sp.v_enu[0]), float(sp.v_enu[1])
        kind = T.kind(uav)
        ref = T._ref.get(uav)
        hold = self.hold.get(uav)
        if kind == "hover" and hold is not None and sp.enabled:                     # station keeping
            d = hold - pos[:2]
            dist = float(np.hypot(d[0], d[1]))
            if dist > HOLD_M:
                sc = min(K_HOLD * dist, V_HOLD_MAX, self.params[uav].v_max) / dist
                vx, vy = float(d[0] * sc), float(d[1] * sc)
        speed = math.hypot(vx, vy)
        tgt = None
        if ref is not None and kind == "goto":
            tgt = ref.cmd.target_enu_m
        elif ref is not None and kind == "follow_path":
            path = np.asarray(ref.cmd.path_enu_m, np.float64).reshape(-1, 3)
            if ref.phase >= len(path) - 1:
                tgt = path[-1]
        elif ref is not None and kind == "orbit" and ref.phase == 2:
            o = ref.cmd.orbit
            tgt = (o.center_enu_m[0] + o.radius_m * math.cos(ref.th), o.center_enu_m[1] + o.radius_m * math.sin(ref.th), o.center_enu_m[2])
        if tgt is not None:
            dx, dy = float(tgt[0]) - pos[0], float(tgt[1]) - pos[1]
            dist = math.hypot(dx, dy)
            psi = float(RB.yaw_of(np.asarray(q, np.float64).reshape(1, 4))[0])
            behind = dx * math.cos(psi) + dy * math.sin(psi) < 0.0
            # inside the arrival radius, or overshot (the point is astern within a hull length): a boat cannot turn on the spot
            if dist < ARRIVE_M or (behind and dist < max(self.params[uav].length_m, 2.0 * ARRIVE_M)):
                speed = 0.0
        if speed < V_SP_MIN:
            vx = vy = 0.0
        rx, ry = vx - float(current[0]), vy - float(current[1])                      # through the water
        speed = math.hypot(rx, ry)
        if speed < V_SP_MIN:
            speed = 0.0
        heading = math.atan2(ry, rx) if speed > 0.0 else math.nan
        self.stats["plans"] += 1
        return speed, heading, bool(sp.enabled)
