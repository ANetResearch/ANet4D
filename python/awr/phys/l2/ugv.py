"""Ground vehicle domain model of the unified L2 physics engine (``awr.phys.l2.ugv``): wheeled vehicles on the shared rigid body.

gz-sim has no vehicle model worth lifting, so this domain integrates the vehicle-dynamics literature instead: the
single-track (bicycle) model with linear tyres and longitudinal load transfer as published by CommonRoad
(``vehicle_dynamics_st``) and used by f1tenth_gym, plus what a sandbox needs and the ST model leaves out (friction
saturation, rolling resistance and drag, slope, terrain following, world collision, a path follower):

- **actuation**: the steering angle δ follows its command through the steering-rate limit within ``±steer_max``; the
  drive / brake acceleration comes from a speed reference ramped at 90 % of the available acceleration (feed-forward) plus
  the PI loop ``k_speed·e + ∫(k_speed/2)·e`` on the tracking error (the integrator holds the speed on a slope, frozen against
  the limit it pushes on), within ``[−a_brake, a_max]``, with the power limit ``a_max·v_switch/|v|`` above ``v_switch`` and
  no acceleration past the speed limits (CommonRoad ``accl_constraints``), and the traction limit ``μ·g``;
- **tyres**: lateral forces linear in the slip, written in body velocities so they hold in reverse,
  ``F_yf = C_f·(v_x·δ − (v_y + l_f·r)) / |v_x|``, ``F_yr = −C_r·(v_y − l_r·r) / |v_x|`` with ``C = μ·c_s·F_z`` and the axle
  loads ``F_zf = m(g·l_r − a·h)/L``, ``F_zr = m(g·l_f + a·h)/L`` (load transfer on the commanded acceleration, as in
  CommonRoad), saturated at ``μ·F_z``. The lateral / yaw pair is solved implicitly each step (backward Euler on the linear
  2×2 system), so the stiff low-speed limit degenerates into the kinematic single-track (``r = v·δ/L``) instead of blowing
  up — f1tenth_gym switches models below 0.5 m/s for the same reason;
- **resistance**: rolling ``c_rr·g``, aerodynamic ``½ρ·CdA·|v_r|·v_r/m`` on the wind-relative velocity, gravity along the slope;
- **rigid body**: the body forces and the yaw moment ``l_f·F_yf − l_r·F_yr`` go through the shared ``RigidState.integrate``
  (world frame, so the kinematic coupling ``−r·v_x`` of the slip-angle equation comes from the frame change); the vertical
  degree of freedom and roll / pitch are then constrained to the terrain (z = ground + cg height, body z along the terrain
  normal under the wheelbase and track), so the PoseHistory of a car on a hill is a car on a hill;
- **references**: the tier's command set is the same (velocity, goto, follow_path, orbit, hover, kill); ``GroundTracker``
  turns the shared M24 tracker's velocity set-points into speed and heading, and the engine's heading loop turns heading
  error into a yaw-rate demand and a steering angle ``δ = atan(L·r_sp/v_x)``. takeoff and land are rejected (630); kill is an
  emergency stop (drive off, full brake).

Parameters come from the ground vehicle package (``awr.phys.l2.ground``); nothing is tuned per vehicle here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import numpy as np

from awr.phys.l2 import rigid as RB
from awr.phys.l2.ground import UgvParams
from awr.phys.tracker import Tracker, TrackerLimits

__all__ = ["ARRIVE_M", "K_HEADING", "V_EPS", "V_ROLL", "V_SP_MIN", "V_STOP", "GroundTracker", "UgvFleet", "terrain"]

G = RB.G
V_EPS = 0.05                     # |v_x| floor in the slip terms (the implicit solve gives the kinematic single-track below it)
V_STOP = 0.05                    # a vehicle commanded to stop and slower than this is at rest (static friction holds it)
V_ROLL = 0.1                     # rolling resistance ramps linearly through zero below this speed (no sign chatter)
K_HEADING = 3.0                  # yaw-rate demand per heading error, 1/s
ARRIVE_M = 0.5                   # goto / final waypoint: stop inside this radius
V_SP_MIN = 0.1                   # a speed reference below this is a stop
TWO_PI = 2.0 * math.pi


def terrain(ground: Callable[[np.ndarray], float], xy: np.ndarray, psi: np.ndarray, half_l: np.ndarray, half_w: np.ndarray,
            max_step: float = math.inf) -> tuple[np.ndarray, np.ndarray]:
    """Height under the cg and the terrain gradient (dz/dx, dz/dy) from the heights under the four wheel positions. A probe
    more than ``max_step`` above or below the cg's ground is a wall or a drop, not terrain the wheels ride: it does not enter
    the gradient (the tier's DSM rule crashes the vehicle when the cg reaches it)."""
    n = xy.shape[0]
    z = np.empty(n)
    grad = np.empty((n, 2))
    for i in range(n):
        c, s = math.cos(psi[i]), math.sin(psi[i])
        x, y = float(xy[i, 0]), float(xy[i, 1])
        z0 = z[i] = ground(xy[i])
        fx, fy = half_l[i] * c, half_l[i] * s                     # along the heading
        rx, ry = -half_w[i] * s, half_w[i] * c                    # across (left)
        zf, zb = ground(np.array([x + fx, y + fy])), ground(np.array([x - fx, y - fy]))
        zl, zr = ground(np.array([x + rx, y + ry])), ground(np.array([x - rx, y - ry]))
        gs = (zf - zb) / (2.0 * half_l[i]) if max(abs(zf - z0), abs(zb - z0)) <= max_step else 0.0    # slope along the heading
        gl = (zl - zr) / (2.0 * half_w[i]) if max(abs(zl - z0), abs(zr - z0)) <= max_step else 0.0    # slope to the left
        grad[i, 0] = gs * c - gl * s
        grad[i, 1] = gs * s + gl * c
    return z, grad


class UgvFleet:
    """Vectorised single-track dynamics and the speed / heading / steering loops for ``n`` ground vehicles."""

    def __init__(self, params: list[UgvParams]) -> None:
        self.params = list(params)
        n = self.n = len(params)
        f = lambda a: np.array([float(getattr(p, a)) for p in params])  # noqa: E731
        self.m, self.I = f("mass_kg"), f("izz")
        self.lf, self.lr, self.L = f("lf"), f("lr"), f("wheelbase")
        self.track, self.h, self.wheel_r = f("track_m"), f("cg_h"), f("wheel_r")
        self.collision_r = f("collision_radius_m")
        self.mu, self.csf, self.csr = f("mu"), f("c_sf"), f("c_sr")
        self.s_max, self.sv_max = f("steer_max"), f("steer_rate_max")
        self.v_max, self.v_min = f("v_max"), f("v_min")
        self.a_max, self.a_brake, self.v_switch, self.k_speed = f("a_max"), f("a_brake"), f("v_switch"), f("k_speed")
        self.c_rr, self.cda = f("c_rr"), f("cda")
        self.delta = np.zeros(n)                                  # steering angle, rad (+ left)
        self.v_int = np.zeros(n)                                  # speed loop integrator, m/s² (holds speed on a slope)
        self.v_ref = np.zeros(n)                                  # ramped speed reference, m/s
        self.a_cmd = np.zeros(n)                                  # drive / brake command of the last step, m/s²
        self.a_x = np.zeros(n)                                    # realised longitudinal acceleration
        self.F_y = np.zeros((n, 2))                               # axle lateral forces (front, rear) of the last step
        self.F_z = np.zeros((n, 2))
        self.at_rest = np.ones(n, bool)

    # ------------------------------------------------------------------------------------------------------ control
    def control(self, v_x: np.ndarray, psi: np.ndarray, delta: np.ndarray, speed_sp: np.ndarray, heading_sp: np.ndarray,
                enabled: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Speed and heading references → (drive / brake acceleration, steering rate). Shared with the ST oracle, so the two
        differ in dynamics and integration only. ``heading_sp`` NaN holds the wheels straight."""
        v_sp = np.where(enabled, np.clip(speed_sp, self.v_min, self.v_max), 0.0)
        # speed reference ramp at 90 % of the available acceleration (the power limit on the reference) and braking, with
        # the ramp rate as feed-forward: the P / I loop only sees the tracking error, so it neither lags the ramp nor winds up
        up = 0.9 * np.minimum(self.a_max * np.minimum(1.0, self.v_switch / np.maximum(np.abs(self.v_ref), 1e-9)), self.mu * G)
        dn = 0.9 * np.minimum(self.a_brake, self.mu * G)
        dv = np.clip(v_sp - self.v_ref, -dn * dt, up * dt)
        self.v_ref = np.where(enabled, self.v_ref + dv, 0.0)
        a_ff = np.where(enabled, dv / dt, 0.0)
        e = self.v_ref - v_x
        a = a_ff + self.k_speed * e + self.v_int
        a = np.where(enabled, a, -self.a_brake * np.sign(v_x))                       # kill: drive off, full brake
        hi = np.minimum(self.a_max * np.minimum(1.0, self.v_switch / np.maximum(np.abs(v_x), 1e-9)), self.mu * G)
        lo = -np.minimum(self.a_brake, self.mu * G)
        a = np.clip(a, lo, hi)
        a = np.where((v_x >= self.v_max) & (a > 0.0), 0.0, a)
        a = np.where((v_x <= self.v_min) & (a < 0.0), 0.0, a)
        # integrator (k_i = k_speed / 2) with anti-windup: frozen against the limit it pushes on, cleared when disabled or stopped
        sat = ((a >= hi - 1e-9) & (e > 0.0)) | ((a <= lo + 1e-9) & (e < 0.0))
        live = enabled & ((np.abs(v_sp) >= V_SP_MIN) | (np.abs(v_x) > V_STOP))
        self.v_int = np.where(live, np.where(sat, self.v_int, np.clip(self.v_int + 0.5 * self.k_speed * e * dt, -self.a_max, self.a_max)), 0.0)
        self.v_ref = np.where(live, self.v_ref, 0.0)
        err = np.where(np.isnan(heading_sp), 0.0, (heading_sp - psi + math.pi) % TWO_PI - math.pi)
        vt = np.maximum(np.abs(v_x), V_EPS)
        r_sp = np.clip(K_HEADING * err, -self.mu * G / vt, self.mu * G / vt)
        vs = np.where(v_x < 0.0, -vt, vt)
        d_sp = np.clip(np.arctan(self.L * r_sp / vs), -self.s_max, self.s_max)
        sv = np.clip((d_sp - delta) / dt, -self.sv_max, self.sv_max)
        sv = np.where(((delta <= -self.s_max) & (sv < 0.0)) | ((delta >= self.s_max) & (sv > 0.0)), 0.0, sv)
        return a, sv

    # ------------------------------------------------------------------------------------------------------ dynamics
    def step(self, S: RB.RigidState, speed_sp: np.ndarray, heading_sp: np.ndarray, enabled: np.ndarray, wind: np.ndarray, rho: np.ndarray,
             grade: np.ndarray, dt: float) -> None:
        """One physics step of the horizontal dynamics on the shared body (the caller constrains z and the attitude to
        the terrain afterwards). ``grade`` is the terrain gradient (n, 2) under each vehicle."""
        psi = RB.yaw_of(S.q)
        c, s = np.cos(psi), np.sin(psi)
        v_x = S.v[:, 0] * c + S.v[:, 1] * s
        v_y = -S.v[:, 0] * s + S.v[:, 1] * c
        r = S.w[:, 2]
        a_cmd, sv = self.control(v_x, psi, self.delta, speed_sp, heading_sp, enabled, dt)
        self.delta = np.clip(self.delta + sv * dt, -self.s_max, self.s_max)
        self.a_cmd = a_cmd
        # axle loads with longitudinal transfer (CommonRoad: on the acceleration input), cornering stiffnesses
        Fzf = np.maximum(self.m * (G * self.lr - a_cmd * self.h) / self.L, 0.0)
        Fzr = np.maximum(self.m * (G * self.lf + a_cmd * self.h) / self.L, 0.0)
        Cf, Cr = self.mu * self.csf * Fzf, self.mu * self.csr * Fzr
        vt = np.maximum(np.abs(v_x), V_EPS)
        # implicit (backward Euler) lateral / yaw update on the linear tyre law
        a11 = 1.0 + dt * (Cf + Cr) / (self.m * vt)
        a12 = dt * ((Cf * self.lf - Cr * self.lr) / (self.m * vt) + v_x)
        a21 = dt * (Cf * self.lf - Cr * self.lr) / (self.I * vt)
        a22 = 1.0 + dt * (Cf * self.lf ** 2 + Cr * self.lr ** 2) / (self.I * vt)
        b1 = v_y + dt * Cf * self.delta * v_x / (self.m * vt)
        b2 = r + dt * self.lf * Cf * self.delta * v_x / (self.I * vt)
        det = a11 * a22 - a12 * a21
        vy1 = (b1 * a22 - a12 * b2) / det
        r1 = (a11 * b2 - a21 * b1) / det
        Fyf = np.clip(Cf * (v_x * self.delta - vy1 - self.lf * r1) / vt, -self.mu * Fzf, self.mu * Fzf)
        Fyr = np.clip(-Cr * (vy1 - self.lr * r1) / vt, -self.mu * Fzr, self.mu * Fzr)
        # longitudinal: drive / brake, rolling resistance, drag on the wind-relative velocity, gravity along the slope
        vr = S.v[:, :2] - wind[:, :2]
        vr_x = vr[:, 0] * c + vr[:, 1] * s
        drag = 0.5 * rho * self.cda * np.hypot(vr[:, 0], vr[:, 1]) * vr_x / self.m
        roll = self.c_rr * G * np.clip(v_x / V_ROLL, -1.0, 1.0)
        slope = G * (grade[:, 0] * c + grade[:, 1] * s)
        a_x = a_cmd - drag - roll - slope
        # static friction: a vehicle told to stop does stop (and the brake holds it on the slope)
        want_stop = (np.abs(speed_sp) < V_SP_MIN) | ~enabled
        rest = want_stop & (np.abs(v_x) <= V_STOP) & (np.abs(v_x + a_x * dt) <= V_STOP + 1e-12)
        a_x = np.where(rest, -v_x / dt, a_x)
        Fyf = np.where(rest, 0.0, Fyf)
        Fyr = np.where(rest, 0.0, Fyr)
        self.at_rest = rest
        self.a_x = a_x
        self.F_y[:, 0], self.F_y[:, 1] = Fyf, Fyr
        self.F_z[:, 0], self.F_z[:, 1] = Fzf, Fzr
        Fb_x, Fb_y = self.m * a_x, Fyf + Fyr
        Fw = np.stack([Fb_x * c - Fb_y * s, Fb_x * s + Fb_y * c, np.zeros(self.n)], axis=1)
        tau = np.stack([np.zeros(self.n), np.zeros(self.n), self.lf * Fyf - self.lr * Fyr], axis=1)
        S.integrate(Fw, tau, dt)
        if rest.any():                                                               # exactly at rest: no residual velocity or yaw rate
            S.v[rest] = 0.0
            S.w[rest] = 0.0

    def stop(self, idx: np.ndarray) -> None:
        self.delta[idx] = 0.0
        self.v_int[idx] = 0.0
        self.v_ref[idx] = 0.0
        self.a_cmd[idx] = 0.0
        self.a_x[idx] = 0.0


class GroundTracker:
    """Per-tick speed and heading references for ground vehicles from the shared M24 command mapping
    (``awr.phys.tracker.Tracker``: goto / follow_path / orbit carrots and limits, the same reference logic the kinematic
    tier uses), with the vehicle's own speed limit and a stop inside ``ARRIVE_M`` of a goto target or the final waypoint."""

    def __init__(self, params: Mapping[str, UgvParams], tick_s: float = 0.1) -> None:
        self.params = dict(params)
        self.T: dict[str, Tracker] = {}
        for u, p in self.params.items():
            lim = TrackerLimits(v_xy_mps=p.v_max, v_z_up_mps=0.0, v_z_dn_mps=0.0, yaw_rate_rps=p.v_max * math.tan(p.steer_max) / p.wheelbase,
                                takeoff_vz_mps=0.0, land_vz_mps=0.0, arrive_m=ARRIVE_M, acc_mps2=min(p.a_max, 0.5 * p.mu * G),
                                orbit_entry_m=1.0, path_switch_m=1.0)
            self.T[u] = Tracker(lim, tick_s=tick_s)
        self.stats = {"plans": 0}

    def set(self, cmd, pos: np.ndarray) -> None:
        self.T[cmd.uav].set(cmd, pos)

    def kind(self, uav: str) -> str:
        return self.T[uav].kind(uav)

    def setpoint(self, uav: str, pos: np.ndarray, vel: np.ndarray, q: np.ndarray) -> tuple[float, float, bool]:
        """tick start state → (speed reference m/s, heading reference rad or NaN, enabled)."""
        T = self.T[uav]
        sp = T.setpoint(uav, pos, vel, q)
        vx, vy = sp.v_enu[0], sp.v_enu[1]
        speed = math.hypot(vx, vy)
        kind = T.kind(uav)
        ref = T._ref.get(uav)
        tgt = None
        if ref is not None and kind == "goto":
            tgt = ref.cmd.target_enu_m
        elif ref is not None and kind == "follow_path":
            path = np.asarray(ref.cmd.path_enu_m, np.float64).reshape(-1, 3)
            if ref.phase >= len(path) - 1:                                      # on the last leg: stop at the final waypoint
                tgt = path[-1]
        elif ref is not None and kind == "orbit" and ref.phase == 2:            # turns done: stop at the end point of the circle
            o = ref.cmd.orbit
            tgt = (o.center_enu_m[0] + o.radius_m * math.cos(ref.th), o.center_enu_m[1] + o.radius_m * math.sin(ref.th), o.center_enu_m[2])
        if tgt is not None and math.hypot(float(tgt[0]) - pos[0], float(tgt[1]) - pos[1]) < ARRIVE_M:
            speed = 0.0
        if speed < V_SP_MIN:
            speed = 0.0
        heading = math.atan2(vy, vx) if speed > 0.0 else math.nan
        self.stats["plans"] += 1
        return speed, heading, bool(sp.enabled)
