"""Gimbal servo tiers: one model for every gimbal in Anet4D.

Both gimbal implementations (``awr.sim.sensors.gimbal.GimbalBank`` inside sim-core and the agent-env camera gimbal
``awr.agentsdk.env.camera.Gimbal``) advance their axes with the functions here, so a tier means the same motion in both.

- ``l1`` (rate-limited, the tier every gimbal had before): the axis moves towards the clamped target by at most
  ``rate_max · dt`` per step and stops exactly on it. Instant acceleration, no overshoot, no settling: fine for
  pointing logic, wrong for what a camera sees during a slew.
- ``l2`` (second-order servo): the axis has an angular velocity ``omega``. The commanded velocity is ``err / tau``
  clamped to ``±rate_max`` and to the braking speed (``√(h² + 2·acc_max·|err|) − h``, ``h = acc_max·dt/2``: the
  discrete deceleration ramp that still stops on the target);
  ``omega`` moves towards it by at most ``acc_max · dt`` per step; the angle advances by ``omega · dt``, landing on
  the target instead of crossing it. Finite acceleration, a braking profile and a first-order velocity loop give the
  ramp-up, the constant-rate slew, the deceleration and the exponential settling (time constant ``tau``) of a real
  pan-tilt unit, without overshoot. The ``l1`` motion is the limit ``tau → 0``, ``acc_max → ∞``.

The functions take floats or numpy arrays with the same expressions in both forms (the scalar and vector paths of
the gimbal bank must agree bit for bit) and are deterministic.
"""

from __future__ import annotations

import math

import numpy as np

TIERS = ("l1", "l2")
L2_DEFAULTS = {"acc_max_deg_s2": 400.0, "tau_s": 0.08}


def _is_array(*xs) -> bool:
    return any(isinstance(x, np.ndarray) for x in xs)


def step_l1(angle, target, rate_max, dt):
    """Rate-limited approach: ``angle + clip(target − angle, ±rate_max·dt)``. Works for floats and arrays."""
    step = rate_max * dt
    if _is_array(angle, target, rate_max):
        return angle + np.minimum(np.maximum(target - angle, -step), step)
    return angle + min(max(target - angle, -step), step)


def step_l2(angle, omega, target, rate_max, acc_max, tau, dt):
    """Second-order servo step → ``(angle, omega)``; ``tau`` > 0 (s), ``acc_max`` > 0, units as given (rad or deg).

    Velocity command ``err / tau`` saturated at ``rate_max`` and at the discrete braking speed; the
    velocity follows it through the acceleration limit;
    the position integrates the new velocity (semi-implicit Euler). A step that would cross the target lands on it
    and zeroes the velocity, so a converged axis rests instead of chattering."""
    err = target - angle
    dv = acc_max * dt
    if _is_array(angle, target, omega, rate_max):
        h = 0.5 * dv
        v_lim = np.minimum(rate_max, np.sqrt(h * h + 2.0 * acc_max * np.abs(err)) - h)
        v_cmd = np.minimum(np.maximum(err / tau, -v_lim), v_lim)
        omega = omega + np.minimum(np.maximum(v_cmd - omega, -dv), dv)
        move = omega * dt
        over = np.where(err >= 0, move > err, move < err)
        move = np.where(over, err, move)
        omega = np.where(over, 0.0, omega)
        return angle + move, omega
    h = 0.5 * dv
    v_lim = min(rate_max, math.sqrt(h * h + 2.0 * acc_max * abs(err)) - h)
    v_cmd = min(max(err / tau, -v_lim), v_lim)
    omega = omega + min(max(v_cmd - omega, -dv), dv)
    move = omega * dt
    over = move > err if err >= 0 else move < err
    if over:
        move, omega = err, 0.0
    return angle + move, omega


def settle_time_s(delta, rate_max, acc_max, tau, dt, tol):
    """Time for ``step_l2`` from rest to bring an axis within ``tol`` of a target ``delta`` away (diagnostics)."""
    a, w, t = 0.0, 0.0, 0.0
    for _ in range(int(60.0 / dt)):
        if abs(delta - a) <= tol:
            return t
        a, w = step_l2(a, w, delta, rate_max, acc_max, tau, dt)
        t += dt
    return math.inf


def response(delta, rate_max, dt, n, tier="l1", acc_max=None, tau=None):
    """Step response of one axis from rest: ``n`` samples of the angle (diagnostics and the device page)."""
    a, w = 0.0, 0.0
    out = []
    for _ in range(n):
        if tier == "l1":
            a = step_l1(a, delta, rate_max, dt)
        else:
            a, w = step_l2(a, w, delta, rate_max, acc_max, tau, dt)
        out.append(a)
    return out


__all__ = ["L2_DEFAULTS", "TIERS", "response", "settle_time_s", "step_l1", "step_l2"]
