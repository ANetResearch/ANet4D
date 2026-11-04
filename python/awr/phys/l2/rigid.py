"""Rigid-body core of the unified L2 physics engine (``awr.phys.l2.rigid``), shared by every device domain.

World frame ENU (m, m/s), body frame FLU, attitude as a unit quaternion ``q`` (xyzw, WORLD←FLU), angular rate in the body
frame. All helpers are vectorised over the first axis (``n`` bodies) and are pure functions of their inputs; the integrator
is the same semi-implicit Euler step for every domain (forces and torques come from the domain model, the integration and
the quaternion update do not):

    v ← v + (F / m)·dt            p ← p + v·dt
    ω ← ω + J⁻¹(τ − ω × Jω)·dt    q ← q ⊗ exp(½·ω·dt)   (normalised)

Nothing here reads the clock or draws random numbers; two runs with the same inputs produce the same bytes.
"""

from __future__ import annotations

import numpy as np

__all__ = ["G", "RigidState", "cross", "q_from_R", "q_from_yaw", "q_from_z_yaw", "q_integrate", "q_inv", "q_mul", "q_rot", "q_rot_inv",
           "tilt_of", "yaw_of"]

G = 9.80665
E3 = np.array([0.0, 0.0, 1.0])


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cross product of (n, 3) arrays (``np.cross`` spends most of its time on axis bookkeeping for small inputs)."""
    return np.stack([a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
                     a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                     a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]], axis=1)


# ------------------------------------------------------------------------------------------------------------ quaternions
def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b for xyzw quaternions, (n, 4)."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=1)


def q_inv(q: np.ndarray) -> np.ndarray:
    out = -q
    out[:, 3] = q[:, 3]
    return out


def q_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """R(q)·v: body → world, (n, 3)."""
    u, w = q[:, 0:3], q[:, 3:4]
    t = 2.0 * cross(u, v)
    return v + w * t + cross(u, t)


def q_rot_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """R(q)ᵀ·v: world → body."""
    u, w = -q[:, 0:3], q[:, 3:4]
    t = 2.0 * cross(u, v)
    return v + w * t + cross(u, t)


def q_from_yaw(yaw) -> np.ndarray:
    h = np.asarray(yaw, np.float64) / 2.0
    z = np.zeros_like(h)
    return np.stack([z, z, np.sin(h), np.cos(h)], axis=-1)


def yaw_of(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def tilt_of(q: np.ndarray) -> np.ndarray:
    """Angle between the body z axis and the vertical, rad."""
    x, y = q[:, 0], q[:, 1]
    return np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))


def q_from_R(R: np.ndarray) -> np.ndarray:
    """Rotation matrices (n, 3, 3) → unit quaternions xyzw (Shepperd's method, branch-free over the batch)."""
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    w = np.sqrt(np.maximum(0.0, 1.0 + tr)) / 2.0
    x = np.sqrt(np.maximum(0.0, 1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2])) / 2.0
    y = np.sqrt(np.maximum(0.0, 1.0 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2])) / 2.0
    z = np.sqrt(np.maximum(0.0, 1.0 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2])) / 2.0
    x = np.copysign(x, R[:, 2, 1] - R[:, 1, 2])
    y = np.copysign(y, R[:, 0, 2] - R[:, 2, 0])
    z = np.copysign(z, R[:, 1, 0] - R[:, 0, 1])
    q = np.stack([x, y, z, w], axis=1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def q_from_z_yaw(b3: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Attitude whose body z axis is ``b3`` (unit, (n, 3)) and whose heading is ``yaw`` (the projection of body x onto the
    horizontal plane points along yaw): R = [b1 b2 b3], b2 = b3 × c / ‖·‖ with c = (cos ψ, sin ψ, 0), b1 = b2 × b3."""
    c = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=1)
    b2 = cross(b3, c)
    n = np.linalg.norm(b2, axis=1, keepdims=True)
    b2 = b2 / np.maximum(n, 1e-12)
    b1 = cross(b2, b3)
    return q_from_R(np.stack([b1, b2, b3], axis=2))


def q_integrate(q: np.ndarray, omega_body: np.ndarray, dt: float) -> np.ndarray:
    """q ⊗ exp(½·ω·dt) for body-frame rates; the result is renormalised."""
    rv = omega_body * dt
    an = np.linalg.norm(rv, axis=1)
    half = an / 2.0
    fs = np.where(an > 1e-12, np.sin(half) / np.maximum(an, 1e-300), 0.5)
    dq = np.concatenate([rv * fs[:, None], np.cos(half)[:, None]], axis=1)
    out = q_mul(q, dq)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


# ------------------------------------------------------------------------------------------------------------ state
class RigidState:
    """Stacked rigid-body state of ``n`` bodies: ``p`` (ENU), ``v`` (ENU), ``q`` (xyzw), ``w`` (body rates), ``m``, ``J``
    (diagonal body inertia). ``integrate`` applies one semi-implicit Euler step and returns the world acceleration."""

    def __init__(self, n: int) -> None:
        self.n = int(n)
        self.p = np.zeros((n, 3))
        self.v = np.zeros((n, 3))
        self.q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
        self.w = np.zeros((n, 3))
        self.m = np.ones(n)
        self.J = np.ones((n, 3))

    def integrate(self, F_world: np.ndarray, tau_body: np.ndarray, dt: float) -> np.ndarray:
        a = F_world / self.m[:, None]
        self.v = self.v + a * dt
        self.p = self.p + self.v * dt
        Jw = self.J * self.w
        wdot = (tau_body - cross(self.w, Jw)) / self.J
        self.w = self.w + wdot * dt
        self.q = q_integrate(self.q, self.w, dt)
        return a
