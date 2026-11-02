"""Bounded-curvature low-speed manoeuvres; unsupported geometry is an error."""

import math

import numpy as np

from awr.life.civil.collision import rectangle


def plan(start, goal, yaw_start, yaw_goal, size, surface):
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    distance = np.linalg.norm(goal - start)
    if distance < 0.03:
        if abs((yaw_goal - yaw_start + math.pi) % (2 * math.pi) - math.pi) > 0.03:
            raise ValueError("A car cannot rotate in place")
        return np.array([start, goal]), np.array([yaw_start, yaw_start]), np.array([0.0, distance])
    a = np.array([math.cos(yaw_start), math.sin(yaw_start)])
    b = np.array([math.cos(yaw_goal), math.sin(yaw_goal)])
    wheelbase = float(size[0]) * 0.6
    max_curvature = math.tan(math.radians(35)) / wheelbase
    for reverse in (False, True):
        sign = -1 if reverse else 1
        for tension in (0.7, 1.0, 1.5, 2.0):
            u = np.linspace(0, 1, max(20, int(distance / 0.1) + 1))[:, None]
            t0 = a * distance * tension * sign
            t1 = b * distance * tension * sign
            points = (
                (2 * u**3 - 3 * u**2 + 1) * start
                + (u**3 - 2 * u**2 + u) * t0
                + (-2 * u**3 + 3 * u**2) * goal
                + (u**3 - u**2) * t1
            )
            delta = np.diff(points, axis=0)
            ds = np.linalg.norm(delta, axis=1)
            angles = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]) + (math.pi if reverse else 0))
            angles = np.r_[angles, angles[-1]]
            curvature = np.abs(np.diff(angles)) / np.maximum(ds, 1e-9)
            if curvature.max(initial=0) > max_curvature:
                continue
            try:
                for p, yaw in zip(points, angles, strict=False):
                    surface.sample(rectangle(p, yaw, size), "vehicle")
            except ValueError:
                continue
            return points, angles, np.r_[0, np.cumsum(ds)]
    raise ValueError(
        "No supported manoeuvre satisfying steering radius; multi-point parking must be planned explicitly"
    )
