"""The pose a motion proposal will have after runtime recording and tape encoding.

This is a prediction for admission, never a correction to motion or output data.
Keep float32 storage and floor(x + .5) rounding in sync with civil.assemble._fix.
The assembler remains the independent authority (covered by differential tests).
"""

import math

import numpy as np
from numba import njit


@njit(cache=True)
def fixed_recorded(value, scale):
    return math.floor(float(np.float32(value)) * scale + 0.5)


@njit(cache=True)
def recorded_pose(x, y, yaw, vx, vy):
    """Return decoded x/y/radian yaw/speed, including runtime float32 storage."""
    heading = (yaw + math.pi) % (2 * math.pi) - math.pi
    cdeg = (fixed_recorded(heading * (180.0 / math.pi), 100.0) + 18000) % 36000 - 18000
    ux, uy = fixed_recorded(vx, 100.0), fixed_recorded(vy, 100.0)
    return (
        fixed_recorded(x, 1000.0) * 0.001,
        fixed_recorded(y, 1000.0) * 0.001,
        cdeg * (math.pi / 18000.0),
        math.hypot(ux, uy) * 0.01,
    )
