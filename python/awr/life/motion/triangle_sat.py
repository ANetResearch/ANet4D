"""Compiled equivalent of the source triangle/AABB separating-axis test.

All 13 axes and the original closed-contact tolerances are retained. No
fast-math or approximate geometry; the standalone NumPy auditor is unchanged.
"""

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _overlap(v, edge, half):
    result = np.ones(len(v), np.bool_)
    for i in range(len(v)):
        for k in range(3):
            low = min(v[i, 0, k], v[i, 1, k], v[i, 2, k])
            high = max(v[i, 0, k], v[i, 1, k], v[i, 2, k])
            if low > half[k] + 1e-9 or high < -half[k] - 1e-9:
                result[i] = False
        if not result[i]:
            continue
        for axis in range(10):
            if axis == 0:
                a = edge[i, 0]
                b = edge[i, 1]
                x = a[1] * b[2] - a[2] * b[1]
                y = a[2] * b[0] - a[0] * b[2]
                z = a[0] * b[1] - a[1] * b[0]
            else:
                e = edge[i, (axis - 1) // 3]
                k = (axis - 1) % 3
                if k == 0:
                    x = 0.0
                    y = e[2]
                    z = -e[1]
                elif k == 1:
                    x = -e[2]
                    y = 0.0
                    z = e[0]
                else:
                    x = e[1]
                    y = -e[0]
                    z = 0.0
            p0 = v[i, 0, 0] * x + v[i, 0, 1] * y + v[i, 0, 2] * z
            p1 = v[i, 1, 0] * x + v[i, 1, 1] * y + v[i, 1, 2] * z
            p2 = v[i, 2, 0] * x + v[i, 2, 1] * y + v[i, 2, 2] * z
            r = abs(x) * half[0] + abs(y) * half[1] + abs(z) * half[2]
            eps = 1e-9 * math.sqrt(x * x + y * y + z * z)
            if min(p0, p1, p2) > r + eps or max(p0, p1, p2) < -r - eps:
                result[i] = False
                break
    return result


def triangle_box_intersections(triangles, lo, hi):
    tri = np.asarray(triangles, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    if (
        tri.ndim != 3
        or tri.shape[1:] != (3, 3)
        or lo.shape != (3,)
        or hi.shape != (3,)
        or not np.isfinite(tri).all()
        or not np.isfinite([lo, hi]).all()
        or np.any(lo > hi)
    ):
        raise ValueError("Finite triangles and ordered 3D box required")
    half = (hi - lo) / 2
    v = tri - (lo + hi) / 2
    edge = np.roll(v, -1, axis=1) - v
    return _overlap(v, edge, half)
