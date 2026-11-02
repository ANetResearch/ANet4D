"""Fused human pose/normal packing for NumPy scalar promotion, preserving authored triangles.

The coefficients and intermediate operations deliberately retain the reference
NumPy intermediate rounding. No fastmath, parallel reduction, temporal quantization,
normal approximation or mesh simplification is used.
"""

from functools import lru_cache

import numpy as np
from numba import njit


@njit(cache=True, nogil=True, fastmath=False)
def vertices(rest, first, second, color, uv, scale, u, blend, rest_float64=False):
    out = np.empty((len(rest), 11), np.float32)
    p = np.empty((3, 3), np.float64)
    # q/u are NumPy float64 scalars in Mesh.vertices: NumPy 2 promotes
    # the interpolated pose to float64, but Python blend leaves the rest
    # product in float32. Keep both stages, including that one rounding.
    a, b, c, d = 1.0 - u, u, 1.0 - blend, blend
    for base in range(0, len(rest), 3):
        for row in range(3):
            for axis in range(3):
                i = base + row
                posed = first[i, axis] * a + second[i, axis] * b
                resting = rest[i, axis] * c if rest_float64 else float(np.float32(rest[i, axis] * np.float32(c)))
                p[row, axis] = resting + posed * d
        x1, y1, z1 = p[1, 0] - p[0, 0], p[1, 1] - p[0, 1], p[1, 2] - p[0, 2]
        x2, y2, z2 = p[2, 0] - p[0, 0], p[2, 1] - p[0, 1], p[2, 2] - p[0, 2]
        nx = y1 * z2 - z1 * y2
        ny = z1 * x2 - x1 * z2
        nz = x1 * y2 - y1 * x2
        norm = max(np.sqrt((nx * nx + ny * ny) + nz * nz), 1e-12)
        nx, ny, nz = nx / norm, ny / norm, nz / norm
        for row in range(3):
            i = base + row
            for axis in range(3):
                out[i, axis] = p[row, axis] * scale[axis]
                out[i, 6 + axis] = color[i, axis]
            out[i, 3], out[i, 4], out[i, 5] = nx, ny, nz
            out[i, 9], out[i, 10] = uv[i, 0], uv[i, 1]
    return out


@lru_cache(maxsize=1)
def warmup():
    """Compile during asset initialization, before real-time camera delivery."""
    p = np.zeros((3, 3), np.float32)
    vertices(p, p, p, p, np.zeros((3, 2), np.float32), np.ones(3), 0.0, 1.0, False)
