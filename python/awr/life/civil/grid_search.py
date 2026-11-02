"""The same four-neighbour A* in Python and optional compiled execution.

Only integer grid search is accelerated. Exact pose connectors, body clearance,
turning, smoothing and dynamic admission remain the caller's responsibility.
No fast-math, threads, changed heuristic or wall-time search limit is used.
"""

import heapq
import math
from functools import lru_cache

import numpy as np


def search(free, weight, source, target, field_index, field_cost):
    """Target-to-source cells; heap ties retain the original (cost, y, x) order."""
    sy, sx = source
    ty, tx = target
    frontier = [(0.0, sy, sx)]
    distance = {(sy, sx): 0.0}
    parent = {(sy, sx): (sy, sx)}
    closed = set()
    while frontier:
        _, uy, ux = heapq.heappop(frontier)
        u = (uy, ux)
        if u == (ty, tx):
            break
        if u in closed:
            continue
        closed.add(u)
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            vy, vx = uy + dy, ux + dx
            if not (0 <= vy < free.shape[0] and 0 <= vx < free.shape[1]) or not free[vy, vx]:
                continue
            v = (vy, vx)
            # Numba's float(float32) can retain float32, unlike Python float().
            # Promote BEFORE addition to preserve costs and equal-cost ties.
            cost = 1.0 if not weight.size else (np.float64(weight[uy, ux]) + np.float64(weight[vy, vx])) / 2
            nd = distance[u] + cost
            if nd < distance.get(v, math.inf):
                parent[v] = u
                distance[v] = nd
                heuristic = (
                    np.float64(field_cost[field_index[vy, vx]])
                    if field_cost.size
                    else float(abs(vy - ty) + abs(vx - tx))
                )
                heapq.heappush(frontier, (nd + heuristic, vy, vx))
    cells = [(ty, tx)]
    if (ty, tx) not in distance:
        return cells[:0]
    while cells[-1] != (sy, sx):
        cells.append(parent[cells[-1]])
    return cells


@lru_cache(maxsize=3)
def implementation(backend):
    if backend not in ("auto", "python", "numba"):
        raise ValueError("Unknown grid search backend")
    if backend == "python":
        return search, "python"
    try:
        from numba import njit
    except ImportError:
        if backend == "numba":
            raise RuntimeError("The requested numba grid search backend is unavailable") from None
        return search, "python"
    # Compilation/runtime failures are surfaced. Silently switching backends
    # after an error would hide broken release environments.
    return njit(cache=True, fastmath=False)(search), "numba"


def grid_path(free, weight, source, target, *, region, field=None, backend="auto"):
    """Crop to the existing component, preserving global coordinate tie order.

    Working dictionaries grow with visited cells; no full-world distance arrays
    are allocated per walker. The caller has already checked exact endpoints
    and dynamic component connectivity. Free cells must be a network subset.
    """
    origin = (region[0].start, region[1].start)
    local_source = tuple(int(v - o) for v, o in zip(source, origin, strict=False))
    local_target = tuple(int(v - o) for v, o in zip(target, origin, strict=False))
    index = np.empty((0, 0), np.int32)
    cost = np.empty(0, np.float32)
    if field is not None:
        f_region, index, cost = field
        if f_region != region:
            raise ValueError("A* heuristic must use the same static component")
    weights = np.empty((0, 0), np.float32) if weight is None else np.asarray(weight)[region]
    execute, _ = implementation(backend)
    cells = execute(np.asarray(free, bool)[region], weights, local_source, local_target, index, cost)
    return [(y + origin[0], x + origin[1]) for y, x in cells]


def segment_cells_clear(q0, q1, e0, e1, geometry, weight, max_weight):
    """Grid part of WalkNetwork.clear_segment in grid units: the segment q -> e visits only free cells (and,
    with a non-empty weight array, none costlier than max_weight). Same sample times, offsets and rounding."""
    h, w = geometry.shape
    d0, d1 = e0 - q0, e1 - q1
    nb0 = 0
    nb1 = 0
    if abs(d0) >= 1e-12:
        nb0 = max(0, math.ceil(max(q0, e0) - 0.5) - (math.floor(min(q0, e0) - 0.5) + 1))
    if abs(d1) >= 1e-12:
        nb1 = max(0, math.ceil(max(q1, e1) - 0.5) - (math.floor(min(q1, e1) - 0.5) + 1))
    times = np.empty(2 + nb0 + nb1)
    times[0] = 0.0
    times[1] = 1.0
    m = 2
    for axis in range(2):
        q, e, d, nb = (q0, e0, d0, nb0) if axis == 0 else (q1, e1, d1, nb1)
        start = math.floor(min(q, e) - 0.5) + 1
        for k in range(nb):
            times[m] = ((start + k) + 0.5 - q) / d
            m += 1
    times = np.unique(times)
    n = len(times)
    for j in range(2 * n - 1):
        t = times[j] if j < n else (times[j - n] + times[j - n + 1]) / 2
        px, py = q0 + t * d0, q1 + t * d1
        for sx in (-1e-8, 1e-8):
            for sy in (-1e-8, 1e-8):
                c = math.floor(px + sx + 0.5)
                r = math.floor(py + sy + 0.5)
                if r < 0 or c < 0 or r >= h or c >= w or not geometry[r, c]:
                    return False
                if weight.size and np.float64(weight[r, c]) > max_weight + 1e-6:
                    return False
    return True


def segment_bodies_clear(a0, a1, b0, b1, length, positions, sizes, yaw_cos, yaw_sin, circles, radius, limit):
    """Body part of WalkNetwork.clear_segment (WalkNetwork._bodies_clear_numpy) in metres, compiled: the segment
    a -> b (of the given length) keeps `limit` from every exact body, or leaves the margin of a body it starts in
    (never in contact within `radius`). Same culling, samples and distances; cos/sin of the yaws come from NumPy."""
    dx, dy = b0 - a0, b1 - a1
    den = max(length * length, 1e-24)
    m = max(2, math.ceil(length / 0.1) + 1)
    step = 1.0 / (m - 1)
    for j in range(positions.shape[0]):
        px, py = positions[j, 0], positions[j, 1]
        round_body = math.isfinite(circles[j])
        hx, hy = sizes[j, 0] / 2, sizes[j, 1] / 2
        reach = circles[j] if round_body else math.sqrt(sizes[j, 0] * sizes[j, 0] + sizes[j, 1] * sizes[j, 1]) / 2
        along = min(max(((px - a0) * dx + (py - a1) * dy) / den, 0.0), 1.0)
        ex, ey = px - (a0 + along * dx), py - (a1 + along * dy)
        if not math.sqrt(ex * ex + ey * ey) <= reach + limit + 1e-9:
            continue
        c, s = yaw_cos[j], yaw_sin[j]
        first = 0.0
        for k in range(m):
            t = 1.0 if k == m - 1 else k * step
            qx, qy = (a0 + t * dx) - px, (a1 + t * dy) - py
            if round_body:
                d = math.sqrt(qx * qx + qy * qy) - circles[j]
            else:
                u = max(abs(qx * c + qy * s) - hx, 0.0)
                v = max(abs(-qx * s + qy * c) - hy, 0.0)
                d = math.sqrt(u * u + v * v)
            if k == 0:
                first = d
                if d <= limit and d <= radius:
                    return False
            elif first <= limit:
                if d < first - 1e-6:
                    return False
            elif d <= limit:
                return False
    return True


@lru_cache(maxsize=1)
def compiled_bodies_check():
    """The compiled body check, or None when numba is unavailable (callers keep the NumPy version)."""
    try:
        from numba import njit
    except ImportError:
        return None
    return njit(cache=True, fastmath=False)(segment_bodies_clear)


@lru_cache(maxsize=1)
def compiled_segment_check():
    """The compiled segment check, or None when numba is unavailable (callers keep the NumPy version)."""
    try:
        from numba import njit
    except ImportError:
        return None
    return njit(cache=True, fastmath=False)(segment_cells_clear)
