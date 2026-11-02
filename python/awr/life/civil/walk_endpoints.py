"""Connect exact walking endpoints to free graph nodes without snapping them.

An obstacle raster describes node centres. A valid continuous endpoint may
share a cell with an invalid centre. Both connectors still use the network's
original static-ground and full-body checks, in their actual travel direction.
"""

import math

import numpy as np

from .walking import PLANNING_MARGIN_M


def standing_anchor_clear(network, surface, free, point, radius):
    """Exact standing/turning clearance, retaining the full planning margin.

    Static grid cells still describe forbidden terrain. Dynamic blocked node
    centres do not describe solid squares; evaluate the actual endpoint against
    every oriented body instead. This alone does not establish a walking path.
    """
    point = np.asarray(point, float)
    if point.shape != (2,) or not np.isfinite(point).all() or not math.isfinite(radius) or radius <= 0:
        raise ValueError("Finite standing point and positive turning radius required")
    cell = tuple(network.cell(point))
    static = getattr(free, "static_free", free)
    if any(i < 0 or i >= n for i, n in zip(cell, free.shape, strict=False)) or not static[cell]:
        return False
    positions = getattr(free, "positions", np.empty((0, 2)))
    if len(positions):
        delta = point - positions
        c = np.cos(free.yaws)
        s = np.sin(free.yaws)
        local = np.abs(np.column_stack([delta[:, 0] * c + delta[:, 1] * s, -delta[:, 0] * s + delta[:, 1] * c]))
        distances = np.linalg.norm(np.maximum(local - free.sizes[:, :2] / 2, 0), axis=1)
        circles = getattr(free, "circle_radii", np.full(len(positions), np.nan))
        round_ids = np.isfinite(circles)
        distances[round_ids] = np.linalg.norm(delta[round_ids], axis=1) - circles[round_ids]
        if np.any(distances <= radius + 0.08 + PLANNING_MARGIN_M):
            return False
    angles = np.linspace(0, 2 * np.pi, 49)
    ring = point + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return bool(surface.supported(np.vstack([point, ring]), "person").all())


def endpoint_cells(network, free, point, *, arriving, prefer_cell=False):
    point = np.asarray(point, float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("Finite exact walking endpoint required")
    candidates = []
    seen = set()

    def consider(cell, xy):
        cell = tuple(map(int, cell))
        if cell in seen:
            return
        seen.add(cell)
        if any(i < 0 or i >= n for i, n in zip(cell, free.shape, strict=False)) or not free[cell]:
            return
        a, b = (xy, point) if arriving else (point, xy)
        if network.clear_segment(a, b, free):
            candidates.append(cell)

    if prefer_cell:
        cell = network.cell(point)
        consider(cell, network.origin + cell[::-1] * network.step)
    distances, indices = network.tree.query(point, k=min(24, len(network.xy)))
    for distance, i in zip(np.atleast_1d(distances), np.atleast_1d(indices), strict=False):
        if distance <= 1.5:
            consider(network.nodes[int(i)], network.xy[int(i)])
    return candidates


def endpoint_pair(network, free, index, components, start, goal):
    """Keep the original target cell when usable; otherwise try safe connectors.

    This is a bounded endpoint search, not a claim of continuous-path optimality.
    Returned cells are graph endpoints; the final route keeps exact start/goal.
    """
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    if start.shape != (2,) or goal.shape != (2,) or not np.isfinite(np.r_[start, goal]).all():
        raise ValueError("Finite exact walking endpoints required")
    # Ordinary facility slots already occupy free centres. Preserve that fast
    # path rather than running two full endpoint searches for every activity.
    target = tuple(network.cell(goal))
    if all(0 <= i < n for i, n in zip(target, free.shape, strict=False)) and free[target]:
        point = network.origin + np.array(target[::-1]) * network.step
        ti = int(index[target])
        if ti >= 0 and network.clear_segment(point, goal, free):
            distances, indices = network.tree.query(start, k=min(24, len(network.xy)))
            for distance, i in zip(np.atleast_1d(distances), np.atleast_1d(indices), strict=False):
                if distance > 1.5:
                    continue
                source = tuple(network.nodes[int(i)])
                si = int(index[source])
                if (
                    si >= 0
                    and components[si] == components[ti]
                    and network.labels[source] == network.labels[target]
                    and network.clear_segment(start, network.xy[int(i)], free)
                ):
                    return source, target
    targets = endpoint_cells(network, free, goal, arriving=True, prefer_cell=True)
    if not targets:
        return None
    sources = endpoint_cells(network, free, start, arriving=False)
    for target in targets:
        ti = int(index[target])
        for source in sources:
            si = int(index[source])
            if (
                si >= 0
                and ti >= 0
                and components[si] == components[ti]
                and network.labels[source] == network.labels[target]
            ):
                return source, target
    return None
