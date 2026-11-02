"""Offline recovery of missed mesh samples from nearby measured ground.

This does not interpolate ground: the callback must return actual mesh hits.
New samples advance one four-connected cell per round from an immutable snapshot,
and must agree with EVERY measured neighbour within the declared grade bound.
Callers must re-run slope, clearance, body and route admission on the result.
"""

from __future__ import annotations

import numpy as np


def recover(z, allowed, cell_m, ground_at, *, origin=(0.0, 0.0), max_grade=0.25, max_rounds=64, batch=80000):
    """Return a new height array and diagnostics; never alter existing samples.

    ``ground_at(xy, hint)`` has the measured support compiler's contract. Hints
    only select intersections; a returned hint with found=False is rejected.
    No tolerance is added to the grade bound and no unknown height is inferred.
    """
    a = np.asarray(z)
    mask = np.asarray(allowed)
    if a.ndim != 2 or mask.shape != a.shape or mask.dtype != np.bool_:
        raise ValueError("Expected a 2D height grid and same-shape boolean permission")
    if not np.issubdtype(a.dtype, np.floating) or np.isinf(a).any():
        raise ValueError("Heights must be floating, with NaN for unknown")
    if (
        not np.isfinite([cell_m, max_grade, *origin]).all()
        or cell_m <= 0
        or max_grade < 0
        or len(origin) != 2
        or not isinstance(max_rounds, int)
        or max_rounds < 0
        or not isinstance(batch, int)
        or batch < 1
    ):
        raise ValueError("Invalid recovery geometry or budget")
    out = a.copy()
    step = cell_m * max_grade
    rows = []
    for iteration in range(max_rounds):
        known = mask & np.isfinite(out)
        # Fixed four-neighbour order; the entire round uses the same snapshot.
        lo = np.full(a.shape, np.inf)
        hi = np.full(a.shape, -np.inf)
        count = np.zeros(a.shape, np.uint8)
        for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            dst = (
                slice(max(0, -dy), min(a.shape[0], a.shape[0] - dy)),
                slice(max(0, -dx), min(a.shape[1], a.shape[1] - dx)),
            )
            src = (
                slice(max(0, dy), min(a.shape[0], a.shape[0] + dy)),
                slice(max(0, dx), min(a.shape[1], a.shape[1] + dx)),
            )
            k = known[src]
            lo[dst] = np.minimum(lo[dst], np.where(k, out[src], np.inf))
            hi[dst] = np.maximum(hi[dst], np.where(k, out[src], -np.inf))
            count[dst] += k
        candidate = mask & ~np.isfinite(out) & (count > 0) & (hi - lo <= 2 * step)
        yy, xx = np.nonzero(candidate)
        if not len(xx):
            break
        accepted = 0
        for start in range(0, len(xx), batch):
            y, x = yy[start : start + batch], xx[start : start + batch]
            hint = (lo[y, x] + hi[y, x]) * 0.5
            xy = np.column_stack([origin[0] + (x + 0.5) * cell_m, origin[1] + (y + 0.5) * cell_m])
            measured, found = ground_at(xy, hint)
            measured, found = np.asarray(measured), np.asarray(found)
            if measured.shape != hint.shape or found.shape != hint.shape or found.dtype != np.bool_:
                raise ValueError("Malformed measured ground callback")
            # Check stored precision too, so conversion cannot cross the bound.
            stored = measured.astype(out.dtype)
            ok = (
                found
                & np.isfinite(measured)
                & np.isfinite(stored)
                & (measured >= hi[y, x] - step)
                & (measured <= lo[y, x] + step)
                & (stored >= hi[y, x] - step)
                & (stored <= lo[y, x] + step)
            )
            out[y[ok], x[ok]] = stored[ok]
            accepted += int(ok.sum())
        rows.append(dict(round=iteration + 1, candidates=len(xx), recovered=accepted))
        if accepted == 0:
            break
    return out, dict(
        rounds=rows,
        recovered_cells=int((~np.isfinite(a) & np.isfinite(out)).sum()),
        max_grade=max_grade,
        max_rounds=max_rounds,
        admission=False,
        method="Neighbour-guided actual mesh intersections; no interpolated heights",
    )
