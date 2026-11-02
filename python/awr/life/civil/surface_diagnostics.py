"""Explain measured-grid rejection without changing terrain or admission.

These categories describe the support representation, not physical capacity:
a false vehicle mask can encode genuine clearance, reconstruction noise, or a
data-processing error. Mesh/semantic evidence is needed to distinguish them.
"""

import numpy as np

from .surface import SurfaceMap


def support_evidence(surface, xy, profile="vehicle", *, examples=8):
    """Partition all queried points and retain bounded rejected-point evidence.

    Counts refer to queried points, including repetitions; unique mask cells
    are separately reported. Missing height takes precedence over mask denial,
    with their overlap retained. Agree with the runtime gate on every call.
    """
    if not isinstance(surface, SurfaceMap):
        raise ValueError("Measured SurfaceMap required for grid decomposition")
    p = np.asarray(xy, float)
    if p.ndim != 2 or p.shape[1] != 2 or not isinstance(examples, int) or examples < 0:
        raise ValueError("An Nx2 point array and nonnegative example count are required")
    if (
        surface.nx < 2
        or surface.ny < 2
        or surface.z.shape != (surface.ny, surface.nx)
        or surface.walk[profile].shape != surface.z.shape
        or not np.isfinite(surface.cell_m)
        or surface.cell_m <= 0
    ):
        raise ValueError("Invalid support grid")
    finite = np.isfinite(p).all(1)
    safe = np.where(np.isfinite(p), p, 0.0)
    grid = (safe - [surface.x0, surface.y0]) / surface.cell_m
    base = np.floor(grid - 0.5).astype(int)
    inside = finite & (base >= 0).all(1) & (base[:, 0] + 1 < surface.nx) & (base[:, 1] + 1 < surface.ny)
    ix = np.clip(base[:, 0], 0, surface.nx - 2)
    iy = np.clip(base[:, 1], 0, surface.ny - 2)
    heights = np.stack(
        [surface.z[iy, ix], surface.z[iy, ix + 1], surface.z[iy + 1, ix], surface.z[iy + 1, ix + 1]], axis=1
    )
    measured = np.isfinite(heights).all(1)
    cell = np.floor(grid).astype(int)
    cx, cy = np.clip(cell[:, 0], 0, surface.nx - 1), np.clip(cell[:, 1], 0, surface.ny - 1)
    allowed = surface.walk[profile][cy, cx].astype(bool)
    groups = dict(
        nonfinite=~finite,
        outside_height_stencil=finite & ~inside,
        missing_height=inside & ~measured,
        measured_mask_denied=inside & measured & ~allowed,
        supported=inside & measured & allowed,
    )
    if not np.array_equal(groups["supported"], surface.supported(p, profile)):
        raise ValueError("Grid diagnostic differs from production support gate")
    rejected_cells = np.unique(cell[groups["measured_mask_denied"]], axis=0)
    rows = []
    for label, mask in groups.items():
        if label == "supported":
            continue
        for i in np.flatnonzero(mask)[:examples]:
            rows.append(
                dict(
                    point_index=int(i),
                    reason=label,
                    xy=[float(v) if np.isfinite(v) else None for v in p[i]],
                    mask_cell_xy=cell[i].tolist() if inside[i] else None,
                    height_stencil_xy=base[i].tolist() if inside[i] else None,
                    heights_m=[float(v) if np.isfinite(v) else None for v in heights[i]] if inside[i] else None,
                    mask_allowed=bool(allowed[i]) if inside[i] else None,
                )
            )
    return dict(
        schema="awr.life.support_rejection.v1",
        profile=profile,
        points=len(p),
        counts={k: int(v.sum()) for k, v in groups.items()},
        missing_height_and_mask_denied=int((inside & ~measured & ~allowed).sum()),
        unique_measured_mask_denied_cells=len(rejected_cells),
        examples=rows,
        scope="Exact support-grid categories; not a mesh, semantic permission or physical-capacity verdict.",
    )
