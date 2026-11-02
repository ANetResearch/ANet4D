"""Progressive terrain hint, adapted from upstream bc0c43b groundfield.py.

Unlike that module's final z_actor substitution, this output is ONLY a hint for
an independent mesh intersection. Missing ground remains missing after raycast.
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator


def terrain_hint(top):
    finite = np.isfinite(top)
    if finite.sum() < 3:
        return np.full_like(top, np.nan, dtype=float)
    nearest = ndimage.distance_transform_edt(~finite, return_distances=False, return_indices=True)
    surface = np.asarray(top, float)[tuple(nearest)]
    ground = finite.copy()
    previous = 1
    for width in (3, 5, 9, 17, 33, 65):
        opened = ndimage.grey_opening(surface, size=(width, width), mode="nearest")
        ground &= surface - opened <= min(10.0, 0.5 + 0.15 * (width - previous))
        surface = opened
        previous = width
    y, x = np.nonzero(ground)
    if len(x) < 3:
        return np.full_like(top, np.nan, dtype=float)
    qy, qx = np.indices(top.shape)
    try:
        inferred = LinearNDInterpolator(np.column_stack([x, y]), top[y, x])(
            np.column_stack([qx.ravel(), qy.ravel()])
        ).reshape(top.shape)
    except Exception as error:
        # Degenerate observations are not a reason to fabricate a plane.
        from scipy.spatial import QhullError

        if isinstance(error, QhullError):
            return np.where(ground, top, np.nan)
        raise
    return np.where(ground, top, inferred)
