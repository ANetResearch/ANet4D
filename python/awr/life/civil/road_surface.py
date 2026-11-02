"""Carriageway surface reconstruction from measured photogrammetric ground.

Two estimators share this module:

* :func:`plane_fit` - a weighted local plane per raster cell over a square window, used
  for the robust (Tukey IRLS) corridor fit; weights are zero outside the corridor so that
  kerbs, walls and gardens never enter the fit.
* :func:`lane_profile` - a longitudinal road profile along one lane polyline: robust
  smoothing of the available heights, bounded gap interpolation across unmeasured
  stretches (tree canopy, baked objects), longitudinal grade and cross slope.

Both are deterministic functions of their inputs and carry no actor or tape data.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def window_sums(weights, values, size):
    """Sum of weights*values over a size x size window (zero outside the raster)."""
    return uniform_filter((weights * values).astype(np.float64), size, mode="constant") * size * size


def plane_fit(w, Z, cell, size):
    """Weighted local plane per cell: fitted height, slope magnitude and total weight."""
    ny, nx = Z.shape
    yy, xx = np.mgrid[:ny, :nx]
    xx = (xx * cell).astype(np.float64)
    yy = (yy * cell).astype(np.float64)
    Zf = np.where(w > 0, Z, 0.0).astype(np.float64)
    n = window_sums(w, 1.0, size)
    sx = window_sums(w, xx, size)
    sy = window_sums(w, yy, size)
    sz = window_sums(w, Zf, size)
    sxx = window_sums(w, xx * xx, size)
    syy = window_sums(w, yy * yy, size)
    sxy = window_sums(w, xx * yy, size)
    sxz = window_sums(w, xx * Zf, size)
    syz = window_sums(w, yy * Zf, size)
    with np.errstate(divide="ignore", invalid="ignore"):
        mx, my, mz = sx / n, sy / n, sz / n
        cxx = sxx / n - mx * mx
        cyy = syy / n - my * my
        cxy = sxy / n - mx * my
        cxz = sxz / n - mx * mz
        cyz = syz / n - my * mz
        det = cxx * cyy - cxy * cxy
        a = (cxz * cyy - cyz * cxy) / det
        b = (cyz * cxx - cxz * cxy) / det
        fitted = mz + a * (xx - mx) + b * (yy - my)
        slope = np.hypot(a, b)
    bad = ~np.isfinite(fitted) | ~np.isfinite(slope) | (n < 1e-9) | (np.abs(det) < 1e-6)
    fitted = np.where(bad & (n > 1e-9), mz, fitted)
    slope = np.where(bad, np.nan, slope)
    fitted = np.where(n > 1e-9, fitted, np.nan)
    return fitted.astype(np.float32), slope.astype(np.float32), n


def running_median(values, half):
    """Median over a +-half sample window ignoring NaN; NaN where nothing is available."""
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        seg = values[max(0, i - half) : i + half + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = np.median(seg)
    return out


def running_linear(values, half, outlier_m=0.4):
    """Local least-squares line per sample over a +-half window, after rejecting samples
    that deviate from the running median by more than ``outlier_m``.

    A running median alone is biased at the edge of a gap or of the profile (the window
    becomes one-sided on a ramp); a local line is not. NaN where the sample itself has no
    height or fewer than three neighbours remain.
    """
    med = running_median(values, half)
    clean = np.where(np.isfinite(values) & np.isfinite(med) & (np.abs(values - med) <= outlier_m), values, np.nan)
    out = np.full(len(values), np.nan)
    x = np.arange(-half, half + 1, dtype=float)
    for i in range(len(values)):
        if not np.isfinite(values[i]):
            continue
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        seg = clean[lo:hi]
        m = np.isfinite(seg)
        if m.sum() >= 3:
            xs = x[half - (i - lo) : half + (hi - i)][m]
            a, b = np.polyfit(xs, seg[m], 1)
            out[i] = b
        elif m.sum() > 0:
            out[i] = np.nanmean(seg)
    return out


def fill_gaps(values, spacing, max_gap_m):
    """Linear interpolation across interior NaN runs no longer than ``max_gap_m``.

    Returns the filled array and a boolean mask of the filled samples. Runs at either
    end of the profile (no measured section on one side) stay NaN.
    """
    v = values.copy()
    filled = np.zeros(len(v), bool)
    finite = np.isfinite(v)
    if finite.sum() < 2:
        return v, filled
    idx = np.flatnonzero(finite)
    for a, b in zip(idx[:-1], idx[1:], strict=False):
        if b - a <= 1:
            continue
        if (b - a) * spacing > max_gap_m:
            continue
        u = (np.arange(a + 1, b) - a) / (b - a)
        v[a + 1 : b] = v[a] + u * (v[b] - v[a])
        filled[a + 1 : b] = True
    return v, filled


def lane_profile(z_centre, z_left, z_right, lateral_m, spacing, *, median_half=5, max_gap_m=40.0, grade_half=3):
    """Longitudinal profile, grade and cross slope of one lane from sampled heights.

    ``z_centre``, ``z_left`` and ``z_right`` are heights (NaN where unknown) at the lane
    centre and at ``+-lateral_m``; ``spacing`` is the sample spacing in metres.
    Returns a dict with ``z`` (profile, NaN where unknown), ``filled`` (interpolated
    samples), ``grade`` (central difference over ``+-grade_half`` samples, NaN where the
    profile is unknown) and ``cross`` (cross slope, zero where unknown).
    """
    stack = np.vstack([z_centre, z_left, z_right])
    with np.errstate(all="ignore"):
        centre = np.where(np.isfinite(stack).any(0), np.nanmedian(stack, 0), np.nan)
    # Smoothing never extends a measured section into a gap: the median is kept only
    # where the sample itself has a height, so gaps are interpolated (and flagged)
    # rather than flat-extrapolated from their edges.
    smooth = running_linear(centre, median_half)
    z, filled = fill_gaps(smooth, spacing, max_gap_m)
    grade = np.full(len(z), np.nan)
    finite = np.isfinite(z)
    h = grade_half
    for i in range(len(z)):
        a, b = max(0, i - h), min(len(z) - 1, i + h)
        if b > a and finite[a] and finite[b]:
            grade[i] = (z[b] - z[a]) / ((b - a) * spacing)
    # Cross slope from robust left/right profiles (outliers such as a baked car on one
    # track are rejected before differencing), then smoothed; zero where unknown.
    left = running_linear(np.asarray(z_left, float), median_half)
    right = running_linear(np.asarray(z_right, float), median_half)
    cross_raw = (right - left) / (2 * lateral_m)
    cross = running_median(cross_raw, median_half)
    cross = np.where(np.isfinite(cross), cross, 0.0)
    return dict(z=z, filled=filled, grade=grade, cross=cross)
