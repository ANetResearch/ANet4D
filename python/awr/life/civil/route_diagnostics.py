"""Explain the existing sampled vehicle-route gate without relaxing it.

This is an offline feature/implementation diagnostic. A rejected support grid
probe is not proof that the reconstructed physical terrain is impassable.
"""

import math

import numpy as np


def footprint_evidence(surface, points, yaw, size, *, examples=8, explain_grid=False):
    from .grounded import footprint_support, probe_offsets

    pp = np.asarray(points, float)
    yy = np.asarray(yaw, float)
    if (
        pp.ndim != 2
        or pp.shape[1] != 2
        or yy.shape != (len(pp),)
        or not np.isfinite(pp).all()
        or not np.isfinite(yy).all()
    ):
        raise ValueError("Finite independent poses required")
    offsets = probe_offsets(size, "vehicle")
    valid = footprint_support(surface, pp, np.cos(yy), np.sin(yy), offsets, "vehicle")
    # Record only the failed poses, and cap examples independently of population.
    rows = []
    for i in np.flatnonzero(~valid)[:examples]:
        c, s = math.cos(yy[i]), math.sin(yy[i])
        xy = pp[i] + np.column_stack([offsets[:, 0] * c - offsets[:, 1] * s, offsets[:, 0] * s + offsets[:, 1] * c])
        failed = np.flatnonzero(~surface.supported(xy, "vehicle"))
        rows.append(
            dict(
                pose_index=int(i),
                xy=pp[i].tolist(),
                yaw=float(yy[i]),
                failed_probes=[dict(index=int(k), offset=offsets[k].tolist(), xy=xy[k].tolist()) for k in failed],
            )
        )
    detail = dict(poses=len(pp), failed_poses=int((~valid).sum()), examples=rows)
    if explain_grid:
        from .surface_diagnostics import support_evidence

        c, s = np.cos(yy)[:, None], np.sin(yy)[:, None]
        probes = pp[:, None, :] + np.stack(
            [offsets[:, 0] * c - offsets[:, 1] * s, offsets[:, 0] * s + offsets[:, 1] * c], axis=-1
        )
        detail["grid"] = support_evidence(surface, probes.reshape(-1, 2), examples=examples)
    return valid, detail


def route_evidence(surface, route, size, *, explain_grid=False):
    """Separate endpoint/interpolated support and both current steering checks.

    The result is checked against the production boolean on every call. Drift
    in the duplicated diagnostic decomposition fails explicitly. Native proof
    backends with different footprint semantics require their own explanation.
    """
    from .curb import valid_route

    if hasattr(surface, "vehicle_support_mask") or hasattr(surface, "vehicle_route_supported"):
        raise ValueError("This decomposition requires the ordinary sampled support backend")
    pp, yaw, arc, size = (np.asarray(v, float) for v in (route["points"], route["yaw"], route["arc"], size))
    if (
        len(pp) < 2
        or arc.shape != (len(pp),)
        or size.shape != (3,)
        or not np.isfinite(arc).all()
        or not np.isfinite(size).all()
        or (np.diff(arc) <= 0).any()
        or (size <= 0).any()
    ):
        raise ValueError("Finite positive route distances and vehicle dimensions required")
    valid, poses = footprint_evidence(surface, pp, yaw, size, explain_grid=explain_grid)
    u = np.linspace(0, 1, 8, endpoint=False)[1:]
    delta = np.diff(np.unwrap(yaw))
    mid = (pp[:-1, None, :] + u[None, :, None] * np.diff(pp, axis=0)[:, None, :]).reshape(-1, 2)
    mid_yaw = (yaw[:-1, None] + u[None, :] * delta[:, None]).ravel()
    mid_valid, between = footprint_evidence(surface, mid, mid_yaw, size, explain_grid=explain_grid)
    limit = min(0.26, math.tan(math.radians(35.0)) / (float(size[0]) * 0.65))
    # Preserve raw-yaw actual-arc and unwrapped-yaw / .35 semantics exactly.
    actual = np.abs(np.diff(yaw)) / np.diff(arc)
    sampled = np.abs(delta) / 0.35
    reasons = []
    if not valid.all():
        reasons.append("sampled_pose_support")
    if not mid_valid.all():
        reasons.append("between_pose_support")
    if not (actual < limit).all():
        reasons.append("actual_arc_steering")
    if not (sampled < limit).all():
        reasons.append("legacy_spacing_steering")
    expected = valid_route(surface, route, size)
    if bool(expected) != (not reasons):
        raise ValueError("Diagnostic decomposition differs from production route gate")
    return dict(
        valid=bool(expected),
        reasons=reasons,
        endpoints_supported=[bool(valid[0]), bool(valid[-1])],
        pose_support=poses,
        between_pose_support=between,
        steering=dict(
            limit_per_m=limit,
            max_actual_per_m=float(actual.max()),
            max_legacy_spacing_per_m=float(sampled.max()),
            actual_failed_segments=np.flatnonzero(actual >= limit).tolist(),
            legacy_failed_segments=np.flatnonzero(sampled >= limit).tolist(),
        ),
        scope="Existing sampled support/steering gate only; no permission, occupancy, "
        "mesh-ground physical impossibility, executable trip or release claim.",
    )
