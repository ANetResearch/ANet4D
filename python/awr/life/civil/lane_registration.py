"""Lateral registration of SUMO lane geometry to the measured carriageway.

The SUMO network is derived from OpenStreetMap centrelines, which are one to three metres
off the photogrammetric road surface on many streets of this town; a lane drawn along a
kerb or a retaining wall fails every honest support test although the road beside it is
flat and measured. This module shifts each road group (forward and backward edges of the
same way) laterally by a slowly varying offset chosen from the measured ground alone:

* at every 0.5 m sample of the group's reference line, candidate offsets in
  ``[-max_shift_m, max_shift_m]`` are scored by the cross-section of measured heights over
  the group's full carriageway width: known coverage, residual to a straight cross
  profile and cross slope;
* dynamic programming along the line selects the offset sequence with the best total
  score under a smoothness penalty, so the lane never jumps between two levels;
* lanes of the group move together; internal junction connectors are re-shaped by
  interpolating the end shifts of the lanes they join.

Scores use the measured v1 ground only (no fit, no inference), and every shift is
recorded, so the registration is an audited geometric correction, not new ground truth.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString

SPACING = 0.5


def group_key(edge_id):
    """Forward and backward edges of one way share a registration."""
    m = re.match(r"^(e-\d+-\d+)-[fb]$", edge_id)
    return m.group(1) if m else edge_id


def section_score(z_row, step, cross_limit=0.08):
    """Score of one cross-section (heights at regular lateral steps, NaN unknown)."""
    known = np.isfinite(z_row)
    share = known.mean()
    if known.sum() < 3:
        return share - 1.0
    x = np.arange(len(z_row)) * step
    a, b = np.polyfit(x[known], z_row[known], 1)
    resid = z_row[known] - (a * x[known] + b)
    rms = float(np.sqrt(np.mean(resid**2)))
    return share - 2.0 * rms - 2.0 * max(0.0, abs(a) - cross_limit)


def register_lanes(
    net,
    Z,
    x0,
    y0,
    cell,
    *,
    max_shift_m=2.5,
    shift_step_m=0.25,
    smooth_penalty=0.12,
    prior_penalty=0.04,
    smooth_sigma_m=2.0,
    section_step_m=0.25,
    margin_m=0.25,
):
    """Return (shifted_points, audit) for every lane of ``net``.

    ``Z`` is the measured ground raster (NaN unknown) with origin ``x0, y0`` and ``cell``.
    """
    ny, nx = Z.shape

    def heights(p):
        gx = np.floor((p[:, 0] - x0) / cell).astype(int)
        gy = np.floor((p[:, 1] - y0) / cell).astype(int)
        inside = (gx >= 0) & (gy >= 0) & (gx < nx) & (gy < ny)
        out = np.full(len(p), np.nan)
        out[inside] = Z[gy[inside], gx[inside]]
        return out

    # Each lane is registered on its own width: on many streets of this town the measured
    # flat carriageway is one lane wide although the map carries two directions, and a
    # two-lane section can then never be flat. Overlaps between registered lanes of one
    # way are detected afterwards (single-track streets).
    groups = {}
    for name, lane in net.lanes.items():
        if lane["internal"]:
            continue
        groups[name] = [name]
    offsets = np.arange(-max_shift_m, max_shift_m + 1e-9, shift_step_m)
    shifted = {}
    audit = {}
    delta_at_end = {}
    delta_at_start = {}
    for key, names in groups.items():
        # Reference line: the longest lane; lateral positions of the others relative to it.
        ref_name = max(names, key=lambda n: LineString(net.lanes[n]["points"]).length)
        ref = LineString(net.lanes[ref_name]["points"])
        if ref.length < 2 * SPACING:
            for n in names:
                shifted[n] = np.asarray(net.lanes[n]["points"], float)
            continue
        ds = np.arange(0, ref.length + 1e-9, SPACING)
        centre = shapely.get_coordinates(shapely.line_interpolate_point(ref, ds))
        tangent = np.gradient(centre, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])  # left of travel
        lateral = {}
        for n in names:
            pts = np.asarray(net.lanes[n]["points"], float)
            proj = np.array([ref.project(shapely.Point(p)) for p in pts])
            on_ref = shapely.get_coordinates(shapely.line_interpolate_point(ref, proj))
            k = np.clip(np.round(proj / SPACING).astype(int), 0, len(ds) - 1)
            lateral[n] = float(np.median(np.einsum("ij,ij->i", pts - on_ref, normal[k])))
        widths = {n: float(net.lanes[n]["width"]) for n in names}
        lo = min(lateral[n] - widths[n] / 2 for n in names) - margin_m
        hi = max(lateral[n] + widths[n] / 2 for n in names) + margin_m
        section = np.arange(lo, hi + 1e-9, section_step_m)
        # Score matrix: samples x offsets.
        score = np.zeros((len(ds), len(offsets)))
        for j, d in enumerate(offsets):
            pts = centre[:, None, :] + normal[:, None, :] * (section + d)[None, :, None]
            z = heights(pts.reshape(-1, 2)).reshape(len(ds), len(section))
            for i in range(len(ds)):
                score[i, j] = section_score(z[i], section_step_m)
        # The mapped geometry is the prior: a shift must be paid for by section evidence
        # (quadratic penalty), so flat ground beside the street does not attract the lane.
        score = score - prior_penalty * offsets[None, :] ** 2
        # Dynamic programming with a smoothness penalty per step of offset change.
        cost = np.full((len(ds), len(offsets)), np.inf)
        back = np.zeros((len(ds), len(offsets)), int)
        cost[0] = -score[0]
        for i in range(1, len(ds)):
            for j in range(len(offsets)):
                prev = cost[i - 1] + smooth_penalty * np.abs(np.arange(len(offsets)) - j)
                k = int(np.argmin(prev))
                cost[i, j] = prev[k] - score[i, j]
                back[i, j] = k
        path = np.zeros(len(ds), int)
        path[-1] = int(np.argmin(cost[-1]))
        for i in range(len(ds) - 1, 0, -1):
            path[i - 1] = back[i, path[i]]
        delta = offsets[path]
        # Gaussian smoothing removes the quantisation steps; a 0.25 m step over 0.5 m
        # would otherwise be a 26 degree kink that fails the curvature limit.
        radius = int(np.ceil(3 * smooth_sigma_m / SPACING))
        kx = np.arange(-radius, radius + 1) * SPACING
        kernel = np.exp(-0.5 * (kx / smooth_sigma_m) ** 2)
        kernel /= kernel.sum()
        delta_s = np.convolve(np.pad(delta, radius, mode="edge"), kernel, mode="valid")
        score_before = float(np.mean(score[:, len(offsets) // 2]))
        score_after = float(np.mean(score[np.arange(len(ds)), path]))
        for n in names:
            pts = np.asarray(net.lanes[n]["points"], float)
            proj = np.array([ref.project(shapely.Point(p)) for p in pts])
            k = np.clip(np.round(proj / SPACING).astype(int), 0, len(ds) - 1)
            # Densify the lane so the shift follows the reference sample by sample.
            line = LineString(pts)
            dd = np.unique(np.r_[np.arange(0, line.length, SPACING), line.length])
            dense = shapely.get_coordinates(shapely.line_interpolate_point(line, dd))
            projd = np.array([ref.project(shapely.Point(p)) for p in dense])
            kd = np.clip(np.round(projd / SPACING).astype(int), 0, len(ds) - 1)
            new = dense + normal[kd] * delta_s[kd][:, None]
            shifted[n] = new
            delta_at_start[n] = new[0] - dense[0]
            delta_at_end[n] = new[-1] - dense[-1]
        audit[key] = dict(
            lanes=names,
            reference=ref_name,
            length_m=round(float(ref.length), 1),
            carriageway_span_m=[round(lo, 2), round(hi, 2)],
            shift_m=dict(
                min=round(float(delta_s.min()), 2),
                max=round(float(delta_s.max()), 2),
                mean=round(float(delta_s.mean()), 2),
                abs_mean=round(float(np.abs(delta_s).mean()), 2),
            ),
            score_before=round(score_before, 3),
            score_after=round(score_after, 3),
        )
    # Internal connectors: interpolate the end shift of the lane they leave and the start
    # shift of the lane they enter. Chains of connectors converge over a few passes.
    predecessors = {}
    for u, vs in net.links.items():
        for v in vs:
            predecessors.setdefault(v, []).append(u)
    internal = [n for n, l in net.lanes.items() if l["internal"]]
    for _ in range(4):
        for n in internal:
            pts = np.asarray(net.lanes[n]["points"], float)
            pre = [u for u in predecessors.get(n, []) if u in delta_at_end]
            suc = [w for w in net.links.get(n, []) if w in delta_at_start]
            d0 = np.mean([delta_at_end[u] for u in pre], 0) if pre else None
            d1 = np.mean([delta_at_start[w] for w in suc], 0) if suc else None
            if d0 is None and d1 is None:
                continue
            d0 = d1 if d0 is None else d0
            d1 = d0 if d1 is None else d1
            # Densify the connector and blend the two end shifts with a smooth step so
            # that the join to each lane keeps its tangent.
            line = LineString(pts)
            dd = (
                np.unique(np.r_[np.arange(0, line.length, SPACING), line.length])
                if line.length > SPACING
                else np.array([0.0, line.length])
            )
            pts = shapely.get_coordinates(shapely.line_interpolate_point(line, dd)) if line.length > 0 else pts
            seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
            u = seg / seg[-1] if seg[-1] > 0 else np.zeros(len(pts))
            u = u * u * (3 - 2 * u)
            new = pts + (1 - u)[:, None] * d0 + u[:, None] * d1
            shifted[n] = new
            delta_at_start[n] = new[0] - pts[0]
            delta_at_end[n] = new[-1] - pts[-1]
    for n in net.lanes:
        shifted.setdefault(n, np.asarray(net.lanes[n]["points"], float))
    # Single-track detection: registered lanes of the same way closer than a vehicle
    # body plus margin share one physical carriageway. The lane with the weaker section
    # evidence is excluded when more than half of it overlaps; shorter overlaps are
    # recorded and left to the collision model.
    ways = {}
    for n, lane in net.lanes.items():
        if not lane["internal"]:
            ways.setdefault(group_key(lane["edge"]), []).append(n)
    excluded = {}
    overlaps = []
    for way, names in ways.items():
        for i, a_ in enumerate(names):
            for b_ in names[i + 1 :]:
                la, lb = LineString(shifted[a_]), LineString(shifted[b_])
                if la.length < 2 or lb.length < 2:
                    continue
                da = np.array(
                    [
                        lb.distance(shapely.Point(p))
                        for p in shapely.get_coordinates(
                            shapely.line_interpolate_point(la, np.arange(0, la.length, 2.0))
                        )
                    ]
                )
                share = float((da < 2.4).mean())
                if share <= 0.0:
                    continue
                weaker = min((a_, b_), key=lambda n: audit.get(n, {}).get("score_after", 0.0))
                overlaps.append(dict(way=way, lanes=[a_, b_], overlap_share=round(share, 3), weaker=weaker))
                if share > 0.5 and weaker not in excluded:
                    excluded[weaker] = dict(
                        reason="single_track_overlap",
                        with_lane=b_ if weaker == a_ else a_,
                        overlap_share=round(share, 3),
                    )
    audit["_overlaps"] = overlaps
    audit["_excluded"] = excluded
    return shifted, audit


def write_registration(path, shifted, audit, net_path, parameters, ground_sha256):
    groups = {k: v for k, v in audit.items() if not k.startswith("_")}
    doc = dict(
        schema="awr.life.lane_registration.v1",
        net=str(net_path),
        ground_sha256=ground_sha256,
        parameters=parameters,
        lanes={n: np.round(p, 3).tolist() for n, p in shifted.items()},
        groups=groups,
        excluded_lanes=audit.get("_excluded", {}),
        overlaps=audit.get("_overlaps", []),
    )
    Path(path).write_text(json.dumps(doc) + "\n")
    return doc
