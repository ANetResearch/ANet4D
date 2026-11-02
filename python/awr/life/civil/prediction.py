"""Batched equivalents of route interpolation and full-body lookahead collision."""

import numpy as np

from awr.life.civil.collision import obb_overlap


def route_at_many(route, distances):
    pp = np.asarray(route["points"])
    d = np.asarray(distances)
    if "arc" in route:
        arc = route["arc"]
        d = np.clip(d, 0, arc[-1])
        k = np.clip(np.searchsorted(arc, d, side="right") - 1, 0, len(pp) - 2)
        u = (d - arc[k]) / (arc[k + 1] - arc[k])
        delta = pp[k + 1] - pp[k]
        yaw = np.arctan2(delta[:, 1], delta[:, 0])
        if route.get("smooth_vehicle_heading"):
            yaw = np.interp(d, arc, route["yaw"])
    else:
        q = np.clip(d / 0.35, 0, len(pp) - 1)
        k = np.minimum(q.astype(int), len(pp) - 2)
        u = q - k
        delta = pp[k + 1] - pp[k]
        yaw = np.arctan2(delta[:, 1], delta[:, 0])
        if "yaw" in route:
            y = route["yaw"]
            yaw = y[k] + u * ((y[k + 1] - y[k] + np.pi) % (2 * np.pi) - np.pi)
    return pp[k] + u[:, None] * delta, yaw


def blocked_many(actor, points, angles, body, positions, yaws, sizes, tree, radii):
    points = np.asarray(points)
    # This enclosing ball contains every individual query ball; exact OBB SAT
    # is still applied to each pose/body pair. No sample is dropped.
    r = float(np.linalg.norm(points - points[0], axis=1).max() + np.linalg.norm(body[:2]) / 2 + radii.max())
    ids = np.asarray(tree.query_ball_point(points[0], r), int)
    ids = ids[ids != actor]
    if not len(ids):
        return np.zeros(len(points), bool)
    return obb_overlap(
        points[:, None, :],
        angles[:, None],
        body,
        positions[ids][None, :, :],
        yaws[ids][None, :],
        sizes[ids][None, :, :],
    ).any(1)


def blocked_disks(actor, points, radius, positions, yaws, sizes, tree, radii, circle_radii):
    """One rotating-body disk per pedestrian, exact boxes for other actors."""
    points = np.asarray(points)
    search = float(np.linalg.norm(points - points[0], axis=1).max() + radius + radii.max() + 0.08)
    ids = np.asarray(tree.query_ball_point(points[0], search), int)
    ids = ids[ids != actor]
    if not len(ids):
        return np.zeros(len(points), bool)
    delta = points[:, None] - positions[ids]
    c, s = np.cos(yaws[ids]), np.sin(yaws[ids])
    local = np.abs(np.stack([delta[..., 0] * c + delta[..., 1] * s, -delta[..., 0] * s + delta[..., 1] * c], -1))
    clearance = np.linalg.norm(np.maximum(local - sizes[ids, :2] / 2, 0), axis=-1)
    circles = np.asarray(circle_radii)[ids]
    rounded = np.isfinite(circles)
    clearance[:, rounded] = np.linalg.norm(delta[:, rounded], axis=-1) - circles[rounded]
    return (clearance <= radius + 0.08).any(1)


def blocked_ahead(
    actor,
    points,
    angles,
    body,
    positions,
    yaws,
    sizes,
    tree,
    radii,
    still=None,
    still_body=None,
    return_blocker=False,
    headings=None,
    starting=False,
):
    """Like blocked_many, but a body that already overlaps the (inflated) own body
    at the current pose and lies behind it does not block forward motion: moving
    on only increases that separation. Otherwise a walker followed closely by
    another could never move away from the follower.

    `still` (bool per actor) with `still_body` (own size for them): bodies at rest
    are tested against `still_body` instead of the inflated `body`. The inflation
    covers a neighbour's turning step during the tick; a neighbour at rest turns
    only after checking moving bodies itself, so the walker may pass it as
    closely as planning allows. With `return_blocker` the actor index of a body hit
    at the first blocked sample (or -1) is returned as well.

    `headings` = (centres, angles), arrays of shape (k, len(points), 2) and
    (k, len(points)): further poses of the own body at each sample (NaN angle where
    there is none). A sample is blocked if the body overlaps at any of its poses.

    `starting` (with `still`): the own body is at rest and about to start. Only
    bodies at rest are then exempt behind it: its walking pose reaches further back
    than its rest pose, and a body passing behind it relies on it staying at rest
    (see `still_body`). A moving follower brakes for the resting body ahead and is
    exempt once it stands."""
    points = np.asarray(points)
    extra = None
    if headings is not None:
        sel = ~np.isnan(headings[1])
        if sel.any():
            extra = (np.nonzero(sel)[1], np.asarray(headings[0])[sel], np.asarray(headings[1])[sel])
    reach = points if extra is None else np.vstack([points, extra[1]])
    r = float(np.linalg.norm(reach - points[0], axis=1).max() + np.linalg.norm(body[:2]) / 2 + radii.max())
    ids = np.asarray(tree.query_ball_point(points[0], r), int)
    ids = ids[ids != actor]
    if not len(ids):
        return (np.zeros(len(points), bool), -1) if return_blocker else np.zeros(len(points), bool)
    rest = np.asarray(still)[ids] if still is not None and still_body is not None else np.zeros(len(ids), bool)

    def overlap(centres, angs):
        h = obb_overlap(
            centres[:, None, :],
            angs[:, None],
            body,
            positions[ids][None, :, :],
            yaws[ids][None, :],
            sizes[ids][None, :, :],
        )
        if rest.any():
            h[:, rest] = obb_overlap(
                centres[:, None, :],
                angs[:, None],
                still_body,
                positions[ids[rest]][None, :, :],
                yaws[ids[rest]][None, :],
                sizes[ids[rest]][None, :, :],
            )
        return h

    hit = overlap(points, np.asarray(angles))
    if extra is not None:
        np.logical_or.at(hit, extra[0], overlap(extra[1], extra[2]))
    heading = np.array([np.cos(angles[0]), np.sin(angles[0])])
    behind = ((positions[ids] - points[0]) @ heading) < 0
    if starting and still is not None:
        behind &= np.asarray(still)[ids]
    hit[:, hit[0] & behind] = False
    blocked = hit.any(1)
    if not return_blocker:
        return blocked
    first = np.flatnonzero(blocked)
    return blocked, (int(ids[np.flatnonzero(hit[first[0]])[0]]) if len(first) else -1)
